"""Graph-based workflow orchestration for ADK.

GraphAgent is ADK's fourth workflow agent type (alongside Sequential, Loop, Parallel),
enabling directed graph-based orchestration with conditional routing and complex branching.

GraphAgent enables workflow creation using directed graphs where:
- Nodes are agents or functions
- Edges define allowed transitions with optional conditions
- State flows through the graph with configurable reducers
- Full checkpointing support via CheckpointService integration

Key features:
- Directed graph workflows with conditional routing
- State management with custom reducers (OVERWRITE, APPEND, SUM, CUSTOM)
- CheckpointService integration for checkpoint/resume
- Human-in-the-loop interrupts
- DatabaseSessionService support for persistence
- Cyclic execution with max_iterations
- Event-based state persistence (ADK-native)

Inspired by adk-graph (Rust) and LangGraph patterns.

Checkpointing Integration:
    For checkpoint/resume functionality, use CheckpointService with CheckpointCallback:

    ```python
    from google.adk.agents.graph import GraphAgent
    from google.adk.checkpoints import CheckpointService, CheckpointCallback
    from google.adk.sessions import InMemorySessionService

    # Create services
    session_service = InMemorySessionService()
    checkpoint_service = CheckpointService(session_service)

    # Create graph with checkpoint callback
    graph = GraphAgent(name="workflow", checkpointing=True)
    graph.add_node(...)
    graph.set_callbacks([
        CheckpointCallback(checkpoint_service, checkpoint_after=True)
    ])

    # Checkpoints are created automatically after each node
    # Use checkpoint_service to list/delete/export/import checkpoints
    ```
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from google.genai import types
from pydantic import ConfigDict, Field
from typing_extensions import override

from ...events.event import Event
from ...events.event_actions import EventActions
from ...utils.feature_decorator import experimental
from ..base_agent import BaseAgent
from ..invocation_context import InvocationContext
from ..llm_agent import LlmAgent
from .graph_edge import EdgeCondition
from .graph_node import GraphNode
from .graph_state import GraphState, StateReducer
from .interrupt import InterruptMode
from .interrupt_service import InterruptService

logger = logging.getLogger("google_adk." + __name__)

# Sentinel constants for graph boundaries
START = "__start__"
END = "__end__"


@experimental
class GraphAgent(BaseAgent):  # type: ignore[misc]
    """Graph-based workflow agent for ADK.

    GraphAgent is the fourth workflow agent type in ADK (alongside SequentialAgent,
    LoopAgent, and ParallelAgent), enabling directed graph-based orchestration with
    conditional routing, state management, and full checkpointing support.

    Workflow agents control execution flow through deterministic logic rather than LLM
    reasoning, providing predictable, reliable, and structured agent orchestration.

    Features:
    - Directed graph workflow with nodes (agents/functions) and edges
    - Conditional routing based on state predicates
    - Cyclic execution support (loops, iterative refinement, ReAct pattern)
    - Human-in-the-loop interrupts (BEFORE/AFTER/BOTH)
    - CheckpointService integration for state persistence
    - DatabaseSessionService support for persistence
    - Full ADK event system integration

    Example:
        >>> from google.adk.agents.graph import GraphAgent, GraphNode, START, END
        >>> from google.adk.agents import LlmAgent
        >>> from google.adk.checkpoints import CheckpointService, CheckpointCallback
        >>> from google.adk.runners import Runner
        >>>
        >>> # Create a simple workflow
        >>> graph = GraphAgent(name="workflow", checkpointing=True)
        >>> graph.add_node(GraphNode(name="analyze", agent=LlmAgent(...)))
        >>> graph.add_node(GraphNode(name="process", agent=LlmAgent(...)))
        >>> graph.add_edge("analyze", "process")
        >>> graph.set_start("analyze")
        >>> graph.set_end("process")
        >>>
        >>> # Add checkpoint callback for automatic checkpointing
        >>> checkpoint_service = CheckpointService(session_service)
        >>> graph.set_callbacks([CheckpointCallback(checkpoint_service)])
        >>>
        >>> # Run with checkpointing
        >>> runner = Runner(app_name="app", agent=graph)
        >>> async for event in runner.run_async(...):
        ...     print(event)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    nodes: Dict[str, GraphNode] = Field(default_factory=dict)
    start_node: Optional[str] = None
    end_nodes: List[str] = Field(default_factory=list)
    max_iterations: int = 50  # Prevent infinite loops
    interrupt_before: List[str] = Field(default_factory=list)
    interrupt_after: List[str] = Field(default_factory=list)
    checkpointing: bool = False
    interrupt_service: Optional[InterruptService] = Field(
        default=None,
        description="Optional InterruptService for dynamic runtime interrupts",
    )

    def __init__(  # type: ignore[no-untyped-def]
        self,
        name: str,
        description: str = "",
        max_iterations: int = 50,
        checkpointing: bool = False,
        interrupt_service: Optional[InterruptService] = None,
        **kwargs,
    ):
        """Initialize GraphAgent.

        Args:
            name: Agent name
            description: Agent description
            max_iterations: Max iterations to prevent infinite loops
            checkpointing: Enable state checkpointing after each node
                Note: For full checkpoint/resume, use CheckpointCallback
            interrupt_service: Optional InterruptService for dynamic runtime interrupts
        """
        super().__init__(name=name, description=description, **kwargs)
        self.nodes = {}
        self.start_node = None
        self.end_nodes = []
        self.max_iterations = max_iterations
        self.interrupt_service = interrupt_service
        self.interrupt_before = []
        self.interrupt_after = []
        self.checkpointing = checkpointing

    def add_node(self, node: GraphNode) -> "GraphAgent":
        """Add a node to the graph.

        Args:
            node: GraphNode to add

        Returns:
            Self for chaining
        """
        self.nodes[node.name] = node
        return self

    def set_start(self, node_name: str) -> "GraphAgent":
        """Set the starting node.

        Args:
            node_name: Name of the start node

        Returns:
            Self for chaining

        Raises:
            ValueError: If node not found in graph
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} not found in graph")
        self.start_node = node_name
        return self

    def set_end(self, node_name: str) -> "GraphAgent":
        """Mark a node as an end node.

        Args:
            node_name: Name of the end node

        Returns:
            Self for chaining

        Raises:
            ValueError: If node not found in graph
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} not found in graph")
        if node_name not in self.end_nodes:
            self.end_nodes.append(node_name)
        return self

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        condition: Optional[Callable[[GraphState], bool]] = None,
    ) -> "GraphAgent":
        """Add an edge between nodes.

        Args:
            from_node: Source node name
            to_node: Target node name
            condition: Optional condition for conditional routing

        Returns:
            Self for chaining

        Raises:
            ValueError: If nodes not found in graph
        """
        if from_node not in self.nodes:
            raise ValueError(f"Source node {from_node} not found")
        if to_node not in self.nodes:
            raise ValueError(f"Target node {to_node} not found")

        self.nodes[from_node].add_edge(to_node, condition)
        return self

    def add_interrupt(
        self, node_name: str, mode: InterruptMode = InterruptMode.BEFORE
    ) -> "GraphAgent":
        """Add human-in-the-loop interrupt point.

        Args:
            node_name: Node to add interrupt to
            mode: When to interrupt (BEFORE, AFTER, or BOTH)

        Returns:
            Self for chaining

        Raises:
            ValueError: If node not found in graph
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} not found")

        if mode in (InterruptMode.BEFORE, InterruptMode.BOTH):
            if node_name not in self.interrupt_before:
                self.interrupt_before.append(node_name)

        if mode in (InterruptMode.AFTER, InterruptMode.BOTH):
            if node_name not in self.interrupt_after:
                self.interrupt_after.append(node_name)

        return self

    def export_graph_structure(self) -> Dict[str, Any]:
        """Export graph structure in D3-compatible JSON format.

        Returns a dictionary containing nodes and links suitable for
        visualization with D3.js or other graph visualization tools.

        Returns:
            Dictionary with structure:
            {
                "nodes": [{"id": "node1", "type": "agent"}, ...],
                "links": [{"source": "node1", "target": "node2"}, ...],
                "metadata": {
                    "start_node": "node1",
                    "end_nodes": ["node3"],
                    "checkpointing": True
                }
            }

        Example:
            >>> graph = GraphAgent(name="workflow")
            >>> graph.add_node(GraphNode(name="start", ...))
            >>> graph.add_node(GraphNode(name="process", ...))
            >>> graph.add_edge("start", "process")
            >>> structure = graph.export_graph_structure()
            >>> # Use structure with D3.js for visualization
        """
        nodes = []
        links = []

        # Export nodes
        for node_name, node in self.nodes.items():
            node_data = {
                "id": node_name,
                "type": "agent" if node.agent else "function",
                "name": node.name,
            }
            nodes.append(node_data)

        # Export edges
        for node_name, node in self.nodes.items():
            for edge in node.edges:
                link_data = {
                    "source": node_name,
                    "target": edge.target_node,
                    "conditional": edge.has_condition,
                }
                links.append(link_data)

        # Export metadata
        metadata = {
            "start_node": self.start_node,
            "end_nodes": self.end_nodes,
            "checkpointing": self.checkpointing,
            "max_iterations": self.max_iterations,
            "interrupt_before": self.interrupt_before,
            "interrupt_after": self.interrupt_after,
        }

        return {"nodes": nodes, "links": links, "metadata": metadata, "directed": True}

    async def _execute_node(  # type: ignore[no-untyped-def]
        self, node: GraphNode, state: GraphState, ctx: InvocationContext
    ):
        """Execute a single node and yield events, returns output via state update.

        Args:
            node: GraphNode to execute
            state: Current graph state
            ctx: Invocation context

        Yields:
            Events from node execution
        """
        # Map state to node input
        node_input = node.input_mapper(state)

        # Execute node (agent or function)
        output = ""
        if node.agent:
            # Create new context with updated user_content for this node
            node_content = types.Content(
                role="user", parts=[types.Part(text=node_input)]
            )
            node_ctx = ctx.model_copy(update={"user_content": node_content})

            # Execute ADK agent with updated context
            async for event in node.agent.run_async(node_ctx):
                # Extract output from final response
                if event.content and event.content.parts:
                    output = event.content.parts[0].text or ""
                yield event
        elif node.function:
            # Execute custom function
            if asyncio.iscoroutinefunction(node.function):
                output = await node.function(state, ctx)
            else:
                output = node.function(state, ctx)
        else:  # pragma: no cover
            # Defensive: This should never happen due to GraphNode validation
            raise ValueError(f"Node {node.name} has no agent or function")

        # Store output in metadata for retrieval
        state.metadata["_last_output"] = output

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Core graph execution logic.

        Executes nodes in graph order, following conditional edges,
        supporting loops and human-in-the-loop interrupts.

        Args:
            ctx: Invocation context

        Yields:
            Events from graph execution

        Raises:
            ValueError: If start node not set or graph structure invalid
        """
        if not self.start_node:
            raise ValueError("Start node not set. Call set_start() first.")

        # Register session with InterruptService if enabled
        if self.interrupt_service:
            self.interrupt_service.register_session(ctx.session.id)

        try:
            # Initialize state from session or create new
            state_dict = ctx.session.state.get("graph_state")
            if state_dict is None:
                # Extract text from Content object (ADK 1.22.1)
                user_text = ""
                if (
                    hasattr(ctx, "user_content")
                    and ctx.user_content
                    and ctx.user_content.parts
                ):
                    user_text = (
                        ctx.user_content.parts[0].text
                        if ctx.user_content.parts[0].text
                        else ""
                    )

                state = GraphState(
                    data={"input": user_text}, metadata={"iteration": 0, "path": []}
                )
            else:
                # Restore GraphState from dict
                state = GraphState(**state_dict)

            current_node_name = self.start_node
            iteration = 0

            while current_node_name and iteration < self.max_iterations:
                iteration += 1
                current_node = self.nodes[current_node_name]

                # Check for dynamic runtime interrupts
                if self.interrupt_service:
                    interrupt_message = await self.interrupt_service.check_interrupt(
                        ctx.session.id
                    )
                    if interrupt_message:
                        # Yield interrupt event with escalate=True for actual pause
                        yield Event(
                            author=self.name,
                            content=types.Content(
                                parts=[
                                    types.Part(
                                        text=f"🛑 INTERRUPT: {interrupt_message.text}"
                                    )
                                ]
                            ),
                            actions=EventActions(
                                escalate=True,
                                state_delta={
                                    "interrupt_message": interrupt_message.text,
                                    "interrupt_metadata": interrupt_message.metadata,
                                    "interrupt_action": interrupt_message.action,
                                    "interrupt_node": current_node_name,
                                    "interrupt_state": state.model_dump(),
                                },
                            ),
                        )

                        # Process interrupt message (update state, change conditions, etc.)
                        await self._process_interrupt_message(
                            interrupt_message, state, ctx
                        )

                        # Wait if paused (blocks until resume or cancel)
                        try:
                            resumed = await self.interrupt_service.wait_if_paused(
                                ctx.session.id
                            )
                            if not resumed:  # Cancelled
                                logger.info(
                                    f"GraphAgent execution cancelled for session {ctx.session.id}"
                                )
                                break
                        except TimeoutError as e:
                            logger.warning(f"Interrupt wait timeout: {e}")
                            # Continue execution after timeout
                            break

                # Track execution path
                state.metadata["path"].append(current_node_name)
                state.metadata["iteration"] = iteration

                # Human-in-the-loop: Before node execution
                if current_node_name in self.interrupt_before:
                    interrupt_msg = (
                        f"Interrupt before executing node: {current_node_name}"
                    )
                    # Use escalate=False for informational interrupts
                    # Set escalate=True to actually pause execution (requires InterruptService)
                    yield Event(
                        author=self.name,
                        content=types.Content(parts=[types.Part(text=interrupt_msg)]),
                        actions=EventActions(
                            escalate=False,
                            state_delta={
                                "interrupt_type": "before",
                                "interrupt_node": current_node_name,
                                "interrupt_state": state.model_dump(),
                            },
                        ),
                    )

                # Execute node
                async for event in self._execute_node(current_node, state, ctx):
                    yield event

                # Update state with node output
                output = state.metadata.get("_last_output", "")
                if output:
                    state = current_node.output_mapper(output, state)

                # Checkpointing - yield event with state_delta to persist checkpoint
                # Note: For full checkpoint/resume functionality, use CheckpointCallback
                if self.checkpointing:
                    checkpoint_data = {
                        "graph_state": state.model_dump(),
                        "graph_checkpoint": {
                            "node": current_node_name,
                            "iteration": iteration,
                        },
                    }
                    # Update session state directly for immediate access
                    ctx.session.state.update(checkpoint_data)
                    # Also yield event with state_delta for proper persistence
                    yield Event(
                        author=self.name,
                        content=types.Content(
                            parts=[types.Part(text=f"Checkpoint: {current_node_name}")]
                        ),
                        actions=EventActions(state_delta=checkpoint_data),
                    )

                # Human-in-the-loop: After node execution
                if current_node_name in self.interrupt_after:
                    interrupt_msg = (
                        f"Interrupt after executing node: {current_node_name}"
                    )
                    # Use escalate=False for informational interrupts
                    # Set escalate=True to actually pause execution (requires InterruptService)
                    yield Event(
                        author=self.name,
                        content=types.Content(parts=[types.Part(text=interrupt_msg)]),
                        actions=EventActions(
                            escalate=False,
                            state_delta={
                                "interrupt_type": "after",
                                "interrupt_node": current_node_name,
                                "interrupt_state": state.model_dump(),
                            },
                        ),
                    )

                # Get next node via conditional routing
                next_node_name = current_node.get_next_node(state)
                if next_node_name is None:
                    # No more edges - check if we're at an end node
                    if current_node_name in self.end_nodes:
                        break
                    else:
                        # Not at an end node and no edges - error
                        raise ValueError(
                            f"Node {current_node_name} has no outgoing edges and is not an end node"
                        )

                current_node_name = next_node_name

            # Final response
            final_output = state.data.get(
                "final_output", state.data.get(current_node_name, "")
            )

            # Format final response with execution metadata
            response_text = f"{final_output}"

            yield Event(
                author=self.name,
                content=types.Content(parts=[types.Part(text=response_text)]),
                actions=EventActions(
                    state_delta={
                        "graph_state": state.model_dump(),
                        "graph_iterations": iteration,
                        "graph_path": state.metadata["path"],
                    }
                ),
            )

        finally:
            # Unregister session from InterruptService
            if self.interrupt_service:
                self.interrupt_service.unregister_session(ctx.session.id)

    async def _process_interrupt_message(  # type: ignore[no-untyped-def]
        self, message, state: GraphState, ctx: InvocationContext
    ):
        """Process interrupt message and update state.

        This method handles interrupt messages from humans/clients and updates
        the graph state accordingly. Common actions include updating state values,
        changing conditions, or modifying execution parameters.

        Args:
            message: InterruptMessage from human
            state: Current graph state
            ctx: Invocation context
        """
        # Store interrupt message in state
        state.data["_last_interrupt"] = message.text
        state.data["_last_interrupt_action"] = message.action

        # Process based on action type
        if message.action == "update_state":
            # Update state with metadata
            if message.metadata:
                state.data.update(message.metadata)
                logger.info(f"Interrupt updated state: {list(message.metadata.keys())}")

        elif message.action == "change_condition":
            # Modify condition parameters
            if message.metadata:
                for key, value in message.metadata.items():
                    state.data[f"condition_{key}"] = value
                logger.info(f"Interrupt changed conditions: {message.metadata}")

        # Default: just store message for agent to process
        else:
            logger.info(f"Interrupt message stored: {message.text}")


# ============================================================================
# Example Usage: ReAct Pattern with Graph
# ============================================================================


async def example_react_pattern():  # type: ignore[no-untyped-def]  # pragma: no cover
    """Example: Implement ReAct pattern (Reasoning + Acting) using GraphAgent.

    Flow:
    1. Reason about the task
    2. Execute action (tool use)
    3. Observe results
    4. Loop until task is complete (conditional edge)
    """
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    # Define agents for each node
    reasoner = LlmAgent(
        name="reasoner",
        model="gemini-2.0-flash-exp",
        instruction=(
            "Analyze the task and current observations. "
            "Decide what action to take next. "
            "Store your reasoning in 'reasoning' key."
        ),
        output_key="reasoning",
    )

    actor = LlmAgent(
        name="actor",
        model="gemini-2.0-flash-exp",
        instruction=(
            "Based on the reasoning, execute the appropriate action. "
            "Return the action result."
        ),
        output_key="action_result",
    )

    observer = LlmAgent(
        name="observer",
        model="gemini-2.0-flash-exp",
        instruction=(
            "Observe the action result. "
            "Determine if the task is complete (output 'COMPLETE') "
            "or if more reasoning is needed (output 'CONTINUE')."
        ),
        output_key="observation",
    )

    # Create graph nodes
    reason_node = GraphNode(
        name="reason",
        agent=reasoner,
        input_mapper=lambda s: s.data.get("input", ""),
        reducer=StateReducer.OVERWRITE,
    )

    act_node = GraphNode(
        name="act",
        agent=actor,
        input_mapper=lambda s: s.data.get("reasoning", ""),
        reducer=StateReducer.OVERWRITE,
    )

    observe_node = GraphNode(
        name="observe",
        agent=observer,
        input_mapper=lambda s: s.data.get("action_result", ""),
        reducer=StateReducer.OVERWRITE,
    )

    # Create graph agent
    graph = GraphAgent(
        name="react_agent",
        description="ReAct pattern: Reasoning + Acting loop",
        max_iterations=10,
        checkpointing=True,
    )

    # Build graph structure
    graph.add_node(reason_node)
    graph.add_node(act_node)
    graph.add_node(observe_node)

    # Define edges
    graph.set_start("reason")
    graph.add_edge("reason", "act")
    graph.add_edge("act", "observe")

    # Conditional edge: loop back if not complete
    graph.add_edge(
        "observe",
        "reason",
        condition=lambda s: "CONTINUE" in s.data.get("observation", "").upper(),
    )

    # Conditional edge: end if complete
    graph.add_edge(
        "observe",
        "observe",  # Self-loop to end node
        condition=lambda s: "COMPLETE" in s.data.get("observation", "").upper(),
    )
    graph.set_end("observe")

    # Optional: Add human-in-the-loop
    # graph.add_interrupt("observe", InterruptMode.AFTER)

    # Run the graph
    runner = Runner(agent=graph, session_service=InMemorySessionService())

    session_id = "react-session-001"
    user_input = "Find information about the latest Python ADK release and summarize key features."

    print(f"Running ReAct pattern for: {user_input}\n")

    async for event in runner.run_async(session_id=session_id, user_message=user_input):
        if event.action == EventActions.AGENT_RESPONSE:
            print(f"\nFinal Response:")
            print(f"Content: {event.data.get('content')}")
            print(f"Iterations: {event.data.get('iterations')}")
            print(f"Path: {' -> '.join(event.data.get('path', []))}")
        elif event.action == EventActions.AGENT_ACTION:
            print(f"\nAction: {event.data}")


# ============================================================================
# Example Usage: Multi-Agent Collaboration Graph
# ============================================================================


async def example_multi_agent_graph():  # type: ignore[no-untyped-def]  # pragma: no cover
    """Example: Multi-agent research system with parallel execution and merging.

    Flow:
    1. Coordinator breaks down task
    2. Parallel researchers investigate different aspects
    3. Merger combines findings
    4. Critic reviews (conditional loop back if quality is low)
    """
    from google.adk.runners import Runner

    coordinator = LlmAgent(
        name="coordinator",
        model="gemini-2.0-flash-exp",
        instruction="Break down the research task into subtopics.",
        output_key="subtopics",
    )

    researcher1 = LlmAgent(
        name="researcher1",
        model="gemini-2.0-flash-exp",
        instruction="Research the first subtopic thoroughly.",
        output_key="research1",
    )

    researcher2 = LlmAgent(
        name="researcher2",
        model="gemini-2.0-flash-exp",
        instruction="Research the second subtopic thoroughly.",
        output_key="research2",
    )

    merger = LlmAgent(
        name="merger",
        model="gemini-2.0-flash-exp",
        instruction="Combine all research findings into a cohesive report.",
        output_key="merged_report",
    )

    critic = LlmAgent(
        name="critic",
        model="gemini-2.0-flash-exp",
        instruction=(
            "Review the merged report. "
            "If quality is good, output 'APPROVED'. "
            "If improvements needed, output 'REVISE' and explain why."
        ),
        output_key="review",
    )

    # Build graph
    graph = GraphAgent(
        name="research_graph",
        description="Multi-agent research with quality review loop",
        max_iterations=20,
    )

    # Note: For true parallel execution, you'd need to implement
    # a custom ParallelNode or use ParallelAgent as a wrapper

    graph.add_node(GraphNode(name="coordinator", agent=coordinator))
    graph.add_node(GraphNode(name="researcher1", agent=researcher1))
    graph.add_node(GraphNode(name="researcher2", agent=researcher2))
    graph.add_node(GraphNode(name="merger", agent=merger))
    graph.add_node(GraphNode(name="critic", agent=critic))

    graph.set_start("coordinator")
    graph.add_edge("coordinator", "researcher1")
    graph.add_edge("coordinator", "researcher2")
    graph.add_edge("researcher1", "merger")
    graph.add_edge("researcher2", "merger")
    graph.add_edge("merger", "critic")

    # Conditional loop: revise if not approved
    graph.add_edge(
        "critic",
        "merger",
        condition=lambda s: "REVISE" in s.data.get("review", "").upper(),
    )

    graph.add_edge(
        "critic",
        "critic",
        condition=lambda s: "APPROVED" in s.data.get("review", "").upper(),
    )
    graph.set_end("critic")

    return graph


if __name__ == "__main__":  # pragma: no cover
    # Run example
    import asyncio

    asyncio.run(example_react_pattern())  # type: ignore[no-untyped-call]
