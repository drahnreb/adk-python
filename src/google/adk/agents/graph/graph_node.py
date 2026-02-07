"""Graph node wrapper for agents and functions."""

from typing import Any, Callable, List, Optional

from .graph_edge import EdgeCondition
from .graph_state import GraphState, StateReducer
from ..base_agent import BaseAgent


class GraphNode:
    """A node in the graph that can wrap ANY ADK agent or custom function.

    Supports all BaseAgent types:
    - LLMAgent: Single agent with tools
    - SequentialAgent: Chain of agents executed in sequence
    - ParallelAgent: Multiple agents executed concurrently
    - LoopAgent: Iterative agent execution
    - GraphAgent: Recursive graph workflows (graphs within graphs!)
    - Custom agents: Any subclass of BaseAgent

    This enables powerful composition patterns like:
    - Nested workflows (GraphAgent containing other GraphAgents)
    - Validation pipelines (SequentialAgent as a node)
    - Parallel analysis (ParallelAgent as a node)
    """

    def __init__(
        self,
        name: str,
        agent: Optional[BaseAgent] = None,
        function: Optional[Callable[..., Any]] = None,
        input_mapper: Optional[Callable[[GraphState], str]] = None,
        output_mapper: Optional[Callable[[str, GraphState], GraphState]] = None,
        reducer: StateReducer = StateReducer.OVERWRITE,
        custom_reducer: Optional[Callable[..., Any]] = None,
    ):
        """Initialize graph node.

        Args:
            name: Node name
            agent: ANY ADK BaseAgent subclass (LLMAgent, SequentialAgent, ParallelAgent,
                   LoopAgent, GraphAgent, or custom agents)
            function: Custom async function to execute (alternative to agent)
            input_mapper: Maps GraphState to agent input
            output_mapper: Maps agent output back to GraphState
            reducer: Strategy for merging output into state
            custom_reducer: Custom reduction function
        """
        if agent is None and function is None:
            raise ValueError("Either agent or function must be provided")

        self.name = name
        self.agent = agent
        self.function = function
        self.input_mapper = input_mapper or self._default_input_mapper
        self.output_mapper = output_mapper or self._default_output_mapper
        self.reducer = reducer
        self.custom_reducer = custom_reducer
        self.edges: List[EdgeCondition] = []

    def add_edge(
        self, target_node: str, condition: Optional[Callable[..., Any]] = None
    ) -> None:
        """Add an edge to another node.

        Args:
            target_node: Name of the target node
            condition: Optional condition function for conditional routing
        """
        self.edges.append(EdgeCondition(target_node, condition))

    def _default_input_mapper(self, state: GraphState) -> str:
        """Default input mapper: extract 'input' or 'messages' from state.

        Args:
            state: Current graph state

        Returns:
            Input string for the node
        """
        return str(state.data.get("input", state.data.get("messages", "")))

    def _default_output_mapper(self, output: str, state: GraphState) -> GraphState:
        """Default output mapper: store output in state with node name as key.

        Applies the configured reducer strategy to merge the output into state.

        Args:
            output: Node execution output
            state: Current graph state

        Returns:
            New graph state with output merged
        """
        new_state = GraphState(data=state.data.copy(), metadata=state.metadata.copy())

        if self.reducer == StateReducer.OVERWRITE:
            new_state.data[self.name] = output
        elif self.reducer == StateReducer.APPEND:
            if self.name not in new_state.data:
                new_state.data[self.name] = []
            new_state.data[self.name].append(output)
        elif self.reducer == StateReducer.SUM:
            new_state.data[self.name] = new_state.data.get(self.name, 0) + output
        elif self.reducer == StateReducer.CUSTOM and self.custom_reducer:
            new_state.data[self.name] = self.custom_reducer(
                new_state.data.get(self.name), output
            )

        return new_state

    def get_next_node(self, state: GraphState) -> Optional[str]:
        """Determine next node based on conditional edges.

        Evaluates edge conditions in order and returns the first matching target.

        Args:
            state: Current graph state

        Returns:
            Name of next node, or None if no edge matches
        """
        for edge in self.edges:
            if edge.should_route(state):
                return edge.target_node
        return None
