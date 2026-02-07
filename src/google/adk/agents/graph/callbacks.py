"""Callback infrastructure for graph observability and extensibility.

This module provides callback primitives for customizing graph behavior:
- NodeCallbackContext: Context passed to node lifecycle callbacks
- EdgeCallbackContext: Context passed to edge condition callbacks
- NodeCallback: Type for before/after node callbacks
- EdgeCallback: Type for edge condition callbacks

Callbacks enable custom observability, logging, debugging, and control flow
without modifying the core GraphAgent implementation.

Example:
    ```python
    from google.adk.agents.graph import GraphAgent
    from google.adk.agents.graph.callbacks import NodeCallbackContext
    from google import genai

    async def my_observability(ctx: NodeCallbackContext):
        '''Custom observability callback.'''
        return genai.types.Event(
            author="observability",
            content=genai.types.Content(parts=[
                genai.types.Part(text=f"→ Executing: {ctx.node.name}"),
                genai.types.Part(text=f"State: {ctx.state.data}"),
            ]),
            actions=genai.types.EventActions(
                escalate=False,
                state_delta={
                    "observability_node": ctx.node.name,
                    "observability_iteration": ctx.iteration,
                }
            )
        )

    graph = GraphAgent(
        name="my_graph",
        before_node_callback=my_observability,
    )
    ```
"""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...events.event import Event

from google import genai

from .graph_state import GraphState


@dataclass
class NodeCallbackContext:
    """Context passed to node lifecycle callbacks.

    Contains all information about the current node execution,
    including the node itself, current state, iteration number,
    and full invocation context.

    Attributes:
        node: The GraphNode being executed
        state: Current graph state (before or after execution)
        iteration: Current iteration number in graph execution
        invocation_context: Full ADK invocation context
        metadata: Extensible metadata dictionary for custom use
    """

    node: Any  # GraphNode (avoiding circular import)
    state: GraphState
    iteration: int
    invocation_context: Any  # InvocationContext (from genai.types)
    metadata: Dict[str, Any]


@dataclass
class EdgeCallbackContext:
    """Context passed to edge condition callbacks.

    Contains all information about an edge condition evaluation,
    including source/target nodes, the condition function,
    evaluation result, and current state.

    Attributes:
        from_node: Name of the source node
        to_node: Name of the target node
        condition: The condition function being evaluated (or None for unconditional)
        condition_result: Boolean result of condition evaluation
        state: Current graph state
        invocation_context: Full ADK invocation context
        metadata: Extensible metadata dictionary for custom use
    """

    from_node: str
    to_node: str
    condition: Optional[Callable[[GraphState], bool]]
    condition_result: bool
    state: GraphState
    invocation_context: Any  # InvocationContext (from genai.types)
    metadata: Dict[str, Any]


# Type aliases for callbacks
NodeCallback = Callable[[NodeCallbackContext], Awaitable[Optional["Event"]]]
"""Callback function type for node lifecycle events.

Receives NodeCallbackContext and optionally returns an Event to emit.
Returning None skips event emission.

Example:
    ```python
    from google.adk.events.event import Event

    async def my_node_callback(ctx: NodeCallbackContext) -> Optional[Event]:
        # Custom logic here
        if should_emit:
            return Event(...)
        return None  # Skip event
    ```
"""

EdgeCallback = Callable[[EdgeCallbackContext], Awaitable[Optional["Event"]]]
"""Callback function type for edge condition events.

Receives EdgeCallbackContext and optionally returns an Event to emit.
Returning None skips event emission.

Example:
    ```python
    from google.adk.events.event import Event
    from google.genai.types import Content, Part

    async def my_edge_callback(ctx: EdgeCallbackContext) -> Optional[Event]:
        # Log conditional routing decisions
        if ctx.condition_result:
            return Event(
                content=Content(parts=[
                    Part(text=f"Routing: {ctx.from_node} → {ctx.to_node}")
                ])
            )
        return None
    ```
"""


def create_nested_observability_callback() -> NodeCallback:
    """Create a callback that shows nested graph hierarchy.

    Returns a NodeCallback that includes agent path information
    in observability events, making nested graph execution visible.

    Example:
        ```python
        graph = GraphAgent(
            name="my_graph",
            before_node_callback=create_nested_observability_callback(),
        )
        ```

    Returns:
        NodeCallback that emits events with nesting hierarchy
    """

    async def nested_callback(ctx: NodeCallbackContext) -> Optional["Event"]:
        """Emit observability event with nested graph hierarchy."""
        from ...events.event import Event
        from ...events.event_actions import EventActions

        # Get agent path from session state
        agent_path = ctx.invocation_context.session.state.get("_agent_path", [])
        hierarchy = " → ".join(agent_path) if agent_path else ctx.node.name

        return Event(
            author="observability",
            content=genai.types.Content(
                parts=[
                    genai.types.Part(text=f"[{hierarchy}] → {ctx.node.name}"),
                    genai.types.Part(text=f"Iteration: {ctx.iteration}"),
                ]
            ),
            actions=EventActions(
                escalate=False,
                state_delta={
                    "observability_hierarchy": hierarchy,
                    "observability_level": len(agent_path),
                    "observability_node": ctx.node.name,
                },
            ),
        )

    return nested_callback
