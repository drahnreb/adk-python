"""Conditional edges for graph routing."""

from typing import Callable
from typing import Optional

from .graph_state import GraphState


class EdgeCondition:
    """Conditional edge that routes based on state.

    Edges connect nodes in the graph and can have optional conditions
    that determine whether the edge should be taken based on the current state.

    Example:
        ```python
        # Unconditional edge (always taken)
        edge = EdgeCondition(target_node="next_node")

        # Conditional edge (taken if score > 0.8)
        edge = EdgeCondition(
            target_node="high_score_handler",
            condition=lambda state: state.data.get("score", 0) > 0.8
        )
        ```
    """

    def __init__(
        self, target_node: str, condition: Optional[Callable[[GraphState], bool]] = None
    ):
        """Initialize edge condition.

        Args:
            target_node: Name of the target node
            condition: Function that returns True if this edge should be taken.
                If None, edge is always taken (unconditional).
        """
        self.target_node = target_node
        self.has_condition = condition is not None
        self.condition = condition or (lambda _: True)

    def should_route(self, state: GraphState) -> bool:
        """Check if this edge should be taken given the current state.

        Args:
            state: Current graph state

        Returns:
            True if edge condition is satisfied, False otherwise
        """
        return self.condition(state)
