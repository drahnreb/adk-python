"""Human-in-the-loop interrupt modes for graph execution."""

from enum import Enum


class InterruptMode(str, Enum):
    """When to interrupt execution for human-in-the-loop.

    Interrupts allow pausing graph execution at specific nodes
    to enable human review, approval, or intervention before continuing.

    Example:
        ```python
        from google.adk.agents.graph import GraphAgent, InterruptMode

        graph = GraphAgent(name="workflow")
        # Add interrupt before critical decision node
        graph.add_interrupt("decision_node", InterruptMode.BEFORE)
        # Add interrupt after sensitive operation
        graph.add_interrupt("sensitive_op", InterruptMode.AFTER)
        # Add interrupt both before and after validation
        graph.add_interrupt("validation", InterruptMode.BOTH)
        ```
    """

    BEFORE = "before"  # Interrupt before node execution
    AFTER = "after"  # Interrupt after node execution
    BOTH = "both"  # Interrupt both before and after node execution
