"""Graph-based agent components.

This module contains components for graph-based workflow orchestration:
- GraphAgent: Main graph workflow agent
- GraphState: State container with typed data and metadata
- GraphNode: Node wrapper for agents and functions
- EdgeCondition: Conditional routing between nodes
- StateReducer: State merge strategies
- InterruptMode: Human-in-the-loop interrupt modes
- InterruptService: Dynamic runtime interrupts with queue bounds and metrics
- InterruptServiceConfig: Configuration for interrupt service
- InterruptMessage: Message from human to agent
- QueueStatus: Queue status information
- SessionMetrics: Per-session interrupt metrics
- GraphEvent: Typed events for streaming
- GraphEventType: Event type enumeration
- GraphStreamMode: Stream mode enumeration
- NodeCallbackContext: Context for node lifecycle callbacks
- EdgeCallbackContext: Context for edge condition callbacks
- NodeCallback: Type for node lifecycle callbacks
- EdgeCallback: Type for edge condition callbacks
"""

from .callbacks import EdgeCallback
from .callbacks import EdgeCallbackContext
from .callbacks import NodeCallback
from .callbacks import NodeCallbackContext
from .callbacks import create_nested_observability_callback
from .graph_agent import GraphAgent
from .graph_edge import EdgeCondition
from .graph_events import GraphEvent
from .graph_events import GraphEventType
from .graph_events import GraphStreamMode
from .graph_node import GraphNode
from .graph_state import GraphState
from .graph_state import StateReducer
from .interrupt import InterruptAction
from .interrupt import InterruptConfig
from .interrupt import InterruptMode
from .interrupt_reasoner import InterruptReasoner
from .interrupt_reasoner import InterruptReasonerConfig
from .interrupt_service import InterruptMessage
from .interrupt_service import InterruptService
from .interrupt_service import InterruptServiceConfig
from .interrupt_service import QueueStatus
from .interrupt_service import SessionMetrics

# Sentinel constants for graph boundaries
START = "__start__"
END = "__end__"

__all__ = [
    "GraphAgent",
    "GraphState",
    "GraphNode",
    "EdgeCondition",
    "StateReducer",
    "InterruptMode",
    "InterruptConfig",
    "InterruptAction",
    "InterruptReasoner",
    "InterruptReasonerConfig",
    "InterruptService",
    "InterruptServiceConfig",
    "InterruptMessage",
    "QueueStatus",
    "SessionMetrics",
    "GraphEvent",
    "GraphEventType",
    "GraphStreamMode",
    "NodeCallbackContext",
    "EdgeCallbackContext",
    "NodeCallback",
    "EdgeCallback",
    "create_nested_observability_callback",
    "START",
    "END",
]
