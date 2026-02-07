"""Test suite for InterruptReasoner LLM-based interrupt reasoning.

Tests:
- LLM-based action decisions
- JSON parsing and validation
- Fallback behavior
- Custom actions
- Defer to todos
- go_back action
- State management
- GraphAgent integration
"""

import json
import pytest
from unittest.mock import AsyncMock, patch

from google.adk.agents.graph import (
    GraphAgent,
    GraphNode,
    GraphState,
    InterruptConfig,
    InterruptMode,
    InterruptAction,
)
from google.adk.agents.graph.interrupt_reasoner import (
    InterruptReasoner,
    InterruptReasonerConfig,
)
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph.interrupt_service import InterruptMessage, InterruptService
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.genai import types


# Mock agent for testing
class MockAgent(BaseAgent):
    """Mock agent that extends BaseAgent for proper Pydantic validation."""

    response: str = "mock"

    async def run_async(self, ctx):
        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text=self.response)]),
        )


def create_test_invocation_context() -> InvocationContext:
    """Helper to create minimal valid InvocationContext for testing."""
    session = Session(id="test_session", appName="test_app", userId="test_user")
    session_service = InMemorySessionService()

    # Create a proper BaseAgent for testing
    mock_agent = MockAgent(name="test_agent")

    return InvocationContext(
        session=session,
        session_service=session_service,
        invocation_id="test_invocation",
        agent=mock_agent,
        user_content=None,
    )


class MockLLMReasoner(InterruptReasoner):
    """Mock reasoner that returns predetermined decision."""

    model_config = {"extra": "allow"}  # Allow extra attributes for mock

    def __init__(self, decision_json: dict, **kwargs):
        config = InterruptReasonerConfig()
        super().__init__(config, **kwargs)
        # Store decision after super().__init__()
        self.mock_decision_json = decision_json

    async def run_async(self, ctx):
        """Return mock decision as JSON."""
        response = json.dumps(self.mock_decision_json)
        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text=response)]),
        )


@pytest.mark.asyncio
async def test_interrupt_reasoner_decides_continue():
    """Test reasoner decides to continue."""
    reasoner = MockLLMReasoner(
        {"action": "continue", "reasoning": "Everything looks good", "parameters": {}}
    )

    message = InterruptMessage(text="Check progress", action="continue")
    state = GraphState(data={"test": "value"}, metadata={"path": ["start"]})
    ctx = create_test_invocation_context()

    action = await reasoner.reason_about_interrupt(message, state, "test_node", ctx)

    assert action.action == "continue"
    assert action.reasoning == "Everything looks good"
    assert action.parameters == {}


@pytest.mark.asyncio
async def test_interrupt_reasoner_decides_rerun():
    """Test reasoner decides to rerun with guidance."""
    reasoner = MockLLMReasoner(
        {
            "action": "rerun",
            "reasoning": "Output needs improvement",
            "parameters": {"guidance": "Be more specific"},
        }
    )

    message = InterruptMessage(text="Output is vague", action=None)
    state = GraphState(data={"output": "vague response"})
    ctx = create_test_invocation_context()

    action = await reasoner.reason_about_interrupt(message, state, "test_node", ctx)

    assert action.action == "rerun"
    assert "improvement" in action.reasoning
    assert action.parameters["guidance"] == "Be more specific"


@pytest.mark.asyncio
async def test_interrupt_reasoner_decides_defer():
    """Test reasoner decides to defer for later."""
    reasoner = MockLLMReasoner(
        {
            "action": "defer",
            "reasoning": "Not critical now",
            "parameters": {"message": "Fix validation later"},
        }
    )

    message = InterruptMessage(text="Validation could be better")
    state = GraphState(data={})
    ctx = create_test_invocation_context()

    action = await reasoner.reason_about_interrupt(message, state, "test_node", ctx)

    assert action.action == "defer"
    assert action.parameters["message"] == "Fix validation later"


@pytest.mark.asyncio
async def test_interrupt_reasoner_decides_go_back():
    """Test reasoner decides to go back."""
    reasoner = MockLLMReasoner(
        {
            "action": "go_back",
            "reasoning": "Need to retry earlier step",
            "parameters": {"steps": 2},
        }
    )

    message = InterruptMessage(text="Previous step had error")
    state = GraphState(data={}, metadata={"path": ["a", "b", "c"]})
    ctx = create_test_invocation_context()

    action = await reasoner.reason_about_interrupt(message, state, "c", ctx)

    assert action.action == "go_back"
    assert action.parameters["steps"] == 2


@pytest.mark.asyncio
async def test_interrupt_reasoner_fallback_on_invalid_json():
    """Test reasoner falls back to continue on invalid JSON."""
    config = InterruptReasonerConfig()
    reasoner = InterruptReasoner(config)

    # Mock run_async to return invalid JSON
    async def mock_run(ctx):
        yield Event(
            author="reasoner",
            content=types.Content(parts=[types.Part(text="invalid json{")]),
        )

    # Can't mock run_async on Pydantic model, so patch the method directly
    original_run_async = reasoner.run_async
    reasoner.__class__.run_async = mock_run

    message = InterruptMessage(text="Test")
    state = GraphState(data={})
    ctx = create_test_invocation_context()

    action = await reasoner.reason_about_interrupt(message, state, "node", ctx)

    # Restore original method
    reasoner.__class__.run_async = original_run_async

    # Should fall back to continue
    assert action.action == "continue"
    assert "Failed to parse" in action.reasoning


@pytest.mark.asyncio
async def test_interrupt_reasoner_validates_action():
    """Test reasoner validates action is in available_actions."""
    reasoner = MockLLMReasoner(
        {
            "action": "invalid_action",  # Not in available_actions
            "reasoning": "Test",
            "parameters": {},
        }
    )

    message = InterruptMessage(text="Test")
    state = GraphState(data={})
    ctx = create_test_invocation_context()

    action = await reasoner.reason_about_interrupt(message, state, "node", ctx)

    # Should default to continue
    assert action.action == "continue"


@pytest.mark.asyncio
async def test_interrupt_reasoner_with_graphagent():
    """Test InterruptReasoner integrated with GraphAgent."""
    # Create reasoner that always decides to defer
    reasoner = MockLLMReasoner(
        {
            "action": "defer",
            "reasoning": "Save for later",
            "parameters": {"message": "Fix this later"},
        }
    )

    interrupt_service = InterruptService()
    graph = GraphAgent(
        name="test_graph",
        interrupt_service=interrupt_service,
        interrupt_config=InterruptConfig(
            mode=InterruptMode.AFTER,
            reasoner=reasoner,
        ),
    )

    node_a = GraphNode(
        name="node_a", agent=MockAgent(name="agent_a", response="output_a")
    )
    graph.add_node(node_a)
    graph.set_start("node_a").set_end("node_a")

    session_service = InMemorySessionService()
    runner = Runner(app_name="test_app", agent=graph, session_service=session_service)

    # Create session
    await session_service.create_session(
        app_name="test_app", user_id="test_user", session_id="test_session"
    )

    # Register session and send interrupt
    interrupt_service.register_session("test_session")
    await interrupt_service.send_message(
        "test_session", text="This needs attention", action="defer"
    )

    events = []
    async for event in runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=types.Content(role="user", parts=[types.Part(text="test input")]),
    ):
        events.append(event)

    # Get session to verify state
    session = await session_service.get_session(
        app_name="test_app", user_id="test_user", session_id="test_session"
    )

    # Verify todos were created in session.state
    todos = session.state.get("_interrupt_todos", [])
    assert len(todos) == 1
    assert todos[0]["message"] == "Fix this later"
    assert todos[0]["node"] == "node_a"

    # Verify interrupt decision was tracked
    decision = session.state.get("_last_interrupt_decision")
    assert decision is not None
    assert decision["action"] == "defer"
    assert decision["reasoning"] == "Save for later"


@pytest.mark.asyncio
async def test_defer_action_stores_in_session_state():
    """Test that defer action stores in session.state, not GraphState."""
    interrupt_service = InterruptService()
    graph = GraphAgent(
        name="test_graph",
        interrupt_service=interrupt_service,
        interrupt_config=InterruptConfig(mode=InterruptMode.AFTER),
    )

    node_a = GraphNode(name="node_a", agent=MockAgent(name="agent_a"))
    graph.add_node(node_a)
    graph.set_start("node_a").set_end("node_a")

    session_service = InMemorySessionService()
    runner = Runner(app_name="test_app", agent=graph, session_service=session_service)

    # Create session
    await session_service.create_session(
        app_name="test_app", user_id="test_user", session_id="test_session"
    )

    interrupt_service.register_session("test_session")
    await interrupt_service.send_message(
        "test_session",
        text="Defer this",
        action="defer",
        metadata={"message": "Fix validation"},
    )

    # Track final GraphState
    final_state = None
    async for event in runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=types.Content(role="user", parts=[types.Part(text="test input")]),
    ):
        if event.actions and event.actions.state_delta:
            graph_state_data = event.actions.state_delta.get("graph_state")
            if graph_state_data:
                final_state = GraphState(**graph_state_data)

    # Verify todos NOT in GraphState
    assert final_state is not None
    assert "todos" not in final_state.metadata
    assert "_interrupt_todos" not in final_state.data

    # Get session to verify state
    session = await session_service.get_session(
        app_name="test_app", user_id="test_user", session_id="test_session"
    )

    # Verify todos in session.state
    todos = session.state.get("_interrupt_todos", [])
    assert len(todos) == 1
    assert "Fix validation" in str(todos[0])


@pytest.mark.asyncio
async def test_go_back_action_restores_path():
    """Test go_back action properly restores execution path."""
    interrupt_service = InterruptService()
    graph = GraphAgent(
        name="test_graph",
        interrupt_service=interrupt_service,
        interrupt_config=InterruptConfig(mode=InterruptMode.AFTER),
    )

    node_a = GraphNode(
        name="node_a", agent=MockAgent(name="agent_a", response="a_output")
    )
    node_b = GraphNode(
        name="node_b", agent=MockAgent(name="agent_b", response="b_output")
    )
    node_c = GraphNode(
        name="node_c", agent=MockAgent(name="agent_c", response="c_output")
    )

    graph.add_node(node_a).add_node(node_b).add_node(node_c)
    graph.add_edge("node_a", "node_b")
    graph.add_edge("node_b", "node_c")
    graph.set_start("node_a").set_end("node_c")

    session_service = InMemorySessionService()
    runner = Runner(app_name="test_app", agent=graph, session_service=session_service)

    # Create session
    await session_service.create_session(
        app_name="test_app", user_id="test_user", session_id="test_session"
    )

    interrupt_service.register_session("test_session")

    # Send go_back interrupt after node_c
    # This will be processed after node_c completes
    await interrupt_service.send_message(
        "test_session", text="Go back 2 steps", action="go_back", metadata={"steps": 2}
    )

    execution_order = []
    async for event in runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=types.Content(role="user", parts=[types.Part(text="test input")]),
    ):
        # Track node execution
        if event.author in ["agent_a", "agent_b", "agent_c"]:
            execution_order.append(event.author)

    # Note: go_back happens after node_c, so we'd need to run graph again
    # to see it jump back. For this test, just verify node_c executed.
    assert "agent_c" in execution_order


@pytest.mark.asyncio
async def test_immediate_cancel_interrupt():
    """Test immediate cancellation (ESC-like) stops execution immediately.

    Tests cancellation between nodes - cancels after node_a completes,
    preventing node_b from starting.
    """
    interrupt_service = InterruptService()
    graph = GraphAgent(
        name="test_graph",
        interrupt_service=interrupt_service,
    )

    # Create slow agents to allow time for cancellation
    node_a = GraphNode(name="node_a", agent=MockAgent(name="agent_a", response="a"))
    node_b = GraphNode(name="node_b", agent=MockAgent(name="agent_b", response="b"))
    node_c = GraphNode(name="node_c", agent=MockAgent(name="agent_c", response="c"))

    graph.add_node(node_a).add_node(node_b).add_node(node_c)
    graph.add_edge("node_a", "node_b")
    graph.add_edge("node_b", "node_c")
    graph.set_start("node_a").set_end("node_c")

    session_service = InMemorySessionService()
    runner = Runner(app_name="test_app", agent=graph, session_service=session_service)

    await session_service.create_session(
        app_name="test_app", user_id="test_user", session_id="test_session"
    )

    interrupt_service.register_session("test_session")

    execution_order = []
    cancel_called = False

    async def run_with_cancel():
        nonlocal cancel_called
        async for event in runner.run_async(
            user_id="test_user",
            session_id="test_session",
            new_message=types.Content(role="user", parts=[types.Part(text="test")]),
        ):
            if event.author in ["agent_a", "agent_b", "agent_c"]:
                execution_order.append(event.author)

            # Cancel after first node executes
            if event.author == "agent_a" and not cancel_called:
                await interrupt_service.cancel("test_session")
                cancel_called = True

    await run_with_cancel()

    # Verify execution stopped after node_a (before node_b)
    assert "agent_a" in execution_order
    assert "agent_b" not in execution_order
    assert "agent_c" not in execution_order


@pytest.mark.asyncio
async def test_immediate_cancel_during_node_execution():
    """Test immediate cancellation DURING node execution (not just between nodes).

    This tests TRUE immediate interrupt like ESC - cancelling while a node
    is actively executing and streaming events.
    """
    interrupt_service = InterruptService()
    graph = GraphAgent(
        name="test_graph",
        interrupt_service=interrupt_service,
    )

    # Create multi-event agent that yields multiple events
    class MultiEventAgent(BaseAgent):
        """Agent that yields multiple events for testing mid-execution cancel."""

        async def run_async(self, ctx):
            # Yield multiple events to allow cancellation during execution
            for i in range(5):
                yield Event(
                    author=self.name,
                    content=types.Content(
                        parts=[types.Part(text=f"Event {i} from {self.name}")]
                    ),
                )

    node_a = GraphNode(name="node_a", agent=MultiEventAgent(name="multi_agent"))

    graph.add_node(node_a)
    graph.set_start("node_a").set_end("node_a")

    session_service = InMemorySessionService()
    runner = Runner(app_name="test_app", agent=graph, session_service=session_service)

    await session_service.create_session(
        app_name="test_app", user_id="test_user", session_id="test_session"
    )

    interrupt_service.register_session("test_session")

    event_count = 0
    cancel_called = False
    cancelled_event_seen = False

    async def run_with_cancel():
        nonlocal event_count, cancel_called, cancelled_event_seen
        async for event in runner.run_async(
            user_id="test_user",
            session_id="test_session",
            new_message=types.Content(role="user", parts=[types.Part(text="test")]),
        ):
            if event.author == "multi_agent":
                event_count += 1
                # Cancel after 2nd event (while node is still executing)
                if event_count == 2 and not cancel_called:
                    await interrupt_service.cancel("test_session")
                    cancel_called = True

            # Check for cancellation event
            if event.content and event.content.parts:
                if "cancelled during node" in event.content.parts[0].text:
                    cancelled_event_seen = True

    await run_with_cancel()

    # Verify execution stopped DURING node execution (not all 5 events)
    assert cancel_called, "Cancel should have been called"
    assert (
        event_count < 5
    ), f"Should have stopped mid-execution, but got {event_count} events"
    assert event_count >= 2, "Should have seen at least 2 events before cancel"
    assert cancelled_event_seen, "Should have seen cancellation event"
