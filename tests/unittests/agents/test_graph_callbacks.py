"""Test suite for GraphAgent callback infrastructure.

Tests callback-based observability and extensibility:
- NodeCallback (before/after node execution)
- EdgeCallback (on edge condition evaluation)
- Custom observability patterns
- Nested graph hierarchy tracking
"""

import pytest
from typing import Optional

from google.adk.agents.graph import (
    GraphAgent,
    GraphNode,
    NodeCallbackContext,
    create_nested_observability_callback,
)
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types


# Mock agent for testing
class MockAgent:
    def __init__(self, name: str, response: str = "mock"):
        self.name = name
        self.response = response

    async def run_async(self, ctx):
        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text=self.response)]),
        )


@pytest.mark.asyncio
async def test_before_node_callback_invoked():
    """Test that before_node_callback is invoked before node execution."""
    callback_invocations = []

    async def before_callback(ctx: NodeCallbackContext) -> Optional[Event]:
        callback_invocations.append(("before", ctx.node.name))
        return Event(
            author="test",
            content=types.Content(parts=[types.Part(text=f"Before: {ctx.node.name}")]),
        )

    graph = GraphAgent(name="test_graph", before_node_callback=before_callback)
    node_a = GraphNode(name="node_a", agent=MockAgent("agent_a", "output_a"))
    node_b = GraphNode(name="node_b", agent=MockAgent("agent_b", "output_b"))

    graph.add_node(node_a).add_node(node_b)
    graph.add_edge("node_a", "node_b")
    graph.set_start("node_a").set_end("node_b")

    session_service = InMemorySessionService()
    runner = Runner(
        app_name="test_app",
        agent=graph,
        session_service=session_service
    )

    # Create session first
    await session_service.create_session(
        app_name="test_app", user_id="test_user", session_id="test_session"
    )

    events = []
    async for event in runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=types.Content(role="user", parts=[types.Part(text="test input")]),
    ):
        events.append(event)

    # Verify callback was invoked for both nodes
    assert len(callback_invocations) == 2
    assert callback_invocations[0] == ("before", "node_a")
    assert callback_invocations[1] == ("before", "node_b")

    # Verify callback events were emitted
    before_events = [e for e in events if e.content and "Before:" in (e.content.parts[0].text or "")]
    assert len(before_events) == 2


@pytest.mark.asyncio
async def test_after_node_callback_invoked():
    """Test that after_node_callback is invoked after node execution."""
    callback_invocations = []

    async def after_callback(ctx: NodeCallbackContext) -> Optional[Event]:
        callback_invocations.append(("after", ctx.node.name, ctx.metadata.get("output")))
        return Event(
            author="test",
            content=types.Content(parts=[types.Part(text=f"After: {ctx.node.name}")]),
        )

    graph = GraphAgent(name="test_graph", after_node_callback=after_callback)
    node_a = GraphNode(name="node_a", agent=MockAgent("agent_a", "output_a"))
    graph.add_node(node_a)
    graph.set_start("node_a").set_end("node_a")

    session_service = InMemorySessionService()
    runner = Runner(
        app_name="test_app",
        agent=graph,
        session_service=session_service
    )

    # Create session first
    await session_service.create_session(
        app_name="test_app", user_id="test_user", session_id="test_session"
    )

    events = []
    async for event in runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=types.Content(role="user", parts=[types.Part(text="test input")]),
    ):
        events.append(event)

    # Verify callback was invoked with output
    assert len(callback_invocations) == 1
    assert callback_invocations[0][0] == "after"
    assert callback_invocations[0][1] == "node_a"
    assert callback_invocations[0][2] == "output_a"


@pytest.mark.asyncio
async def test_callback_returning_none_skips_event():
    """Test that callback returning None skips event emission."""
    callback_invocations = []

    async def selective_callback(ctx: NodeCallbackContext) -> Optional[Event]:
        callback_invocations.append(ctx.node.name)
        # Only emit for node_a
        if ctx.node.name == "node_a":
            return Event(
                author="test",
                content=types.Content(parts=[types.Part(text="Event")]),
            )
        return None  # Skip for node_b

    graph = GraphAgent(name="test_graph", before_node_callback=selective_callback)
    node_a = GraphNode(name="node_a", agent=MockAgent("agent_a"))
    node_b = GraphNode(name="node_b", agent=MockAgent("agent_b"))

    graph.add_node(node_a).add_node(node_b)
    graph.add_edge("node_a", "node_b")
    graph.set_start("node_a").set_end("node_b")

    session_service = InMemorySessionService()
    runner = Runner(
        app_name="test_app",
        agent=graph,
        session_service=session_service
    )

    # Create session first
    await session_service.create_session(
        app_name="test_app", user_id="test_user", session_id="test_session"
    )

    events = []
    async for event in runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=types.Content(role="user", parts=[types.Part(text="test input")]),
    ):
        events.append(event)

    # Callback invoked for both
    assert len(callback_invocations) == 2

    # But only one event emitted
    test_events = [e for e in events if e.author == "test"]
    assert len(test_events) == 1


@pytest.mark.asyncio
async def test_callback_has_full_context():
    """Test that callback receives full context including state and iteration."""
    captured_contexts = []

    async def capture_callback(ctx: NodeCallbackContext) -> Optional[Event]:
        captured_contexts.append({
            "node_name": ctx.node.name,
            "iteration": ctx.iteration,
            "state_data_keys": list(ctx.state.data.keys()),
            "state_path": ctx.state.metadata.get("path", []).copy(),  # Copy to avoid mutation
        })
        return None

    graph = GraphAgent(name="test_graph", before_node_callback=capture_callback)
    node_a = GraphNode(name="node_a", agent=MockAgent("agent_a", "output_a"))
    node_b = GraphNode(name="node_b", agent=MockAgent("agent_b", "output_b"))

    graph.add_node(node_a).add_node(node_b)
    graph.add_edge("node_a", "node_b")
    graph.set_start("node_a").set_end("node_b")

    session_service = InMemorySessionService()
    runner = Runner(
        app_name="test_app",
        agent=graph,
        session_service=session_service
    )

    # Create session first
    await session_service.create_session(
        app_name="test_app", user_id="test_user", session_id="test_session"
    )

    async for _ in runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=types.Content(role="user", parts=[types.Part(text="test input")]),
    ):
        pass

    # Verify contexts
    assert len(captured_contexts) == 2

    # First node
    assert captured_contexts[0]["node_name"] == "node_a"
    assert captured_contexts[0]["iteration"] == 1
    assert "input" in captured_contexts[0]["state_data_keys"]
    assert captured_contexts[0]["state_path"] == ["node_a"]

    # Second node
    assert captured_contexts[1]["node_name"] == "node_b"
    assert captured_contexts[1]["iteration"] == 2
    assert captured_contexts[1]["state_path"] == ["node_a", "node_b"]


@pytest.mark.asyncio
async def test_nested_observability_callback():
    """Test create_nested_observability_callback shows hierarchy."""
    graph = GraphAgent(
        name="outer_graph",
        before_node_callback=create_nested_observability_callback()
    )
    node_a = GraphNode(name="node_a", agent=MockAgent("agent_a"))
    graph.add_node(node_a)
    graph.set_start("node_a").set_end("node_a")

    session_service = InMemorySessionService()
    runner = Runner(
        app_name="test_app",
        agent=graph,
        session_service=session_service
    )

    # Create session first
    await session_service.create_session(
        app_name="test_app", user_id="test_user", session_id="test_session"
    )

    events = []
    async for event in runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=types.Content(role="user", parts=[types.Part(text="test input")]),
    ):
        events.append(event)

    # Find observability event
    obs_events = [e for e in events if e.author == "observability"]
    assert len(obs_events) == 1

    # Check hierarchy is shown
    event_text = obs_events[0].content.parts[0].text
    assert "outer_graph" in event_text
    assert "node_a" in event_text


@pytest.mark.asyncio
async def test_both_callbacks_invoked_in_order():
    """Test that before and after callbacks are invoked in correct order."""
    invocation_order = []

    async def before_callback(ctx: NodeCallbackContext) -> Optional[Event]:
        invocation_order.append(f"before_{ctx.node.name}")
        return None

    async def after_callback(ctx: NodeCallbackContext) -> Optional[Event]:
        invocation_order.append(f"after_{ctx.node.name}")
        return None

    graph = GraphAgent(
        name="test_graph",
        before_node_callback=before_callback,
        after_node_callback=after_callback,
    )
    node_a = GraphNode(name="node_a", agent=MockAgent("agent_a"))
    node_b = GraphNode(name="node_b", agent=MockAgent("agent_b"))

    graph.add_node(node_a).add_node(node_b)
    graph.add_edge("node_a", "node_b")
    graph.set_start("node_a").set_end("node_b")

    session_service = InMemorySessionService()
    runner = Runner(
        app_name="test_app",
        agent=graph,
        session_service=session_service
    )

    # Create session first
    await session_service.create_session(
        app_name="test_app", user_id="test_user", session_id="test_session"
    )

    async for _ in runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=types.Content(role="user", parts=[types.Part(text="test input")]),
    ):
        pass

    # Verify order: before_a, after_a, before_b, after_b
    assert invocation_order == [
        "before_node_a",
        "after_node_a",
        "before_node_b",
        "after_node_b",
    ]
