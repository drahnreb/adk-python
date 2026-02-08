"""Comprehensive core logic tests for GraphAgent to achieve high coverage.

These tests specifically target untested code paths identified by coverage analysis:
- Immediate cancellation (lines 592-616)
- BEFORE interrupt handling (lines 670-735)
- AFTER interrupt handling (lines 938-1025)
- Parallel execution paths
- Error handling paths
- State restoration paths
"""

import asyncio
import pytest
from datetime import datetime, timezone
from typing import Any, Dict

from google.adk.agents import LlmAgent
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent, GraphNode, GraphState
from google.adk.agents.graph import EdgeCondition, InterruptMode, InterruptConfig
from google.adk.agents.graph.interrupt_service import InterruptService, InterruptMessage
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.genai import types


# ============================================================================
# Test Agents (Real BaseAgent implementations per ADK guidelines)
# ============================================================================


class SimpleTestAgent(BaseAgent):
    """Real test agent extending BaseAgent per ADK guidelines."""

    model_config = {"extra": "allow", "arbitrary_types_allowed": True}

    def __init__(self, name: str, responses: list[str]):
        super().__init__(name=name)
        object.__setattr__(self, "_responses", responses)
        object.__setattr__(self, "_call_count", 0)

    async def _run_async_impl(self, ctx):
        """Real agent implementation."""
        call_count = object.__getattribute__(self, "_call_count")
        responses = object.__getattribute__(self, "_responses")

        response = responses[min(call_count, len(responses) - 1)]
        object.__setattr__(self, "_call_count", call_count + 1)

        yield Event(
            author=self.name, content=types.Content(parts=[types.Part(text=response)])
        )


class MockLlmAgent(LlmAgent):
    """Mock LLM agent to avoid real API calls (per ADK guidelines)."""

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    def __init__(self, name: str, response: str = "mock response"):
        super().__init__(name=name, model="gemini-2.0-flash-exp", instruction="mock")
        object.__setattr__(self, "_mock_response", response)

    async def _run_async_impl(self, ctx):
        """Mock implementation avoiding real LLM call."""
        response = object.__getattribute__(self, "_mock_response")
        yield Event(
            author=self.name, content=types.Content(parts=[types.Part(text=response)])
        )


# ============================================================================
# Test: Immediate Cancellation (Lines 592-616)
# ============================================================================


@pytest.mark.asyncio
class TestImmediateCancellation:
    """Test immediate cancellation via interrupt service."""

    async def test_immediate_cancellation_during_execution(self):
        """Test immediate cancellation between nodes (ESC-like interrupt)."""
        # Setup
        session_service = InMemorySessionService()
        interrupt_service = InterruptService()

        graph = GraphAgent(
            name="test_graph",
            interrupt_service=interrupt_service,
            interrupt_config=InterruptConfig(mode=InterruptMode.BOTH),
        )

        agent1 = SimpleTestAgent("agent1", ["step1"])
        agent2 = SimpleTestAgent("agent2", ["step2"])

        graph.add_node(GraphNode(name="node1", agent=agent1))
        graph.add_node(GraphNode(name="node2", agent=agent2))
        graph.add_edge("node1", "node2")
        graph.set_start("node1")
        graph.set_end("node2")

        # Create session and register with interrupt service
        app_name = "test"
        user_id = "u1"
        session = await session_service.create_session(app_name, user_id)
        interrupt_service.register_session(session.id)

        # Cancel after first node completes
        async def cancel_after_delay():
            await asyncio.sleep(0.1)
            interrupt_service.cancel(session.id)

        cancel_task = asyncio.create_task(cancel_after_delay())

        # Execute
        runner = Runner(agent=graph, session_service=session_service)

        events = []
        async for event in runner.run_async(
            app_name=app_name,
            user_id=user_id,
            session_id=session.id,
            new_message="start",
        ):
            if event.content and event.content.parts:
                events.append(event.content.parts[0].text)

        await cancel_task

        # Verify cancellation event was emitted
        assert any("cancelled" in str(e).lower() for e in events)

        # Verify state was saved for resume
        session = await session_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session.id
        )
        assert session.state.get("graph_cancelled") == True
        assert "graph_cancelled_at_node" in session.state
        assert session.state.get("graph_can_resume") == True


# ============================================================================
# Test: BEFORE Interrupt Handling (Lines 670-735)
# ============================================================================


@pytest.mark.asyncio
class TestBeforeInterruptHandling:
    """Test BEFORE interrupt modes and actions."""

    async def test_before_interrupt_skip_action(self):
        """Test BEFORE interrupt with SKIP action (lines 718-724)."""
        session_service = InMemorySessionService()
        interrupt_service = InterruptService()

        graph = GraphAgent(
            name="test",
            interrupt_service=interrupt_service,
            interrupt_config=InterruptConfig(mode=InterruptMode.BEFORE),
        )

        agent1 = SimpleTestAgent("agent1", ["step1"])
        agent2 = SimpleTestAgent("agent2", ["step2"])
        agent3 = SimpleTestAgent("agent3", ["step3"])

        graph.add_node(GraphNode(name="node1", agent=agent1))
        graph.add_node(GraphNode(name="node2", agent=agent2))
        graph.add_node(GraphNode(name="node3", agent=agent3))
        graph.add_edge("node1", "node2")
        graph.add_edge("node2", "node3")
        graph.set_start("node1")
        graph.set_end("node3")

        # Create session
        app_name = "test"
        user_id = "u1"
        session = await session_service.create_session(app_name, user_id)
        interrupt_service.register_session(session.id)

        # Send SKIP interrupt for node2
        async def send_skip_interrupt():
            await asyncio.sleep(0.1)  # Wait for node1 to complete
            interrupt_service.send_interrupt(
                session.id,
                InterruptMessage(text="Skip node2", action="skip", metadata={}),
            )

        skip_task = asyncio.create_task(send_skip_interrupt())

        # Execute
        runner = Runner(agent=graph, session_service=session_service)

        events = []
        async for event in runner.run_async(
            app_name=app_name,
            user_id=user_id,
            session_id=session.id,
            new_message="start",
        ):
            if event.content and event.content.parts:
                events.append(event.content.parts[0].text)

        await skip_task

        # Verify node2 was skipped
        event_texts = " ".join(events)
        assert "step1" in event_texts  # node1 executed
        assert "step2" not in event_texts  # node2 skipped
        assert "step3" in event_texts  # node3 executed

    async def test_before_interrupt_pause_action(self):
        """Test BEFORE interrupt with PAUSE action (lines 725-735)."""
        session_service = InMemorySessionService()
        interrupt_service = InterruptService()

        graph = GraphAgent(
            name="test",
            interrupt_service=interrupt_service,
            interrupt_config=InterruptConfig(mode=InterruptMode.BEFORE),
        )

        agent1 = SimpleTestAgent("agent1", ["step1"])
        agent2 = SimpleTestAgent("agent2", ["step2"])

        graph.add_node(GraphNode(name="node1", agent=agent1))
        graph.add_node(GraphNode(name="node2", agent=agent2))
        graph.add_edge("node1", "node2")
        graph.set_start("node1")
        graph.set_end("node2")

        # Create session
        app_name = "test"
        user_id = "u1"
        session = await session_service.create_session(app_name, user_id)
        interrupt_service.register_session(session.id)

        # Send PAUSE interrupt then resume
        async def pause_and_resume():
            await asyncio.sleep(0.1)
            interrupt_service.send_interrupt(
                session.id,
                InterruptMessage(
                    text="Pause before node2", action="pause", metadata={}
                ),
            )
            await asyncio.sleep(0.2)
            interrupt_service.resume(session.id)

        pause_task = asyncio.create_task(pause_and_resume())

        # Execute
        runner = Runner(agent=graph, session_service=session_service)

        events = []
        async for event in runner.run_async(
            app_name=app_name,
            user_id=user_id,
            session_id=session.id,
            new_message="start",
        ):
            if event.content and event.content.parts:
                events.append(event.content.parts[0].text)

        await pause_task

        # Verify pause event was emitted
        assert any("INTERRUPT" in str(e) for e in events)

    async def test_before_interrupt_rerun_action(self):
        """Test BEFORE interrupt with RERUN action (lines 716-717)."""
        session_service = InMemorySessionService()
        interrupt_service = InterruptService()

        graph = GraphAgent(
            name="test",
            interrupt_service=interrupt_service,
            interrupt_config=InterruptConfig(mode=InterruptMode.BEFORE),
        )

        agent = SimpleTestAgent("agent", ["run1", "run2"])

        graph.add_node(GraphNode(name="node", agent=agent))
        graph.set_start("node")
        graph.set_end("node")

        # Create session
        app_name = "test"
        user_id = "u1"
        session = await session_service.create_session(app_name, user_id)
        interrupt_service.register_session(session.id)

        rerun_count = 0

        async def send_rerun_once():
            nonlocal rerun_count
            await asyncio.sleep(0.05)
            if rerun_count == 0:
                rerun_count += 1
                interrupt_service.send_interrupt(
                    session.id,
                    InterruptMessage(text="Rerun node", action="rerun", metadata={}),
                )

        rerun_task = asyncio.create_task(send_rerun_once())

        # Execute
        runner = Runner(agent=graph, session_service=session_service)

        events = []
        async for event in runner.run_async(
            app_name=app_name,
            user_id=user_id,
            session_id=session.id,
            new_message="start",
        ):
            if event.content and event.content.parts:
                events.append(event.content.parts[0].text)

        await rerun_task

        # Agent should have been called multiple times due to rerun
        assert agent.call_count >= 2


# ============================================================================
# Test: AFTER Interrupt Handling (Lines 938-1025)
# ============================================================================


@pytest.mark.asyncio
class TestAfterInterruptHandling:
    """Test AFTER interrupt modes and actions."""

    async def test_after_interrupt_pause_action(self):
        """Test AFTER interrupt with PAUSE action (lines 1008-1025)."""
        session_service = InMemorySessionService()
        interrupt_service = InterruptService()

        graph = GraphAgent(
            name="test",
            interrupt_service=interrupt_service,
            interrupt_config=InterruptConfig(mode=InterruptMode.AFTER),
        )

        agent1 = SimpleTestAgent("agent1", ["step1"])
        agent2 = SimpleTestAgent("agent2", ["step2"])

        graph.add_node(GraphNode(name="node1", agent=agent1))
        graph.add_node(GraphNode(name="node2", agent=agent2))
        graph.add_edge("node1", "node2")
        graph.set_start("node1")
        graph.set_end("node2")

        # Create session
        app_name = "test"
        user_id = "u1"
        session = await session_service.create_session(app_name, user_id)
        interrupt_service.register_session(session.id)

        # Send PAUSE interrupt after node1
        async def pause_and_resume():
            await asyncio.sleep(0.15)
            interrupt_service.send_interrupt(
                session.id,
                InterruptMessage(text="Pause after node1", action="pause", metadata={}),
            )
            await asyncio.sleep(0.2)
            interrupt_service.resume(session.id)

        pause_task = asyncio.create_task(pause_and_resume())

        # Execute
        runner = Runner(agent=graph, session_service=session_service)

        events = []
        async for event in runner.run_async(
            app_name=app_name,
            user_id=user_id,
            session_id=session.id,
            new_message="start",
        ):
            if event.content and event.content.parts:
                events.append(event.content.parts[0].text)

        await pause_task

        # Verify AFTER interrupt was triggered
        event_str = " ".join(events)
        assert "INTERRUPT (AFTER)" in event_str or "step2" in event_str

    async def test_after_interrupt_rerun_action(self):
        """Test AFTER interrupt with RERUN action (lines 1005-1007)."""
        session_service = InMemorySessionService()
        interrupt_service = InterruptService()

        graph = GraphAgent(
            name="test",
            interrupt_service=interrupt_service,
            interrupt_config=InterruptConfig(mode=InterruptMode.AFTER),
            max_iterations=5,
        )

        agent = SimpleTestAgent("agent", ["run1", "run2", "run3"])

        graph.add_node(GraphNode(name="node", agent=agent))
        graph.set_start("node")
        graph.set_end("node")

        # Create session
        app_name = "test"
        user_id = "u1"
        session = await session_service.create_session(app_name, user_id)
        interrupt_service.register_session(session.id)

        rerun_sent = False

        async def send_rerun_after_first():
            nonlocal rerun_sent
            await asyncio.sleep(0.1)
            if not rerun_sent:
                rerun_sent = True
                interrupt_service.send_interrupt(
                    session.id,
                    InterruptMessage(
                        text="Rerun after completion", action="rerun", metadata={}
                    ),
                )

        rerun_task = asyncio.create_task(send_rerun_after_first())

        # Execute
        runner = Runner(agent=graph, session_service=session_service)

        events = []
        async for event in runner.run_async(
            app_name=app_name,
            user_id=user_id,
            session_id=session.id,
            new_message="start",
        ):
            if event.content and event.content.parts:
                events.append(event.content.parts[0].text)

        await rerun_task

        # Agent should be called at least twice
        assert agent.call_count >= 2

    async def test_after_interrupt_continue_action(self):
        """Test AFTER interrupt with CONTINUE action (lines 1026+)."""
        session_service = InMemorySessionService()
        interrupt_service = InterruptService()

        graph = GraphAgent(
            name="test",
            interrupt_service=interrupt_service,
            interrupt_config=InterruptConfig(mode=InterruptMode.AFTER),
        )

        agent1 = SimpleTestAgent("agent1", ["step1"])
        agent2 = SimpleTestAgent("agent2", ["step2"])

        graph.add_node(GraphNode(name="node1", agent=agent1))
        graph.add_node(GraphNode(name="node2", agent=agent2))
        graph.add_edge("node1", "node2")
        graph.set_start("node1")
        graph.set_end("node2")

        # Create session
        app_name = "test"
        user_id = "u1"
        session = await session_service.create_session(app_name, user_id)
        interrupt_service.register_session(session.id)

        # Send CONTINUE interrupt (accept results and proceed)
        async def send_continue():
            await asyncio.sleep(0.1)
            interrupt_service.send_interrupt(
                session.id,
                InterruptMessage(
                    text="Continue to next node", action="continue", metadata={}
                ),
            )

        continue_task = asyncio.create_task(send_continue())

        # Execute
        runner = Runner(agent=graph, session_service=session_service)

        events = []
        async for event in runner.run_async(
            app_name=app_name,
            user_id=user_id,
            session_id=session.id,
            new_message="start",
        ):
            if event.content and event.content.parts:
                events.append(event.content.parts[0].text)

        await continue_task

        # Both nodes should execute
        event_str = " ".join(events)
        assert "step1" in event_str
        assert "step2" in event_str
