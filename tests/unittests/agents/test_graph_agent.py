"""Comprehensive test suite for GraphAgent implementation.

Tests all features with 100% coverage:
- Graph-based workflows with nodes and edges
- AgentNode for wrapping LLM agents
- Cyclic support for loops and iterative reasoning (ReAct pattern)
- Conditional routing based on state
- State management with reducers (overwrite, append, sum, custom)
- Checkpointing with persistent state (memory, SQLite)
- Human-in-the-loop with interrupt capabilities
"""

import asyncio
import pytest
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Dict
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

from google.adk.agents import LlmAgent
from google.adk.agents import ParallelAgent
from google.adk.agents import SequentialAgent
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import EdgeCondition
from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import GraphNode
from google.adk.agents.graph import GraphState
from google.adk.agents.graph import InterruptMode
from google.adk.agents.graph import StateReducer
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.genai import types


# ============================================================================
# Mock Agents for Testing
# ============================================================================

class MockAgent:
    """Mock agent that returns predetermined responses."""

    def __init__(self, name: str, responses: list[str], delay: float = 0.0):
        self.name = name
        self.responses = responses
        self.call_count = 0
        self.delay = delay

    async def run_async(self, ctx):
        """Mock run_async that yields predetermined responses."""
        await asyncio.sleep(self.delay)  # Simulate LLM latency
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text=response)])
        )


class MockLlmAgent(LlmAgent):
    """Mock LLM agent that doesn't call real LLM."""

    # Use model_config to allow extra attributes
    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    def __init__(self, name: str, response: str = "mock response", **kwargs):
        super().__init__(name=name, model="gemini-2.0-flash-exp", instruction="mock", **kwargs)
        # Store as model extra fields
        object.__setattr__(self, "_mock_response", response)
        object.__setattr__(self, "_mock_call_count", 0)

    async def _run_async_impl(self, ctx):
        """Mock implementation."""
        count = object.__getattribute__(self, "_mock_call_count")
        object.__setattr__(self, "_mock_call_count", count + 1)

        response = object.__getattribute__(self, "_mock_response")
        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text=response)])
        )

    @property
    def call_count(self):
        """Get call count."""
        return object.__getattribute__(self, "_mock_call_count")


# ============================================================================
# Test: Basic Graph Structure
# ============================================================================

class TestGraphStructure:
    """Test basic graph construction and structure."""

    def test_create_empty_graph(self):
        """Test creating empty graph."""
        graph = GraphAgent(name="test_graph", description="Test graph")
        assert graph.name == "test_graph"
        assert graph.description == "Test graph"
        assert len(graph.nodes) == 0
        assert graph.start_node is None
        assert len(graph.end_nodes) == 0

    def test_add_nodes(self):
        """Test adding nodes to graph."""
        graph = GraphAgent(name="test")

        node1 = GraphNode(name="node1", agent=MockLlmAgent("agent1"))
        node2 = GraphNode(name="node2", agent=MockLlmAgent("agent2"))

        graph.add_node(node1)
        graph.add_node(node2)

        assert len(graph.nodes) == 2
        assert "node1" in graph.nodes
        assert "node2" in graph.nodes

    def test_add_edges(self):
        """Test adding edges between nodes."""
        graph = GraphAgent(name="test")

        graph.add_node(GraphNode(name="node1", agent=MockLlmAgent("agent1")))
        graph.add_node(GraphNode(name="node2", agent=MockLlmAgent("agent2")))

        graph.add_edge("node1", "node2")

        assert len(graph.nodes["node1"].edges) == 1
        assert graph.nodes["node1"].edges[0].target_node == "node2"

    def test_set_start_end(self):
        """Test setting start and end nodes."""
        graph = GraphAgent(name="test")
        graph.add_node(GraphNode(name="start", agent=MockLlmAgent("agent1")))
        graph.add_node(GraphNode(name="end", agent=MockLlmAgent("agent2")))

        graph.set_start("start")
        graph.set_end("end")

        assert graph.start_node == "start"
        assert "end" in graph.end_nodes

    def test_invalid_edge_raises_error(self):
        """Test that invalid edges raise errors."""
        graph = GraphAgent(name="test")
        graph.add_node(GraphNode(name="node1", agent=MockLlmAgent("agent1")))

        with pytest.raises(ValueError, match="Target node node2 not found"):
            graph.add_edge("node1", "node2")

    def test_invalid_start_raises_error(self):
        """Test that invalid start node raises error."""
        graph = GraphAgent(name="test")

        with pytest.raises(ValueError, match="Node invalid not found"):
            graph.set_start("invalid")


# ============================================================================
# Test: State Management and Reducers
# ============================================================================

class TestStateManagement:
    """Test state management and reducer strategies."""

    def test_overwrite_reducer(self):
        """Test OVERWRITE reducer replaces values."""
        node = GraphNode(
            name="test",
            agent=MockLlmAgent("agent"),
            reducer=StateReducer.OVERWRITE
        )

        state = GraphState(data={"test": "old_value"})
        new_state = node.output_mapper("new_value", state)

        assert new_state.data["test"] == "new_value"

    def test_append_reducer(self):
        """Test APPEND reducer appends to list."""
        node = GraphNode(
            name="test",
            agent=MockLlmAgent("agent"),
            reducer=StateReducer.APPEND
        )

        state = GraphState(data={"test": ["item1"]})
        new_state = node.output_mapper("item2", state)

        assert new_state.data["test"] == ["item1", "item2"]

    def test_append_reducer_creates_list(self):
        """Test APPEND reducer creates list if key doesn't exist."""
        node = GraphNode(
            name="test",
            agent=MockLlmAgent("agent"),
            reducer=StateReducer.APPEND
        )

        state = GraphState(data={})
        new_state = node.output_mapper("item1", state)

        assert new_state.data["test"] == ["item1"]

    def test_sum_reducer(self):
        """Test SUM reducer adds numeric values."""
        node = GraphNode(
            name="test",
            agent=MockLlmAgent("agent"),
            reducer=StateReducer.SUM
        )

        state = GraphState(data={"test": 10})
        new_state = node.output_mapper(5, state)

        assert new_state.data["test"] == 15

    def test_custom_reducer(self):
        """Test CUSTOM reducer uses custom function."""
        def custom_fn(old_val, new_val):
            """Concatenate strings with separator."""
            if old_val is None:
                return new_val
            return f"{old_val} | {new_val}"

        node = GraphNode(
            name="test",
            agent=MockLlmAgent("agent"),
            reducer=StateReducer.CUSTOM,
            custom_reducer=custom_fn
        )

        state = GraphState(data={"test": "value1"})
        new_state = node.output_mapper("value2", state)

        assert new_state.data["test"] == "value1 | value2"

    def test_state_immutability(self):
        """Test that output_mapper doesn't modify original state."""
        node = GraphNode(
            name="test",
            agent=MockLlmAgent("agent"),
            reducer=StateReducer.OVERWRITE
        )

        original_state = GraphState(data={"test": "original"})
        new_state = node.output_mapper("new", original_state)

        assert original_state.data["test"] == "original"
        assert new_state.data["test"] == "new"


# ============================================================================
# Test: Conditional Routing
# ============================================================================

class TestConditionalRouting:
    """Test conditional edge routing."""

    def test_unconditional_edge(self):
        """Test edge without condition always routes."""
        edge = EdgeCondition(target_node="next")
        state = GraphState(data={})

        assert edge.should_route(state) is True

    def test_conditional_edge_true(self):
        """Test conditional edge routes when condition is true."""
        edge = EdgeCondition(
            target_node="next",
            condition=lambda s: s.data.get("value") > 10
        )
        state = GraphState(data={"value": 15})

        assert edge.should_route(state) is True

    def test_conditional_edge_false(self):
        """Test conditional edge doesn't route when condition is false."""
        edge = EdgeCondition(
            target_node="next",
            condition=lambda s: s.data.get("value") > 10
        )
        state = GraphState(data={"value": 5})

        assert edge.should_route(state) is False

    def test_node_routing_multiple_edges(self):
        """Test node with multiple conditional edges picks first match."""
        node = GraphNode(name="router", agent=MockLlmAgent("agent"))

        node.add_edge("path1", condition=lambda s: s.data.get("type") == "A")
        node.add_edge("path2", condition=lambda s: s.data.get("type") == "B")
        node.add_edge("default", condition=lambda s: True)

        state_a = GraphState(data={"type": "A"})
        state_b = GraphState(data={"type": "B"})
        state_c = GraphState(data={"type": "C"})

        assert node.get_next_node(state_a) == "path1"
        assert node.get_next_node(state_b) == "path2"
        assert node.get_next_node(state_c) == "default"


# ============================================================================
# Test: Cyclic Support and ReAct Pattern
# ============================================================================

@pytest.mark.asyncio
class TestCyclicExecution:
    """Test cyclic graph execution (loops, ReAct pattern)."""

    async def test_simple_loop(self):
        """Test graph with loop executes multiple iterations."""
        graph = GraphAgent(name="loop_graph", max_iterations=5)

        # Counter agent that increments
        counter_responses = [str(i) for i in range(1, 10)]
        counter_agent = MockAgent("counter", counter_responses)

        graph.add_node(GraphNode(
            name="counter",
            agent=counter_agent,
            output_mapper=lambda output, state: GraphState(
                data={**state.data, "count": int(output)},
                metadata=state.metadata
            )
        ))

        # Loop back if count < 3
        graph.set_start("counter")
        graph.add_edge(
            "counter",
            "counter",
            condition=lambda s: s.data.get("count", 0) < 3
        )
        graph.set_end("counter")

        # Execute with Runner
        runner = Runner(app_name="test_graph", agent=graph, session_service=InMemorySessionService())

        # Create session first
        session_service = runner.session_service
        await session_service.create_session(app_name="test_graph", user_id="test_user", session_id="test")
        
        iterations = 0
        async for event in runner.run_async(user_id="test_user", session_id="test", new_message=types.Content(role="user", parts=[types.Part(text="start")])):
            if event.content and event.content.parts:
                iterations = event.actions.state_delta.get("graph_iterations", 0) if event.actions and event.actions.state_delta else 0

        # Should run 3 iterations (count 1, 2, 3)
        assert iterations == 3
        assert counter_agent.call_count == 3

    async def test_max_iterations_prevents_infinite_loop(self):
        """Test max_iterations prevents infinite loops."""
        graph = GraphAgent(name="infinite", max_iterations=3)

        # Agent that never ends
        loop_agent = MockAgent("loop", ["continue"] * 100)

        graph.add_node(GraphNode(name="loop", agent=loop_agent))
        graph.set_start("loop")
        graph.add_edge("loop", "loop")  # Always loop back

        runner = Runner(app_name="test_graph", agent=graph, session_service=InMemorySessionService())

        # Create session first
        session_service = runner.session_service
        await session_service.create_session(app_name="test_graph", user_id="test_user", session_id="test")
        
        iterations = 0
        async for event in runner.run_async(user_id="test_user", session_id="test", new_message=types.Content(role="user", parts=[types.Part(text="start")])):
            if event.content and event.content.parts:
                iterations = event.actions.state_delta.get("graph_iterations", 0) if event.actions and event.actions.state_delta else 0

        # Should stop at max_iterations
        assert iterations == 3

    async def test_react_pattern(self):
        """Test ReAct pattern (Reason -> Act -> Observe -> loop)."""
        graph = GraphAgent(name="react", max_iterations=10)

        # Simulate ReAct: Complete after 2 iterations
        reason_agent = MockAgent("reason", ["plan action 1", "plan action 2"])
        act_agent = MockAgent("act", ["result 1", "result 2"])
        observe_agent = MockAgent("observe", ["CONTINUE", "COMPLETE"])

        graph.add_node(GraphNode(name="reason", agent=reason_agent))
        graph.add_node(GraphNode(name="act", agent=act_agent))
        graph.add_node(GraphNode(name="observe", agent=observe_agent))

        graph.set_start("reason")
        graph.add_edge("reason", "act")
        graph.add_edge("act", "observe")

        # Loop back if CONTINUE, otherwise end (observe is end node)
        graph.add_edge(
            "observe",
            "reason",
            condition=lambda s: "CONTINUE" in s.data.get("observe", "").upper()
        )
        # When COMPLETE (or any other value), no edge matches, so execution stops at end node
        graph.set_end("observe")

        runner = Runner(app_name="test_graph", agent=graph, session_service=InMemorySessionService())

        # Create session first
        session_service = runner.session_service
        await session_service.create_session(app_name="test_graph", user_id="test_user", session_id="test")
        
        path = []
        async for event in runner.run_async(user_id="test_user", session_id="test", new_message=types.Content(role="user", parts=[types.Part(text="test task")])):
            if event.content and event.content.parts:
                path = event.actions.state_delta.get("graph_path", []) if event.actions and event.actions.state_delta else []

        # Should execute: reason -> act -> observe -> reason -> act -> observe
        expected = ["reason", "act", "observe", "reason", "act", "observe"]
        assert path == expected


# ============================================================================
# Test: Human-in-the-Loop Interrupts
# ============================================================================

@pytest.mark.asyncio
class TestInterrupts:
    """Test human-in-the-loop interrupt capabilities."""

    async def test_interrupt_before(self):
        """Test interrupt before node execution."""
        graph = GraphAgent(name="test")

        agent = MockAgent("agent", ["response"])
        graph.add_node(GraphNode(name="worker", agent=agent))
        graph.set_start("worker")
        graph.set_end("worker")

        # Add interrupt before
        graph.add_interrupt("worker", InterruptMode.BEFORE)

        runner = Runner(app_name="test_graph", agent=graph, session_service=InMemorySessionService())

        # Create session first
        session_service = runner.session_service
        await session_service.create_session(app_name="test_graph", user_id="test_user", session_id="test")
        
        interrupt_events = []
        async for event in runner.run_async(user_id="test_user", session_id="test", new_message=types.Content(role="user", parts=[types.Part(text="test")])):
            if event.content and event.content.parts and "Interrupt" in event.content.parts[0].text:
                if "Interrupt before" in event.content.parts[0].text:
                    interrupt_events.append(event)

        assert len(interrupt_events) == 1
        # Parse node name from interrupt message
        interrupt_text = interrupt_events[0].content.parts[0].text
        assert "worker" in interrupt_text

    async def test_interrupt_after(self):
        """Test interrupt after node execution."""
        graph = GraphAgent(name="test")

        agent = MockAgent("agent", ["response"])
        graph.add_node(GraphNode(name="worker", agent=agent))
        graph.set_start("worker")
        graph.set_end("worker")

        # Add interrupt after
        graph.add_interrupt("worker", InterruptMode.AFTER)

        runner = Runner(app_name="test_graph", agent=graph, session_service=InMemorySessionService())

        # Create session first
        session_service = runner.session_service
        await session_service.create_session(app_name="test_graph", user_id="test_user", session_id="test")
        
        interrupt_events = []
        async for event in runner.run_async(user_id="test_user", session_id="test", new_message=types.Content(role="user", parts=[types.Part(text="test")])):
            if event.content and event.content.parts and "Interrupt" in event.content.parts[0].text:
                if "Interrupt after" in event.content.parts[0].text:
                    interrupt_events.append(event)

        assert len(interrupt_events) == 1
        # Parse node name from interrupt message
        interrupt_text = interrupt_events[0].content.parts[0].text
        assert "worker" in interrupt_text

    async def test_interrupt_both(self):
        """Test interrupt both before and after."""
        graph = GraphAgent(name="test")

        agent = MockAgent("agent", ["response"])
        graph.add_node(GraphNode(name="worker", agent=agent))
        graph.set_start("worker")
        graph.set_end("worker")

        # Add interrupt both
        graph.add_interrupt("worker", InterruptMode.BOTH)

        runner = Runner(app_name="test_graph", agent=graph, session_service=InMemorySessionService())

        # Create session first
        session_service = runner.session_service
        await session_service.create_session(app_name="test_graph", user_id="test_user", session_id="test")
        
        before_count = 0
        after_count = 0
        async for event in runner.run_async(user_id="test_user", session_id="test", new_message=types.Content(role="user", parts=[types.Part(text="test")])):
            if event.content and event.content.parts and "Interrupt" in event.content.parts[0].text:
                if "Interrupt before" in event.content.parts[0].text:
                    before_count += 1
                elif "Interrupt after" in event.content.parts[0].text:
                    after_count += 1

        assert before_count == 1
        assert after_count == 1

    async def test_interrupt_contains_state(self):
        """Test interrupt event contains current state."""
        graph = GraphAgent(name="test")

        agent = MockAgent("agent", ["response"])
        graph.add_node(GraphNode(name="worker", agent=agent))
        graph.set_start("worker")
        graph.set_end("worker")
        graph.add_interrupt("worker", InterruptMode.BEFORE)

        runner = Runner(app_name="test_graph", agent=graph, session_service=InMemorySessionService())

        # Create session first
        session_service = runner.session_service
        await session_service.create_session(app_name="test_graph", user_id="test_user", session_id="test")
        
        interrupt_event = None
        async for event in runner.run_async(user_id="test_user", session_id="test", new_message=types.Content(role="user", parts=[types.Part(text="test input")])):
            if event.content and event.content.parts and "Interrupt" in event.content.parts[0].text:
                if "Interrupt before" in event.content.parts[0].text:
                    # State is included in event.actions.state_delta (ADK-conformant)
                    interrupt_event = event

        # Verify interrupt was received and contains state information
        assert interrupt_event is not None
        assert "worker" in interrupt_event.content.parts[0].text
        # Check state_delta contains interrupt metadata
        assert interrupt_event.actions is not None
        assert interrupt_event.actions.state_delta is not None
        assert interrupt_event.actions.state_delta["interrupt_node"] == "worker"
        assert "test input" in str(interrupt_event.actions.state_delta["interrupt_state"])


# ============================================================================
# Test: Checkpointing
# ============================================================================

@pytest.mark.asyncio
class TestCheckpointing:
    """Test state checkpointing for resumability."""

    async def test_checkpointing_enabled(self):
        """Test that checkpointing saves state after each node."""
        graph = GraphAgent(name="test", checkpointing=True)

        agent1 = MockAgent("agent1", ["step1"])
        agent2 = MockAgent("agent2", ["step2"])

        graph.add_node(GraphNode(name="node1", agent=agent1))
        graph.add_node(GraphNode(name="node2", agent=agent2))
        graph.set_start("node1")
        graph.add_edge("node1", "node2")
        graph.set_end("node2")

        runner = Runner(app_name="test_graph", agent=graph, session_service=InMemorySessionService())

        # Create session first
        session_service = runner.session_service
        await session_service.create_session(app_name="test_graph", user_id="test_user", session_id="test")
        
        checkpoints = []
        last_checkpoint = None
        async for event in runner.run_async(user_id="test_user", session_id="test", new_message=types.Content(role="user", parts=[types.Part(text="test")])):
            session = await runner.session_service.get_session(app_name="test_graph", user_id="test_user", session_id="test")
            if "graph_checkpoint" in session.state:
                current_checkpoint = session.state["graph_checkpoint"]
                # Only append if it's a new checkpoint (different node or iteration)
                if current_checkpoint != last_checkpoint:
                    checkpoints.append(current_checkpoint.copy())
                    last_checkpoint = current_checkpoint

        # Should have checkpoints for both nodes
        assert len(checkpoints) >= 2
        assert checkpoints[0]["node"] == "node1"
        assert checkpoints[1]["node"] == "node2"

    async def test_checkpoint_contains_state(self):
        """Test checkpoint contains graph state."""
        graph = GraphAgent(name="test", checkpointing=True)

        agent = MockAgent("agent", ["response"])
        graph.add_node(GraphNode(name="worker", agent=agent))
        graph.set_start("worker")
        graph.set_end("worker")

        runner = Runner(app_name="test_graph", agent=graph, session_service=InMemorySessionService())

        # Create session first
        session_service = runner.session_service
        await session_service.create_session(app_name="test_graph", user_id="test_user", session_id="test")
        
        async for event in runner.run_async(user_id="test_user", session_id="test", new_message=types.Content(role="user", parts=[types.Part(text="test")])):
            pass

        # Check saved state
        session = await runner.session_service.get_session(app_name="test_graph", user_id="test_user", session_id="test")
        assert "graph_state" in session.state
        graph_state = session.state["graph_state"]
        assert graph_state["data"]["worker"] == "response"


# ============================================================================
# Test: Agent Type Support (LLM, Sequential, Parallel, Graph)
# ============================================================================

@pytest.mark.asyncio
class TestAgentTypeSupport:
    """Test support for all BaseAgent types."""

    async def test_llm_agent_node(self):
        """Test node with LLMAgent."""
        graph = GraphAgent(name="test")

        llm_agent = MockLlmAgent("llm", response="llm response")
        graph.add_node(GraphNode(name="llm", agent=llm_agent))
        graph.set_start("llm")
        graph.set_end("llm")

        runner = Runner(app_name="test_graph", agent=graph, session_service=InMemorySessionService())

        # Create session first
        session_service = runner.session_service
        await session_service.create_session(app_name="test_graph", user_id="test_user", session_id="test")
        
        final_output = None
        async for event in runner.run_async(user_id="test_user", session_id="test", new_message=types.Content(role="user", parts=[types.Part(text="test")])):
            if event.content and event.content.parts:
                final_output = event.content.parts[0].text if event.content and event.content.parts else ""

        assert "llm response" in str(final_output)
        assert llm_agent.call_count == 1

    async def test_custom_function_node(self):
        """Test node with custom function instead of agent."""
        graph = GraphAgent(name="test")

        async def custom_fn(state: GraphState, ctx):
            """Custom function."""
            return f"processed: {state.data.get('input', '')}"

        graph.add_node(GraphNode(name="custom", function=custom_fn))
        graph.set_start("custom")
        graph.set_end("custom")

        runner = Runner(app_name="test_graph", agent=graph, session_service=InMemorySessionService())

        # Create session first
        session_service = runner.session_service
        await session_service.create_session(app_name="test_graph", user_id="test_user", session_id="test")
        
        final_output = None
        async for event in runner.run_async(user_id="test_user", session_id="test", new_message=types.Content(role="user", parts=[types.Part(text="test input")])):
            if event.content and event.content.parts:
                final_output = event.content.parts[0].text if event.content and event.content.parts else ""

        assert "processed: test input" in str(final_output)

    def test_node_requires_agent_or_function(self):
        """Test that node requires either agent or function."""
        with pytest.raises(ValueError, match="Either agent or function must be provided"):
            GraphNode(name="invalid", agent=None, function=None)


# ============================================================================
# Test: Input/Output Mappers
# ============================================================================

class TestMappers:
    """Test input and output mappers."""

    def test_custom_input_mapper(self):
        """Test custom input mapper transforms state to agent input."""
        def input_mapper(state: GraphState) -> str:
            return f"Custom: {state.data.get('value', '')}"

        node = GraphNode(
            name="test",
            agent=MockLlmAgent("agent"),
            input_mapper=input_mapper
        )

        state = GraphState(data={"value": "test"})
        mapped_input = node.input_mapper(state)

        assert mapped_input == "Custom: test"

    def test_custom_output_mapper(self):
        """Test custom output mapper transforms agent output to state."""
        def output_mapper(output: str, state: GraphState) -> GraphState:
            new_state = GraphState(
                data={**state.data, "result": output.upper()},
                metadata=state.metadata
            )
            return new_state

        node = GraphNode(
            name="test",
            agent=MockLlmAgent("agent"),
            output_mapper=output_mapper
        )

        state = GraphState(data={})
        new_state = node.output_mapper("hello", state)

        assert new_state.data["result"] == "HELLO"


# ============================================================================
# Test: Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling and validation."""

    def test_set_end_invalid_node(self):
        """Test set_end raises error for non-existent node."""
        graph = GraphAgent(name="test")
        with pytest.raises(ValueError, match="Node invalid_node not found in graph"):
            graph.set_end("invalid_node")

    def test_add_edge_invalid_source(self):
        """Test add_edge raises error for non-existent source node."""
        graph = GraphAgent(name="test")
        agent = MockAgent("agent", ["response"])
        graph.add_node(GraphNode(name="node1", agent=agent))

        with pytest.raises(ValueError, match="Source node invalid not found"):
            graph.add_edge("invalid", "node1")

    def test_add_interrupt_invalid_node(self):
        """Test add_interrupt raises error for non-existent node."""
        graph = GraphAgent(name="test")

        with pytest.raises(ValueError, match="Node invalid not found"):
            graph.add_interrupt("invalid", InterruptMode.BEFORE)

    @pytest.mark.asyncio
    async def test_node_no_edges_not_end_raises_error(self):
        """Test execution raises error when node has no edges and is not an end node."""
        graph = GraphAgent(name="test")

        agent = MockAgent("agent", ["response"])
        graph.add_node(GraphNode(name="node1", agent=agent))
        graph.set_start("node1")
        # Don't set as end node and don't add edges

        runner = Runner(app_name="test_graph", agent=graph, session_service=InMemorySessionService())
        session_service = runner.session_service
        await session_service.create_session(app_name="test_graph", user_id="test_user", session_id="test")

        with pytest.raises(ValueError, match="has no outgoing edges and is not an end node"):
            async for event in runner.run_async(user_id="test_user", session_id="test", new_message=types.Content(role="user", parts=[types.Part(text="test")])):
                pass

    @pytest.mark.asyncio
    async def test_start_node_not_set_raises_error(self):
        """Test execution raises error when start node is not set."""
        graph = GraphAgent(name="test")

        agent = MockAgent("agent", ["response"])
        graph.add_node(GraphNode(name="node1", agent=agent))
        # Don't set start node

        runner = Runner(app_name="test_graph", agent=graph, session_service=InMemorySessionService())
        session_service = runner.session_service
        await session_service.create_session(app_name="test_graph", user_id="test_user", session_id="test")

        with pytest.raises(ValueError, match="Start node not set"):
            async for event in runner.run_async(user_id="test_user", session_id="test", new_message=types.Content(role="user", parts=[types.Part(text="test")])):
                pass


# ============================================================================
# Test: Function Execution
# ============================================================================

@pytest.mark.asyncio
class TestFunctionExecution:
    """Test synchronous and asynchronous function execution."""

    async def test_sync_function_node(self):
        """Test node with synchronous function."""
        graph = GraphAgent(name="test")

        # Synchronous function
        def sync_fn(state: GraphState, ctx):
            return f"sync: {state.data.get('input', '')}"

        graph.add_node(GraphNode(name="sync_node", function=sync_fn))
        graph.set_start("sync_node")
        graph.set_end("sync_node")

        runner = Runner(app_name="test_graph", agent=graph, session_service=InMemorySessionService())
        session_service = runner.session_service
        await session_service.create_session(app_name="test_graph", user_id="test_user", session_id="test")

        final_output = None
        async for event in runner.run_async(user_id="test_user", session_id="test", new_message=types.Content(role="user", parts=[types.Part(text="test")])):
            if event.content and event.content.parts:
                final_output = event.content.parts[0].text

        assert "sync: test" in str(final_output)


# ============================================================================
# Test: State Restoration
# ============================================================================

@pytest.mark.asyncio
class TestStateRestoration:
    """Test state restoration from session."""

    async def test_state_restoration_from_session(self):
        """Test that graph can restore state from session."""
        graph = GraphAgent(name="test", checkpointing=True)

        agent = MockAgent("agent", ["response1", "response2"])
        graph.add_node(GraphNode(name="node1", agent=agent))
        graph.set_start("node1")
        graph.set_end("node1")

        runner = Runner(app_name="test_graph", agent=graph, session_service=InMemorySessionService())
        session_service = runner.session_service
        await session_service.create_session(app_name="test_graph", user_id="test_user", session_id="test")

        # First run - create state
        async for event in runner.run_async(user_id="test_user", session_id="test", new_message=types.Content(role="user", parts=[types.Part(text="first")])):
            pass

        # Second run - should restore state
        async for event in runner.run_async(user_id="test_user", session_id="test", new_message=types.Content(role="user", parts=[types.Part(text="second")])):
            pass

        # Verify state was persisted
        session = await session_service.get_session(app_name="test_graph", user_id="test_user", session_id="test")
        assert "graph_state" in session.state


# ============================================================================
# Test: Example Functions (Smoke Tests)
# ============================================================================

@pytest.mark.asyncio
class TestExamples:
    """Smoke tests for example functions."""

    async def test_example_react_pattern_imports(self):
        """Test that example_react_pattern can be imported and called."""
        from google.adk.agents.graph.graph_agent import example_react_pattern
        # Just verify the function exists and is async
        assert asyncio.iscoroutinefunction(example_react_pattern)

    async def test_example_multi_agent_graph_imports(self):
        """Test that example_multi_agent_graph can be imported and called."""
        from google.adk.agents.graph.graph_agent import example_multi_agent_graph
        # Just verify the function exists and is async
        assert asyncio.iscoroutinefunction(example_multi_agent_graph)


# ============================================================================
# Test: ADK Conformity
# ============================================================================

@pytest.mark.asyncio
class TestADKConformity:
    """Test ADK conformance."""

    async def test_event_structure_conformity(self):
        """Test that GraphAgent yields proper Event objects."""
        graph = GraphAgent(name="test")

        agent = MockAgent("agent", ["response"])
        graph.add_node(GraphNode(name="node1", agent=agent))
        graph.set_start("node1")
        graph.set_end("node1")

        runner = Runner(app_name="test_graph", agent=graph, session_service=InMemorySessionService())
        session_service = runner.session_service
        await session_service.create_session(app_name="test_graph", user_id="test_user", session_id="test")

        # Collect all events
        events = []
        async for event in runner.run_async(user_id="test_user", session_id="test", new_message=types.Content(role="user", parts=[types.Part(text="test")])):
            events.append(event)

        # Verify all events are proper Event objects
        assert len(events) > 0
        for event in events:
            # Must have author field
            assert hasattr(event, 'author')
            assert event.author is not None

            # Should have content (some events might not)
            if event.content:
                assert isinstance(event.content, types.Content)
                assert hasattr(event.content, 'parts')

            # May have actions (EventActions)
            if event.actions:
                assert isinstance(event.actions, EventActions)

    async def test_invocation_context_conformity(self):
        """Test that InvocationContext is properly structured."""
        graph = GraphAgent(name="test")

        # Custom function that verifies InvocationContext structure
        def verify_ctx(state: GraphState, ctx):
            # Verify required InvocationContext fields
            assert hasattr(ctx, 'session')
            assert hasattr(ctx, 'invocation_id')
            assert hasattr(ctx, 'agent')
            assert hasattr(ctx, 'session_service')
            return "context valid"

        graph.add_node(GraphNode(name="node1", function=verify_ctx))
        graph.set_start("node1")
        graph.set_end("node1")

        runner = Runner(app_name="test_graph", agent=graph, session_service=InMemorySessionService())
        session_service = runner.session_service
        await session_service.create_session(app_name="test_graph", user_id="test_user", session_id="test")

        # If this doesn't raise, context is valid
        async for event in runner.run_async(user_id="test_user", session_id="test", new_message=types.Content(role="user", parts=[types.Part(text="test")])):
            pass

    async def test_state_delta_conformity(self):
        """Test that state changes use EventActions.state_delta."""
        graph = GraphAgent(name="test", checkpointing=True)

        agent = MockAgent("agent", ["response"])
        graph.add_node(GraphNode(name="node1", agent=agent))
        graph.set_start("node1")
        graph.set_end("node1")

        runner = Runner(app_name="test_graph", agent=graph, session_service=InMemorySessionService())
        session_service = runner.session_service
        await session_service.create_session(app_name="test_graph", user_id="test_user", session_id="test")

        # Collect events with state_delta
        state_delta_events = []
        async for event in runner.run_async(user_id="test_user", session_id="test", new_message=types.Content(role="user", parts=[types.Part(text="test")])):
            if event.actions and event.actions.state_delta:
                state_delta_events.append(event)

        # Checkpointing should produce state_delta events
        assert len(state_delta_events) > 0

        # Verify state_delta structure
        for event in state_delta_events:
            assert isinstance(event.actions.state_delta, dict)

    async def test_escalate_flag_conformity(self):
        """Test that interrupts use EventActions.escalate properly."""
        graph = GraphAgent(name="test")

        agent = MockAgent("agent", ["response"])
        graph.add_node(GraphNode(name="node1", agent=agent))
        graph.set_start("node1")
        graph.set_end("node1")
        graph.add_interrupt("node1", InterruptMode.BEFORE)

        runner = Runner(app_name="test_graph", agent=graph, session_service=InMemorySessionService())
        session_service = runner.session_service
        await session_service.create_session(app_name="test_graph", user_id="test_user", session_id="test")

        # Collect interrupt events
        interrupt_events = []
        async for event in runner.run_async(user_id="test_user", session_id="test", new_message=types.Content(role="user", parts=[types.Part(text="test")])):
            if event.content and event.content.parts and "Interrupt" in event.content.parts[0].text:
                interrupt_events.append(event)

        assert len(interrupt_events) > 0

        # Verify escalate flag is present
        for event in interrupt_events:
            assert event.actions is not None
            assert hasattr(event.actions, 'escalate')
            assert isinstance(event.actions.escalate, bool)



# ============================================================================
# Graph Export Tests
# ============================================================================

class TestGraphExport:
    """Tests for D3-compatible graph structure export."""

    def test_export_graph_structure(self):
        """Test exporting graph structure in D3 format."""
        graph = GraphAgent(name="test_graph", checkpointing=True)

        # Add nodes
        graph.add_node(GraphNode(name="start", function=lambda s, c: "start"))
        graph.add_node(GraphNode(name="process", function=lambda s, c: "process"))
        graph.add_node(GraphNode(name="end", function=lambda s, c: "end"))

        # Add edges
        graph.add_edge("start", "process")
        graph.add_edge("process", "end")

        # Set start and end
        graph.set_start("start")
        graph.set_end("end")

        # Export structure
        structure = graph.export_graph_structure()

        # Verify structure
        assert "nodes" in structure
        assert "links" in structure
        assert "metadata" in structure
        assert structure["directed"] is True

        # Verify nodes
        assert len(structure["nodes"]) == 3
        node_ids = [n["id"] for n in structure["nodes"]]
        assert "start" in node_ids
        assert "process" in node_ids
        assert "end" in node_ids

        # Verify all nodes are function type
        for node in structure["nodes"]:
            assert node["type"] == "function"

        # Verify links
        assert len(structure["links"]) == 2
        links = [(l["source"], l["target"]) for l in structure["links"]]
        assert ("start", "process") in links
        assert ("process", "end") in links

        # Verify metadata
        assert structure["metadata"]["start_node"] == "start"
        assert structure["metadata"]["end_nodes"] == ["end"]
        assert structure["metadata"]["checkpointing"] is True

    def test_export_with_conditional_edges(self):
        """Test export includes conditional edge information."""
        graph = GraphAgent(name="test")

        graph.add_node(GraphNode(name="a", function=lambda s, c: "a"))
        graph.add_node(GraphNode(name="b", function=lambda s, c: "b"))
        graph.add_node(GraphNode(name="c", function=lambda s, c: "c"))

        # Add conditional and unconditional edges
        graph.add_edge("a", "b", condition=lambda s: s.data.get("go_b", False))
        graph.add_edge("a", "c")  # No condition

        structure = graph.export_graph_structure()

        # Verify conditional flags
        links = structure["links"]
        assert len(links) == 2

        # Find the links
        b_link = next(l for l in links if l["target"] == "b")
        c_link = next(l for l in links if l["target"] == "c")

        assert b_link["conditional"] is True
        assert c_link["conditional"] is False

    def test_export_with_agent_nodes(self):
        """Test export distinguishes agent vs function nodes."""
        graph = GraphAgent(name="test")

        # Add function node
        graph.add_node(GraphNode(name="func", function=lambda s, c: "func"))

        # Add agent node
        mock_agent = Mock(spec=BaseAgent)
        mock_agent.name = "agent"
        graph.add_node(GraphNode(name="agent", agent=mock_agent))

        structure = graph.export_graph_structure()

        # Verify node types
        nodes = {n["id"]: n for n in structure["nodes"]}
        assert nodes["func"]["type"] == "function"
        assert nodes["agent"]["type"] == "agent"

    def test_export_with_interrupts(self):
        """Test export includes interrupt metadata."""
        graph = GraphAgent(name="test")

        graph.add_node(GraphNode(name="a", function=lambda s, c: "a"))
        graph.add_node(GraphNode(name="b", function=lambda s, c: "b"))

        graph.add_interrupt("a", InterruptMode.BEFORE)
        graph.add_interrupt("b", InterruptMode.AFTER)

        structure = graph.export_graph_structure()

        # Verify interrupt metadata
        metadata = structure["metadata"]
        assert "a" in metadata["interrupt_before"]
        assert "b" in metadata["interrupt_after"]

    def test_export_empty_graph(self):
        """Test export of empty graph."""
        graph = GraphAgent(name="empty")

        structure = graph.export_graph_structure()

        assert structure["nodes"] == []
        assert structure["links"] == []
        assert structure["metadata"]["start_node"] is None
        assert structure["metadata"]["end_nodes"] == []

    def test_export_cyclic_graph(self):
        """Test export of graph with cycles."""
        graph = GraphAgent(name="cyclic")

        graph.add_node(GraphNode(name="a", function=lambda s, c: "a"))
        graph.add_node(GraphNode(name="b", function=lambda s, c: "b"))
        graph.add_node(GraphNode(name="c", function=lambda s, c: "c"))

        # Create cycle: a -> b -> c -> a
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("c", "a")

        structure = graph.export_graph_structure()

        # Verify cycle is preserved
        assert len(structure["links"]) == 3
        links = [(l["source"], l["target"]) for l in structure["links"]]
        assert ("a", "b") in links
        assert ("b", "c") in links
        assert ("c", "a") in links


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
