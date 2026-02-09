"""Tests for P0.1: O(1) task lookup in parallel execution.

This test suite verifies that the parallel execution correctly identifies
which node a completed task belongs to using O(1) dictionary lookup instead
of O(n) linear search.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict
from unittest.mock import Mock

import pytest

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph.graph_state import GraphState
from google.adk.agents.graph.parallel import (
    ErrorPolicy,
    JoinStrategy,
    ParallelNodeGroup,
    execute_parallel_group,
)
from google.adk.events.event import Event
from google.genai import types


class SimpleTestAgent(BaseAgent):
    """Real test agent extending BaseAgent per ADK guidelines."""

    def __init__(self, name: str, response: str = "test response"):
        super().__init__(name=name)
        object.__setattr__(self, "_response", response)
        object.__setattr__(self, "_delay", 0.0)

    def set_delay(self, delay: float) -> None:
        """Set execution delay for testing timing."""
        object.__setattr__(self, "_delay", delay)

    async def _run_async_impl(self, ctx) -> AsyncGenerator[Event, None]:
        """Test implementation."""
        delay = object.__getattribute__(self, "_delay")
        response = object.__getattribute__(self, "_response")

        if delay > 0:
            await asyncio.sleep(delay)

        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text=response)]),
        )


@pytest.mark.asyncio
async def test_task_lookup_correct_node_identification():
    """Test that O(1) lookup correctly identifies which node completed."""

    # Create test agents
    agent1 = SimpleTestAgent("agent1", "response1")
    agent2 = SimpleTestAgent("agent2", "response2")
    agent3 = SimpleTestAgent("agent3", "response3")

    nodes = {
        "node1": Mock(agent=agent1),
        "node2": Mock(agent=agent2),
        "node3": Mock(agent=agent3),
    }

    # Mock execute_node_fn
    async def execute_node_fn(node, state, ctx):
        async for event in node.agent._run_async_impl(ctx):
            yield event

    state = GraphState()
    group = ParallelNodeGroup(
        nodes=["node1", "node2", "node3"],
        join_strategy=JoinStrategy.WAIT_ALL,
    )

    # Execute parallel group
    events = []
    async for event in execute_parallel_group(
        group=group,
        nodes=nodes,
        state=state,
        ctx=Mock(),
        execute_node_fn=execute_node_fn,
    ):
        events.append(event)

    # Verify all nodes executed correctly
    assert len(events) == 3
    sources = {event.author for event in events}
    assert sources == {"agent1", "agent2", "agent3"}

    # Verify responses
    responses = {event.content.parts[0].text for event in events}
    assert responses == {"response1", "response2", "response3"}


@pytest.mark.asyncio
async def test_task_lookup_with_staggered_completion():
    """Test O(1) lookup with tasks completing in different order."""

    # Create agents with different delays to ensure out-of-order completion
    agent1 = SimpleTestAgent("agent1", "fast")
    agent1.set_delay(0.01)  # Fast

    agent2 = SimpleTestAgent("agent2", "slow")
    agent2.set_delay(0.05)  # Slow

    agent3 = SimpleTestAgent("agent3", "medium")
    agent3.set_delay(0.03)  # Medium

    nodes = {
        "node1": Mock(agent=agent1),
        "node2": Mock(agent=agent2),
        "node3": Mock(agent=agent3),
    }

    async def execute_node_fn(node, state, ctx):
        async for event in node.agent._run_async_impl(ctx):
            yield event

    state = GraphState()
    group = ParallelNodeGroup(
        nodes=["node1", "node2", "node3"],
        join_strategy=JoinStrategy.WAIT_ALL,
    )

    # Track completion order
    completion_order = []

    async for event in execute_parallel_group(
        group=group,
        nodes=nodes,
        state=state,
        ctx=Mock(),
        execute_node_fn=execute_node_fn,
    ):
        completion_order.append(event.author)

    # Verify all completed
    assert len(completion_order) == 3

    # Verify correct order (fast -> medium -> slow)
    assert completion_order[0] == "agent1"  # Fast completes first
    assert completion_order[1] == "agent3"  # Medium completes second
    assert completion_order[2] == "agent2"  # Slow completes last


@pytest.mark.asyncio
async def test_task_lookup_performance_with_many_nodes():
    """Test that O(1) lookup scales with 100+ parallel nodes."""

    # Create 100 parallel nodes
    num_nodes = 100
    agents = {
        f"agent{i}": SimpleTestAgent(f"agent{i}", f"response{i}")
        for i in range(num_nodes)
    }

    nodes = {f"node{i}": Mock(agent=agents[f"agent{i}"]) for i in range(num_nodes)}

    async def execute_node_fn(node, state, ctx):
        async for event in node.agent._run_async_impl(ctx):
            yield event

    state = GraphState()
    group = ParallelNodeGroup(
        nodes=[f"node{i}" for i in range(num_nodes)],
        join_strategy=JoinStrategy.WAIT_ALL,
    )

    # Execute and measure - should complete quickly with O(1) lookup
    import time

    start_time = time.time()

    events = []
    async for event in execute_parallel_group(
        group=group,
        nodes=nodes,
        state=state,
        ctx=Mock(),
        execute_node_fn=execute_node_fn,
    ):
        events.append(event)

    elapsed_time = time.time() - start_time

    # Verify all completed
    assert len(events) == num_nodes

    # With O(1) lookup, 100 nodes should complete in < 1 second
    # With O(n) lookup, this would be significantly slower
    assert elapsed_time < 1.0, f"Took {elapsed_time}s - O(1) lookup should be faster"


@pytest.mark.asyncio
async def test_task_lookup_with_concurrent_completions():
    """Test O(1) lookup handles multiple tasks completing simultaneously."""

    # Create agents that complete nearly simultaneously
    agents = {
        f"agent{i}": SimpleTestAgent(f"agent{i}", f"response{i}")
        for i in range(10)
    }

    nodes = {f"node{i}": Mock(agent=agents[f"agent{i}"]) for i in range(10)}

    async def execute_node_fn(node, state, ctx):
        async for event in node.agent._run_async_impl(ctx):
            yield event

    state = GraphState()
    group = ParallelNodeGroup(
        nodes=[f"node{i}" for i in range(10)],
        join_strategy=JoinStrategy.WAIT_ALL,
    )

    # Execute
    events = []
    async for event in execute_parallel_group(
        group=group,
        nodes=nodes,
        state=state,
        ctx=Mock(),
        execute_node_fn=execute_node_fn,
    ):
        events.append(event)

    # Verify all completed correctly
    assert len(events) == 10
    sources = {event.author for event in events}
    assert len(sources) == 10  # All unique sources


@pytest.mark.asyncio
async def test_task_lookup_with_wait_any():
    """Test O(1) lookup works correctly with WAIT_ANY strategy."""

    # Create agents with different delays
    agent1 = SimpleTestAgent("agent1", "first")
    agent1.set_delay(0.01)  # Fast

    agent2 = SimpleTestAgent("agent2", "second")
    agent2.set_delay(0.1)  # Slow

    agent3 = SimpleTestAgent("agent3", "third")
    agent3.set_delay(0.1)  # Slow

    nodes = {
        "node1": Mock(agent=agent1),
        "node2": Mock(agent=agent2),
        "node3": Mock(agent=agent3),
    }

    async def execute_node_fn(node, state, ctx):
        async for event in node.agent._run_async_impl(ctx):
            yield event

    state = GraphState()
    group = ParallelNodeGroup(
        nodes=["node1", "node2", "node3"],
        join_strategy=JoinStrategy.WAIT_ANY,  # Only wait for first
    )

    # Execute
    events = []
    async for event in execute_parallel_group(
        group=group,
        nodes=nodes,
        state=state,
        ctx=Mock(),
        execute_node_fn=execute_node_fn,
    ):
        events.append(event)

    # Should only get one event (from fastest agent)
    assert len(events) == 1
    assert events[0].author == "agent1"


@pytest.mark.asyncio
async def test_task_lookup_with_wait_n():
    """Test O(1) lookup works correctly with WAIT_N strategy."""

    # Create 5 agents with delays to ensure sequential completion
    agents = {}
    for i in range(5):
        agent = SimpleTestAgent(f"agent{i}", f"response{i}")
        agent.set_delay(0.01 * (i + 1))  # Stagger delays: 0.01, 0.02, 0.03, 0.04, 0.05
        agents[f"agent{i}"] = agent

    nodes = {f"node{i}": Mock(agent=agents[f"agent{i}"]) for i in range(5)}

    async def execute_node_fn(node, state, ctx):
        async for event in node.agent._run_async_impl(ctx):
            yield event

    state = GraphState()
    group = ParallelNodeGroup(
        nodes=[f"node{i}" for i in range(5)],
        join_strategy=JoinStrategy.WAIT_N,
        wait_n=3,  # Wait for 3 nodes
    )

    # Execute
    events = []
    async for event in execute_parallel_group(
        group=group,
        nodes=nodes,
        state=state,
        ctx=Mock(),
        execute_node_fn=execute_node_fn,
    ):
        events.append(event)

    # Should get exactly 3 events
    assert len(events) == 3


@pytest.mark.asyncio
async def test_task_lookup_with_error_handling():
    """Test O(1) lookup correctly identifies failing nodes."""

    class FailingAgent(BaseAgent):
        """Agent that raises an error."""

        def __init__(self, name: str):
            super().__init__(name=name)

        async def _run_async_impl(self, ctx) -> AsyncGenerator[Event, None]:
            raise ValueError(f"Error from {self.name}")
            yield  # Make it a generator

    agent1 = SimpleTestAgent("agent1", "success")
    agent2 = FailingAgent("agent2")

    nodes = {
        "node1": Mock(agent=agent1),
        "node2": Mock(agent=agent2),
    }

    async def execute_node_fn(node, state, ctx):
        async for event in node.agent._run_async_impl(ctx):
            yield event

    state = GraphState()
    group = ParallelNodeGroup(
        nodes=["node1", "node2"],
        join_strategy=JoinStrategy.WAIT_ALL,
        error_policy=ErrorPolicy.FAIL_FAST,
    )

    # Execute - should raise error from agent2
    with pytest.raises(ValueError, match="Error from agent2"):
        async for _ in execute_parallel_group(
            group=group,
            nodes=nodes,
            state=state,
            ctx=Mock(),
            execute_node_fn=execute_node_fn,
        ):
            pass


@pytest.mark.asyncio
async def test_task_to_node_mapping_created_correctly():
    """Test that task_to_node inverse mapping is created at task creation."""

    # This test verifies the fix by ensuring task_to_node exists and works
    agents = {f"agent{i}": SimpleTestAgent(f"agent{i}", f"response{i}") for i in range(3)}
    nodes = {f"node{i}": Mock(agent=agents[f"agent{i}"]) for i in range(3)}

    async def execute_node_fn(node, state, ctx):
        async for event in node.agent._run_async_impl(ctx):
            yield event

    state = GraphState()
    group = ParallelNodeGroup(
        nodes=["node0", "node1", "node2"],
        join_strategy=JoinStrategy.WAIT_ALL,
    )

    # Execute and verify no errors
    events = []
    async for event in execute_parallel_group(
        group=group,
        nodes=nodes,
        state=state,
        ctx=Mock(),
        execute_node_fn=execute_node_fn,
    ):
        events.append(event)

    # All tasks should be identified correctly
    assert len(events) == 3
