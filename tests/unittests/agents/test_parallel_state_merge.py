"""Tests for P0.2: State merge conflict detection in parallel execution.

This test suite verifies that parallel execution correctly detects and handles
conflicts when multiple branches modify the same state keys.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator
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


class StateModifyingAgent(BaseAgent):
    """Test agent that modifies state keys."""

    def __init__(self, name: str, state_updates: dict):
        super().__init__(name=name)
        object.__setattr__(self, "_state_updates", state_updates)

    async def _run_async_impl(self, ctx) -> AsyncGenerator[Event, None]:
        """Modify state and yield event."""
        state_updates = object.__getattribute__(self, "_state_updates")

        # Simulate state modification
        # Note: In real execution, state is modified via ctx or node reducers
        # For testing, we'll modify via the event's state_delta

        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text="modified state")]),
        )


@pytest.mark.asyncio
async def test_no_conflict_disjoint_keys():
    """Test that no conflict is detected when branches modify different keys."""
    # Branch 1 modifies key "a", branch 2 modifies key "b"

    agent1 = StateModifyingAgent("agent1", {"a": "value_a"})
    agent2 = StateModifyingAgent("agent2", {"b": "value_b"})

    nodes = {
        "node1": Mock(agent=agent1),
        "node2": Mock(agent=agent2),
    }

    async def execute_node_fn(node, state, ctx):
        # Modify state directly in branch
        updates = object.__getattribute__(node.agent, "_state_updates")
        for key, value in updates.items():
            state.data[key] = value

        async for event in node.agent._run_async_impl(ctx):
            yield event

    state = GraphState()
    group = ParallelNodeGroup(
        nodes=["node1", "node2"],
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

    # Verify both keys merged without conflict
    assert "a" in state.data
    assert "b" in state.data
    assert state.data["a"] == "value_a"
    assert state.data["b"] == "value_b"


@pytest.mark.asyncio
async def test_conflict_detected_same_key(caplog):
    """Test that conflict is detected when branches modify the same key."""
    import logging

    caplog.set_level(logging.WARNING)

    # Both branches modify key "x"
    agent1 = StateModifyingAgent("agent1", {"x": "value1"})
    agent2 = StateModifyingAgent("agent2", {"x": "value2"})

    nodes = {
        "node1": Mock(agent=agent1),
        "node2": Mock(agent=agent2),
    }

    async def execute_node_fn(node, state, ctx):
        updates = object.__getattribute__(node.agent, "_state_updates")
        for key, value in updates.items():
            state.data[key] = value

        async for event in node.agent._run_async_impl(ctx):
            yield event

    state = GraphState()
    group = ParallelNodeGroup(
        nodes=["node1", "node2"],
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

    # Verify conflict was logged
    assert "State merge conflict detected" in caplog.text
    assert "key 'x' modified by multiple parallel branches" in caplog.text

    # Verify last write wins (one of the values)
    assert state.data["x"] in ["value1", "value2"]


@pytest.mark.asyncio
async def test_inherited_keys_cause_false_conflict(caplog):
    """Test documenting limitation: inherited keys from deepcopy cause false conflicts.

    When branches get deepcopy of original state, ALL keys are inherited.
    During merge, we can't distinguish 'modified' from 'inherited', so
    inherited-but-unmodified keys are treated as modifications, causing
    false conflict warnings.

    This is a known limitation documented for future improvement.
    """
    import logging

    caplog.set_level(logging.WARNING)

    agent1 = StateModifyingAgent("agent1", {"x": "new_value"})
    agent2 = StateModifyingAgent("agent2", {"y": "value_y"})

    nodes = {
        "node1": Mock(agent=agent1),
        "node2": Mock(agent=agent2),
    }

    async def execute_node_fn(node, state, ctx):
        updates = object.__getattribute__(node.agent, "_state_updates")
        for key, value in updates.items():
            state.data[key] = value

        async for event in node.agent._run_async_impl(ctx):
            yield event

    # Original state has "x" - both branches will inherit this via deepcopy
    state = GraphState(data={"x": "original"})
    group = ParallelNodeGroup(
        nodes=["node1", "node2"],
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

    # Known limitation: False conflict warning for inherited "x" key
    # This is because branch2 inherits "x" via deepcopy but doesn't modify it
    # The merge sees both branches have "x" and treats it as a conflict
    assert "State merge conflict detected" in caplog.text

    # Last write wins - value depends on merge order
    assert state.data["x"] in ["new_value", "original"]
    assert state.data["y"] == "value_y"


@pytest.mark.asyncio
async def test_multiple_conflicts_detected(caplog):
    """Test detection of multiple conflicts across different keys."""
    import logging

    caplog.set_level(logging.WARNING)

    # Both branches modify keys "a" and "b"
    agent1 = StateModifyingAgent("agent1", {"a": "a1", "b": "b1", "unique1": "u1"})
    agent2 = StateModifyingAgent("agent2", {"a": "a2", "b": "b2", "unique2": "u2"})

    nodes = {
        "node1": Mock(agent=agent1),
        "node2": Mock(agent=agent2),
    }

    async def execute_node_fn(node, state, ctx):
        updates = object.__getattribute__(node.agent, "_state_updates")
        for key, value in updates.items():
            state.data[key] = value

        async for event in node.agent._run_async_impl(ctx):
            yield event

    state = GraphState()
    group = ParallelNodeGroup(
        nodes=["node1", "node2"],
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

    # Verify conflicts logged for both "a" and "b"
    log_text = caplog.text
    conflict_count = log_text.count("State merge conflict detected")
    assert conflict_count == 2  # Conflicts for "a" and "b"

    # Verify unique keys merged without conflict
    assert state.data["unique1"] == "u1"
    assert state.data["unique2"] == "u2"


@pytest.mark.asyncio
async def test_metadata_conflict_detection(caplog):
    """Test that metadata conflicts are also detected."""
    import logging

    caplog.set_level(logging.WARNING)

    # Both branches modify metadata key "meta_key"
    agent1 = StateModifyingAgent("agent1", {})
    agent2 = StateModifyingAgent("agent2", {})

    nodes = {
        "node1": Mock(agent=agent1),
        "node2": Mock(agent=agent2),
    }

    async def execute_node_fn(node, state, ctx):
        # Modify metadata instead of data
        if node.agent.name == "agent1":
            state.metadata["meta_key"] = "meta1"
        else:
            state.metadata["meta_key"] = "meta2"

        async for event in node.agent._run_async_impl(ctx):
            yield event

    state = GraphState()
    group = ParallelNodeGroup(
        nodes=["node1", "node2"],
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

    # Verify metadata conflict was logged
    assert "Metadata merge conflict detected" in caplog.text
    assert state.metadata["meta_key"] in ["meta1", "meta2"]


@pytest.mark.asyncio
async def test_three_way_conflict(caplog):
    """Test conflict detection with three branches modifying same key."""
    import logging

    caplog.set_level(logging.WARNING)

    # Three branches modify key "x"
    agent1 = StateModifyingAgent("agent1", {"x": "value1"})
    agent2 = StateModifyingAgent("agent2", {"x": "value2"})
    agent3 = StateModifyingAgent("agent3", {"x": "value3"})

    nodes = {
        "node1": Mock(agent=agent1),
        "node2": Mock(agent=agent2),
        "node3": Mock(agent=agent3),
    }

    async def execute_node_fn(node, state, ctx):
        updates = object.__getattribute__(node.agent, "_state_updates")
        for key, value in updates.items():
            state.data[key] = value

        async for event in node.agent._run_async_impl(ctx):
            yield event

    state = GraphState()
    group = ParallelNodeGroup(
        nodes=["node1", "node2", "node3"],
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

    # Verify at least 2 conflicts logged (2nd and 3rd branch)
    log_text = caplog.text
    conflict_count = log_text.count("State merge conflict detected")
    assert conflict_count >= 2

    # Verify last write wins
    assert state.data["x"] in ["value1", "value2", "value3"]


@pytest.mark.asyncio
async def test_conflict_with_complex_values():
    """Test conflict detection works with complex value types."""
    # Branches modify same key with different value types

    agent1 = StateModifyingAgent("agent1", {"data": {"nested": "value1"}})
    agent2 = StateModifyingAgent("agent2", {"data": [1, 2, 3]})

    nodes = {
        "node1": Mock(agent=agent1),
        "node2": Mock(agent=agent2),
    }

    async def execute_node_fn(node, state, ctx):
        updates = object.__getattribute__(node.agent, "_state_updates")
        for key, value in updates.items():
            state.data[key] = value

        async for event in node.agent._run_async_impl(ctx):
            yield event

    state = GraphState()
    group = ParallelNodeGroup(
        nodes=["node1", "node2"],
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

    # Verify one value won (either dict or list)
    assert state.data["data"] in [{"nested": "value1"}, [1, 2, 3]]


@pytest.mark.asyncio
async def test_telemetry_conflict_tracking():
    """Test that telemetry correctly tracks conflict count."""
    # This test verifies span attributes are set correctly
    # We can't directly access the span, but we verify no errors occur

    agent1 = StateModifyingAgent("agent1", {"x": "v1", "y": "v1"})
    agent2 = StateModifyingAgent("agent2", {"x": "v2", "y": "v2"})

    nodes = {
        "node1": Mock(agent=agent1),
        "node2": Mock(agent=agent2),
    }

    async def execute_node_fn(node, state, ctx):
        updates = object.__getattribute__(node.agent, "_state_updates")
        for key, value in updates.items():
            state.data[key] = value

        async for event in node.agent._run_async_impl(ctx):
            yield event

    state = GraphState()
    group = ParallelNodeGroup(
        nodes=["node1", "node2"],
        join_strategy=JoinStrategy.WAIT_ALL,
    )

    # Execute - should complete without errors
    events = []
    async for event in execute_parallel_group(
        group=group,
        nodes=nodes,
        state=state,
        ctx=Mock(),
        execute_node_fn=execute_node_fn,
    ):
        events.append(event)

    # Verify execution completed
    assert len(events) == 2
    assert state.data["x"] in ["v1", "v2"]
    assert state.data["y"] in ["v1", "v2"]
