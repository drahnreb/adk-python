"""Comprehensive state management tests for GraphAgent.

Tests all state reducers and state propagation patterns:
- StateReducer.OVERWRITE
- StateReducer.APPEND
- StateReducer.SUM
- StateReducer.CUSTOM
- State propagation through graph
- State isolation in parallel execution

These are unit tests focusing on state reducer logic, not full integration tests.
Full integration tests are in test_graph_agent.py and test_parallel_execution.py.
"""

import pytest
from typing import Any, Dict

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent, GraphNode, GraphState, StateReducer
from google.adk.events.event import Event
from google.genai import types


# ============================================================================
# Test Agents (Real BaseAgent implementations per ADK guidelines)
# ============================================================================


class TextAgent(BaseAgent):
    """Agent that outputs text."""

    model_config = {"extra": "allow", "arbitrary_types_allowed": True}

    def __init__(self, name: str, text: str):
        super().__init__(name=name)
        object.__setattr__(self, "_text", text)

    async def _run_async_impl(self, ctx):
        """Output text."""
        text = object.__getattribute__(self, "_text")
        yield Event(
            author=self.name, content=types.Content(parts=[types.Part(text=text)])
        )


# ============================================================================
# Test: StateReducer.OVERWRITE
# ============================================================================


@pytest.mark.asyncio
class TestStateReducerOverwrite:
    """Test OVERWRITE reducer - replaces existing value."""

    async def test_overwrite_reducer_basic(self):
        """Test basic OVERWRITE behavior - new value replaces old."""
        node = GraphNode(
            name="test_node",
            agent=TextAgent("agent", "new"),
            reducer=StateReducer.OVERWRITE,
        )

        # Initial state with existing value
        state = GraphState(data={"test_node": "old"}, metadata={})

        # Apply output with OVERWRITE reducer
        new_state = node._default_output_mapper("new", state)

        # Verify value was overwritten
        assert new_state.data["test_node"] == "new"
        assert "old" not in str(new_state.data["test_node"])

    async def test_overwrite_reducer_new_key(self):
        """Test OVERWRITE creates key if it doesn't exist."""
        node = GraphNode(
            name="new_key",
            agent=TextAgent("agent", "value"),
            reducer=StateReducer.OVERWRITE,
        )

        state = GraphState(data={}, metadata={})
        new_state = node._default_output_mapper("value", state)

        assert new_state.data["new_key"] == "value"

    async def test_overwrite_preserves_other_keys(self):
        """Test OVERWRITE doesn't affect other state keys."""
        node = GraphNode(
            name="key1", agent=TextAgent("agent", "new"), reducer=StateReducer.OVERWRITE
        )

        state = GraphState(data={"key1": "old", "key2": "preserved"}, metadata={})
        new_state = node._default_output_mapper("new", state)

        assert new_state.data["key1"] == "new"
        assert new_state.data["key2"] == "preserved"


# ============================================================================
# Test: StateReducer.APPEND
# ============================================================================


@pytest.mark.asyncio
class TestStateReducerAppend:
    """Test APPEND reducer - appends to list."""

    async def test_append_reducer_creates_list(self):
        """Test APPEND reducer creates list when key doesn't exist."""
        node = GraphNode(
            name="collector",
            agent=TextAgent("agent", "item"),
            reducer=StateReducer.APPEND,
        )

        state = GraphState(data={}, metadata={})
        new_state = node._default_output_mapper("first_item", state)

        # Verify list was created with first item
        assert "collector" in new_state.data
        assert isinstance(new_state.data["collector"], list)
        assert new_state.data["collector"] == ["first_item"]

    async def test_append_reducer_appends_to_existing_list(self):
        """Test APPEND adds to existing list."""
        node = GraphNode(
            name="collector",
            agent=TextAgent("agent", "item"),
            reducer=StateReducer.APPEND,
        )

        state = GraphState(data={"collector": ["item1", "item2"]}, metadata={})
        new_state = node._default_output_mapper("item3", state)

        assert new_state.data["collector"] == ["item1", "item2", "item3"]

    async def test_append_multiple_values(self):
        """Test APPEND accumulates multiple values."""
        node = GraphNode(
            name="results",
            agent=TextAgent("agent", "item"),
            reducer=StateReducer.APPEND,
        )

        # First append
        state1 = GraphState(data={}, metadata={})
        state2 = node._default_output_mapper("first", state1)

        # Second append
        state3 = node._default_output_mapper("second", state2)

        # Third append
        state4 = node._default_output_mapper("third", state3)

        assert state4.data["results"] == ["first", "second", "third"]


# ============================================================================
# Test: StateReducer.SUM
# ============================================================================


@pytest.mark.asyncio
class TestStateReducerSum:
    """Test SUM reducer - sums numeric values."""

    async def test_sum_reducer_string_concatenation(self):
        """Test SUM reducer fails with string output (documents limitation)."""
        # NOTE: SUM reducer in graph_node.py line 104 does:
        # new_state.data[self.name] = new_state.data.get(self.name, 0) + output
        # With string output, this raises TypeError: unsupported operand type(s) for +: 'int' and 'str'

        node = GraphNode(
            name="counter", agent=TextAgent("agent", "5"), reducer=StateReducer.SUM
        )

        state = GraphState(data={}, metadata={})

        # SUM reducer with string output raises TypeError
        with pytest.raises(TypeError, match="unsupported operand type"):
            node._default_output_mapper("5", state)

    async def test_sum_reducer_with_existing_string(self):
        """Test SUM concatenates strings when existing value is string."""
        node = GraphNode(
            name="counter", agent=TextAgent("agent", "5"), reducer=StateReducer.SUM
        )

        # If existing value is already a string, Python allows string + string
        state = GraphState(data={"counter": "prefix"}, metadata={})
        new_state = node._default_output_mapper("_suffix", state)

        # Result: "prefix" + "_suffix" = "prefix_suffix" (string concatenation)
        assert new_state.data.get("counter") == "prefix_suffix"

    async def test_sum_reducer_limitation(self):
        """Document that SUM doesn't work with string agent outputs.

        This test documents the limitation that StateReducer.SUM expects
        numeric output but agents return strings. Users should use a custom
        output_mapper for numeric summing.
        """

        # This is a known limitation - agents return strings
        # For true numeric summing, users need custom output_mapper:
        def numeric_sum_mapper(output: str, state: GraphState) -> GraphState:
            new_state = GraphState(
                data=state.data.copy(), metadata=state.metadata.copy()
            )
            try:
                value = int(output)
                new_state.data["total"] = new_state.data.get("total", 0) + value
            except ValueError:
                pass
            return new_state

        node = GraphNode(
            name="numeric",
            agent=TextAgent("agent", "10"),
            output_mapper=numeric_sum_mapper,
        )

        state1 = GraphState(data={}, metadata={})
        state2 = node.output_mapper("10", state1)
        state3 = node.output_mapper("20", state2)

        assert state3.data["total"] == 30  # True numeric sum


# ============================================================================
# Test: StateReducer.CUSTOM
# ============================================================================


@pytest.mark.asyncio
class TestStateReducerCustom:
    """Test CUSTOM reducer - uses custom reduction function."""

    async def test_custom_reducer_basic(self):
        """Test CUSTOM reducer with simple concatenation."""

        def concat_reducer(existing, new_value):
            if existing is None:
                return new_value
            return f"{existing}|{new_value}"

        node = GraphNode(
            name="custom",
            agent=TextAgent("agent", "test"),
            reducer=StateReducer.CUSTOM,
            custom_reducer=concat_reducer,
        )

        # First call - no existing value
        state1 = GraphState(data={}, metadata={})
        new_state1 = node._default_output_mapper("A", state1)
        assert new_state1.data["custom"] == "A"

        # Second call - merge with existing
        state2 = GraphState(data={"custom": "A"}, metadata={})
        new_state2 = node._default_output_mapper("B", state2)
        assert new_state2.data["custom"] == "A|B"

    async def test_custom_reducer_dict_merge(self):
        """Test CUSTOM reducer for merging dictionaries."""

        def dict_merge_reducer(existing, new_value):
            """Merge dict-like string representations."""
            if existing is None:
                return {"data": [new_value]}
            if isinstance(existing, dict):
                existing["data"].append(new_value)
                return existing
            return {"data": [existing, new_value]}

        node = GraphNode(
            name="merger",
            agent=TextAgent("agent", "test"),
            reducer=StateReducer.CUSTOM,
            custom_reducer=dict_merge_reducer,
        )

        state1 = GraphState(data={}, metadata={})
        new_state1 = node._default_output_mapper("item1", state1)
        assert new_state1.data["merger"] == {"data": ["item1"]}

        new_state2 = node._default_output_mapper("item2", new_state1)
        assert new_state2.data["merger"] == {"data": ["item1", "item2"]}

    async def test_custom_reducer_counter(self):
        """Test CUSTOM reducer for counting."""

        def count_reducer(existing, new_value):
            """Count occurrences."""
            if existing is None:
                return 1
            return existing + 1

        node = GraphNode(
            name="counter",
            agent=TextAgent("agent", "test"),
            reducer=StateReducer.CUSTOM,
            custom_reducer=count_reducer,
        )

        state1 = GraphState(data={}, metadata={})
        new_state1 = node._default_output_mapper("ignored", state1)
        assert new_state1.data["counter"] == 1

        new_state2 = node._default_output_mapper("ignored", new_state1)
        assert new_state2.data["counter"] == 2

        new_state3 = node._default_output_mapper("ignored", new_state2)
        assert new_state3.data["counter"] == 3


# ============================================================================
# Test: State Propagation (Unit Tests)
# ============================================================================


@pytest.mark.asyncio
class TestStatePropagation:
    """Test how state flows through graph nodes (unit tests)."""

    async def test_output_mapper_preserves_existing_state(self):
        """Test that output mapper preserves existing state data."""
        node = GraphNode(name="node1", agent=TextAgent("agent", "new"))

        state = GraphState(
            data={"existing_key": "existing_value"}, metadata={"meta": "data"}
        )

        new_state = node._default_output_mapper("new_output", state)

        # New output added
        assert new_state.data["node1"] == "new_output"

        # Existing state preserved
        assert new_state.data["existing_key"] == "existing_value"
        assert new_state.metadata["meta"] == "data"

    async def test_metadata_preserved_across_state_updates(self):
        """Test that metadata is preserved during state updates."""
        node = GraphNode(name="node", agent=TextAgent("agent", "output"))

        state = GraphState(
            data={"key": "value"},
            metadata={"iteration": 1, "path": ["start", "middle"]},
        )

        new_state = node._default_output_mapper("output", state)

        assert new_state.metadata["iteration"] == 1
        assert new_state.metadata["path"] == ["start", "middle"]

    async def test_state_isolation_between_nodes(self):
        """Test that each node gets its own state copy."""
        node1 = GraphNode(name="node1", agent=TextAgent("agent1", "output1"))
        node2 = GraphNode(name="node2", agent=TextAgent("agent2", "output2"))

        state = GraphState(data={}, metadata={})

        # Node 1 processes state
        state1 = node1._default_output_mapper("output1", state)

        # Node 2 processes original state (not state1)
        state2 = node2._default_output_mapper("output2", state)

        # Verify isolation - state2 doesn't have node1's output
        assert "node1" in state1.data
        assert "node1" not in state2.data
        assert "node2" in state2.data


# ============================================================================
# Test: Custom Output Mappers
# ============================================================================


@pytest.mark.asyncio
class TestCustomOutputMappers:
    """Test custom output mapper functionality."""

    async def test_custom_output_mapper_override(self):
        """Test custom output mapper completely overrides default."""

        def custom_mapper(output: str, state: GraphState) -> GraphState:
            # Completely custom logic
            new_state = GraphState(
                data={"custom_key": f"CUSTOM_{output}"}, metadata={"custom": True}
            )
            return new_state

        node = GraphNode(
            name="custom", agent=TextAgent("agent", "test"), output_mapper=custom_mapper
        )

        state = GraphState(data={"existing": "data"}, metadata={})
        new_state = node.output_mapper("output", state)

        # Custom mapper replaced everything
        assert "custom_key" in new_state.data
        assert new_state.data["custom_key"] == "CUSTOM_output"
        assert new_state.metadata.get("custom") == True
        # Original state data gone (custom mapper replaced it)
        assert "existing" not in new_state.data

    async def test_custom_output_mapper_with_state_merge(self):
        """Test custom output mapper that merges with existing state."""

        def merging_mapper(output: str, state: GraphState) -> GraphState:
            # Preserve existing state and add new data
            new_state = GraphState(
                data=state.data.copy(), metadata=state.metadata.copy()
            )
            new_state.data["processed"] = output.upper()
            new_state.metadata["processed_count"] = (
                new_state.metadata.get("processed_count", 0) + 1
            )
            return new_state

        node = GraphNode(
            name="merger",
            agent=TextAgent("agent", "test"),
            output_mapper=merging_mapper,
        )

        state = GraphState(data={"existing": "value"}, metadata={"processed_count": 5})
        new_state = node.output_mapper("hello", state)

        # Existing state preserved
        assert new_state.data["existing"] == "value"
        # New data added
        assert new_state.data["processed"] == "HELLO"
        # Metadata updated
        assert new_state.metadata["processed_count"] == 6


# ============================================================================
# Test: Edge Cases
# ============================================================================


@pytest.mark.asyncio
class TestStateEdgeCases:
    """Test edge cases in state management."""

    async def test_empty_state_initialization(self):
        """Test graph node with empty initial state."""
        node = GraphNode(name="solo", agent=TextAgent("agent", "output"))

        state = GraphState(data={}, metadata={})
        new_state = node._default_output_mapper("output", state)

        assert new_state.data["solo"] == "output"

    async def test_state_copy_safety(self):
        """Test that state copies don't share references for simple types."""
        state1 = GraphState(data={"key": "value"}, metadata={"meta": "data"})

        # GraphNode does .copy() for data and metadata
        state2 = GraphState(data=state1.data.copy(), metadata=state1.metadata.copy())

        # Modify state2
        state2.data["key"] = "modified"
        state2.metadata["meta"] = "modified"

        # State1 unchanged (shallow copy works for simple types)
        assert state1.data["key"] == "value"
        assert state1.metadata["meta"] == "data"

    async def test_state_nested_dict_shallow_copy_limitation(self):
        """Document shallow copy limitation with nested dicts.

        This test documents that GraphNode._default_output_mapper uses .copy()
        which is a shallow copy. For nested structures, this can cause issues.
        """
        state1 = GraphState(data={"nested": {"key": "value"}}, metadata={})

        # Shallow copy (what GraphNode does)
        state2 = GraphState(data=state1.data.copy(), metadata=state1.metadata.copy())

        # Modify nested structure in state2
        state2.data["nested"]["key"] = "modified"

        # BUG: state1 is also modified (shared reference)
        # This is a known limitation of shallow copy
        assert state1.data["nested"]["key"] == "modified"  # Unintended side effect

        # NOTE: parallel.py uses deepcopy to avoid this issue
        # For regular sequential execution, users should avoid nested mutations

    async def test_reducer_with_none_output(self):
        """Test reducer behavior with None or empty output."""
        node = GraphNode(
            name="test", agent=TextAgent("agent", ""), reducer=StateReducer.OVERWRITE
        )

        state = GraphState(data={}, metadata={})
        new_state = node._default_output_mapper("", state)

        # Empty string is still stored
        assert new_state.data["test"] == ""
