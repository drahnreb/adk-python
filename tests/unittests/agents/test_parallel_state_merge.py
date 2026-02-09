"""Unit tests for parallel execution state merging.

Tests parallel execution state management:
- Deep copy isolation between branches
- State merge logic with conflict detection
- Reducers for different merge strategies
"""

import pytest
from copy import deepcopy

from google.adk.agents.graph.graph_state import GraphState
from google.adk.agents.graph.parallel import ParallelNodeGroup, JoinStrategy, ErrorPolicy


@pytest.mark.asyncio
async def test_deep_copy_isolation():
    """Test that nested structures are truly isolated in parallel branches."""
    original_state = GraphState(
        data={"results": [1, 2, 3], "meta": {"count": 0}},
        metadata={}
    )

    # Simulate parallel execution deep copy (from parallel.py line 162)
    branch1 = GraphState(
        data=deepcopy(original_state.data),
        metadata=deepcopy(original_state.metadata)
    )
    branch2 = GraphState(
        data=deepcopy(original_state.data),
        metadata=deepcopy(original_state.metadata)
    )

    # Modify branches
    branch1.data["results"].append(4)
    branch2.data["results"].append(5)
    branch1.data["meta"]["count"] = 1
    branch2.data["meta"]["count"] = 2

    # Original unchanged
    assert original_state.data["results"] == [1, 2, 3]
    assert original_state.data["meta"]["count"] == 0

    # Branches isolated
    assert branch1.data["results"] == [1, 2, 3, 4]
    assert branch2.data["results"] == [1, 2, 3, 5]
    assert branch1.data["meta"]["count"] == 1
    assert branch2.data["meta"]["count"] == 2


@pytest.mark.asyncio
async def test_shallow_vs_deep_copy_bug():
    """Test that shallow copy would cause state mutation (the bug we fixed)."""
    original_state = GraphState(
        data={"nested_list": [1, 2, 3]},
        metadata={}
    )

    # Shallow copy (BUG - mutations affect original)
    shallow_branch = GraphState(
        data=original_state.data.copy(),  # Shallow copy
        metadata=original_state.metadata.copy()
    )

    # Deep copy (FIXED - mutations isolated)
    deep_branch = GraphState(
        data=deepcopy(original_state.data),  # Deep copy
        metadata=deepcopy(original_state.metadata)
    )

    # Modify both branches
    shallow_branch.data["nested_list"].append(4)
    deep_branch.data["nested_list"].append(5)

    # Shallow copy MUTATES original (BUG!)
    assert original_state.data["nested_list"] == [1, 2, 3, 4]

    # Deep copy is isolated (made before shallow mutation)
    assert deep_branch.data["nested_list"] == [1, 2, 3, 5]


@pytest.mark.asyncio
async def test_state_merge_no_conflicts():
    """Test state merge when branches modify different keys."""
    # Simulate two branches with no conflicts
    state = GraphState(data={}, metadata={})

    branch1_state = GraphState(
        data={"branch1_result": "value1"},
        metadata={"branch1_meta": "meta1"}
    )

    branch2_state = GraphState(
        data={"branch2_result": "value2"},
        metadata={"branch2_meta": "meta2"}
    )

    # Simulate merge (from parallel.py lines 276-320)
    results = {
        "node1": {"state": branch1_state},
        "node2": {"state": branch2_state}
    }

    for node_name, result in results.items():
        branch_state = result["state"]

        # Merge data keys
        for key, value in branch_state.data.items():
            state.data[key] = value

        # Merge metadata keys
        for key, value in branch_state.metadata.items():
            state.metadata[key] = value

    # Both branches merged
    assert state.data["branch1_result"] == "value1"
    assert state.data["branch2_result"] == "value2"
    assert state.metadata["branch1_meta"] == "meta1"
    assert state.metadata["branch2_meta"] == "meta2"


@pytest.mark.asyncio
async def test_state_merge_with_conflicts():
    """Test state merge when branches modify same keys (last write wins)."""
    state = GraphState(data={"shared_key": "original"}, metadata={})

    branch1_state = GraphState(
        data={"shared_key": "branch1_value"},
        metadata={}
    )

    branch2_state = GraphState(
        data={"shared_key": "branch2_value"},
        metadata={}
    )

    # Simulate merge with conflict detection
    results = {
        "node1": {"state": branch1_state},
        "node2": {"state": branch2_state}
    }

    conflicts_detected = []
    keys_merged = set()

    for node_name, result in results.items():
        branch_state = result["state"]

        for key, value in branch_state.data.items():
            if key in state.data and key in keys_merged:
                # Conflict detected!
                conflicts_detected.append({
                    "key": key,
                    "node": node_name,
                    "old_value": state.data[key],
                    "new_value": value,
                })

            state.data[key] = value  # Last write wins
            keys_merged.add(key)

    # Conflict was detected
    assert len(conflicts_detected) == 1
    assert conflicts_detected[0]["key"] == "shared_key"
    assert conflicts_detected[0]["node"] == "node2"
    assert conflicts_detected[0]["old_value"] == "branch1_value"
    assert conflicts_detected[0]["new_value"] == "branch2_value"

    # Last write wins (node2 overwrote node1)
    assert state.data["shared_key"] == "branch2_value"


@pytest.mark.asyncio
async def test_parallel_group_config():
    """Test ParallelNodeGroup configuration."""
    # Test WAIT_ALL strategy
    group1 = ParallelNodeGroup(
        nodes=["node1", "node2"],
        join_strategy=JoinStrategy.WAIT_ALL,
        error_policy=ErrorPolicy.FAIL_FAST
    )
    assert group1.join_strategy == JoinStrategy.WAIT_ALL
    assert group1.error_policy == ErrorPolicy.FAIL_FAST
    assert group1.nodes == ["node1", "node2"]

    # Test WAIT_ANY strategy
    group2 = ParallelNodeGroup(
        nodes=["node3", "node4"],
        join_strategy=JoinStrategy.WAIT_ANY,
        error_policy=ErrorPolicy.CONTINUE
    )
    assert group2.join_strategy == JoinStrategy.WAIT_ANY
    assert group2.error_policy == ErrorPolicy.CONTINUE

    # Test WAIT_N strategy
    group3 = ParallelNodeGroup(
        nodes=["node5", "node6", "node7"],
        join_strategy=JoinStrategy.WAIT_N,
        wait_n=2,
        error_policy=ErrorPolicy.COLLECT
    )
    assert group3.join_strategy == JoinStrategy.WAIT_N
    assert group3.wait_n == 2
    assert group3.error_policy == ErrorPolicy.COLLECT


@pytest.mark.asyncio
async def test_metadata_merge_with_conflicts():
    """Test metadata merge with conflict detection."""
    state = GraphState(data={}, metadata={"iteration": 0})

    branch1_state = GraphState(
        data={},
        metadata={"iteration": 1, "path": ["node1"]}
    )

    branch2_state = GraphState(
        data={},
        metadata={"iteration": 2, "path": ["node2"]}
    )

    # Simulate merge
    results = {
        "node1": {"state": branch1_state},
        "node2": {"state": branch2_state}
    }

    metadata_conflicts = []
    keys_merged = set()

    for node_name, result in results.items():
        branch_state = result["state"]

        for key, value in branch_state.metadata.items():
            if key in state.metadata and key in keys_merged:
                # Conflict detected!
                metadata_conflicts.append({
                    "key": key,
                    "node": node_name,
                    "old_value": state.metadata[key],
                    "new_value": value,
                })

            state.metadata[key] = value
            keys_merged.add(key)

    # Conflicts detected (both iteration and path keys conflict)
    assert len(metadata_conflicts) == 2
    assert metadata_conflicts[0]["key"] == "iteration"
    assert metadata_conflicts[1]["key"] == "path"

    # Last write wins (node2 overwrites node1)
    assert state.metadata["iteration"] == 2
    assert state.metadata["path"] == ["node2"]
