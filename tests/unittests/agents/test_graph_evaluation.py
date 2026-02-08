"""Tests for GraphAgent evaluation metrics."""

import pytest
from types import SimpleNamespace
from google.genai import types
from google.adk.agents.graph.evaluation_metrics import (
    graph_path_match,
    state_contains_keys,
    node_execution_count,
)
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalStatus


@pytest.mark.asyncio
async def test_graph_path_match_exact():
    """Test graph_path_match metric with exact path match."""
    invocation = Invocation(
        userContent=types.Content(parts=[types.Part(text="test")]),
        finalResponse=types.Content(parts=[types.Part(text="response")]),
    )

    # Create metric with custom attributes (SimpleNamespace for testing)
    # NOTE: In production, actual_graph_path would come from intermediate_data
    metric = SimpleNamespace(
        metric_name="graph_path",
        expected_graph_path=["n1", "n2", "n3"],
        actual_graph_path=["n1", "n2", "n3"],  # Exact match
    )

    # Evaluate
    result = graph_path_match(metric, [invocation], None, None)

    # Should pass with perfect score
    assert result.overall_score == 1.0
    assert result.overall_eval_status == EvalStatus.PASSED
    assert len(result.per_invocation_results) == 1
    assert result.per_invocation_results[0].score == 1.0


@pytest.mark.asyncio
async def test_graph_path_match_partial():
    """Test graph_path_match with partial path match."""
    invocation = Invocation(
        userContent=types.Content(parts=[types.Part(text="test")]),
        finalResponse=types.Content(parts=[types.Part(text="response")]),
    )

    metric = SimpleNamespace(
        metric_name="graph_path",
        expected_graph_path=["n1", "n3", "n4"],
        actual_graph_path=["n1", "n2"],  # Partial match
    )

    result = graph_path_match(metric, [invocation], None, None)

    # Should have partial score (1 match out of 3 expected)
    assert result.overall_score < 1.0
    assert result.overall_score > 0.0  # At least n1 matches


@pytest.mark.asyncio
async def test_state_contains_keys_exact():
    """Test state_contains_keys metric with exact match."""
    invocation = Invocation(
        userContent=types.Content(parts=[types.Part(text="test")]),
        finalResponse=types.Content(parts=[types.Part(text="done")]),
    )

    metric = SimpleNamespace(
        metric_name="state_check",
        expected_state={"key1": "value1", "key2": 42},
        actual_state={"key1": "value1", "key2": 42},  # Exact match
    )

    result = state_contains_keys(metric, [invocation], None, None)

    # Should pass with perfect score
    assert result.overall_score == 1.0
    assert result.overall_eval_status == EvalStatus.PASSED


@pytest.mark.asyncio
async def test_state_contains_keys_partial():
    """Test state_contains_keys with partial match."""
    invocation = Invocation(
        userContent=types.Content(parts=[types.Part(text="test")]),
        finalResponse=types.Content(parts=[types.Part(text="done")]),
    )

    metric = SimpleNamespace(
        metric_name="state_check",
        expected_state={"key1": "value1", "key2": 42},
        actual_state={"key1": "value1", "key2": 999},  # key2 wrong
    )

    result = state_contains_keys(metric, [invocation], None, None)

    # Should have partial score (1 out of 2 keys match)
    assert result.overall_score == 0.5
    assert result.overall_eval_status == EvalStatus.FAILED


@pytest.mark.asyncio
async def test_node_execution_count_exact():
    """Test node_execution_count with exact counts."""
    invocation = Invocation(
        userContent=types.Content(parts=[types.Part(text="test")]),
        finalResponse=types.Content(parts=[types.Part(text="done")]),
    )

    metric = SimpleNamespace(
        metric_name="execution_count",
        expected_node_counts={"loop_node": 3},
        actual_node_counts={"loop_node": 3},  # Exact match
    )

    result = node_execution_count(metric, [invocation], None, None)

    # Should pass if count matches
    assert result.overall_score == 1.0
    assert result.overall_eval_status == EvalStatus.PASSED


@pytest.mark.asyncio
async def test_metrics_with_no_expected_data():
    """Test metrics skip when no expected data provided."""
    invocation = Invocation(
        userContent=types.Content(parts=[types.Part(text="test")]),
        finalResponse=types.Content(parts=[types.Part(text="done")]),
    )

    metric = SimpleNamespace(metric_name="test")  # No custom fields

    # All metrics should return NOT_EVALUATED when no expected data
    result1 = graph_path_match(metric, [invocation], None, None)
    assert result1.overall_eval_status == EvalStatus.NOT_EVALUATED

    result2 = state_contains_keys(metric, [invocation], None, None)
    assert result2.overall_eval_status == EvalStatus.NOT_EVALUATED

    result3 = node_execution_count(metric, [invocation], None, None)
    assert result3.overall_eval_status == EvalStatus.NOT_EVALUATED
