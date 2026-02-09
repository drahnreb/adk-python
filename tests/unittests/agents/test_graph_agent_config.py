"""Test suite for GraphAgent configuration classes.

Tests Pydantic model validation for all graph-related config classes:
- GraphNodeConfig
- GraphEdgeConfig
- InterruptConfigYaml
- ParallelGroupConfig
- GraphAgentConfig
"""

import pytest
from pydantic import ValidationError

from google.adk.agents.graph.graph_agent_config import (
    GraphAgentConfig,
    GraphEdgeConfig,
    GraphNodeConfig,
    InterruptConfigYaml,
    ParallelGroupConfig,
)


class TestGraphNodeConfig:
    """Tests for GraphNodeConfig validation."""

    def test_minimal_node_config(self):
        """Test minimal valid node configuration."""
        config = GraphNodeConfig(name="test_node")
        assert config.name == "test_node"
        assert config.function_ref is None
        assert config.reducer == "overwrite"

    def test_node_with_function_ref(self):
        """Test node with function reference."""
        config = GraphNodeConfig(
            name="test_node",
            function_ref="my_module.my_function",
        )
        assert config.function_ref == "my_module.my_function"

    def test_node_with_mappers(self):
        """Test node with input/output mappers."""
        config = GraphNodeConfig(
            name="test_node",
            input_mapper_ref="mappers.input_fn",
            output_mapper_ref="mappers.output_fn",
        )
        assert config.input_mapper_ref == "mappers.input_fn"
        assert config.output_mapper_ref == "mappers.output_fn"

    def test_node_with_reducers(self):
        """Test node with different reducer strategies."""
        # Overwrite (default)
        config1 = GraphNodeConfig(name="node1")
        assert config1.reducer == "overwrite"

        # Append
        config2 = GraphNodeConfig(name="node2", reducer="append")
        assert config2.reducer == "append"

        # Sum
        config3 = GraphNodeConfig(name="node3", reducer="sum")
        assert config3.reducer == "sum"

        # Custom
        config4 = GraphNodeConfig(
            name="node4",
            reducer="custom",
            custom_reducer_ref="reducers.my_reducer",
        )
        assert config4.reducer == "custom"
        assert config4.custom_reducer_ref == "reducers.my_reducer"

    def test_node_extra_forbid(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            GraphNodeConfig(name="test", invalid_field="value")
        assert "Extra inputs are not permitted" in str(exc_info.value)


class TestGraphEdgeConfig:
    """Tests for GraphEdgeConfig validation."""

    def test_minimal_edge_config(self):
        """Test minimal valid edge configuration."""
        config = GraphEdgeConfig(name="edge1", from_node="start", to_node="end")
        assert config.name == "edge1"
        assert config.from_node == "start"
        assert config.to_node == "end"
        assert config.condition_ref is None
        assert config.priority == 1
        assert config.weight == 1.0

    def test_edge_with_condition(self):
        """Test edge with condition function."""
        config = GraphEdgeConfig(
            name="edge1",
            from_node="start",
            to_node="end",
            condition_ref="conditions.check_success",
        )
        assert config.condition_ref == "conditions.check_success"

    def test_edge_with_priority(self):
        """Test edge with custom priority."""
        config = GraphEdgeConfig(
            name="edge1",
            from_node="start",
            to_node="end",
            priority=10,
        )
        assert config.priority == 10

    def test_edge_with_weight(self):
        """Test edge with custom weight for weighted routing."""
        config = GraphEdgeConfig(
            name="edge1",
            from_node="start",
            to_node="end",
            weight=0.75,
        )
        assert config.weight == 0.75

    def test_edge_extra_forbid(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            GraphEdgeConfig(
                name="edge1",
                from_node="start",
                to_node="end",
                invalid_field="value",
            )
        assert "Extra inputs are not permitted" in str(exc_info.value)


class TestInterruptConfigYaml:
    """Tests for InterruptConfigYaml validation."""

    def test_minimal_interrupt_config(self):
        """Test minimal valid interrupt configuration."""
        config = InterruptConfigYaml(name="interrupt1")
        assert config.name == "interrupt1"
        assert config.mode == "none"
        assert config.interrupt_service_ref is None

    def test_interrupt_modes(self):
        """Test different interrupt modes."""
        # None (default)
        config1 = InterruptConfigYaml(name="int1", mode="none")
        assert config1.mode == "none"

        # Before
        config2 = InterruptConfigYaml(name="int2", mode="before")
        assert config2.mode == "before"

        # After
        config3 = InterruptConfigYaml(name="int3", mode="after")
        assert config3.mode == "after"

        # Both
        config4 = InterruptConfigYaml(name="int4", mode="both")
        assert config4.mode == "both"

    def test_interrupt_with_service_ref(self):
        """Test interrupt config with service reference."""
        config = InterruptConfigYaml(
            name="interrupt1",
            mode="both",
            interrupt_service_ref="services.interrupt_service",
        )
        assert config.mode == "both"
        assert config.interrupt_service_ref == "services.interrupt_service"

    def test_interrupt_extra_forbid(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            InterruptConfigYaml(name="int1", invalid_field="value")
        assert "Extra inputs are not permitted" in str(exc_info.value)


class TestParallelGroupConfig:
    """Tests for ParallelGroupConfig validation."""

    def test_minimal_parallel_config(self):
        """Test minimal valid parallel group configuration."""
        config = ParallelGroupConfig(name="pg1", nodes=["node1", "node2"])
        assert config.name == "pg1"
        assert config.nodes == ["node1", "node2"]
        assert config.join_strategy == "all"
        assert config.error_policy == "fail_fast"
        assert config.wait_n == 1

    def test_parallel_join_strategies(self):
        """Test different join strategies."""
        # All (default)
        config1 = ParallelGroupConfig(name="pg1", nodes=["n1", "n2"])
        assert config1.join_strategy == "all"

        # Any
        config2 = ParallelGroupConfig(name="pg2", nodes=["n1", "n2"], join_strategy="any")
        assert config2.join_strategy == "any"

        # N
        config3 = ParallelGroupConfig(
            name="pg3",
            nodes=["n1", "n2", "n3"],
            join_strategy="n",
            wait_n=2,
        )
        assert config3.join_strategy == "n"
        assert config3.wait_n == 2

    def test_parallel_error_policies(self):
        """Test different error policies."""
        # Fail fast (default)
        config1 = ParallelGroupConfig(name="pg1", nodes=["n1", "n2"])
        assert config1.error_policy == "fail_fast"

        # Continue
        config2 = ParallelGroupConfig(name="pg2", nodes=["n1", "n2"], error_policy="continue")
        assert config2.error_policy == "continue"

        # Collect
        config3 = ParallelGroupConfig(name="pg3", nodes=["n1", "n2"], error_policy="collect")
        assert config3.error_policy == "collect"

    def test_parallel_extra_forbid(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            ParallelGroupConfig(name="pg1", nodes=["n1"], invalid_field="value")
        assert "Extra inputs are not permitted" in str(exc_info.value)


class TestGraphAgentConfig:
    """Tests for GraphAgentConfig validation."""

    def test_minimal_graph_config(self):
        """Test minimal valid graph configuration."""
        config = GraphAgentConfig(
            name="test_graph",
            start_node="start",
        )
        assert config.name == "test_graph"
        assert config.agent_class == "GraphAgent"
        assert config.start_node == "start"
        assert config.end_nodes == []
        assert config.max_iterations == 20
        assert config.checkpointing is False

    def test_graph_with_end_nodes(self):
        """Test graph with multiple end nodes."""
        config = GraphAgentConfig(
            name="test_graph",
            start_node="start",
            end_nodes=["end1", "end2"],
        )
        assert config.end_nodes == ["end1", "end2"]

    def test_graph_with_max_iterations(self):
        """Test graph with custom max_iterations."""
        config = GraphAgentConfig(
            name="test_graph",
            start_node="start",
            max_iterations=50,
        )
        assert config.max_iterations == 50

    def test_graph_with_checkpointing(self):
        """Test graph with checkpointing enabled."""
        config = GraphAgentConfig(
            name="test_graph",
            start_node="start",
            checkpointing=True,
            checkpoint_service_ref="services.checkpoint_service",
        )
        assert config.checkpointing is True
        assert config.checkpoint_service_ref == "services.checkpoint_service"

    def test_graph_with_nodes(self):
        """Test graph with node configurations."""
        config = GraphAgentConfig(
            name="test_graph",
            start_node="start",
            nodes=[
                {"name": "start", "sub_agents": ["agent1"]},
                {"name": "middle", "sub_agents": ["agent2"]},
                {"name": "end", "sub_agents": ["agent3"]},
            ],
        )
        assert len(config.nodes) == 3
        assert config.nodes[0]["name"] == "start"
        assert config.nodes[1]["sub_agents"] == ["agent2"]

    def test_graph_with_edges(self):
        """Test graph with edge configurations."""
        config = GraphAgentConfig(
            name="test_graph",
            start_node="start",
            edges=[
                {"from_node": "start", "to_node": "middle"},
                {"from_node": "middle", "to_node": "end"},
            ],
        )
        assert len(config.edges) == 2
        assert config.edges[0]["from_node"] == "start"
        assert config.edges[1]["to_node"] == "end"

    def test_graph_with_interrupt_config(self):
        """Test graph with interrupt configuration."""
        config = GraphAgentConfig(
            name="test_graph",
            start_node="start",
            interrupt_config={
                "mode": "both",
                "interrupt_service_ref": "services.interrupt",
            },
        )
        assert config.interrupt_config is not None
        assert config.interrupt_config["mode"] == "both"

    def test_graph_with_parallel_groups(self):
        """Test graph with parallel execution groups."""
        config = GraphAgentConfig(
            name="test_graph",
            start_node="start",
            parallel_groups=[
                {
                    "nodes": ["parallel1", "parallel2"],
                    "join_strategy": "all",
                    "error_policy": "fail_fast",
                }
            ],
        )
        assert len(config.parallel_groups) == 1
        assert config.parallel_groups[0]["nodes"] == ["parallel1", "parallel2"]

    def test_graph_with_callbacks(self):
        """Test graph with callback references."""
        config = GraphAgentConfig(
            name="test_graph",
            start_node="start",
            before_node_callback_ref="callbacks.before",
            after_node_callback_ref="callbacks.after",
            on_edge_condition_callback_ref="callbacks.on_edge",
        )
        assert config.before_node_callback_ref == "callbacks.before"
        assert config.after_node_callback_ref == "callbacks.after"
        assert config.on_edge_condition_callback_ref == "callbacks.on_edge"

    def test_graph_complete_config(self):
        """Test complete graph configuration with all features."""
        config = GraphAgentConfig(
            name="complete_graph",
            description="A complete graph configuration",
            start_node="start",
            end_nodes=["end"],
            max_iterations=30,
            checkpointing=True,
            checkpoint_service_ref="services.checkpoint",
            nodes=[
                {
                    "name": "start",
                    "sub_agents": ["agent1"],
                    "reducer": "overwrite",
                },
                {
                    "name": "middle",
                    "function_ref": "functions.process",
                    "input_mapper_ref": "mappers.input",
                    "output_mapper_ref": "mappers.output",
                },
                {"name": "end", "sub_agents": ["agent3"]},
            ],
            edges=[
                {"from_node": "start", "to_node": "middle", "priority": 1},
                {
                    "from_node": "middle",
                    "to_node": "end",
                    "condition_ref": "conditions.check",
                },
            ],
            interrupt_config={
                "mode": "both",
                "interrupt_service_ref": "services.interrupt",
            },
            parallel_groups=[
                {
                    "nodes": ["parallel1", "parallel2"],
                    "join_strategy": "all",
                }
            ],
            before_node_callback_ref="callbacks.before",
            after_node_callback_ref="callbacks.after",
        )

        # Verify all fields
        assert config.name == "complete_graph"
        assert config.start_node == "start"
        assert config.end_nodes == ["end"]
        assert config.max_iterations == 30
        assert config.checkpointing is True
        assert len(config.nodes) == 3
        assert len(config.edges) == 2
        assert config.interrupt_config is not None
        assert len(config.parallel_groups) == 1
        assert config.before_node_callback_ref == "callbacks.before"
        assert config.after_node_callback_ref == "callbacks.after"

    def test_graph_extra_forbid(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            GraphAgentConfig(
                name="test",
                start_node="start",
                invalid_field="value",
            )
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_graph_agent_class_default(self):
        """Test that agent_class defaults to GraphAgent."""
        config = GraphAgentConfig(name="test", start_node="start")
        assert config.agent_class == "GraphAgent"

    def test_graph_missing_required_fields(self):
        """Test validation fails when required fields are missing."""
        # Missing name
        with pytest.raises(ValidationError) as exc_info:
            GraphAgentConfig(start_node="start")
        assert "name" in str(exc_info.value).lower()

        # Missing start_node
        with pytest.raises(ValidationError) as exc_info:
            GraphAgentConfig(name="test")
        assert "start_node" in str(exc_info.value).lower()
