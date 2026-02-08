# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Config definition for GraphAgent."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import ConfigDict
from pydantic import Field

from ...utils.feature_decorator import experimental
from ..base_agent_config import BaseAgentConfig  # type: ignore[attr-defined]


@experimental
class GraphNodeConfig(BaseAgentConfig):  # type: ignore[misc]
    """Configuration for a single node in the graph.

    A node can contain either an agent reference or a function reference,
    plus optional mappers and reducers for state management.
    """

    model_config = ConfigDict(extra="forbid")

    # Node can reference an agent (sub_agents) OR a function
    function_ref: Optional[str] = Field(
        default=None,
        description="Reference to a function (e.g., 'module.function_name')",
    )

    input_mapper_ref: Optional[str] = Field(
        default=None,
        description="Reference to custom input mapper function",
    )

    output_mapper_ref: Optional[str] = Field(
        default=None,
        description="Reference to custom output mapper function",
    )

    reducer: str = Field(
        default="overwrite",
        description="State reducer strategy: overwrite|append|sum|custom",
    )

    custom_reducer_ref: Optional[str] = Field(
        default=None,
        description="Reference to custom reducer function (if reducer=custom)",
    )


@experimental
class GraphEdgeConfig(BaseAgentConfig):  # type: ignore[misc]
    """Configuration for an edge between nodes.

    Edges can have optional conditions for conditional routing.
    """

    model_config = ConfigDict(extra="forbid")

    from_node: str = Field(description="Source node name")

    to_node: str = Field(description="Target node name")

    condition_ref: Optional[str] = Field(
        default=None,
        description="Reference to condition function for conditional routing",
    )

    priority: int = Field(
        default=1,
        description="Edge priority for routing (higher = evaluated first)",
    )

    weight: float = Field(
        default=1.0,
        description="Edge weight for weighted random routing",
    )


@experimental
class InterruptConfigYaml(BaseAgentConfig):  # type: ignore[misc]
    """Configuration for interrupt handling."""

    model_config = ConfigDict(extra="forbid")

    mode: str = Field(
        default="none",
        description="Interrupt mode: none|before|after|both",
    )

    interrupt_service_ref: Optional[str] = Field(
        default=None,
        description="Reference to InterruptService instance",
    )


@experimental
class ParallelGroupConfig(BaseAgentConfig):  # type: ignore[misc]
    """Configuration for parallel node execution."""

    model_config = ConfigDict(extra="forbid")

    nodes: List[str] = Field(description="List of node names to execute in parallel")

    join_strategy: str = Field(
        default="all",
        description="Join strategy: all|any|n",
    )

    error_policy: str = Field(
        default="fail_fast",
        description="Error policy: fail_fast|continue|collect",
    )

    wait_n: int = Field(
        default=1,
        description="Number of nodes to wait for (when join_strategy=n)",
    )


@experimental
class GraphAgentConfig(BaseAgentConfig):  # type: ignore[misc]
    """The config for the YAML schema of a GraphAgent.

    This config supports defining graph structure, nodes, edges, and
    advanced features like interrupts and parallel execution.

    Example YAML:
        ```yaml
        agent_class: GraphAgent
        name: my_graph
        description: My graph workflow
        start_node: start
        end_nodes:
          - end
        max_iterations: 10
        checkpointing: true
        nodes:
          - name: start
            sub_agents:
              - agent1
          - name: middle
            sub_agents:
              - agent2
          - name: end
            sub_agents:
              - agent3
        edges:
          - from_node: start
            to_node: middle
          - from_node: middle
            to_node: end
        ```
    """

    model_config = ConfigDict(extra="forbid")

    agent_class: str = Field(
        default="GraphAgent",
        description="The value is used to uniquely identify the GraphAgent class.",
    )

    start_node: str = Field(description="Name of the starting node")

    end_nodes: List[str] = Field(
        default_factory=list,
        description="List of end node names",
    )

    max_iterations: int = Field(
        default=20,
        description="Maximum iterations for cyclic graphs",
    )

    checkpointing: bool = Field(
        default=False,
        description="Enable automatic checkpointing",
    )

    checkpoint_service_ref: Optional[str] = Field(
        default=None,
        description="Reference to CheckpointService instance",
    )

    # Graph structure
    nodes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of node configurations",
    )

    edges: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of edge configurations",
    )

    # Advanced features
    interrupt_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Interrupt configuration",
    )

    parallel_groups: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of parallel execution group configurations",
    )

    # Callbacks
    before_node_callback_ref: Optional[str] = Field(
        default=None,
        description="Reference to before_node callback function",
    )

    after_node_callback_ref: Optional[str] = Field(
        default=None,
        description="Reference to after_node callback function",
    )

    on_edge_condition_callback_ref: Optional[str] = Field(
        default=None,
        description="Reference to on_edge_condition callback function",
    )
