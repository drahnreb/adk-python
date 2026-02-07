"""Graph state management with typed state and reducers."""

from enum import Enum
from typing import Any
from typing import Dict

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class StateReducer(str, Enum):
    """State reduction strategies for merging node outputs.

    Defines how node outputs are merged into the graph state:
    - OVERWRITE: Replace existing value with new value
    - APPEND: Append new value to list (creates list if needed)
    - SUM: Sum numeric values
    - CUSTOM: Use custom reducer function
    """

    OVERWRITE = "overwrite"
    APPEND = "append"
    SUM = "sum"
    CUSTOM = "custom"


class GraphState(BaseModel):  # type: ignore[misc]
    """Typed state container for graph execution.

    GraphState holds the evolving state as the graph executes:
    - data: Node outputs and intermediate results
    - metadata: Execution metadata (iteration count, path, etc.)

    Example:
        ```python
        state = GraphState(
            data={"input": "user query", "result": "agent response"},
            metadata={"iteration": 1, "path": ["start", "process", "end"]}
        )
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: Dict[str, Any] = Field(
        default_factory=dict, description="Node outputs and intermediate results"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Execution metadata (iteration, path, etc.)"
    )
