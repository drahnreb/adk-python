"""Data models for checkpoint service."""

from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Optional

from pydantic import BaseModel
from pydantic import Field


class CheckpointMetadata(BaseModel):  # type: ignore[misc]
    """Metadata for a checkpoint.

    Stored in session state and provides information about checkpoint creation
    and artifact versions at checkpoint time.
    """

    checkpoint_id: str = Field(description="Unique identifier for this checkpoint")

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 timestamp when checkpoint was created",
    )

    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of the checkpoint",
    )

    agent_name: Optional[str] = Field(
        default=None,
        description="Name of the agent that created the checkpoint",
    )

    artifact_versions: dict[str, int] = Field(
        default_factory=dict,
        description="Map of filename -> version number at checkpoint time",
    )

    state_snapshot: dict[str, Any] = Field(
        default_factory=dict,
        description="Snapshot of session state at checkpoint time",
    )

    is_delta: bool = Field(
        default=False,
        description="Whether state_snapshot contains only changes (delta) or full state",
    )

    base_checkpoint_id: Optional[str] = Field(
        default=None,
        description="ID of the base checkpoint if this is a delta checkpoint",
    )

    custom_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional user-defined metadata",
    )


class ListCheckpointsResponse(BaseModel):  # type: ignore[misc]
    """Response model for paginated checkpoint listing.

    Following ADK pagination patterns (similar to ConversationListResponse).
    """

    checkpoints: list[CheckpointMetadata] = Field(
        default_factory=list,
        description="List of checkpoint metadata objects for current page",
    )

    total_count: int = Field(
        description="Total number of checkpoints across all pages",
    )

    page: int = Field(
        description="Current page number (1-indexed)",
    )

    page_size: int = Field(
        description="Number of checkpoints per page",
    )

    has_next: bool = Field(
        description="Whether there are more checkpoints on next page",
    )

    has_previous: bool = Field(
        description="Whether there are checkpoints on previous page",
    )
