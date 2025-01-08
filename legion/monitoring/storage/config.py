"""Configuration for storage backends"""

from typing import Optional

from pydantic import BaseModel, Field


class StorageConfig(BaseModel):
    """Configuration for storage backends

    Attributes
    ----------
        retention_days: Number of days to retain events for
        cleanup_interval: Number of minutes between cleanup runs
        max_events: Maximum number of events to store (None for unlimited)

    """

    retention_days: int = Field(
        default=30,
        description="Number of days to retain events for",
        ge=1
    )

    cleanup_interval: float = Field(
        default=60.0,
        description="Number of minutes between cleanup runs",
        gt=0
    )

    max_events: Optional[int] = Field(
        default=None,
        description="Maximum number of events to store (None for unlimited)",
        ge=1
    )
