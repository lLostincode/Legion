import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..memory.base import MemoryProvider
from .state import GraphState


class GraphCheckpoint(BaseModel):
    """Container for graph checkpoint data"""

    version: str = "1.0"
    created_at: datetime = Field(default_factory=datetime.now)
    state_data: Dict[str, Any]

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

class GraphCheckpointer:
    """Handles saving and loading graph state checkpoints"""

    def __init__(self, memory_provider: Optional[MemoryProvider] = None):
        self._memory_provider = memory_provider

    async def save_checkpoint(
        self,
        graph_state: GraphState,
        path: Optional[Path] = None,
        thread_id: Optional[str] = None
    ) -> None:
        """Save graph state checkpoint

        Args:
        ----
            graph_state: The graph state to checkpoint
            path: Optional file path to save checkpoint to
            thread_id: Optional thread ID for memory provider storage

        """
        checkpoint = GraphCheckpoint(
            state_data=graph_state.checkpoint()
        )

        # Save to file if path provided
        if path:
            path = Path(path)
            with path.open("w") as f:
                json_str = checkpoint.model_dump_json(indent=2)
                f.write(json_str)

        # Save to memory provider if available
        if self._memory_provider and thread_id:
            await self._memory_provider.save_state(
                entity_id=graph_state.graph_id,
                thread_id=thread_id,
                state={"graph_checkpoint": checkpoint.dict()}
            )

    async def load_checkpoint(
        self,
        graph_state: GraphState,
        path: Optional[Path] = None,
        thread_id: Optional[str] = None,
        graph_id: Optional[str] = None
    ) -> None:
        """Load graph state from checkpoint

        Args:
        ----
            graph_state: The graph state to restore into
            path: Optional file path to load checkpoint from
            thread_id: Optional thread ID for memory provider storage
            graph_id: Optional graph ID to load from (if different from current state)

        """
        checkpoint_data = None

        # Try loading from file
        if path:
            path = Path(path)
            if path.exists():
                with path.open("r") as f:
                    data = json.load(f)
                    if "created_at" in data:
                        data["created_at"] = datetime.fromisoformat(data["created_at"])
                    checkpoint_data = GraphCheckpoint(**data)

        # Try loading from memory provider
        if not checkpoint_data and self._memory_provider and thread_id:
            # Use provided graph_id or try to get it from checkpoint data
            entity_id = graph_id or (
                checkpoint_data.state_data["metadata"]["graph_id"]
                if checkpoint_data and "metadata" in checkpoint_data.state_data
                else graph_state.graph_id
            )

            state = await self._memory_provider.load_state(
                entity_id=entity_id,
                thread_id=thread_id
            )
            if state and "graph_checkpoint" in state:
                checkpoint_data = GraphCheckpoint(**state["graph_checkpoint"])

        # Restore state if checkpoint found
        if checkpoint_data:
            graph_state.restore(checkpoint_data.state_data)
        else:
            raise ValueError("No checkpoint found at specified location")

    async def list_checkpoints(
        self,
        graph_id: Optional[str] = None
    ) -> Dict[str, GraphCheckpoint]:
        """List available checkpoints in memory provider

        Args:
        ----
            graph_id: Optional graph ID to filter by

        Returns:
        -------
            Dict mapping thread IDs to checkpoints

        """
        if not self._memory_provider:
            return {}

        checkpoints = {}
        threads = await self._memory_provider.list_threads(entity_id=graph_id)

        for thread in threads:
            state = await self._memory_provider.load_state(
                entity_id=thread.entity_id,
                thread_id=thread.thread_id
            )
            if state and "graph_checkpoint" in state:
                checkpoints[thread.thread_id] = GraphCheckpoint(**state["graph_checkpoint"])

        return checkpoints

    async def delete_checkpoint(
        self,
        thread_id: str,
        recursive: bool = False
    ) -> None:
        """Delete a checkpoint from memory provider

        Args:
        ----
            thread_id: Thread ID of checkpoint to delete
            recursive: Whether to recursively delete child threads

        """
        if self._memory_provider:
            await self._memory_provider.delete_thread(
                thread_id=thread_id,
                recursive=recursive
            )
