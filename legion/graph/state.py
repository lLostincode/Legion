from datetime import datetime
from typing import Any, Dict, List, Optional, Type, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .channels import Channel, ChannelMetadata, LastValue, SharedState, ValueSequence

T = TypeVar("T")

class GraphStateMetadata(BaseModel):
    """Metadata for graph state"""

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    version: int = 0
    graph_id: str = Field(default_factory=lambda: str(uuid4()))

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

class GraphState:
    """Manages state for a graph instance"""

    def __init__(self):
        self._metadata = GraphStateMetadata()
        self._channels: Dict[str, Channel] = {}
        self._global_state = SharedState()

    @property
    def metadata(self) -> GraphStateMetadata:
        """Get graph state metadata"""
        return self._metadata

    @property
    def graph_id(self) -> str:
        """Get graph ID"""
        return self._metadata.graph_id

    def _update_metadata(self) -> None:
        """Update metadata after state change"""
        self._metadata.updated_at = datetime.now()
        self._metadata.version += 1

    def create_channel(
        self,
        channel_type: Type[Channel[T]],
        name: str,
        type_hint: Optional[Type[T]] = None,
        **kwargs
    ) -> Channel[T]:
        """Create a new channel"""
        if name in self._channels:
            raise ValueError(f"Channel '{name}' already exists")

        channel = channel_type(type_hint=type_hint, **kwargs)
        self._channels[name] = channel
        self._update_metadata()
        return channel

    def get_channel(self, name: str) -> Optional[Channel]:
        """Get channel by name"""
        return self._channels.get(name)

    def delete_channel(self, name: str) -> None:
        """Delete channel by name"""
        if name in self._channels:
            del self._channels[name]
            self._update_metadata()

    def list_channels(self) -> List[str]:
        """List all channel names"""
        return list(self._channels.keys())

    def get_global_state(self) -> Dict[str, Any]:
        """Get global state"""
        return self._global_state.get()

    def update_global_state(self, updates: Dict[str, Any]) -> None:
        """Update global state"""
        self._global_state.update(updates)
        self._update_metadata()

    def set_global_state(self, state: Dict[str, Any]) -> None:
        """Set global state"""
        self._global_state.set(state)
        self._update_metadata()

    def checkpoint(self) -> Dict[str, Any]:
        """Create a checkpoint of current state"""
        channels_checkpoint = {
            name: channel.checkpoint()
            for name, channel in self._channels.items()
        }

        return {
            "metadata": self._metadata.model_dump(),
            "channels": channels_checkpoint,
            "global_state": self._global_state.checkpoint()
        }

    def restore(self, checkpoint: Dict[str, Any]) -> None:
        """Restore from checkpoint"""
        self._metadata = GraphStateMetadata(**checkpoint["metadata"])

        # Restore channels
        self._channels.clear()
        for name, channel_checkpoint in checkpoint["channels"].items():
            # Determine channel type from checkpoint
            ChannelMetadata(**channel_checkpoint["metadata"])
            if "values" in channel_checkpoint:
                channel = ValueSequence()
            elif "value" in channel_checkpoint:
                channel = LastValue()
            else:
                channel = SharedState()

            channel.restore(channel_checkpoint)
            self._channels[name] = channel

        # Restore global state
        self._global_state.restore(checkpoint["global_state"])

    def clear(self) -> None:
        """Clear all state"""
        self._channels.clear()
        self._global_state = SharedState()
        self._metadata = GraphStateMetadata()

    def merge(self, other: "GraphState") -> None:
        """Merge another graph state into this one"""
        # Merge channels
        for name, channel in other._channels.items():
            if name not in self._channels:
                self._channels[name] = channel
            else:
                # For now, skip conflicting channels
                continue

        # Merge global state
        other_state = other.get_global_state()
        current_state = self.get_global_state()
        merged_state = {**current_state, **other_state}
        self.set_global_state(merged_state)

        self._update_metadata()
