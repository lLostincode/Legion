from datetime import datetime
from typing import Any, Dict, Optional, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field

from ..nodes.base import NodeBase
from ..state import GraphState

T = TypeVar("T")

class EdgeMetadata(BaseModel):
    """Edge metadata"""

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    version: int = 0
    edge_id: str = Field(default_factory=lambda: str(uuid4()))
    edge_type: str = Field(default="")
    error: Optional[str] = None

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to handle datetime serialization"""
        data = super().model_dump(**kwargs)
        data["created_at"] = data["created_at"].isoformat()
        data["updated_at"] = data["updated_at"].isoformat()
        return data

class EdgeBase:
    """Base class for all graph edges"""

    def __init__(
        self,
        graph_state: GraphState,
        source_node: NodeBase,
        target_node: NodeBase,
        source_channel: str,
        target_channel: str
    ):
        """Initialize edge

        Args:
        ----
            graph_state: Graph state manager
            source_node: Source node
            target_node: Target node
            source_channel: Name of source output channel
            target_channel: Name of target input channel

        """
        self._metadata = EdgeMetadata(
            edge_type=self.__class__.__name__
        )
        self._graph_state = graph_state
        self._source_node = source_node
        self._target_node = target_node
        self._source_channel = source_channel
        self._target_channel = target_channel

        # Validate channel existence and compatibility
        self._validate_channels()

    @property
    def metadata(self) -> EdgeMetadata:
        """Get edge metadata"""
        return self._metadata

    @property
    def edge_id(self) -> str:
        """Get edge ID"""
        return self._metadata.edge_id

    @property
    def source_node(self) -> NodeBase:
        """Get source node"""
        return self._source_node

    @property
    def target_node(self) -> NodeBase:
        """Get target node"""
        return self._target_node

    @property
    def source_channel(self) -> str:
        """Get source channel name"""
        return self._source_channel

    @property
    def target_channel(self) -> str:
        """Get target channel name"""
        return self._target_channel

    def _update_metadata(self) -> None:
        """Update metadata after state change"""
        self._metadata.updated_at = datetime.now()
        self._metadata.version += 1

    def _validate_channels(self) -> None:
        """Validate channel existence and compatibility"""
        source_channel = self._source_node.get_output_channel(self._source_channel)
        if not source_channel:
            raise ValueError(
                f"Source channel '{self._source_channel}' not found in node {self._source_node.node_id}"
            )

        target_channel = self._target_node.get_input_channel(self._target_channel)
        if not target_channel:
            raise ValueError(
                f"Target channel '{self._target_channel}' not found in node {self._target_node.node_id}"
            )

        # Check channel type compatibility
        if not isinstance(source_channel, type(target_channel)):
            raise TypeError(
                f"Channel type mismatch: {type(source_channel)} -> {type(target_channel)}"
            )

        # Check value type compatibility if available
        source_type = getattr(source_channel, "type_hint", None)
        target_type = getattr(target_channel, "type_hint", None)

        if source_type and target_type and source_type != target_type:
            raise TypeError(
                f"Channel value type mismatch: {source_type} -> {target_type}"
            )

    def checkpoint(self) -> Dict[str, Any]:
        """Create a checkpoint of current state"""
        return {
            "metadata": self._metadata.model_dump(),
            "source_node": self._source_node.node_id,
            "target_node": self._target_node.node_id,
            "source_channel": self._source_channel,
            "target_channel": self._target_channel
        }

    def restore(self, checkpoint: Dict[str, Any]) -> None:
        """Restore from checkpoint"""
        self._metadata = EdgeMetadata(**checkpoint["metadata"])
        # Note: Nodes and channels are restored through registry
