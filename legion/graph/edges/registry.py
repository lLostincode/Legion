from datetime import datetime
from typing import Any, Dict, Optional, Set, Type, TypeVar

from pydantic import BaseModel, Field

from ..nodes.registry import NodeRegistry
from ..state import GraphState
from .base import EdgeBase

T = TypeVar("T", bound=EdgeBase)

class EdgeRegistryMetadata(BaseModel):
    """Metadata for edge registry"""

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    version: int = 0

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to handle datetime serialization"""
        data = super().model_dump(**kwargs)
        data["created_at"] = data["created_at"].isoformat()
        data["updated_at"] = data["updated_at"].isoformat()
        return data

class EdgeRegistry:
    """Registry for managing graph edges"""

    def __init__(self, graph_state: GraphState, node_registry: NodeRegistry):
        self._metadata = EdgeRegistryMetadata()
        self._graph_state = graph_state
        self._node_registry = node_registry
        self._edges: Dict[str, EdgeBase] = {}
        self._edge_types: Dict[str, Type[EdgeBase]] = {}

        # Edge indices for efficient lookup
        self._source_edges: Dict[str, Set[str]] = {}  # node_id -> edge_ids
        self._target_edges: Dict[str, Set[str]] = {}  # node_id -> edge_ids
        self._channel_edges: Dict[str, Dict[str, Set[str]]] = {}  # node_id -> channel_name -> edge_ids

    @property
    def metadata(self) -> EdgeRegistryMetadata:
        """Get registry metadata"""
        return self._metadata

    def _update_metadata(self) -> None:
        """Update metadata after registry change"""
        self._metadata.updated_at = datetime.now()
        self._metadata.version += 1

    def register_edge_type(self, name: str, edge_type: Type[EdgeBase]) -> None:
        """Register an edge type"""
        if name in self._edge_types:
            raise ValueError(f"Edge type '{name}' already registered")

        self._edge_types[name] = edge_type
        self._update_metadata()

    def create_edge(
        self,
        type_name: str,
        source_node_id: str,
        target_node_id: str,
        source_channel: str,
        target_channel: str,
        edge_id: Optional[str] = None,
        **kwargs
    ) -> EdgeBase:
        """Create a new edge instance"""
        if type_name not in self._edge_types:
            raise ValueError(f"Unknown edge type '{type_name}'")

        # Get nodes from registry
        source_node = self._node_registry.get_node(source_node_id)
        target_node = self._node_registry.get_node(target_node_id)

        if not source_node or not target_node:
            raise ValueError("Source and target nodes must exist")

        # Check for cycles before creating edge
        if self._would_create_cycle(source_node_id, target_node_id):
            raise ValueError("Adding edge would create cycle")

        # Create edge instance
        edge_type = self._edge_types[type_name]
        edge = edge_type(
            self._graph_state,
            source_node,
            target_node,
            source_channel,
            target_channel,
            **kwargs
        )

        if edge_id:
            if edge_id in self._edges:
                raise ValueError(f"Edge ID '{edge_id}' already exists")
            edge._metadata.edge_id = edge_id

        # Update indices
        self._edges[edge.edge_id] = edge
        self._source_edges.setdefault(source_node_id, set()).add(edge.edge_id)
        self._target_edges.setdefault(target_node_id, set()).add(edge.edge_id)

        # Update channel indices
        self._channel_edges.setdefault(source_node_id, {}).setdefault(source_channel, set()).add(edge.edge_id)
        self._channel_edges.setdefault(target_node_id, {}).setdefault(target_channel, set()).add(edge.edge_id)

        # Update node registry dependencies
        self._node_registry.add_dependency(source_node_id, target_node_id)

        self._update_metadata()
        return edge

    def get_edge(self, edge_id: str) -> Optional[EdgeBase]:
        """Get edge by ID"""
        return self._edges.get(edge_id)

    def delete_edge(self, edge_id: str) -> None:
        """Delete an edge"""
        if edge_id not in self._edges:
            return

        edge = self._edges[edge_id]
        source_id = edge.source_node.node_id
        target_id = edge.target_node.node_id

        # Remove from indices
        if source_id in self._source_edges:
            self._source_edges[source_id].discard(edge_id)
        if target_id in self._target_edges:
            self._target_edges[target_id].discard(edge_id)

        # Remove from channel indices
        if source_id in self._channel_edges and edge.source_channel in self._channel_edges[source_id]:
            self._channel_edges[source_id][edge.source_channel].discard(edge_id)
        if target_id in self._channel_edges and edge.target_channel in self._channel_edges[target_id]:
            self._channel_edges[target_id][edge.target_channel].discard(edge_id)

        # Remove node dependency if no other edges exist
        if not self._has_other_edges(source_id, target_id, edge_id):
            self._node_registry.remove_dependency(source_id, target_id)

        del self._edges[edge_id]
        self._update_metadata()

    def get_node_edges(self, node_id: str, as_source: bool = True) -> Set[str]:
        """Get edges connected to a node"""
        if as_source:
            return self._source_edges.get(node_id, set()).copy()
        return self._target_edges.get(node_id, set()).copy()

    def get_channel_edges(self, node_id: str, channel_name: str) -> Set[str]:
        """Get edges connected to a channel"""
        return self._channel_edges.get(node_id, {}).get(channel_name, set()).copy()

    def _would_create_cycle(self, source_id: str, target_id: str) -> bool:
        """Check if adding an edge would create a cycle"""
        # Use node registry's dependency tracking
        try:
            self._node_registry.add_dependency(source_id, target_id)
            self._node_registry.remove_dependency(source_id, target_id)
            return False
        except ValueError:
            return True

    def _has_other_edges(self, source_id: str, target_id: str, exclude_edge_id: str) -> bool:
        """Check if other edges exist between nodes"""
        source_edges = self._source_edges.get(source_id, set())
        for edge_id in source_edges:
            if edge_id != exclude_edge_id and self._edges[edge_id].target_node.node_id == target_id:
                return True
        return False

    def checkpoint(self) -> Dict[str, Any]:
        """Create a checkpoint of current state"""
        return {
            "metadata": self._metadata.model_dump(),
            "edges": {
                edge_id: edge.checkpoint()
                for edge_id, edge in self._edges.items()
            }
        }

    def restore(self, checkpoint: Dict[str, Any]) -> None:
        """Restore from checkpoint"""
        self._metadata = EdgeRegistryMetadata(**checkpoint["metadata"])
        # Note: Edges are restored through graph restore
