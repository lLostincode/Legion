from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Type, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from ..state import GraphState
from .base import NodeBase, NodeStatus

T = TypeVar("T", bound=NodeBase)

class NodeRegistryMetadata(BaseModel):
    """Metadata for node registry"""

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    version: int = 0

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

class NodeRegistry:
    """Registry for managing graph nodes"""

    def __init__(self, graph_state: GraphState):
        self._metadata = NodeRegistryMetadata()
        self._graph_state = graph_state
        self._nodes: Dict[str, NodeBase] = {}
        self._node_types: Dict[str, Type[NodeBase]] = {}
        self._dependencies: Dict[str, Set[str]] = {}  # node_id -> dependent node_ids
        self._reverse_dependencies: Dict[str, Set[str]] = {}  # node_id -> dependency node_ids

    @property
    def metadata(self) -> NodeRegistryMetadata:
        """Get registry metadata"""
        return self._metadata

    def _update_metadata(self) -> None:
        """Update metadata after registry change"""
        self._metadata.updated_at = datetime.now()
        self._metadata.version += 1

    def register_node_type(self, name: str, node_type: Type[NodeBase]) -> None:
        """Register a node type"""
        if name in self._node_types:
            raise ValueError(f"Node type '{name}' already registered")

        self._node_types[name] = node_type
        self._update_metadata()

    def create_node(
        self,
        type_name: str,
        node_id: Optional[str] = None,
        **kwargs
    ) -> NodeBase:
        """Create a new node instance"""
        if type_name not in self._node_types:
            raise ValueError(f"Unknown node type '{type_name}'")

        node_type = self._node_types[type_name]
        node = node_type(self._graph_state, **kwargs)

        if node_id:
            if node_id in self._nodes:
                raise ValueError(f"Node ID '{node_id}' already exists")
            node._metadata.node_id = node_id

        self._nodes[node.node_id] = node
        self._dependencies[node.node_id] = set()
        self._reverse_dependencies[node.node_id] = set()
        self._update_metadata()

        return node

    def register_node(self, node: NodeBase) -> None:
        """Register an existing node"""
        if node.node_id in self._nodes:
            raise ValueError(f"Node ID '{node.node_id}' already exists")

        self._nodes[node.node_id] = node
        self._dependencies[node.node_id] = set()
        self._reverse_dependencies[node.node_id] = set()
        self._update_metadata()

    def get_node(self, node_id: str) -> Optional[NodeBase]:
        """Get node by ID"""
        return self._nodes.get(node_id)

    def delete_node(self, node_id: str) -> None:
        """Delete a node"""
        if node_id not in self._nodes:
            return

        # Remove dependencies
        for dependent in self._dependencies[node_id]:
            self._reverse_dependencies[dependent].remove(node_id)

        for dependency in self._reverse_dependencies[node_id]:
            self._dependencies[dependency].remove(node_id)

        del self._nodes[node_id]
        del self._dependencies[node_id]
        del self._reverse_dependencies[node_id]
        self._update_metadata()

    def add_dependency(self, from_node: str, to_node: str) -> None:
        """Add a dependency between nodes"""
        if from_node not in self._nodes or to_node not in self._nodes:
            raise ValueError("Both nodes must exist")

        # Initialize dependencies if not already present
        if from_node not in self._dependencies:
            self._dependencies[from_node] = set()
        if to_node not in self._reverse_dependencies:
            self._reverse_dependencies[to_node] = set()

        if self._would_create_cycle(from_node, to_node):
            raise ValueError("Adding dependency would create cycle")

        self._dependencies[from_node].add(to_node)
        self._reverse_dependencies[to_node].add(from_node)
        self._update_metadata()

    def remove_dependency(self, from_node: str, to_node: str) -> None:
        """Remove a dependency between nodes"""
        if from_node in self._dependencies:
            self._dependencies[from_node].discard(to_node)

        if to_node in self._reverse_dependencies:
            self._reverse_dependencies[to_node].discard(from_node)

        self._update_metadata()

    def get_dependencies(self, node_id: str) -> Set[str]:
        """Get node's dependencies"""
        return self._dependencies.get(node_id, set()).copy()

    def get_dependents(self, node_id: str) -> Set[str]:
        """Get node's dependents"""
        return self._reverse_dependencies.get(node_id, set()).copy()

    def get_execution_order(self) -> List[str]:
        """Get topologically sorted execution order"""
        visited = set()
        temp_mark = set()
        order = []

        def visit(node_id: str) -> None:
            if node_id in temp_mark:
                raise ValueError("Cycle detected in node dependencies")
            if node_id in visited:
                return

            temp_mark.add(node_id)

            # Visit dependencies in reverse order
            for dependency in sorted(self._reverse_dependencies[node_id], reverse=True):
                visit(dependency)

            temp_mark.remove(node_id)
            visited.add(node_id)
            order.append(node_id)

        # Visit nodes in reverse order for consistent results
        for node_id in sorted(self._nodes.keys(), reverse=True):
            if node_id not in visited:
                visit(node_id)

        return list(reversed(order))

    def get_node_status(self) -> Dict[str, NodeStatus]:
        """Get status of all nodes"""
        return {
            node_id: node.status
            for node_id, node in self._nodes.items()
        }

    def clear(self) -> None:
        """Clear registry"""
        self._nodes.clear()
        self._dependencies.clear()
        self._reverse_dependencies.clear()
        self._metadata = NodeRegistryMetadata()

    def _would_create_cycle(self, from_node: str, to_node: str) -> bool:
        """Check if adding a dependency would create a cycle"""
        visited = set()
        temp_mark = set()

        def visit(node_id: str) -> bool:
            if node_id == from_node:
                return True
            if node_id in temp_mark:
                return False
            if node_id in visited:
                return False

            temp_mark.add(node_id)

            # Check dependencies
            for dependency in self._dependencies.get(node_id, set()):
                if visit(dependency):
                    return True

            temp_mark.remove(node_id)
            visited.add(node_id)
            return False

        # Start from target node
        return visit(to_node)

    def checkpoint(self) -> Dict[str, Any]:
        """Create a checkpoint of current state"""
        return {
            "metadata": self._metadata.model_dump(),
            "nodes": {
                node_id: node.checkpoint()
                for node_id, node in self._nodes.items()
            },
            "node_types": {
                node_id: node.__class__.__name__
                for node_id, node in self._nodes.items()
            },
            "dependencies": {
                node_id: list(deps)
                for node_id, deps in self._dependencies.items()
            }
        }

    def restore(self, checkpoint: Dict[str, Any]) -> None:
        """Restore from checkpoint"""
        self._metadata = NodeRegistryMetadata(**checkpoint["metadata"])

        # Clear current state
        self._nodes.clear()
        self._dependencies.clear()
        self._reverse_dependencies.clear()

        # Create reverse lookup of node types
        type_lookup = {
            type_.__name__: name
            for name, type_ in self._node_types.items()
        }

        # Restore nodes
        for node_id, node_checkpoint in checkpoint["nodes"].items():
            # Get node type from checkpoint
            node_type = checkpoint["node_types"][node_id]
            type_name = type_lookup[node_type]
            node_type = self._node_types[type_name]

            # Create and restore node
            node = node_type(self._graph_state)
            node.restore(node_checkpoint)
            self._nodes[node_id] = node

        # Restore dependencies
        for node_id, deps in checkpoint["dependencies"].items():
            self._dependencies[node_id] = set(deps)
            for dep in deps:
                if dep not in self._reverse_dependencies:
                    self._reverse_dependencies[dep] = set()
                self._reverse_dependencies[dep].add(node_id)
