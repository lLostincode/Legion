from typing import Any, Dict, Optional, Type, TypeVar, Union

from .channels import Channel
from .edges.base import EdgeBase
from .graph import Graph, GraphConfig
from .nodes.base import NodeBase

T = TypeVar("T")

class GraphBuilder:
    """Fluent API for building graphs"""

    def __init__(self, name: str = "", description: str = "", config: Optional[GraphConfig] = None):
        self._graph = Graph(name=name, description=description, config=config)
        self._node_configs: Dict[str, Dict[str, Any]] = {}
        self._edge_configs: Dict[str, Dict[str, Any]] = {}
        self._current_node: Optional[str] = None
        self._current_edge: Optional[str] = None

    @property
    def graph(self) -> Graph:
        """Get the constructed graph"""
        return self._graph

    def add_node(
        self,
        node_type: Union[str, Type[NodeBase]],
        node_id: Optional[str] = None,
        **kwargs
    ) -> "GraphBuilder":
        """Add a node to the graph

        Args:
        ----
            node_type: Node type name or class
            node_id: Optional node ID
            **kwargs: Additional node parameters

        Returns:
        -------
            Self for chaining

        """
        # Create node without configuration
        node = self._graph.add_node(node_type, node_id)
        self._current_node = node.node_id

        # Store configuration for later use
        self._node_configs[node.node_id] = kwargs

        # Apply configuration if any
        if kwargs:
            self.configure_node(**kwargs)

        return self

    def configure_node(self, **kwargs) -> "GraphBuilder":
        """Configure the current node

        Args:
        ----
            **kwargs: Configuration parameters

        Returns:
        -------
            Self for chaining

        Raises:
        ------
            ValueError: If no current node

        """
        if not self._current_node:
            raise ValueError("No current node to configure")

        # Update stored configuration
        self._node_configs[self._current_node].update(kwargs)

        # Apply configuration to node
        node = self._graph.nodes._nodes[self._current_node]
        for key, value in kwargs.items():
            setattr(node, key, value)

        return self

    def with_input_channel(
        self,
        name: str,
        channel_type: Type[Channel[T]],
        **kwargs
    ) -> "GraphBuilder":
        """Add an input channel to the current node

        Args:
        ----
            name: Channel name
            channel_type: Channel type
            **kwargs: Channel configuration

        Returns:
        -------
            Self for chaining

        Raises:
        ------
            ValueError: If no current node

        """
        if not self._current_node:
            raise ValueError("No current node to configure")

        node = self._graph.nodes._nodes[self._current_node]
        node.create_input_channel(name, channel_type, **kwargs)
        return self

    def with_output_channel(
        self,
        name: str,
        channel_type: Type[Channel[T]],
        **kwargs
    ) -> "GraphBuilder":
        """Add an output channel to the current node

        Args:
        ----
            name: Channel name
            channel_type: Channel type
            **kwargs: Channel configuration

        Returns:
        -------
            Self for chaining

        Raises:
        ------
            ValueError: If no current node

        """
        if not self._current_node:
            raise ValueError("No current node to configure")

        node = self._graph.nodes._nodes[self._current_node]
        node.create_output_channel(name, channel_type, **kwargs)
        return self

    def select_node(self, node_id: str) -> "GraphBuilder":
        """Select a node as the current node

        Args:
        ----
            node_id: Node ID

        Returns:
        -------
            Self for chaining

        Raises:
        ------
            ValueError: If node doesn't exist

        """
        if node_id not in self._graph.nodes._nodes:
            raise ValueError(f"Node {node_id} not found")

        self._current_node = node_id
        return self

    def connect(
        self,
        source_node: Union[str, NodeBase],
        target_node: Union[str, NodeBase],
        edge_type: Union[str, Type[EdgeBase]],
        source_channel: Optional[str] = None,
        target_channel: Optional[str] = None,
        edge_id: Optional[str] = None,
        **kwargs
    ) -> "GraphBuilder":
        """Connect two nodes with an edge

        Args:
        ----
            source_node: Source node ID or instance
            target_node: Target node ID or instance
            edge_type: Edge type name or class
            source_channel: Optional source channel name
            target_channel: Optional target channel name
            edge_id: Optional edge ID
            **kwargs: Additional edge parameters

        Returns:
        -------
            Self for chaining

        """
        edge = self._graph.add_edge(
            source_node=source_node,
            target_node=target_node,
            edge_type=edge_type,
            source_channel=source_channel,
            target_channel=target_channel,
            edge_id=edge_id,
            **kwargs
        )
        self._current_edge = edge.edge_id
        self._edge_configs[edge.edge_id] = kwargs
        return self

    def configure_edge(self, **kwargs) -> "GraphBuilder":
        """Configure the current edge

        Args:
        ----
            **kwargs: Configuration parameters

        Returns:
        -------
            Self for chaining

        Raises:
        ------
            ValueError: If no current edge

        """
        if not self._current_edge:
            raise ValueError("No current edge to configure")

        # Update stored configuration
        self._edge_configs[self._current_edge].update(kwargs)

        # Apply configuration to edge
        edge = self._graph.edges._edges[self._current_edge]
        for key, value in kwargs.items():
            setattr(edge, key, value)

        return self

    def select_edge(self, edge_id: str) -> "GraphBuilder":
        """Select an edge as the current edge

        Args:
        ----
            edge_id: Edge ID

        Returns:
        -------
            Self for chaining

        Raises:
        ------
            ValueError: If edge doesn't exist

        """
        if edge_id not in self._graph.edges._edges:
            raise ValueError(f"Edge {edge_id} not found")

        self._current_edge = edge_id
        return self

    def build(self) -> Graph:
        """Build and return the graph

        Returns
        -------
            The constructed graph

        """
        return self._graph
