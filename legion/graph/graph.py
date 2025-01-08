import logging
from datetime import datetime
from enum import Enum
from typing import Optional, Type, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field

from .edges.base import EdgeBase
from .edges.registry import EdgeRegistry
from .edges.validator import EdgeValidator
from .nodes.base import NodeBase
from .nodes.execution import ExecutionMode
from .nodes.registry import NodeRegistry
from .retry import RetryPolicy, RetryStrategy
from .state import GraphState

T = TypeVar("T")

class LogLevel(str, Enum):
    """Log levels for graph execution"""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

class ResourceLimits(BaseModel):
    """Resource limits for graph execution"""

    max_nodes: Optional[int] = None
    max_edges: Optional[int] = None
    max_memory_mb: Optional[int] = None
    max_execution_time_seconds: Optional[int] = None

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

class GraphConfig(BaseModel):
    """Configuration for graph execution"""

    execution_mode: ExecutionMode = Field(default=ExecutionMode.SEQUENTIAL)
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)
    log_level: LogLevel = Field(default=LogLevel.INFO)
    debug_mode: bool = Field(default=False)
    enable_performance_tracking: bool = Field(default=False)
    checkpoint_interval_seconds: Optional[int] = None

    # Retry configuration
    node_retry_policy: RetryPolicy = Field(
        default_factory=lambda: RetryPolicy(
            max_retries=3,
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay=1.0,
            max_delay=30.0
        )
    )
    state_retry_policy: RetryPolicy = Field(
        default_factory=lambda: RetryPolicy(
            max_retries=3,
            strategy=RetryStrategy.LINEAR,
            base_delay=0.5,
            max_delay=5.0
        )
    )
    resource_retry_policy: RetryPolicy = Field(
        default_factory=lambda: RetryPolicy(
            max_retries=2,
            strategy=RetryStrategy.LINEAR,
            base_delay=2.0,
            max_delay=10.0
        )
    )

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

class GraphMetadata(BaseModel):
    """Metadata for graph"""

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    version: int = 0
    name: str = Field(default="")
    description: str = Field(default="")

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

class Graph:
    """Main graph class for managing nodes, edges, and execution"""

    def __init__(
        self,
        name: str = "",
        description: str = "",
        config: Optional[GraphConfig] = None
    ):
        self._metadata = GraphMetadata(name=name, description=description)
        self._config = config or GraphConfig()
        self._state = GraphState()
        self._node_registry = NodeRegistry(self._state)
        self._edge_registry = EdgeRegistry(self._state, self._node_registry)
        self._edge_validator = EdgeValidator()

        # Set up logging
        self._logger = logging.getLogger(f"graph.{self._metadata.name}")
        self._setup_logging()

    async def execute(self, **kwargs) -> None:
        """Execute the graph by calling process()."""
        await self.process()

    async def process(self) -> None:
        """Process the graph. To be implemented by subclasses."""
        pass

    def _setup_logging(self) -> None:
        """Configure logging based on settings"""
        level_map = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR
        }
        self._logger.setLevel(level_map[self._config.log_level])

        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

    @property
    def config(self) -> GraphConfig:
        """Get graph configuration"""
        return self._config

    @config.setter
    def config(self, value: GraphConfig) -> None:
        """Set graph configuration"""
        self._config = value
        self._setup_logging()
        self._update_metadata()

    def _check_resource_limits(self, check_nodes: bool = True, check_edges: bool = True) -> None:
        """Check if resource limits are exceeded

        Args:
        ----
            check_nodes: Whether to check node limits
            check_edges: Whether to check edge limits

        """
        limits = self._config.resource_limits

        if check_nodes and limits.max_nodes and len(self._node_registry._nodes) >= limits.max_nodes:
            raise ValueError(f"Maximum number of nodes ({limits.max_nodes}) exceeded")

        if check_edges and limits.max_edges and len(self._edge_registry._edges) >= limits.max_edges:
            raise ValueError(f"Maximum number of edges ({limits.max_edges}) exceeded")

    def add_node(
        self,
        node_type: Union[str, Type[NodeBase]],
        node_id: Optional[str] = None,
        **kwargs
    ) -> NodeBase:
        """Add a node to the graph

        Args:
        ----
            node_type: Node type name or class
            node_id: Optional node ID
            **kwargs: Additional node parameters

        Returns:
        -------
            Created node instance

        """
        self._check_resource_limits(check_nodes=True, check_edges=False)

        try:
            if isinstance(node_type, str):
                node = self._node_registry.create_node(node_type, node_id, **kwargs)
            else:
                # Register type if not already registered
                type_name = node_type.__name__
                if type_name not in self._node_registry._node_types:
                    self._node_registry.register_node_type(type_name, node_type)
                node = self._node_registry.create_node(type_name, node_id, **kwargs)

            self._update_metadata()
            self._logger.info(f"Added node {node.node_id} of type {type(node).__name__}")
            return node

        except Exception as e:
            self._logger.error(f"Failed to add node: {str(e)}")
            raise

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the graph

        Args:
        ----
            node_id: ID of node to remove

        """
        try:
            # Remove connected edges first
            for edge_id in list(self._edge_registry._edges.keys()):
                edge = self._edge_registry._edges[edge_id]
                if edge.source_node.node_id == node_id or edge.target_node.node_id == node_id:
                    self._edge_registry.delete_edge(edge_id)

            # Remove node
            if node_id in self._node_registry._nodes:
                del self._node_registry._nodes[node_id]
                del self._node_registry._dependencies[node_id]
                del self._node_registry._reverse_dependencies[node_id]
                self._update_metadata()
                self._logger.info(f"Removed node {node_id}")

        except Exception as e:
            self._logger.error(f"Failed to remove node {node_id}: {str(e)}")
            raise

    def add_edge(
        self,
        source_node: Union[str, NodeBase],
        target_node: Union[str, NodeBase],
        edge_type: Union[str, Type[EdgeBase]],
        source_channel: Optional[str] = None,
        target_channel: Optional[str] = None,
        edge_id: Optional[str] = None,
        **kwargs
    ) -> EdgeBase:
        """Add an edge between nodes

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
            Created edge instance

        """
        self._check_resource_limits(check_nodes=False, check_edges=True)

        try:
            # Get node IDs
            source_id = source_node if isinstance(source_node, str) else source_node.node_id
            target_id = target_node if isinstance(target_node, str) else target_node.node_id

            # Handle edge type
            if isinstance(edge_type, str):
                type_name = edge_type
            else:
                type_name = edge_type.__name__
                if type_name not in self._edge_registry._edge_types:
                    self._edge_registry.register_edge_type(type_name, edge_type)

            # Create edge
            edge = self._edge_registry.create_edge(
                type_name=type_name,
                source_node_id=source_id,
                target_node_id=target_id,
                source_channel=source_channel or "default",  # Use default channel if none specified
                target_channel=target_channel or "default",  # Use default channel if none specified
                edge_id=edge_id,
                **kwargs
            )

            self._update_metadata()
            self._logger.info(
                f"Added edge {edge.edge_id} from node {source_id} to {target_id}"
            )
            return edge

        except Exception as e:
            self._logger.error(f"Failed to add edge: {str(e)}")
            raise

    def remove_edge(self, edge_id: str) -> None:
        """Remove an edge from the graph

        Args:
        ----
            edge_id: ID of edge to remove

        """
        try:
            self._edge_registry.delete_edge(edge_id)
            self._update_metadata()
            self._logger.info(f"Removed edge {edge_id}")

        except Exception as e:
            self._logger.error(f"Failed to remove edge {edge_id}: {str(e)}")
            raise

    def clear(self) -> None:
        """Clear all nodes and edges from the graph"""
        try:
            self._node_registry._nodes.clear()
            self._node_registry._dependencies.clear()
            self._node_registry._reverse_dependencies.clear()
            self._edge_registry._edges.clear()
            self._edge_registry._source_edges.clear()
            self._edge_registry._target_edges.clear()
            self._edge_registry._channel_edges.clear()
            self._update_metadata()
            self._logger.info("Cleared all nodes and edges from graph")

        except Exception as e:
            self._logger.error(f"Failed to clear graph: {str(e)}")
            raise

    @property
    def metadata(self) -> GraphMetadata:
        """Get graph metadata"""
        return self._metadata

    @property
    def state(self) -> GraphState:
        """Get graph state"""
        return self._state

    @property
    def nodes(self) -> NodeRegistry:
        """Get node registry"""
        return self._node_registry

    @property
    def edges(self) -> EdgeRegistry:
        """Get edge registry"""
        return self._edge_registry

    def _update_metadata(self) -> None:
        """Update metadata after graph change"""
        self._metadata.updated_at = datetime.now()
        self._metadata.version += 1
