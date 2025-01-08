from copy import deepcopy
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Type

from ..edges.base import EdgeBase
from ..graph import Graph, GraphConfig
from ..state import GraphState
from .base import NodeBase


class SubgraphStatus(str, Enum):
    """Status of subgraph execution"""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

class GraphNodeAdapter(NodeBase):
    """Adapter that allows using a graph as a node."""

    def __init__(
        self,
        graph_class: Type[Graph],
        graph_state: GraphState,
        node_id: Optional[str] = None,
        config: Optional[GraphConfig] = None,
        **kwargs
    ):
        """Initialize graph node adapter.

        Args:
        ----
            graph_class: Graph class to adapt
            graph_state: Parent graph state
            node_id: Optional node ID
            config: Optional subgraph configuration
            **kwargs: Additional configuration

        """
        super().__init__(graph_state)

        # Create isolated graph instance with its own state
        self._graph_state = GraphState()

        # Merge configurations
        parent_config = graph_state.graph.config if hasattr(graph_state, "graph") else None
        merged_config = self._merge_configs(parent_config, config)

        # Initialize graph with merged config
        self._graph = graph_class(config=merged_config)

        # Map channels and edges
        self._map_channels()
        self._map_edges()

        # Track exposed edges
        self._exposed_edges: Dict[str, EdgeBase] = {}

        # Execution state
        self._subgraph_status = SubgraphStatus.IDLE
        self._execution_start: Optional[datetime] = None
        self._last_checkpoint: Optional[Dict[str, Any]] = None
        self._execution_stats: Dict[str, Any] = {
            "total_executions": 0,
            "total_duration": 0.0,
            "avg_duration": 0.0,
            "last_execution": None,
            "error_count": 0
        }

        # Update metadata
        self._metadata.node_type = graph_class.__name__
        if node_id:
            self._metadata.node_id = node_id

    def _merge_configs(
        self,
        parent_config: Optional[GraphConfig],
        subgraph_config: Optional[GraphConfig]
    ) -> GraphConfig:
        """Merge parent and subgraph configurations.

        Args:
        ----
            parent_config: Parent graph configuration
            subgraph_config: Subgraph configuration

        Returns:
        -------
            Merged configuration

        """
        if not parent_config:
            return subgraph_config or GraphConfig()

        if not subgraph_config:
            # Create copy of parent config with reduced resource limits
            config = deepcopy(parent_config)
            if config.resource_limits:
                # Allocate portion of parent resources to subgraph
                for limit_name, limit_value in config.resource_limits.model_dump().items():
                    if limit_value is not None:
                        setattr(config.resource_limits, limit_name, limit_value // 2)
            return config

        # Merge configurations with subgraph taking precedence
        merged = deepcopy(parent_config)
        subgraph_dict = subgraph_config.model_dump()

        for key, value in subgraph_dict.items():
            if value is not None:  # Only override if explicitly set
                setattr(merged, key, value)

        return merged

    def _map_channels(self) -> None:
        """Map graph channels to node channels."""
        # Map channels from graph state
        for name, channel in self._graph._state._channels.items():
            # Skip internal channels (those with node IDs in their names)
            if "_in_" in name or "_out_" in name:
                continue

            # Create corresponding node channel
            if name.startswith("input_"):
                self.create_input_channel(
                    name,
                    type(channel),
                    channel._type_hint
                )
            else:
                self.create_output_channel(
                    name,
                    type(channel),
                    channel._type_hint
                )

    def _map_edges(self) -> None:
        """Map internal edges to external representation."""
        # Get all edges that connect to exposed channels
        exposed_edges: Set[str] = set()

        for edge in self._graph.edges._edges.values():
            # Check if edge connects to an exposed channel
            source_exposed = edge.source_channel in self._input_channels
            target_exposed = edge.target_channel in self._output_channels

            if source_exposed or target_exposed:
                exposed_edges.add(edge.edge_id)

        # Map exposed edges
        for edge_id in exposed_edges:
            self.expose_edge(edge_id)

    def expose_edge(self, edge_id: str) -> None:
        """Expose an internal edge to the parent graph.

        Args:
        ----
            edge_id: ID of edge to expose

        """
        if edge_id not in self._graph.edges._edges:
            raise ValueError(f"Edge {edge_id} not found in subgraph")

        edge = self._graph.edges._edges[edge_id]
        self._exposed_edges[edge_id] = edge

    def get_exposed_edges(self) -> List[EdgeBase]:
        """Get list of exposed edges.

        Returns
        -------
            List of exposed edge instances

        """
        return list(self._exposed_edges.values())

    def _check_resource_limits(self) -> None:
        """Check if resource limits are exceeded."""
        limits = self._graph.config.resource_limits
        if not limits:
            return

        # Check memory usage
        if limits.max_memory_mb:
            # TODO: Implement memory usage tracking
            pass

        # Check execution time
        if (
            limits.max_execution_time_seconds is not None and
            self._execution_start and
            (datetime.now() - self._execution_start).total_seconds() > limits.max_execution_time_seconds
        ):
            raise ResourceLimitExceeded(
                f"Maximum execution time ({limits.max_execution_time_seconds}s) exceeded"
            )

    async def pause(self) -> None:
        """Pause subgraph execution."""
        if self._subgraph_status != SubgraphStatus.RUNNING:
            raise InvalidStateError(
                f"Cannot pause subgraph in {self._subgraph_status} state"
            )

        # Create checkpoint before pausing
        self._last_checkpoint = self.create_checkpoint()
        self._subgraph_status = SubgraphStatus.PAUSED

    async def resume(self) -> None:
        """Resume subgraph execution."""
        if self._subgraph_status != SubgraphStatus.PAUSED:
            raise InvalidStateError(
                f"Cannot resume subgraph in {self._subgraph_status} state"
            )

        if self._last_checkpoint:
            self.restore_checkpoint(self._last_checkpoint)

        self._subgraph_status = SubgraphStatus.RUNNING

    def create_checkpoint(self) -> Dict[str, Any]:
        """Create a checkpoint of current state.

        Returns
        -------
            Dictionary containing checkpoint data

        """
        return {
            "graph_state": self._graph_state.checkpoint(),
            "execution_stats": deepcopy(self._execution_stats),
            "status": self._subgraph_status,
            "metadata": self._metadata.model_dump()
        }

    def restore_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Restore from checkpoint.

        Args:
        ----
            checkpoint: Checkpoint data to restore from

        """
        self._graph_state.restore(checkpoint["graph_state"])
        self._execution_stats = deepcopy(checkpoint["execution_stats"])
        self._subgraph_status = checkpoint["status"]
        self._metadata = type(self._metadata)(**checkpoint["metadata"])

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics.

        Returns
        -------
            Dictionary containing execution statistics

        """
        return deepcopy(self._execution_stats)

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debugging information.

        Returns
        -------
            Dictionary containing debug information

        """
        return {
            "status": self._subgraph_status,
            "execution_start": self._execution_start,
            "has_checkpoint": self._last_checkpoint is not None,
            "stats": self.get_execution_stats(),
            "config": self._graph.config.model_dump(),
            "exposed_edges": len(self._exposed_edges),
            "channels": {
                "input": [name for name in self._input_channels.keys() if "_in_" not in name],
                "output": [name for name in self._output_channels.keys() if "_out_" not in name]
            }
        }

    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the graph node.

        Returns
        -------
            Dictionary mapping output channel names to their values

        """
        self._execution_start = datetime.now()
        self._subgraph_status = SubgraphStatus.RUNNING

        try:
            # Update graph state with input values
            for name, channel in self._input_channels.items():
                value = channel.get()
                self._graph._state.get_channel(name).set(value)

            # Check resource limits before processing
            self._check_resource_limits()

            # Process through graph
            await self._graph.execute()

            # Check resource limits after processing
            self._check_resource_limits()

            # Collect outputs
            outputs = {}
            for name, channel in self._output_channels.items():
                value = self._graph._state.get_channel(name).get()
                channel.set(value)
                outputs[name] = value

            # Update execution stats
            duration = (datetime.now() - self._execution_start).total_seconds()
            self._execution_stats["total_executions"] += 1
            self._execution_stats["total_duration"] += duration
            self._execution_stats["avg_duration"] = (
                self._execution_stats["total_duration"] /
                self._execution_stats["total_executions"]
            )
            self._execution_stats["last_execution"] = self._execution_start

            self._subgraph_status = SubgraphStatus.COMPLETED
            return outputs

        except Exception as e:
            self._execution_stats["error_count"] += 1
            self._subgraph_status = SubgraphStatus.FAILED

            # Transform and propagate subgraph errors
            if isinstance(e, ResourceLimitExceeded):
                raise SubgraphExecutionError(str(e)) from e
            raise SubgraphExecutionError(
                f"Error executing subgraph {self._metadata.node_type}: {str(e)}"
            ) from e

class SubgraphExecutionError(Exception):
    """Error raised when subgraph execution fails."""

    pass

class ResourceLimitExceeded(Exception):
    """Error raised when resource limits are exceeded."""

    pass

class InvalidStateError(Exception):
    """Error raised when operation is invalid for current state."""

    pass
