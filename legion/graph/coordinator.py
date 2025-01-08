from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from .channels import BarrierChannel, BroadcastChannel, Channel, LastValue, MessageChannel
from .nodes.base import NodeBase


class StepMetadata(BaseModel):
    """Metadata for a BSP step"""

    step_id: str = Field(default_factory=lambda: str(uuid4()))
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    node_count: int = 0
    completed_nodes: int = 0
    error_count: int = 0

class BSPCoordinator:
    """Coordinates Bulk Synchronous Parallel execution of graph nodes"""

    def __init__(self, timeout: Optional[float] = None):
        """Initialize BSP coordinator

        Args:
        ----
            timeout: Optional timeout in seconds for synchronization barriers

        """
        self._timeout = timeout
        self._nodes: Dict[str, NodeBase] = {}
        self._node_barriers: Dict[str, BarrierChannel] = {}
        self._node_channels: Dict[str, Dict[str, Channel]] = {}
        self._step_metadata: Optional[StepMetadata] = None
        self._global_barrier: Optional[BarrierChannel] = None
        self._error_channel: MessageChannel[Exception] = MessageChannel(Exception)
        self._status_channel: BroadcastChannel[str] = BroadcastChannel(str)
        self._result_channel: LastValue[Any] = LastValue(Any)

    def register_node(self, node: NodeBase) -> None:
        """Register a node with the coordinator

        Args:
        ----
            node: Node to register

        """
        if node.node_id in self._nodes:
            raise ValueError(f"Node {node.node_id} already registered")

        self._nodes[node.node_id] = node
        self._node_barriers[node.node_id] = BarrierChannel(1, self._timeout)
        self._node_channels[node.node_id] = {}

        # Subscribe node to status updates
        self._status_channel.subscribe(node.node_id)

    def unregister_node(self, node_id: str) -> None:
        """Unregister a node from the coordinator

        Args:
        ----
            node_id: ID of node to unregister

        """
        if node_id not in self._nodes:
            return

        # Clean up node resources
        self._nodes.pop(node_id)
        self._node_barriers.pop(node_id)
        self._node_channels.pop(node_id)
        self._status_channel.unsubscribe(node_id)

    def start_step(self) -> None:
        """Start a new BSP step"""
        if self._step_metadata is not None:
            raise RuntimeError("Step already in progress")

        # Initialize step
        self._step_metadata = StepMetadata(node_count=len(self._nodes))
        self._global_barrier = BarrierChannel(len(self._nodes), self._timeout)

        # Reset channels
        self._error_channel.clear()
        self._result_channel.clear()

        # Notify nodes
        self._status_channel.broadcast("step_started")

    def node_ready(self, node_id: str) -> bool:
        """Mark a node as ready for the current step

        Args:
        ----
            node_id: ID of ready node

        Returns:
        -------
            True if all nodes are ready, False otherwise

        """
        if not self._step_metadata or not self._global_barrier:
            raise RuntimeError("No step in progress")

        if node_id not in self._nodes:
            raise ValueError(f"Unknown node {node_id}")

        # Contribute to global barrier
        return self._global_barrier.contribute(node_id)

    def node_complete(self, node_id: str, result: Any = None, error: Optional[Exception] = None) -> None:
        """Mark a node as complete for the current step

        Args:
        ----
            node_id: ID of completed node
            result: Optional result from node execution
            error: Optional error from node execution

        """
        if not self._step_metadata:
            raise RuntimeError("No step in progress")

        if node_id not in self._nodes:
            raise ValueError(f"Unknown node {node_id}")

        # Update step metadata
        self._step_metadata.completed_nodes += 1
        if error:
            self._step_metadata.error_count += 1
            self._error_channel.push(error)

        # Store result if provided
        if result is not None:
            self._result_channel.set(result)

        # Trigger node barrier
        self._node_barriers[node_id].contribute(node_id)

    def wait_for_completion(self) -> None:
        """Wait for all nodes to complete the current step"""
        if not self._step_metadata:
            raise RuntimeError("No step in progress")

        # Wait for all node barriers
        for node_id, barrier in self._node_barriers.items():
            if not barrier.is_triggered():
                raise TimeoutError(f"Node {node_id} did not complete")

    def end_step(self) -> None:
        """End the current BSP step"""
        if not self._step_metadata:
            raise RuntimeError("No step in progress")

        # Update metadata
        self._step_metadata.completed_at = datetime.now()

        # Reset barriers
        self._global_barrier = None
        for barrier in self._node_barriers.values():
            barrier.reset()

        # Notify nodes
        self._status_channel.broadcast("step_completed")

        # Clear step metadata
        self._step_metadata = None

    @property
    def errors(self) -> List[Exception]:
        """Get errors from current step"""
        return self._error_channel.get()

    @property
    def results(self) -> Any:
        """Get results from current step"""
        return self._result_channel.get()

    @property
    def step_metadata(self) -> Optional[StepMetadata]:
        """Get metadata for current step"""
        return self._step_metadata

    def get_node_channel(self, node_id: str, channel_id: str) -> Optional[Channel]:
        """Get a channel for a specific node

        Args:
        ----
            node_id: ID of node
            channel_id: ID of channel

        Returns:
        -------
            Channel if found, None otherwise

        """
        if node_id not in self._node_channels:
            return None
        return self._node_channels[node_id].get(channel_id)

    def set_node_channel(self, node_id: str, channel_id: str, channel: Channel) -> None:
        """Set a channel for a specific node

        Args:
        ----
            node_id: ID of node
            channel_id: ID of channel
            channel: Channel to set

        """
        if node_id not in self._node_channels:
            raise ValueError(f"Unknown node {node_id}")
        self._node_channels[node_id][channel_id] = channel
