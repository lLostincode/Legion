import logging
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from pydantic import BaseModel, ConfigDict, Field

from ...exceptions import StateError
from ..retry import RetryHandler, RetryPolicy
from ..state import GraphState
from .base import NodeBase, NodeStatus
from .registry import NodeRegistry


class ExecutionMode(str, Enum):
    """Node execution mode"""

    SEQUENTIAL = "sequential"  # Execute nodes one at a time
    PARALLEL = "parallel"      # Execute independent nodes in parallel

class ExecutionHook:
    """Hook for node execution events"""

    def __init__(
        self,
        before: Optional[Callable[[NodeBase], Awaitable[None]]] = None,
        after: Optional[Callable[[NodeBase, Optional[Dict[str, Any]]], Awaitable[None]]] = None,
        on_error: Optional[Callable[[NodeBase, Exception], Awaitable[None]]] = None
    ):
        self.before = before
        self.after = after
        self.on_error = on_error

class ExecutionMetadata(BaseModel):
    """Metadata for execution manager"""

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    version: int = 0
    mode: ExecutionMode = Field(default=ExecutionMode.SEQUENTIAL)
    is_running: bool = False
    current_node: Optional[str] = None
    error: Optional[str] = None

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

class ExecutionManager:
    """Manages node execution"""

    def __init__(
        self,
        graph_state: GraphState,
        registry: NodeRegistry,
        mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
        node_retry_policy: Optional[RetryPolicy] = None,
        state_retry_policy: Optional[RetryPolicy] = None,
        resource_retry_policy: Optional[RetryPolicy] = None
    ):
        self._metadata = ExecutionMetadata(mode=mode)
        self._graph_state = graph_state
        self._registry = registry
        self._hooks: List[ExecutionHook] = []

        # Initialize retry handler and policies
        self._retry_handler = RetryHandler(logger=logging.getLogger(__name__))
        self._node_retry_policy = node_retry_policy or RetryPolicy()
        self._state_retry_policy = state_retry_policy or RetryPolicy()
        self._resource_retry_policy = resource_retry_policy or RetryPolicy()

    @property
    def metadata(self) -> ExecutionMetadata:
        """Get execution metadata"""
        return self._metadata

    def _update_metadata(self) -> None:
        """Update metadata after execution change"""
        self._metadata.updated_at = datetime.now()
        self._metadata.version += 1

    def add_hook(self, hook: ExecutionHook) -> None:
        """Add execution hook"""
        self._hooks.append(hook)
        self._update_metadata()

    def clear_hooks(self) -> None:
        """Clear all execution hooks"""
        self._hooks.clear()
        self._update_metadata()

    async def _execute_node(self, node: NodeBase, **kwargs) -> None:
        """Execute a single node with hooks and retry logic"""
        self._metadata.current_node = node.node_id
        self._update_metadata()

        async def execute_with_hooks():
            try:
                # Run before hooks
                for hook in self._hooks:
                    if hook.before:
                        await hook.before(node)

                # Execute node
                result = await node.execute(**kwargs)

                # Run after hooks
                for hook in self._hooks:
                    if hook.after:
                        await hook.after(node, result)

                return result

            except Exception as e:
                # Run error hooks
                for hook in self._hooks:
                    if hook.on_error:
                        await hook.on_error(node, e)
                raise

        try:
            # Execute with retry
            await self._retry_handler.execute_with_retry(
                f"node_{node.node_id}",
                execute_with_hooks,
                self._node_retry_policy
            )

        finally:
            self._metadata.current_node = None
            self._update_metadata()

    async def execute_node(self, node_id: str, **kwargs) -> None:
        """Execute a single node by ID with retry logic"""
        async def get_node():
            node = self._registry.get_node(node_id)
            if not node:
                raise StateError(f"Node '{node_id}' not found", retry_count=0, max_retries=3)
            return node

        # Get node with retry
        node = await self._retry_handler.execute_with_retry(
            f"get_node_{node_id}",
            get_node,
            self._state_retry_policy
        )

        await self._execute_node(node, **kwargs)

    async def execute_all(self, **kwargs) -> None:
        """Execute all nodes in dependency order with retry logic"""
        if self._metadata.is_running:
            raise RuntimeError("Execution already in progress")

        self._metadata.is_running = True
        self._update_metadata()

        try:
            # Get execution order with retry
            async def get_order():
                try:
                    return self._registry.get_execution_order()
                except Exception as e:
                    raise StateError(f"Failed to get execution order: {e}", retry_count=0, max_retries=3)

            order = await self._retry_handler.execute_with_retry(
                "get_execution_order",
                get_order,
                self._state_retry_policy
            )

            if self._metadata.mode == ExecutionMode.SEQUENTIAL:
                # Execute nodes sequentially
                for node_id in order:
                    node = self._registry.get_node(node_id)
                    if node:
                        await self._execute_node(node, **kwargs)
            else:
                # Execute nodes in parallel (TODO: implement parallel execution)
                raise NotImplementedError("Parallel execution not yet implemented")

        finally:
            self._metadata.is_running = False
            self._update_metadata()

    async def get_ready_nodes(self) -> Set[str]:
        """Get nodes ready for execution with retry logic"""
        async def get_ready():
            try:
                ready = set()
                status = self._registry.get_node_status()

                for node_id in self._registry.get_execution_order():
                    # Node is ready if all dependencies are completed
                    dependencies = self._registry.get_dependencies(node_id)
                    if all(
                        dep not in status or status[dep] == NodeStatus.COMPLETED
                        for dep in dependencies
                    ):
                        ready.add(node_id)

                return ready

            except Exception as e:
                raise StateError(f"Failed to get ready nodes: {e}", retry_count=0, max_retries=3)

        return await self._retry_handler.execute_with_retry(
            "get_ready_nodes",
            get_ready,
            self._state_retry_policy
        )

    def checkpoint(self) -> Dict[str, Any]:
        """Create a checkpoint of current state"""
        return {
            "metadata": self._metadata.model_dump()
        }

    def restore(self, checkpoint: Dict[str, Any]) -> None:
        """Restore from checkpoint"""
        self._metadata = ExecutionMetadata(**checkpoint["metadata"])
