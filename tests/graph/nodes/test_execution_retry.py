"""Tests for execution manager retry functionality."""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from legion.exceptions import FatalError, NodeError, NonRetryableError, ResourceError, StateError
from legion.graph.nodes.base import NodeBase, NodeStatus
from legion.graph.nodes.execution import ExecutionHook, ExecutionManager
from legion.graph.nodes.registry import NodeRegistry
from legion.graph.retry import RetryPolicy, RetryStrategy
from legion.graph.state import GraphState


# Test fixtures
@pytest.fixture
def graph_state():
    """Create graph state for testing"""
    return GraphState()

@pytest.fixture
def node_registry():
    """Create node registry for testing"""
    return NodeRegistry(GraphState())

@pytest.fixture
def execution_manager(graph_state, node_registry):
    """Create execution manager with retry policies"""
    return ExecutionManager(
        graph_state,
        node_registry,
        node_retry_policy=RetryPolicy(
            max_retries=2,
            strategy=RetryStrategy.IMMEDIATE
        ),
        state_retry_policy=RetryPolicy(
            max_retries=2,
            strategy=RetryStrategy.IMMEDIATE
        )
    )

class MockNode(NodeBase):
    """Mock node for testing"""

    def __init__(self, node_id: str, graph_state: GraphState):
        super().__init__(graph_state)
        self._metadata.node_id = node_id
        self.execute_mock = AsyncMock()

    async def _execute(self, **kwargs):
        return await self.execute_mock(**kwargs)

# Test node execution retries
@pytest.mark.asyncio
async def test_node_execution_retry(execution_manager, graph_state):
    """Test node execution retry logic"""
    # Create mock node that fails twice then succeeds
    node = MockNode("test_node", graph_state)
    node.execute_mock.side_effect = [
        NodeError("First failure", "test_node"),
        NodeError("Second failure", "test_node"),
        {"result": "success"}
    ]

    # Add node to registry
    execution_manager._registry.register_node(node)

    # Execute node
    await execution_manager.execute_node("test_node")

    # Verify execution attempts
    assert node.execute_mock.call_count == 3
    assert node.status == NodeStatus.COMPLETED

@pytest.mark.asyncio
async def test_node_execution_fatal(execution_manager, graph_state):
    """Test node execution with fatal error"""
    # Create mock node that fails with non-retryable error
    node = MockNode("test_node", graph_state)
    node.execute_mock.side_effect = NonRetryableError("Fatal error")

    # Add node to registry
    execution_manager._registry.register_node(node)

    # Execute node and verify error
    with pytest.raises(NonRetryableError):
        await execution_manager.execute_node("test_node")

    # Verify single execution attempt
    assert node.execute_mock.call_count == 1
    assert node.status == NodeStatus.FAILED

@pytest.mark.asyncio
async def test_node_execution_max_retries(execution_manager, graph_state):
    """Test node execution with max retries exceeded"""
    # Create mock node that always fails
    node = MockNode("test_node", graph_state)
    node.execute_mock.side_effect = NodeError("Persistent failure", "test_node")

    # Add node to registry
    execution_manager._registry.register_node(node)

    # Execute node and verify error
    with pytest.raises(FatalError) as exc_info:
        await execution_manager.execute_node("test_node")

    assert "Max retries (2) exceeded" in str(exc_info.value)
    assert node.execute_mock.call_count == 3
    assert node.status == NodeStatus.FAILED

# Test execution hooks with retries
@pytest.mark.asyncio
async def test_execution_hooks_with_retry(execution_manager, graph_state):
    """Test execution hooks during retry attempts"""
    # Create mock hooks
    before_hook = AsyncMock()
    after_hook = AsyncMock()
    error_hook = AsyncMock()

    execution_manager.add_hook(ExecutionHook(
        before=before_hook,
        after=after_hook,
        on_error=error_hook
    ))

    # Create mock node that fails once then succeeds
    node = MockNode("test_node", graph_state)
    node.execute_mock.side_effect = [
        NodeError("First failure", "test_node"),
        {"result": "success"}
    ]

    # Add node to registry
    execution_manager._registry.register_node(node)

    # Execute node
    await execution_manager.execute_node("test_node")

    # Verify hook calls
    assert before_hook.call_count == 2  # Called before each attempt
    assert after_hook.call_count == 1   # Called after success
    assert error_hook.call_count == 1   # Called on error

# Test parallel node execution with retries
@pytest.mark.asyncio
async def test_parallel_node_execution(execution_manager, graph_state):
    """Test parallel node execution with retries"""
    # Create mock nodes
    node1 = MockNode("node1", graph_state)
    node1.execute_mock.side_effect = [
        NodeError("Node1 failure", "node1"),
        {"result": "success1"}
    ]

    node2 = MockNode("node2", graph_state)
    node2.execute_mock.side_effect = [
        NodeError("Node2 failure", "node2"),
        NodeError("Node2 failure", "node2"),
        {"result": "success2"}
    ]

    # Add nodes to registry
    execution_manager._registry.register_node(node1)
    execution_manager._registry.register_node(node2)

    # Execute nodes concurrently
    await asyncio.gather(
        execution_manager.execute_node("node1"),
        execution_manager.execute_node("node2")
    )

    # Verify execution attempts
    assert node1.execute_mock.call_count == 2
    assert node2.execute_mock.call_count == 3
    assert node1.status == NodeStatus.COMPLETED
    assert node2.status == NodeStatus.COMPLETED

# Test state operation retries
@pytest.mark.asyncio
async def test_state_operation_retry(execution_manager):
    """Test retry logic for state operations"""
    # Mock registry methods to fail then succeed
    async def mock_get_order():
        if mock_get_order.calls == 0:
            mock_get_order.calls += 1
            raise StateError("First failure")
        return ["node1", "node2"]
    mock_get_order.calls = 0

    with patch.object(
        execution_manager._registry,
        "get_execution_order",
        mock_get_order
    ):
        # Get execution order
        order = await execution_manager._retry_handler.execute_with_retry(
            "get_execution_order",
            mock_get_order,
            execution_manager._state_retry_policy
        )

        assert order == ["node1", "node2"]

@pytest.mark.asyncio
async def test_resource_limit_retry(execution_manager, graph_state):
    """Test retry logic for resource limit errors"""
    # Create mock node that hits resource limits then succeeds
    node = MockNode("test_node", graph_state)
    node.execute_mock.side_effect = [
        ResourceError("Memory limit"),
        {"result": "success"}
    ]

    # Add node to registry
    execution_manager._registry.register_node(node)

    # Execute node
    await execution_manager.execute_node("test_node")

    # Verify execution attempts
    assert node.execute_mock.call_count == 2
    assert node.status == NodeStatus.COMPLETED
