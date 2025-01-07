"""Tests for execution manager."""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from legion.exceptions import (
    NodeError, StateError, ResourceError,
    NonRetryableError, FatalError
)
from legion.graph.nodes.execution import (
    ExecutionMode, ExecutionManager, ExecutionHook
)
from legion.graph.retry import RetryPolicy, RetryStrategy
from legion.graph.state import GraphState
from legion.graph.nodes.registry import NodeRegistry
from legion.graph.nodes.base import NodeBase, NodeStatus

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
    """Create execution manager for testing"""
    return ExecutionManager(
        graph_state,
        node_registry,
        node_retry_policy=RetryPolicy(
            max_retries=2,
            strategy=RetryStrategy.IMMEDIATE
        )
    )

class TestNode(NodeBase):
    """Test node for execution tests"""
    def __init__(self, graph_state: GraphState):
        super().__init__(graph_state)
        self.execute_mock = AsyncMock()
        
    async def _execute(self, **kwargs):
        return await self.execute_mock(**kwargs)

def test_execution_metadata(execution_manager):
    """Test execution metadata"""
    assert execution_manager.metadata.mode == ExecutionMode.SEQUENTIAL
    assert not execution_manager.metadata.is_running
    assert execution_manager.metadata.current_node is None
    assert execution_manager.metadata.error is None

@pytest.mark.asyncio
async def test_single_node_execution(execution_manager, graph_state):
    """Test single node execution"""
    # Create test node
    node = TestNode(graph_state)
    node.execute_mock.return_value = {"result": "success"}
    
    # Add node to registry
    execution_manager._registry.register_node(node)
    
    # Execute node
    await execution_manager.execute_node(node.node_id)
    
    # Verify execution
    node.execute_mock.assert_called_once()
    assert node.status == NodeStatus.COMPLETED

@pytest.mark.asyncio
async def test_sequential_execution(execution_manager, graph_state):
    """Test sequential node execution"""
    # Create test nodes
    node1 = TestNode(graph_state)
    node1.execute_mock.return_value = {"result": "success1"}
    
    node2 = TestNode(graph_state)
    node2.execute_mock.return_value = {"result": "success2"}
    
    # Add nodes to registry
    execution_manager._registry.register_node(node1)
    execution_manager._registry.register_node(node2)
    
    # Execute nodes
    await execution_manager.execute_all()
    
    # Verify execution order
    node1.execute_mock.assert_called_once()
    node2.execute_mock.assert_called_once()
    assert node1.status == NodeStatus.COMPLETED
    assert node2.status == NodeStatus.COMPLETED

@pytest.mark.asyncio
async def test_execution_hooks(execution_manager, graph_state):
    """Test execution hooks"""
    # Create mock hooks
    before_hook = AsyncMock()
    after_hook = AsyncMock()
    error_hook = AsyncMock()
    
    execution_manager.add_hook(ExecutionHook(
        before=before_hook,
        after=after_hook,
        on_error=error_hook
    ))
    
    # Create test node
    node = TestNode(graph_state)
    node.execute_mock.return_value = {"result": "success"}
    
    # Add node to registry
    execution_manager._registry.register_node(node)
    
    # Execute node
    await execution_manager.execute_node(node.node_id)
    
    # Verify hook calls
    before_hook.assert_called_once()
    after_hook.assert_called_once()
    error_hook.assert_not_called()

@pytest.mark.asyncio
async def test_execution_error_handling(execution_manager, graph_state):
    """Test execution error handling"""
    # Create test node that raises an error
    node = TestNode(graph_state)
    node.execute_mock.side_effect = [
        NodeError("Test error", node.node_id),  # First attempt fails
        NodeError("Test error", node.node_id),  # Second attempt fails
        NodeError("Test error", node.node_id)   # Third attempt fails
    ]
    
    # Add node to registry
    execution_manager._registry.register_node(node)
    
    # Execute node and verify error
    with pytest.raises(FatalError) as exc_info:
        await execution_manager.execute_node(node.node_id)
    
    assert "Max retries (2) exceeded" in str(exc_info.value)
    assert node.execute_mock.call_count == 3
    assert node.status == NodeStatus.FAILED

@pytest.mark.asyncio
async def test_ready_nodes(execution_manager, graph_state):
    """Test getting ready nodes"""
    # Create test nodes
    node1 = TestNode(graph_state)
    node2 = TestNode(graph_state)
    
    # Add nodes to registry
    execution_manager._registry.register_node(node1)
    execution_manager._registry.register_node(node2)
    
    # Add dependency
    execution_manager._registry.add_dependency(node2.node_id, node1.node_id)
    
    # Get ready nodes
    ready = await execution_manager.get_ready_nodes()
    
    # Only node1 should be ready (no dependencies)
    assert len(ready) == 1
    assert node1.node_id in ready

@pytest.mark.asyncio
async def test_checkpointing(execution_manager):
    """Test execution manager checkpointing"""
    # Create checkpoint
    checkpoint = execution_manager.checkpoint()
    
    # Verify checkpoint data
    assert 'metadata' in checkpoint
    assert checkpoint['metadata']['mode'] == ExecutionMode.SEQUENTIAL
    assert not checkpoint['metadata']['is_running']
    
    # Modify state
    execution_manager._metadata.is_running = True
    execution_manager._metadata.current_node = "test_node"
    
    # Restore checkpoint
    execution_manager.restore(checkpoint)
    
    # Verify restored state
    assert not execution_manager.metadata.is_running
    assert execution_manager.metadata.current_node is None

if __name__ == "__main__":
    pytest.main([__file__])