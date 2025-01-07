import pytest
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4

from legion.graph.coordinator import BSPCoordinator, StepMetadata
from legion.graph.nodes.base import NodeBase
from legion.graph.channels import Channel, LastValue
from legion.graph.state import GraphState

class TestNode(NodeBase):
    """Test node implementation"""
    
    def __init__(self, graph_state: GraphState, node_id: Optional[str] = None):
        super().__init__(graph_state)
        if node_id:
            self._metadata.node_id = node_id
        self.executed = False
        self.result = None
        self.error = None
    
    async def execute(self, input_data: Optional[Dict[str, Any]] = None) -> Any:
        self.executed = True
        if self.error:
            raise self.error
        return self.result
        
    async def _execute(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Internal execution implementation"""
        result = await self.execute(kwargs)
        return {'result': result} if result is not None else None

@pytest.fixture
def graph_state():
    return GraphState()

def test_coordinator_init():
    """Test coordinator initialization"""
    coordinator = BSPCoordinator(timeout=1.0)
    assert coordinator.step_metadata is None
    assert coordinator.errors == []
    assert coordinator.results is None

def test_node_registration(graph_state):
    """Test node registration and unregistration"""
    coordinator = BSPCoordinator()
    node = TestNode(graph_state)
    
    # Test registration
    coordinator.register_node(node)
    assert node.node_id in coordinator._nodes
    assert node.node_id in coordinator._node_barriers
    assert node.node_id in coordinator._node_channels
    
    # Test duplicate registration
    with pytest.raises(ValueError):
        coordinator.register_node(node)
    
    # Test unregistration
    coordinator.unregister_node(node.node_id)
    assert node.node_id not in coordinator._nodes
    assert node.node_id not in coordinator._node_barriers
    assert node.node_id not in coordinator._node_channels
    
    # Test unregistration of unknown node
    coordinator.unregister_node("unknown")  # Should not raise

def test_step_lifecycle(graph_state):
    """Test BSP step lifecycle"""
    coordinator = BSPCoordinator()
    node1 = TestNode(graph_state)
    node2 = TestNode(graph_state)
    
    coordinator.register_node(node1)
    coordinator.register_node(node2)
    
    # Test start step
    coordinator.start_step()
    assert coordinator.step_metadata is not None
    assert coordinator.step_metadata.node_count == 2
    assert coordinator.step_metadata.completed_nodes == 0
    
    # Test cannot start another step
    with pytest.raises(RuntimeError):
        coordinator.start_step()
    
    # Test node ready
    assert not coordinator.node_ready(node1.node_id)  # First node ready
    assert coordinator.node_ready(node2.node_id)  # All nodes ready
    
    # Test node completion
    coordinator.node_complete(node1.node_id, result="result1")
    assert coordinator.step_metadata.completed_nodes == 1
    
    coordinator.node_complete(node2.node_id, result="result2")
    assert coordinator.step_metadata.completed_nodes == 2
    
    # Test wait for completion
    coordinator.wait_for_completion()  # Should not raise
    
    # Test end step
    coordinator.end_step()
    assert coordinator.step_metadata is None

def test_error_handling(graph_state):
    """Test error handling in coordinator"""
    coordinator = BSPCoordinator()
    node = TestNode(graph_state)
    error = ValueError("Test error")
    
    coordinator.register_node(node)
    coordinator.start_step()
    
    # Test error reporting
    coordinator.node_complete(node.node_id, error=error)
    assert coordinator.step_metadata.error_count == 1
    assert len(coordinator.errors) == 1
    assert isinstance(coordinator.errors[0], ValueError)
    
    coordinator.end_step()

def test_result_aggregation(graph_state):
    """Test result aggregation"""
    coordinator = BSPCoordinator()
    node1 = TestNode(graph_state)
    node2 = TestNode(graph_state)
    
    coordinator.register_node(node1)
    coordinator.register_node(node2)
    coordinator.start_step()
    
    # Test result aggregation
    coordinator.node_complete(node1.node_id, result=1)
    coordinator.node_complete(node2.node_id, result=2)
    
    # Default aggregator uses last value
    assert coordinator.results == 2
    
    coordinator.end_step()

def test_node_channels(graph_state):
    """Test node channel management"""
    coordinator = BSPCoordinator()
    node = TestNode(graph_state)
    channel = LastValue[str](str)
    
    coordinator.register_node(node)
    
    # Test setting channel
    coordinator.set_node_channel(node.node_id, "test_channel", channel)
    assert coordinator.get_node_channel(node.node_id, "test_channel") == channel
    
    # Test getting non-existent channel
    assert coordinator.get_node_channel(node.node_id, "unknown") is None
    assert coordinator.get_node_channel("unknown", "test_channel") is None
    
    # Test setting channel for unknown node
    with pytest.raises(ValueError):
        coordinator.set_node_channel("unknown", "test_channel", channel)

def test_timeout_handling(graph_state):
    """Test timeout handling"""
    coordinator = BSPCoordinator(timeout=0.1)
    node1 = TestNode(graph_state)
    node2 = TestNode(graph_state)
    
    coordinator.register_node(node1)
    coordinator.register_node(node2)
    coordinator.start_step()
    
    # Complete only one node
    coordinator.node_complete(node1.node_id)
    
    # Wait should raise timeout
    with pytest.raises(TimeoutError):
        coordinator.wait_for_completion()
    
    coordinator.end_step()

def test_step_validation(graph_state):
    """Test step state validation"""
    coordinator = BSPCoordinator()
    node = TestNode(graph_state)
    coordinator.register_node(node)
    
    # Test operations without active step
    with pytest.raises(RuntimeError):
        coordinator.node_ready(node.node_id)
    
    with pytest.raises(RuntimeError):
        coordinator.node_complete(node.node_id)
    
    with pytest.raises(RuntimeError):
        coordinator.wait_for_completion()
    
    with pytest.raises(RuntimeError):
        coordinator.end_step()

def test_step_metadata(graph_state):
    """Test step metadata tracking"""
    coordinator = BSPCoordinator()
    node = TestNode(graph_state)
    coordinator.register_node(node)
    
    # Start step and verify metadata
    coordinator.start_step()
    metadata = coordinator.step_metadata
    assert isinstance(metadata, StepMetadata)
    assert isinstance(metadata.step_id, str)
    assert isinstance(metadata.started_at, datetime)
    assert metadata.completed_at is None
    assert metadata.node_count == 1
    assert metadata.completed_nodes == 0
    assert metadata.error_count == 0
    
    # Complete step and verify metadata updates
    coordinator.node_complete(node.node_id, error=ValueError())
    assert metadata.completed_nodes == 1
    assert metadata.error_count == 1
    
    # End step and verify metadata cleared
    coordinator.end_step()
    assert coordinator.step_metadata is None 