import pytest
from datetime import datetime
from typing import Dict, Any, Optional

from legion.graph.state import GraphState
from legion.graph.nodes.base import NodeBase, NodeStatus
from legion.graph.nodes.registry import NodeRegistry, NodeRegistryMetadata

class TestNodeA(NodeBase):
    """Test node implementation A"""
    
    async def _execute(self, **kwargs) -> Optional[Dict[str, Any]]:
        return {"output": "A"}

class TestNodeB(NodeBase):
    """Test node implementation B"""
    
    async def _execute(self, **kwargs) -> Optional[Dict[str, Any]]:
        return {"output": "B"}

@pytest.fixture
def graph_state():
    """Fixture for graph state"""
    return GraphState()

@pytest.fixture
def registry(graph_state):
    """Fixture for node registry"""
    registry = NodeRegistry(graph_state)
    registry.register_node_type("node_a", TestNodeA)
    registry.register_node_type("node_b", TestNodeB)
    return registry

@pytest.fixture
def node_a(registry):
    """Fixture for test node A"""
    return registry.create_node("node_a")

@pytest.fixture
def node_b(registry):
    """Fixture for test node B"""
    return registry.create_node("node_b")

@pytest.fixture
def node_c(registry):
    """Fixture for test node C (another instance of A)"""
    return registry.create_node("node_a")

def test_registry_metadata(registry):
    """Test registry metadata functionality"""
    metadata = registry.metadata
    
    assert isinstance(metadata, NodeRegistryMetadata)
    assert isinstance(metadata.created_at, datetime)
    assert metadata.version == 2  # Two registrations

def test_node_type_registration(registry):
    """Test node type registration"""
    # Test duplicate registration
    with pytest.raises(ValueError):
        registry.register_node_type("node_a", TestNodeA)
    
    # Test unknown type
    with pytest.raises(ValueError):
        registry.create_node("unknown_type")

def test_node_creation(registry):
    """Test node creation"""
    # Create nodes
    node_a = registry.create_node("node_a")
    node_b = registry.create_node("node_b", node_id="custom_id")
    
    assert isinstance(node_a, TestNodeA)
    assert isinstance(node_b, TestNodeB)
    assert node_b.node_id == "custom_id"
    
    # Test duplicate ID
    with pytest.raises(ValueError):
        registry.create_node("node_a", node_id=node_a.node_id)

def test_node_management(registry, node_a, node_b):
    """Test node management"""
    # Test retrieval
    assert registry.get_node(node_a.node_id) == node_a
    assert registry.get_node(node_b.node_id) == node_b
    assert registry.get_node("nonexistent") is None
    
    # Test deletion
    registry.delete_node(node_a.node_id)
    assert registry.get_node(node_a.node_id) is None
    
    # Test deleting nonexistent node
    registry.delete_node("nonexistent")  # Should not raise

def test_dependency_management(registry, node_a, node_b, node_c):
    """Test dependency management"""
    # Add dependencies
    registry.add_dependency(node_a.node_id, node_b.node_id)
    registry.add_dependency(node_b.node_id, node_c.node_id)
    
    # Test dependency retrieval
    assert registry.get_dependencies(node_a.node_id) == {node_b.node_id}
    assert registry.get_dependencies(node_b.node_id) == {node_c.node_id}
    assert registry.get_dependencies(node_c.node_id) == set()
    
    # Test dependent retrieval
    assert registry.get_dependents(node_a.node_id) == set()
    assert registry.get_dependents(node_b.node_id) == {node_a.node_id}
    assert registry.get_dependents(node_c.node_id) == {node_b.node_id}
    
    # Test cycle detection
    with pytest.raises(ValueError):
        registry.add_dependency(node_c.node_id, node_a.node_id)
    
    # Test dependency removal
    registry.remove_dependency(node_a.node_id, node_b.node_id)
    assert registry.get_dependencies(node_a.node_id) == set()
    assert registry.get_dependents(node_b.node_id) == set()

def test_execution_order(registry, node_a, node_b, node_c):
    """Test execution order calculation"""
    # Add dependencies
    registry.add_dependency(node_a.node_id, node_b.node_id)
    registry.add_dependency(node_b.node_id, node_c.node_id)
    
    # Test execution order
    order = registry.get_execution_order()
    assert len(order) == 3
    assert order.index(node_c.node_id) < order.index(node_b.node_id)
    assert order.index(node_b.node_id) < order.index(node_a.node_id)

def test_node_status(registry, node_a, node_b):
    """Test node status tracking"""
    # Test initial status
    status = registry.get_node_status()
    assert status[node_a.node_id] == NodeStatus.IDLE
    assert status[node_b.node_id] == NodeStatus.IDLE
    
    # Update status
    node_a._update_status(NodeStatus.RUNNING)
    node_b._update_status(NodeStatus.COMPLETED)
    
    # Test updated status
    status = registry.get_node_status()
    assert status[node_a.node_id] == NodeStatus.RUNNING
    assert status[node_b.node_id] == NodeStatus.COMPLETED

def test_registry_clearing(registry, node_a, node_b):
    """Test registry clearing"""
    # Add dependencies
    registry.add_dependency(node_a.node_id, node_b.node_id)
    
    # Clear registry
    registry.clear()
    
    # Verify cleared state
    assert registry.get_node(node_a.node_id) is None
    assert registry.get_node(node_b.node_id) is None
    assert registry.get_dependencies(node_a.node_id) == set()
    assert registry.get_dependents(node_b.node_id) == set()
    assert registry.metadata.version == 0

def test_checkpointing(registry, node_a, node_b):
    """Test registry checkpointing"""
    # Add dependencies
    registry.add_dependency(node_a.node_id, node_b.node_id)
    
    # Create checkpoint
    checkpoint = registry.checkpoint()
    
    # Create new registry and restore
    new_registry = NodeRegistry(registry._graph_state)
    new_registry.register_node_type("node_a", TestNodeA)
    new_registry.register_node_type("node_b", TestNodeB)
    new_registry.restore(checkpoint)
    
    # Verify nodes were restored
    assert new_registry.get_node(node_a.node_id) is not None
    assert new_registry.get_node(node_b.node_id) is not None
    
    # Verify dependencies were restored
    assert new_registry.get_dependencies(node_a.node_id) == {node_b.node_id}
    assert new_registry.get_dependents(node_b.node_id) == {node_a.node_id}

if __name__ == "__main__":
    pytest.main([__file__])