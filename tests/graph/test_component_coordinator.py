import pytest
from unittest.mock import MagicMock

from legion.graph.component_coordinator import ComponentCoordinator, ComponentType, ComponentMetadata
from legion.graph.state import GraphState
from legion.graph.nodes.base import NodeBase
from legion.graph.channels import LastValue, MessageChannel

@pytest.fixture
def graph_state():
    return GraphState()

@pytest.fixture
def coordinator(graph_state):
    return ComponentCoordinator(graph_state)

@pytest.fixture
def mock_node():
    node = MagicMock(spec=NodeBase)
    node.node_id = "test_node"
    return node

def test_component_registration(coordinator, mock_node):
    """Test basic component registration"""
    # Register component
    component_id = coordinator.register_component(mock_node, ComponentType.AGENT)
    
    # Verify registration
    assert component_id is not None
    metadata = coordinator.get_component(component_id)
    assert metadata is not None
    assert metadata.component_type == ComponentType.AGENT
    assert metadata.node == mock_node
    
    # Check events
    assert f"component_registered:{component_id}" in coordinator.events

def test_component_unregistration(coordinator, mock_node):
    """Test component unregistration"""
    # Register and then unregister
    component_id = coordinator.register_component(mock_node, ComponentType.BLOCK)
    coordinator.unregister_component(component_id)
    
    # Verify unregistration
    assert coordinator.get_component(component_id) is None
    assert f"component_unregistered:{component_id}" in coordinator.events

def test_component_type_registry(coordinator, mock_node):
    """Test component type registry"""
    # Register components of different types
    agent_id = coordinator.register_component(mock_node, ComponentType.AGENT)
    block_id = coordinator.register_component(mock_node, ComponentType.BLOCK)
    
    # Check type-specific queries
    agent_components = coordinator.get_components_by_type(ComponentType.AGENT)
    block_components = coordinator.get_components_by_type(ComponentType.BLOCK)
    
    assert len(agent_components) == 1
    assert len(block_components) == 1
    assert agent_components[0].component_id == agent_id
    assert block_components[0].component_id == block_id

def test_channel_management(coordinator, mock_node):
    """Test channel management"""
    # Register component
    component_id = coordinator.register_component(mock_node, ComponentType.CHAIN)
    
    # Add channels
    value_channel = LastValue[str](str)
    message_channel = MessageChannel[str](str)
    
    coordinator.add_channel(component_id, "value", value_channel)
    coordinator.add_channel(component_id, "message", message_channel)
    
    # Verify channels
    assert coordinator.get_channel(component_id, "value") == value_channel
    assert coordinator.get_channel(component_id, "message") == message_channel
    
    # Test invalid component
    with pytest.raises(ValueError):
        coordinator.add_channel("invalid", "test", value_channel)

def test_error_reporting(coordinator, mock_node):
    """Test error reporting"""
    # Register component
    component_id = coordinator.register_component(mock_node, ComponentType.TEAM)
    
    # Report error
    error = ValueError("Test error")
    coordinator.report_error(component_id, error)
    
    # Verify error reporting
    assert error in coordinator.errors
    assert f"component_error:{component_id}" in coordinator.events
    
    # Test invalid component
    with pytest.raises(ValueError):
        coordinator.report_error("invalid", error)

def test_component_metadata_immutability(coordinator, mock_node):
    """Test that component metadata is properly encapsulated"""
    # Register component
    component_id = coordinator.register_component(mock_node, ComponentType.GRAPH)
    
    # Get metadata
    metadata = coordinator.get_component(component_id)
    
    # Verify metadata is a copy or immutable
    assert isinstance(metadata, ComponentMetadata)
    
    # Verify type registry consistency
    graph_components = coordinator.get_components_by_type(ComponentType.GRAPH)
    assert len(graph_components) == 1
    assert graph_components[0].component_id == component_id

def test_state_scoping(coordinator, mock_node):
    """Test state scoping functionality"""
    # Register parent component
    parent_id = coordinator.register_component(mock_node, ComponentType.TEAM)
    parent_scope = coordinator.get_component(parent_id).state_scope
    
    # Register child component
    child_id = coordinator.register_component(
        mock_node, 
        ComponentType.AGENT,
        parent_scope=parent_scope
    )
    
    # Set states
    parent_state = {"key": "parent_value"}
    child_state = {"key": "child_value"}
    
    coordinator.set_state(parent_id, parent_state)
    coordinator.set_state(child_id, child_state)
    
    # Verify state isolation
    assert coordinator.get_state(parent_id) == parent_state
    assert coordinator.get_state(child_id) == child_state
    
    # Verify parent state access
    assert coordinator.get_parent_state(child_id) == parent_state
    assert coordinator.get_parent_state(parent_id) is None

def test_state_cleanup(coordinator, mock_node):
    """Test state cleanup on component unregistration"""
    # Create component hierarchy
    parent_id = coordinator.register_component(mock_node, ComponentType.TEAM)
    parent_scope = coordinator.get_component(parent_id).state_scope
    
    child_id = coordinator.register_component(
        mock_node, 
        ComponentType.AGENT,
        parent_scope=parent_scope
    )
    
    # Set states
    coordinator.set_state(parent_id, {"key": "parent"})
    coordinator.set_state(child_id, {"key": "child"})
    
    # Unregister parent
    coordinator.unregister_component(parent_id)
    
    # Verify state cleanup
    assert coordinator.get_state(parent_id) is None
    assert coordinator.get_state(child_id) is None
    assert coordinator.get_parent_state(child_id) is None

def test_state_isolation(coordinator, mock_node):
    """Test state isolation between components"""
    # Create two independent components
    comp1_id = coordinator.register_component(mock_node, ComponentType.AGENT)
    comp2_id = coordinator.register_component(mock_node, ComponentType.AGENT)
    
    # Set states
    state1 = {"key": "value1"}
    state2 = {"key": "value2"}
    
    coordinator.set_state(comp1_id, state1)
    coordinator.set_state(comp2_id, state2)
    
    # Verify isolation
    assert coordinator.get_state(comp1_id) == state1
    assert coordinator.get_state(comp2_id) == state2
    
    # Modify one state
    coordinator.set_state(comp1_id, {"key": "new_value"})
    
    # Verify other state remains unchanged
    assert coordinator.get_state(comp2_id) == state2 