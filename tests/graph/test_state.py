import pytest
from datetime import datetime
from typing import Dict, List, Optional
from legion.graph.state import GraphState, GraphStateMetadata
from legion.graph.channels import LastValue, ValueSequence, SharedState

def test_graph_state_metadata():
    """Test graph state metadata functionality"""
    state = GraphState()
    metadata = state.metadata
    
    assert isinstance(metadata, GraphStateMetadata)
    assert isinstance(metadata.created_at, datetime)
    assert metadata.version == 0
    assert isinstance(metadata.graph_id, str)

def test_channel_creation():
    """Test channel creation and management"""
    state = GraphState()
    
    # Create different types of channels
    last_value = state.create_channel(LastValue, "last_value", type_hint=str)
    value_sequence = state.create_channel(ValueSequence, "value_sequence", type_hint=int, max_size=3)
    shared_state = state.create_channel(SharedState, "shared_state")
    
    # Verify channels were created
    assert state.get_channel("last_value") == last_value
    assert state.get_channel("value_sequence") == value_sequence
    assert state.get_channel("shared_state") == shared_state
    
    # Verify channel listing
    channels = state.list_channels()
    assert set(channels) == {"last_value", "value_sequence", "shared_state"}
    
    # Test duplicate channel creation
    with pytest.raises(ValueError):
        state.create_channel(LastValue, "last_value")
    
    # Test channel deletion
    state.delete_channel("last_value")
    assert state.get_channel("last_value") is None
    assert len(state.list_channels()) == 2

def test_global_state():
    """Test global state management"""
    state = GraphState()
    
    # Test initial state
    assert state.get_global_state() == {}
    
    # Test state setting
    initial_state = {'key': 'value'}
    state.set_global_state(initial_state)
    assert state.get_global_state() == initial_state
    
    # Test state update
    state.update_global_state({'new_key': 'new_value'})
    assert state.get_global_state() == {'key': 'value', 'new_key': 'new_value'}

def test_checkpointing():
    """Test state checkpointing and restoration"""
    state = GraphState()
    
    # Setup some state
    state.create_channel(LastValue, "last_value", type_hint=str).set("test")
    state.create_channel(ValueSequence, "value_sequence", type_hint=int).append(1)
    state.set_global_state({'key': 'value'})
    
    # Create checkpoint
    checkpoint = state.checkpoint()
    
    # Create new state and restore
    new_state = GraphState()
    new_state.restore(checkpoint)
    
    # Verify channels were restored
    assert new_state.get_channel("last_value").get() == "test"
    assert new_state.get_channel("value_sequence").get() == [1]
    assert new_state.get_global_state() == {'key': 'value'}
    
    # Verify metadata was restored
    assert new_state.metadata.version == state.metadata.version
    assert new_state.metadata.graph_id == state.metadata.graph_id

def test_state_clearing():
    """Test state clearing functionality"""
    state = GraphState()
    
    # Setup some state
    state.create_channel(LastValue, "test")
    state.set_global_state({'key': 'value'})
    
    # Clear state
    state.clear()
    
    # Verify state was cleared
    assert len(state.list_channels()) == 0
    assert state.get_global_state() == {}
    assert state.metadata.version == 0

def test_state_merging():
    """Test state merging functionality"""
    state1 = GraphState()
    state2 = GraphState()
    
    # Setup state1
    state1.create_channel(LastValue, "unique1", type_hint=str).set("test1")
    state1.set_global_state({'key1': 'value1'})
    
    # Setup state2
    state2.create_channel(LastValue, "unique2", type_hint=str).set("test2")
    state2.set_global_state({'key2': 'value2'})
    
    # Merge states
    state1.merge(state2)
    
    # Verify merged channels
    assert state1.get_channel("unique1").get() == "test1"
    assert state1.get_channel("unique2").get() == "test2"
    
    # Verify merged global state
    global_state = state1.get_global_state()
    assert global_state['key1'] == 'value1'
    assert global_state['key2'] == 'value2'

def test_version_tracking():
    """Test version tracking in metadata"""
    state = GraphState()
    initial_version = state.metadata.version
    
    # Version should increment on channel creation
    state.create_channel(LastValue, "test")
    assert state.metadata.version == initial_version + 1
    
    # Version should increment on channel deletion
    state.delete_channel("test")
    assert state.metadata.version == initial_version + 2
    
    # Version should increment on global state changes
    state.set_global_state({'key': 'value'})
    assert state.metadata.version == initial_version + 3
    
    state.update_global_state({'new_key': 'new_value'})
    assert state.metadata.version == initial_version + 4
