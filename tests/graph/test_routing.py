import pytest
from typing import Any, Dict, Optional

from legion.graph.state import GraphState
from legion.graph.nodes.base import NodeBase
from legion.graph.channels import LastValue
from legion.graph.edges.routing import (
    RoutingCondition,
    StateCondition,
    ChannelCondition,
    CustomCondition
)

class TestNode(NodeBase):
    """Test node implementation"""
    def __init__(self, graph_state: GraphState):
        super().__init__(graph_state)
        self.create_output_channel('test', LastValue, type_hint=str)
        
    async def _execute(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Execute the node"""
        return {"result": "test"}

@pytest.fixture
def graph_state():
    """Create graph state fixture"""
    return GraphState()

@pytest.fixture
def test_node(graph_state):
    """Create test node fixture"""
    return TestNode(graph_state)

@pytest.mark.asyncio
async def test_state_condition(graph_state, test_node):
    """Test state-based routing condition"""
    # Set up condition
    condition = StateCondition(
        graph_state,
        'test_key',
        lambda x: x == 'test_value'
    )
    
    # Test false case
    graph_state.update_global_state({'test_key': 'wrong_value'})
    assert not await condition.evaluate(test_node)
    
    # Test true case
    graph_state.update_global_state({'test_key': 'test_value'})
    assert await condition.evaluate(test_node)
    
    # Test missing key
    graph_state.clear()
    assert not await condition.evaluate(test_node)

@pytest.mark.asyncio
async def test_channel_condition(graph_state, test_node):
    """Test channel-based routing condition"""
    # Set up condition
    condition = ChannelCondition(
        graph_state,
        'test',
        lambda x: x == 'test_value'
    )
    
    # Test false case
    test_node.get_output_channel('test').set('wrong_value')
    assert not await condition.evaluate(test_node)
    
    # Test true case
    test_node.get_output_channel('test').set('test_value')
    assert await condition.evaluate(test_node)
    
    # Test missing channel
    condition = ChannelCondition(
        graph_state,
        'non_existent',
        lambda x: True
    )
    assert not await condition.evaluate(test_node)

@pytest.mark.asyncio
async def test_custom_condition(graph_state, test_node):
    """Test custom routing condition"""
    # Set up condition
    def custom_evaluator(node: NodeBase, state: GraphState, kwargs: Dict[str, Any]) -> bool:
        state_data = state.get_global_state()
        return (
            state_data.get('test_key') == 'test_value' and
            kwargs.get('extra') == 'test_extra'
        )
    
    condition = CustomCondition(graph_state, custom_evaluator)
    
    # Test false cases
    graph_state.update_global_state({'test_key': 'wrong_value'})
    assert not await condition.evaluate(test_node, extra='test_extra')
    
    graph_state.update_global_state({'test_key': 'test_value'})
    assert not await condition.evaluate(test_node, extra='wrong_extra')
    
    # Test true case
    assert await condition.evaluate(test_node, extra='test_extra')

@pytest.mark.asyncio
async def test_condition_checkpoint_restore(graph_state, test_node):
    """Test condition checkpoint and restore"""
    # Create and configure condition
    condition = StateCondition(
        graph_state,
        'test_key',
        lambda x: x == 'test_value'
    )
    
    # Create checkpoint
    checkpoint = condition.checkpoint()
    assert checkpoint['metadata']['condition_type'] == 'StateCondition'
    assert checkpoint['state_key'] == 'test_key'
    
    # Create new condition and restore
    new_condition = StateCondition(
        graph_state,
        'temp_key',  # Different key
        lambda x: x == 'test_value'
    )
    new_condition.restore(checkpoint)
    
    # Verify metadata restored
    assert new_condition.metadata.condition_type == 'StateCondition'
    
    # Test condition still works
    graph_state.update_global_state({'test_key': 'test_value'})
    assert await new_condition.evaluate(test_node) 