from typing import Any, Dict, Optional

import pytest

from legion.graph.channels import LastValue
from legion.graph.edges.conditional import ConditionalEdge, Target
from legion.graph.edges.routing import StateCondition
from legion.graph.nodes.base import NodeBase
from legion.graph.state import GraphState


class TestNode(NodeBase):
    """Test node implementation"""

    def __init__(self, graph_state: GraphState):
        super().__init__(graph_state)
        self._output_channels["out"] = LastValue[str]()
        self._input_channels["in"] = LastValue[str]()

    async def _execute(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Execute the node"""
        return {"result": "test"}

@pytest.fixture
def graph_state():
    """Create graph state fixture"""
    return GraphState()

@pytest.fixture
def source_node(graph_state):
    """Create source node fixture"""
    return TestNode(graph_state)

@pytest.fixture
def target_node_1(graph_state):
    """Create first target node fixture"""
    return TestNode(graph_state)

@pytest.fixture
def target_node_2(graph_state):
    """Create second target node fixture"""
    return TestNode(graph_state)

@pytest.mark.asyncio
async def test_conditional_edge_basic(graph_state, source_node, target_node_1):
    """Test basic conditional edge functionality"""
    # Create edge with just default target
    default_target = Target(target_node_1, "in")
    edge = ConditionalEdge(
        graph_state,
        source_node,
        "out",
        default_target
    )

    # Verify edge properties
    assert edge.source_node == source_node
    assert edge.target_node == target_node_1
    assert edge.source_channel == "out"
    assert edge.target_channel == "in"

    # Verify active target
    active = await edge.get_active_target()
    assert active == default_target

@pytest.mark.asyncio
async def test_conditional_edge_routing(
    graph_state,
    source_node,
    target_node_1,
    target_node_2
):
    """Test conditional routing logic"""
    # Create condition
    condition = StateCondition(
        graph_state,
        "route_key",
        lambda x: x == "use_conditional"
    )

    # Create edge with conditional target
    default_target = Target(target_node_1, "in")
    conditional_target = Target(
        target_node_2,
        "in",
        condition=condition,
        priority=1
    )

    edge = ConditionalEdge(
        graph_state,
        source_node,
        "out",
        default_target,
        [conditional_target]
    )

    # Test default routing
    graph_state.update_global_state({"route_key": "use_default"})
    active = await edge.get_active_target()
    assert active == default_target

    # Test conditional routing
    graph_state.update_global_state({"route_key": "use_conditional"})
    active = await edge.get_active_target()
    assert active == conditional_target

@pytest.mark.asyncio
async def test_conditional_edge_priority(
    graph_state,
    source_node,
    target_node_1,
    target_node_2
):
    """Test priority-based routing"""
    # Create conditions
    condition1 = StateCondition(
        graph_state,
        "key1",
        lambda x: x == "true"
    )
    condition2 = StateCondition(
        graph_state,
        "key2",
        lambda x: x == "true"
    )

    # Create edge with multiple conditional targets
    default_target = Target(target_node_1, "in")
    target1 = Target(
        target_node_2,
        "in",
        condition=condition1,
        priority=1
    )
    target2 = Target(
        target_node_2,
        "in",
        condition=condition2,
        priority=2  # Higher priority
    )

    edge = ConditionalEdge(
        graph_state,
        source_node,
        "out",
        default_target,
        [target1, target2]
    )

    # Test priority ordering
    graph_state.update_global_state({
        "key1": "true",
        "key2": "true"
    })

    active = await edge.get_active_target()
    assert active == target2  # Higher priority should win

def test_conditional_edge_validation(graph_state, source_node, target_node_1):
    """Test edge validation"""
    # Try to create edge with invalid channel
    default_target = Target(target_node_1, "non_existent")

    with pytest.raises(ValueError, match=r"Target channel 'non_existent' not found in node .*"):
        ConditionalEdge(
            graph_state,
            source_node,
            "out",
            default_target
        )

@pytest.mark.asyncio
async def test_conditional_edge_checkpoint(
    graph_state,
    source_node,
    target_node_1,
    target_node_2
):
    """Test edge checkpoint and restore"""
    # Create edge with conditional target
    condition = StateCondition(
        graph_state,
        "route_key",
        lambda x: x == "use_conditional"
    )

    default_target = Target(target_node_1, "in")
    conditional_target = Target(
        target_node_2,
        "in",
        condition=condition,
        priority=1
    )

    edge = ConditionalEdge(
        graph_state,
        source_node,
        "out",
        default_target,
        [conditional_target]
    )

    # Create checkpoint
    checkpoint = edge.checkpoint()

    # Verify checkpoint contents
    assert checkpoint["default_target"]["node"] == target_node_1.node_id
    assert checkpoint["default_target"]["channel"] == "in"
    assert len(checkpoint["conditional_targets"]) == 1
    assert checkpoint["conditional_targets"][0]["node"] == target_node_2.node_id
    assert checkpoint["conditional_targets"][0]["channel"] == "in"
    assert checkpoint["conditional_targets"][0]["priority"] == 1
