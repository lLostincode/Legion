import pytest
from typing import Optional, Dict, Any

from legion.graph.builder import GraphBuilder
from legion.graph.graph import Graph, GraphConfig
from legion.graph.nodes.base import NodeBase
from legion.graph.edges.base import EdgeBase
from legion.graph.state import GraphState
from legion.graph.channels import LastValue, ValueSequence

class TestNode(NodeBase):
    """Test node implementation"""
    def __init__(self, graph_state: GraphState):
        super().__init__(graph_state)
        self.param1: Optional[str] = None
        self.param2: Optional[str] = None
        
    async def _execute(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Execute the node"""
        return {"result": "test"}

class TestEdge(EdgeBase):
    """Test edge implementation"""
    def __init__(
        self,
        graph_state: GraphState,
        source_node: NodeBase,
        target_node: NodeBase,
        source_channel: str,
        target_channel: str,
        **kwargs
    ):
        super().__init__(
            graph_state,
            source_node,
            target_node,
            source_channel,
            target_channel
        )
        self.param1: Optional[str] = kwargs.get("param1")
        self.param2: Optional[str] = kwargs.get("param2")
        
    async def validate(self) -> bool:
        """Validate the edge"""
        return True

def test_builder_initialization():
    """Test builder initialization"""
    name = "test_graph"
    description = "Test graph description"
    config = GraphConfig()
    
    builder = GraphBuilder(name=name, description=description, config=config)
    graph = builder.build()
    
    assert isinstance(graph, Graph)
    assert graph.metadata.name == name
    assert graph.metadata.description == description
    assert graph.config == config
    
def test_node_management():
    """Test node addition and configuration"""
    builder = GraphBuilder()
    
    # Add node
    node_id = "test_node"
    builder.add_node(TestNode, node_id=node_id, param1="value1")
    
    # Verify node was added
    graph = builder.build()
    node = graph.nodes._nodes[node_id]
    assert isinstance(node, TestNode)
    assert node.param1 == "value1"
    
    # Configure node
    builder.configure_node(param2="value2")
    assert builder._node_configs[node_id] == {"param1": "value1", "param2": "value2"}
    assert node.param2 == "value2"
    
def test_channel_configuration():
    """Test channel configuration"""
    builder = GraphBuilder()
    
    # Add node with channels
    node = builder.add_node(TestNode)\
        .with_input_channel("input1", LastValue)\
        .with_output_channel("output1", ValueSequence)\
        .build()\
        .nodes._nodes[builder._current_node]
    
    # Verify channels
    assert "input1" in node._input_channels
    assert isinstance(node._input_channels["input1"], LastValue)
    assert "output1" in node._output_channels
    assert isinstance(node._output_channels["output1"], ValueSequence)
    
def test_node_selection():
    """Test node selection"""
    builder = GraphBuilder()
    
    # Add nodes
    node1 = builder.add_node(TestNode).build().nodes._nodes[builder._current_node]
    node2_id = builder.add_node(TestNode).build().nodes._nodes[builder._current_node].node_id
    
    # Select first node
    builder.select_node(node1.node_id)
    assert builder._current_node == node1.node_id
    
    # Configure selected node
    builder.configure_node(param1="value")
    assert builder._node_configs[node1.node_id] == {"param1": "value"}
    assert node1.param1 == "value"
    
    # Try selecting non-existent node
    with pytest.raises(ValueError, match="Node .* not found"):
        builder.select_node("non_existent")
        
def test_no_current_node():
    """Test operations with no current node"""
    builder = GraphBuilder()
    
    with pytest.raises(ValueError, match="No current node to configure"):
        builder.configure_node(param="value")
        
    with pytest.raises(ValueError, match="No current node to configure"):
        builder.with_input_channel("input", LastValue)
        
    with pytest.raises(ValueError, match="No current node to configure"):
        builder.with_output_channel("output", LastValue)
        
def test_edge_management():
    """Test edge addition and configuration"""
    builder = GraphBuilder()
    
    # Add nodes
    node1 = builder.add_node(TestNode)\
        .with_output_channel("default", LastValue)\
        .build()\
        .nodes._nodes[builder._current_node]
        
    node2 = builder.add_node(TestNode)\
        .with_input_channel("default", LastValue)\
        .build()\
        .nodes._nodes[builder._current_node]
    
    # Connect nodes
    edge = builder.connect(
        source_node=node1,
        target_node=node2,
        edge_type=TestEdge,
        param1="value1"
    )
    
    # Verify edge was added
    graph = builder.build()
    edge_id = builder._current_edge
    assert edge_id in graph.edges._edges
    edge = graph.edges._edges[edge_id]
    assert isinstance(edge, TestEdge)
    assert edge.param1 == "value1"
    
    # Configure edge
    builder.configure_edge(param2="value2")
    assert builder._edge_configs[edge_id] == {"param1": "value1", "param2": "value2"}
    assert edge.param2 == "value2"
    
def test_edge_selection():
    """Test edge selection"""
    builder = GraphBuilder()
    
    # Add nodes and edge
    node1 = builder.add_node(TestNode)\
        .with_output_channel("default", LastValue)\
        .build()\
        .nodes._nodes[builder._current_node]
        
    node2 = builder.add_node(TestNode)\
        .with_input_channel("default", LastValue)\
        .build()\
        .nodes._nodes[builder._current_node]
        
    edge = builder.connect(node1, node2, TestEdge)
    edge_id = builder._current_edge
    
    # Select edge
    builder.select_edge(edge_id)
    assert builder._current_edge == edge_id
    
    # Configure selected edge
    builder.configure_edge(param1="value")
    graph = builder.build()
    assert builder._edge_configs[edge_id] == {"param1": "value"}
    assert graph.edges._edges[edge_id].param1 == "value"
    
    # Try selecting non-existent edge
    with pytest.raises(ValueError, match="Edge .* not found"):
        builder.select_edge("non_existent")
        
def test_no_current_edge():
    """Test operations with no current edge"""
    builder = GraphBuilder()
    
    with pytest.raises(ValueError, match="No current edge to configure"):
        builder.configure_edge(param="value") 