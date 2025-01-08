from datetime import datetime
from typing import Any, Dict, Optional

import pytest

from legion.graph.channels import LastValue
from legion.graph.edges.base import EdgeBase
from legion.graph.edges.registry import EdgeRegistry
from legion.graph.graph import (
    ExecutionMode,
    Graph,
    GraphConfig,
    GraphMetadata,
    LogLevel,
    ResourceLimits,
)
from legion.graph.nodes.base import NodeBase
from legion.graph.nodes.registry import NodeRegistry
from legion.graph.state import GraphState


class TestNode(NodeBase):
    """Test node implementation"""

    def __init__(self, graph_state: GraphState):
        super().__init__(graph_state)
        # Create default channels
        self.create_input_channel("default", LastValue)
        self.create_output_channel("default", LastValue)

    async def _execute(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Execute the node"""
        return {"result": "test"}

class TestEdge(EdgeBase):
    """Test edge implementation"""

    async def validate(self) -> bool:
        """Validate the edge"""
        return True

def test_graph_initialization():
    """Test basic graph initialization"""
    name = "test_graph"
    description = "Test graph description"
    graph = Graph(name=name, description=description)

    # Check metadata
    assert isinstance(graph.metadata, GraphMetadata)
    assert graph.metadata.name == name
    assert graph.metadata.description == description
    assert isinstance(graph.metadata.created_at, datetime)
    assert isinstance(graph.metadata.updated_at, datetime)
    assert graph.metadata.version == 0

    # Check components
    assert isinstance(graph.state, GraphState)
    assert isinstance(graph.nodes, NodeRegistry)
    assert isinstance(graph.edges, EdgeRegistry)

def test_graph_metadata_update():
    """Test metadata updates when graph changes"""
    graph = Graph()
    initial_version = graph.metadata.version
    initial_updated_at = graph.metadata.updated_at

    # Force metadata update
    graph._update_metadata()

    assert graph.metadata.version == initial_version + 1
    assert graph.metadata.updated_at > initial_updated_at

def test_node_management():
    """Test node addition and removal"""
    graph = Graph()

    # Add node by type
    node1 = graph.add_node(TestNode)
    assert node1.node_id in graph.nodes._nodes
    assert isinstance(node1, TestNode)

    # Add node by string type name
    graph.nodes.register_node_type("test_node", TestNode)
    node2 = graph.add_node("test_node")
    assert node2.node_id in graph.nodes._nodes
    assert isinstance(node2, TestNode)

    # Remove node
    graph.remove_node(node1.node_id)
    assert node1.node_id not in graph.nodes._nodes
    assert node1.node_id not in graph.nodes._dependencies
    assert node1.node_id not in graph.nodes._reverse_dependencies

def test_edge_management():
    """Test edge addition and removal"""
    graph = Graph()

    # Create nodes
    node1 = graph.add_node(TestNode)
    node2 = graph.add_node(TestNode)

    # Add edge by type
    edge1 = graph.add_edge(node1, node2, TestEdge)
    assert edge1.edge_id in graph.edges._edges
    assert isinstance(edge1, TestEdge)

    # Add edge by string type name
    graph.edges.register_edge_type("test_edge", TestEdge)
    edge2 = graph.add_edge(
        node1.node_id,  # Test with node ID
        node2,          # Test with node instance
        "test_edge"
    )
    assert edge2.edge_id in graph.edges._edges
    assert isinstance(edge2, TestEdge)

    # Remove edge
    graph.remove_edge(edge1.edge_id)
    assert edge1.edge_id not in graph.edges._edges

def test_graph_clear():
    """Test clearing the graph"""
    graph = Graph()

    # Add some nodes and edges
    node1 = graph.add_node(TestNode)
    node2 = graph.add_node(TestNode)
    graph.add_edge(node1, node2, TestEdge)

    # Clear graph
    graph.clear()

    # Verify everything is cleared
    assert len(graph.nodes._nodes) == 0
    assert len(graph.nodes._dependencies) == 0
    assert len(graph.nodes._reverse_dependencies) == 0
    assert len(graph.edges._edges) == 0
    assert len(graph.edges._source_edges) == 0
    assert len(graph.edges._target_edges) == 0
    assert len(graph.edges._channel_edges) == 0

def test_graph_configuration():
    """Test graph configuration"""
    # Test default configuration
    graph = Graph()
    assert isinstance(graph.config, GraphConfig)
    assert graph.config.execution_mode == ExecutionMode.SEQUENTIAL
    assert isinstance(graph.config.resource_limits, ResourceLimits)
    assert graph.config.log_level == LogLevel.INFO

    # Test custom configuration
    config = GraphConfig(
        execution_mode=ExecutionMode.PARALLEL,
        log_level=LogLevel.DEBUG,
        debug_mode=True,
        enable_performance_tracking=True,
        checkpoint_interval_seconds=60,
        error_retry_count=3,
        error_retry_delay_seconds=5,
        resource_limits=ResourceLimits(
            max_nodes=10,
            max_edges=20,
            max_memory_mb=1024,
            max_execution_time_seconds=300
        )
    )
    graph = Graph(config=config)
    assert graph.config == config

    # Test configuration update
    new_config = GraphConfig(
        execution_mode=ExecutionMode.SEQUENTIAL,
        log_level=LogLevel.WARNING
    )
    graph.config = new_config
    assert graph.config == new_config

def test_resource_limits():
    """Test resource limits"""
    config = GraphConfig(
        resource_limits=ResourceLimits(
            max_nodes=2,
            max_edges=1
        )
    )
    graph = Graph(config=config)

    # Test node limit
    node1 = graph.add_node(TestNode)
    node2 = graph.add_node(TestNode)
    with pytest.raises(ValueError, match="Maximum number of nodes.*exceeded"):
        graph.add_node(TestNode)

    # Test edge limit
    graph.add_edge(node1, node2, TestEdge)
    with pytest.raises(ValueError, match="Maximum number of edges.*exceeded"):
        graph.add_edge(node2, node1, TestEdge)
