from typing import Any, Dict

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

from legion.agents.base import Agent
from legion.graph.edges.base import EdgeBase
from legion.graph.edges.registry import EdgeRegistry
from legion.graph.edges.validator import EdgeValidator
from legion.graph.nodes.agent import AgentNode
from legion.graph.nodes.registry import NodeRegistry
from legion.graph.state import GraphState


class TestResponse(BaseModel):
    """Test response schema"""

    content: str
    metadata: Dict[str, Any]

@pytest.fixture
def graph_state():
    """Create a graph state instance"""
    return GraphState()

@pytest.fixture
def node_registry(graph_state):
    """Create a node registry instance"""
    return NodeRegistry(graph_state)

@pytest.fixture
def edge_registry(graph_state, node_registry):
    """Create an edge registry instance"""
    registry = EdgeRegistry(graph_state, node_registry)
    registry.register_edge_type("base", EdgeBase)
    return registry

@pytest.fixture
def validator():
    """Create an edge validator instance"""
    return EdgeValidator()

@pytest.fixture
def source_agent():
    """Create a source agent"""
    return Agent(
        name="SourceAgent",
        model="openai:gpt-4o-mini",
        temperature=0.7,
        system_prompt="You are a helpful source agent."
    )

@pytest.fixture
def target_agent():
    """Create a target agent"""
    return Agent(
        name="TargetAgent",
        model="openai:gpt-4o-mini",
        temperature=0.7,
        system_prompt="You are a helpful target agent."
    )

@pytest.fixture
def source_node(graph_state, source_agent):
    """Create a source node"""
    return AgentNode(
        graph_state=graph_state,
        agent=source_agent,
        response_schema=TestResponse
    )

@pytest.fixture
def target_node(graph_state, target_agent):
    """Create a target node"""
    return AgentNode(
        graph_state=graph_state,
        agent=target_agent,
        response_schema=TestResponse
    )

def test_edge_initialization(graph_state, source_node, target_node):
    """Test edge initialization"""
    edge = EdgeBase(
        graph_state=graph_state,
        source_node=source_node,
        target_node=target_node,
        source_channel="output",
        target_channel="input"
    )

    assert edge.source_node == source_node
    assert edge.target_node == target_node
    assert edge.source_channel == "output"
    assert edge.target_channel == "input"
    assert edge.metadata is not None
    assert edge.edge_id is not None

def test_edge_validation_success(validator, source_node, target_node):
    """Test successful edge validation"""
    result = validator.validate_edge(
        source_node=source_node,
        target_node=target_node,
        source_channel="output",
        target_channel="input"
    )

    assert result.is_valid
    assert result.error is None

def test_edge_validation_invalid_source_channel(validator, source_node, target_node):
    """Test edge validation with invalid source channel"""
    result = validator.validate_edge(
        source_node=source_node,
        target_node=target_node,
        source_channel="nonexistent",
        target_channel="input"
    )

    assert not result.is_valid
    assert "Source channel" in result.error

def test_edge_validation_invalid_target_channel(validator, source_node, target_node):
    """Test edge validation with invalid target channel"""
    result = validator.validate_edge(
        source_node=source_node,
        target_node=target_node,
        source_channel="output",
        target_channel="nonexistent"
    )

    assert not result.is_valid
    assert "Target channel" in result.error

def test_edge_registry_create_edge(edge_registry, node_registry, source_node, target_node):
    """Test edge creation through registry"""
    # Register nodes first
    node_registry._nodes[source_node.node_id] = source_node
    node_registry._nodes[target_node.node_id] = target_node

    edge = edge_registry.create_edge(
        type_name="base",
        source_node_id=source_node.node_id,
        target_node_id=target_node.node_id,
        source_channel="output",
        target_channel="input"
    )

    assert edge is not None
    assert edge.source_node == source_node
    assert edge.target_node == target_node
    assert edge_registry.get_edge(edge.edge_id) == edge

def test_edge_registry_delete_edge(edge_registry, node_registry, source_node, target_node):
    """Test edge deletion through registry"""
    # Register nodes first
    node_registry._nodes[source_node.node_id] = source_node
    node_registry._nodes[target_node.node_id] = target_node

    edge = edge_registry.create_edge(
        type_name="base",
        source_node_id=source_node.node_id,
        target_node_id=target_node.node_id,
        source_channel="output",
        target_channel="input"
    )

    edge_id = edge.edge_id
    edge_registry.delete_edge(edge_id)

    assert edge_registry.get_edge(edge_id) is None
    assert edge_id not in edge_registry.get_node_edges(source_node.node_id)
    assert edge_id not in edge_registry.get_node_edges(target_node.node_id, as_source=False)

def test_edge_registry_cycle_detection(edge_registry, node_registry, source_node, target_node):
    """Test cycle detection in edge registry"""
    # Register nodes first
    node_registry._nodes[source_node.node_id] = source_node
    node_registry._nodes[target_node.node_id] = target_node

    # Create first edge
    edge_registry.create_edge(
        type_name="base",
        source_node_id=source_node.node_id,
        target_node_id=target_node.node_id,
        source_channel="output",
        target_channel="input"
    )

    # Attempt to create cycle
    with pytest.raises(ValueError, match="cycle"):
        edge_registry.create_edge(
            type_name="base",
            source_node_id=target_node.node_id,
            target_node_id=source_node.node_id,
            source_channel="output",
            target_channel="input"
        )

def test_edge_registry_channel_edges(edge_registry, node_registry, source_node, target_node):
    """Test getting edges by channel"""
    # Register nodes first
    node_registry._nodes[source_node.node_id] = source_node
    node_registry._nodes[target_node.node_id] = target_node

    edge = edge_registry.create_edge(
        type_name="base",
        source_node_id=source_node.node_id,
        target_node_id=target_node.node_id,
        source_channel="output",
        target_channel="input"
    )

    source_edges = edge_registry.get_channel_edges(source_node.node_id, "output")
    target_edges = edge_registry.get_channel_edges(target_node.node_id, "input")

    assert edge.edge_id in source_edges
    assert edge.edge_id in target_edges

def test_edge_checkpoint_restore(graph_state, source_node, target_node):
    """Test edge checkpoint and restore"""
    edge = EdgeBase(
        graph_state=graph_state,
        source_node=source_node,
        target_node=target_node,
        source_channel="output",
        target_channel="input"
    )

    # Create checkpoint
    checkpoint = edge.checkpoint()

    # Create new edge and restore
    new_edge = EdgeBase(
        graph_state=graph_state,
        source_node=source_node,
        target_node=target_node,
        source_channel="output",
        target_channel="input"
    )
    new_edge.restore(checkpoint)

    assert new_edge.metadata.edge_id == edge.metadata.edge_id
    assert new_edge.metadata.version == edge.metadata.version
    assert new_edge.source_channel == edge.source_channel
    assert new_edge.target_channel == edge.target_channel

def test_edge_validator_cache(validator, source_node, target_node):
    """Test edge validator caching"""
    # First validation
    result1 = validator.validate_edge(
        source_node=source_node,
        target_node=target_node,
        source_channel="output",
        target_channel="input"
    )

    # Second validation (should use cache)
    result2 = validator.validate_edge(
        source_node=source_node,
        target_node=target_node,
        source_channel="output",
        target_channel="input"
    )

    assert result1 is result2  # Should be same object due to caching

    # Invalidate cache
    validator.invalidate_cache(node_id=source_node.node_id)

    # Third validation (should not use cache)
    result3 = validator.validate_edge(
        source_node=source_node,
        target_node=target_node,
        source_channel="output",
        target_channel="input"
    )

    assert result1 is not result3  # Should be different objects

def test_edge_registry_restore(edge_registry, node_registry, source_node, target_node):
    """Test edge registry checkpoint and restore"""
    # Register nodes first
    node_registry._nodes[source_node.node_id] = source_node
    node_registry._nodes[target_node.node_id] = target_node

    # Create edge
    edge_registry.create_edge(
        type_name="base",
        source_node_id=source_node.node_id,
        target_node_id=target_node.node_id,
        source_channel="output",
        target_channel="input"
    )

    # Create checkpoint
    checkpoint = edge_registry.checkpoint()

    # Create new registry and restore
    new_registry = EdgeRegistry(graph_state, node_registry)
    new_registry.register_edge_type("base", EdgeBase)
    new_registry.restore(checkpoint)

    assert new_registry.metadata.version == edge_registry.metadata.version

if __name__ == "__main__":
    pytest.main(["-v", __file__])
