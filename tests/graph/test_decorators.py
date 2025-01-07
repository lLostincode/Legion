import pytest
from legion.graph.decorators import graph
from legion.graph.graph import Graph, GraphConfig, ExecutionMode, LogLevel
from legion.graph.state import GraphState

def test_basic_graph_creation():
    """Test basic graph creation with minimal configuration"""
    
    @graph
    class SimpleGraph:
        """A simple test graph."""
        pass
    
    instance = SimpleGraph()
    assert isinstance(instance.graph, Graph)
    assert instance.graph.metadata.name == "SimpleGraph"
    assert "simple test graph" in instance.graph.metadata.description.lower()

def test_graph_with_config():
    """Test graph creation with configuration"""
    
    @graph(
        name="custom_graph",
        debug_mode=True,
        log_level=LogLevel.DEBUG
    )
    class ConfiguredGraph:
        """A configured test graph."""
        pass
    
    instance = ConfiguredGraph()
    assert instance.graph.metadata.name == "custom_graph"
    assert instance.graph._config.debug_mode is True
    assert instance.graph._config.log_level == LogLevel.DEBUG

def test_graph_with_class_config():
    """Test graph creation with class-level configuration"""
    
    @graph
    class ConfigGraph:
        """A graph with class configuration."""
        
        config = GraphConfig(
            execution_mode=ExecutionMode.PARALLEL,
            debug_mode=True,
            log_level=LogLevel.DEBUG
        )
    
    instance = ConfigGraph()
    assert instance.graph._config.execution_mode == ExecutionMode.PARALLEL
    assert instance.graph._config.debug_mode is True
    assert instance.graph._config.log_level == LogLevel.DEBUG

def test_graph_inheritance():
    """Test graph inheritance"""
    
    @graph
    class BaseGraph:
        """Base graph class."""
        config = GraphConfig(debug_mode=True)
    
    @graph
    class DerivedGraph(BaseGraph):
        """Derived graph class."""
        config = GraphConfig(log_level=LogLevel.DEBUG)
    
    base = BaseGraph()
    derived = DerivedGraph()
    
    assert base.graph._config.debug_mode is True
    assert derived.graph._config.debug_mode is False  # Not inherited
    assert derived.graph._config.log_level == LogLevel.DEBUG

if __name__ == "__main__":
    pytest.main(["-v", __file__]) 