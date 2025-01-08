from typing import Optional

import pytest
from dotenv import load_dotenv

from legion.agents.base import Agent
from legion.graph.channels import LastValue
from legion.graph.decorators import graph
from legion.graph.graph import Graph, GraphConfig, ResourceLimits
from legion.graph.nodes.agent import AgentNode
from legion.graph.nodes.base import NodeBase
from legion.graph.nodes.graph import (
    GraphNodeAdapter,
    InvalidStateError,
    SubgraphExecutionError,
    SubgraphStatus,
)
from legion.graph.state import GraphState

# Load environment variables
load_dotenv()

# Test graph that will be used as a node
@graph
class SimpleProcessor(Graph):
    """A simple processing graph."""

    def __init__(self, config: Optional[GraphConfig] = None):
        """Initialize graph with channels.

        Args:
        ----
            config: Optional graph configuration

        """
        super().__init__(config=config)

        # Create channels using graph state
        self._state.create_channel(
            LastValue,
            "input_channel",
            type_hint=str
        )
        self._state.create_channel(
            LastValue,
            "output_channel",
            type_hint=str
        )

        # Create agent
        agent = Agent(
            name="processor",
            model="gpt-4",
            temperature=0.7
        )

        # Add processor node
        self.processor = self.add_node(
            AgentNode,
            "processor",
            agent=agent,
            input_channel_type=str,
            output_channel_type=str
        )

    async def process(self) -> None:
        """Process the input and write to output."""
        # Read input
        input_value = self._state.get_channel("input_channel").get()

        # Process
        output_value = f"Processed: {input_value}"

        # Write output
        self._state.get_channel("output_channel").set(output_value)

def test_graph_node_adapter_creation():
    """Test creating a GraphNodeAdapter."""
    state = GraphState()
    adapter = GraphNodeAdapter(SimpleProcessor, state)

    assert isinstance(adapter, NodeBase)
    assert adapter.metadata.node_type == "SimpleProcessor"

def test_graph_node_channel_mapping():
    """Test that channels are properly mapped."""
    state = GraphState()
    adapter = GraphNodeAdapter(SimpleProcessor, state)

    # Check that input/output channels are mapped
    assert "input_channel" in adapter._input_channels
    assert "output_channel" in adapter._output_channels

    # Verify channel types
    assert isinstance(adapter._input_channels["input_channel"], LastValue)
    assert isinstance(adapter._output_channels["output_channel"], LastValue)

@pytest.mark.asyncio
async def test_graph_node_processing():
    """Test that the graph node processes data correctly."""
    state = GraphState()
    adapter = GraphNodeAdapter(SimpleProcessor, state)

    # Set input
    adapter._input_channels["input_channel"].set("test")

    # Process
    await adapter.execute()

    # Check output
    result = adapter._output_channels["output_channel"].get()
    assert result == "Processed: test"

def test_graph_node_state_isolation():
    """Test that graph nodes maintain state isolation."""
    state = GraphState()
    adapter1 = GraphNodeAdapter(SimpleProcessor, state)
    adapter2 = GraphNodeAdapter(SimpleProcessor, state)

    assert adapter1._graph_state is not adapter2._graph_state
    assert adapter1._graph_state._channels != adapter2._graph_state._channels

@pytest.mark.asyncio
async def test_graph_node_resource_limits():
    """Test resource limit enforcement."""
    state = GraphState()
    config = GraphConfig(
        resource_limits=ResourceLimits(
            max_execution_time_seconds=0  # Force immediate timeout
        )
    )
    adapter = GraphNodeAdapter(SimpleProcessor, state, config=config)

    # Set input
    adapter._input_channels["input_channel"].set("test")

    # Process should fail due to timeout
    with pytest.raises(SubgraphExecutionError) as exc_info:
        await adapter.execute()
    assert "Maximum execution time" in str(exc_info.value)

@pytest.mark.asyncio
async def test_graph_node_pause_resume():
    """Test pause and resume functionality."""
    state = GraphState()
    adapter = GraphNodeAdapter(SimpleProcessor, state)

    # Cannot pause when not running
    with pytest.raises(InvalidStateError):
        await adapter.pause()

    # Start execution
    adapter._subgraph_status = SubgraphStatus.RUNNING

    # Pause execution
    await adapter.pause()
    assert adapter._subgraph_status == SubgraphStatus.PAUSED
    assert adapter._last_checkpoint is not None

    # Resume execution
    await adapter.resume()
    assert adapter._subgraph_status == SubgraphStatus.RUNNING

@pytest.mark.asyncio
async def test_graph_node_execution_stats():
    """Test execution statistics tracking."""
    state = GraphState()
    adapter = GraphNodeAdapter(SimpleProcessor, state)

    # Initial stats
    stats = adapter.get_execution_stats()
    assert stats["total_executions"] == 0
    assert stats["error_count"] == 0

    # Execute successfully
    adapter._input_channels["input_channel"].set("test")
    await adapter.execute()

    # Check updated stats
    stats = adapter.get_execution_stats()
    assert stats["total_executions"] == 1
    assert stats["error_count"] == 0
    assert stats["last_execution"] is not None
    assert stats["avg_duration"] > 0

@pytest.mark.asyncio
async def test_graph_node_debug_info():
    """Test debug information access."""
    state = GraphState()
    adapter = GraphNodeAdapter(SimpleProcessor, state)

    # Get initial debug info
    debug_info = adapter.get_debug_info()
    assert debug_info["status"] == SubgraphStatus.IDLE
    assert not debug_info["has_checkpoint"]
    assert len(debug_info["channels"]["input"]) == 1
    assert len(debug_info["channels"]["output"]) == 1

    # Execute and check updated debug info
    adapter._input_channels["input_channel"].set("test")
    await adapter.execute()

    debug_info = adapter.get_debug_info()
    assert debug_info["status"] == SubgraphStatus.COMPLETED
    assert debug_info["execution_start"] is not None
