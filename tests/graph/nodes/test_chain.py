import pytest
import asyncio
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock

from legion.graph.nodes.chain import ChainNode, ChainMode
from legion.graph.state import GraphState
from legion.groups.chain import Chain
from legion.interface.schemas import Message, Role, SystemPrompt
from legion.agents.base import Agent
from legion.graph.nodes.base import NodeStatus
from legion.memory.providers.memory import InMemoryProvider

class MockAgent(Agent):
    """Mock agent for testing"""
    def __init__(self, name: str, responses: List[str]):
        super().__init__(
            name=name,
            model="openai:gpt-3.5-turbo",
            temperature=0.0,
            system_prompt=SystemPrompt(sections=[{"content": "Mock agent", "is_dynamic": False}]),
            memory_provider=InMemoryProvider()
        )
        self.responses = responses
        self.call_count = 0
        
    async def aprocess(self, message: Message, **kwargs) -> Message:
        """Override aprocess to return predefined responses"""
        # Store input message
        self.memory.add_message(message)
        
        # Generate response
        response = Message(
            role=Role.ASSISTANT,
            content=self.responses[self.call_count % len(self.responses)]
        )
        
        # Store response
        self.memory.add_message(response)
        
        self.call_count += 1
        return response
        
    def _setup_provider(self, provider_name: str) -> None:
        """Override provider setup to avoid actual API calls"""
        pass

@pytest.fixture
def mock_chain():
    """Create a mock chain for testing"""
    agents = [
        MockAgent("agent1", ["Step 1 output"]),
        MockAgent("agent2", ["Step 2 output"]),
        MockAgent("agent3", ["Final output"])
    ]
    return Chain(name="test_chain", members=agents)

@pytest.fixture
def graph_state():
    """Create a graph state for testing"""
    return GraphState()

@pytest.mark.asyncio
async def test_chain_node_atomic_mode(graph_state, mock_chain):
    """Test chain node in atomic mode"""
    node = ChainNode(
        graph_state=graph_state,
        chain=mock_chain,
        mode=ChainMode.ATOMIC
    )
    
    # Set input
    input_channel = node.get_input_channel("input")
    input_channel.set("Test input")
    
    # Execute
    result = await node.execute()
    
    # Check output
    assert result["output"] == "Final output"
    assert len(result["member_outputs"]) == 3
    assert result["member_outputs"]["step_1"] == "Step 1 output"
    assert result["member_outputs"]["step_2"] == "Step 2 output"
    assert result["member_outputs"]["step_3"] == "Final output"

@pytest.mark.asyncio
async def test_chain_node_expanded_mode(graph_state, mock_chain):
    """Test chain node in expanded mode"""
    node = ChainNode(
        graph_state=graph_state,
        chain=mock_chain,
        mode=ChainMode.EXPANDED
    )
    
    # Set input
    input_channel = node.get_input_channel("input")
    input_channel.set("Test input")
    
    # Execute
    result = await node.execute()
    
    # Check intermediate outputs
    step1_channel = node.get_output_channel("step_1_output")
    step2_channel = node.get_output_channel("step_2_output")
    step3_channel = node.get_output_channel("step_3_output")
    
    assert step1_channel.get() == "Step 1 output"
    assert step2_channel.get() == "Step 2 output"
    assert step3_channel.get() == "Final output"
    
    # Check final output
    assert result["output"] == "Final output"
    assert len(result["member_outputs"]) == 3

@pytest.mark.asyncio
async def test_chain_node_mode_switching(graph_state, mock_chain):
    """Test switching between atomic and expanded modes"""
    node = ChainNode(
        graph_state=graph_state,
        chain=mock_chain,
        mode=ChainMode.ATOMIC
    )
    
    # Check initial channels
    assert "step_1_output" not in node._output_channels
    
    # Switch to expanded mode
    node.mode = ChainMode.EXPANDED
    
    # Check expanded mode channels
    assert "step_1_output" in node._output_channels
    assert "step_2_output" in node._output_channels
    assert "step_3_output" in node._output_channels
    
    # Switch back to atomic
    node.mode = ChainMode.ATOMIC
    
    # Check channels cleaned up
    assert "step_1_output" not in node._output_channels

@pytest.mark.asyncio
async def test_chain_node_pause_resume_expanded(graph_state, mock_chain):
    """Test pause and resume in expanded mode"""
    node = ChainNode(
        graph_state=graph_state,
        chain=mock_chain,
        mode=ChainMode.EXPANDED
    )
    
    # Set input
    input_channel = node.get_input_channel("input")
    input_channel.set("Test input")
    
    # Create a slow mock agent
    slow_agent = MockAgent("slow_agent", ["Slow output"])
    original_aprocess = slow_agent.aprocess
    
    async def slow_aprocess(*args, **kwargs):
        await asyncio.sleep(1.0)  # Longer delay
        return await original_aprocess(*args, **kwargs)
    
    # Replace first agent with slow agent
    mock_chain.members["step_1"] = slow_agent
    slow_agent.aprocess = slow_aprocess
    
    # Start execution
    task = asyncio.create_task(node.execute())
    
    # Wait for execution to start
    await asyncio.sleep(0.2)  # Longer wait
    await node.pause()
    
    # Check paused state
    assert node.status == NodeStatus.PAUSED
    assert "paused_at_member" in node._metadata.custom_data
    
    # Resume execution
    await node.resume()
    result = await task
    
    # Check completed successfully
    assert node.status == NodeStatus.COMPLETED
    assert result["output"] == "Final output"

@pytest.mark.asyncio
async def test_chain_node_checkpoint_restore(graph_state, mock_chain):
    """Test checkpoint and restore with mode preservation"""
    node = ChainNode(
        graph_state=graph_state,
        chain=mock_chain,
        mode=ChainMode.EXPANDED
    )
    
    # Set some state
    input_channel = node.get_input_channel("input")
    input_channel.set("Test input")
    await node.execute()
    
    # Create checkpoint
    checkpoint = node.checkpoint()
    
    # Create new node
    new_node = ChainNode(
        graph_state=graph_state,
        chain=mock_chain,
        mode=ChainMode.ATOMIC  # Different mode
    )
    
    # Restore checkpoint
    new_node.restore(checkpoint)
    
    # Check mode restored
    assert new_node.mode == ChainMode.EXPANDED
    assert "step_1_output" in new_node._output_channels
