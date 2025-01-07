import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from legion.graph.state import GraphState
from legion.graph.nodes.team import TeamNode, TeamMode
from legion.graph.nodes.base import NodeStatus
from legion.interface.schemas import Message, ModelResponse, Role, TokenUsage
from legion.agents.base import Agent
from legion.groups.team import Team
from legion.memory.base import MemoryProvider
from legion.interface.base import ProviderConfig

class MockMemoryProvider(MemoryProvider):
    """Mock memory provider for testing"""
    async def create_thread(self, entity_id: str, parent_thread_id: str = None) -> str:
        return "mock_thread_id"
    
    async def get_thread(self, thread_id: str) -> Dict[str, Any]:
        return {"id": thread_id, "messages": []}
    
    async def add_message(self, thread_id: str, message: Dict[str, Any]) -> None:
        pass
    
    async def get_messages(self, thread_id: str) -> List[Dict[str, Any]]:
        return []

class MockProvider:
    """Mock provider for testing"""
    def __init__(self, config: ProviderConfig = None, debug: bool = False):
        self.config = config or ProviderConfig()
        self.debug = debug
    
    async def agenerate(self, *args, **kwargs) -> ModelResponse:
        return ModelResponse(
            content="Mock response",
            role=Role.ASSISTANT,
            raw_response={"content": "Mock response"},
            usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20)
        )

class MockAgent(Agent):
    """Mock agent for testing"""
    def __init__(self, name: str = "mock_agent"):
        self.name = name
        self.model = "openai:gpt-3.5-turbo"  # Mock model name
        self.system_prompt = "You are a mock agent for testing."
        self.aprocess = AsyncMock()
        self.process = MagicMock()
        
        # Mock the provider
        with patch('legion.agents.base.get_provider', return_value=MockProvider()):
            super().__init__(
                name=name,
                model=self.model,
                system_prompt=self.system_prompt
            )

@pytest.fixture
def graph_state():
    return GraphState()

@pytest.fixture
def mock_team():
    leader = MockAgent("leader")
    members = {
        "member1": MockAgent("member1"),
        "member2": MockAgent("member2")
    }
    team = Team(
        name="test_team",
        leader=leader,
        members=members
    )
    # Mock the team's aprocess method
    team.aprocess = AsyncMock()
    return team

@pytest.fixture
def team_node(graph_state, mock_team):
    return TeamNode(graph_state, mock_team)

@pytest.mark.asyncio
async def test_team_node_atomic_mode(team_node, mock_team):
    """Test team node in atomic mode"""
    # Setup mock response
    mock_response = ModelResponse(
        content="Test response",
        role=Role.ASSISTANT,
        tool_calls=[{"function": {"name": "test_tool"}, "result": "test result"}],
        raw_response={"content": "Test response"},
        usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    )
    mock_team.aprocess.return_value = mock_response
    
    # Set input
    input_channel = team_node.get_input_channel("input")
    input_channel.set("Test input")
    
    # Execute
    result = await team_node._execute()
    
    # Verify
    assert team_node.status == NodeStatus.COMPLETED
    assert result["output"] == "Test response"
    assert len(result["delegations"]) == 1
    
    # Check channels
    output_channel = team_node.get_output_channel("output")
    assert output_channel.get() == "Test response"
    
    delegation_results = team_node.get_output_channel("delegation_results")
    assert len(delegation_results.get_all()) == 1

@pytest.mark.asyncio
async def test_team_node_expanded_mode(team_node, mock_team):
    """Test team node in expanded mode"""
    # Switch to expanded mode
    team_node.mode = TeamMode.EXPANDED
    
    # Setup mock responses
    leader_response = ModelResponse(
        content="Leader response",
        role=Role.ASSISTANT,
        raw_response={"content": "Leader response"},
        usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    )
    member1_response = ModelResponse(
        content="Member 1 response",
        role=Role.ASSISTANT,
        tool_calls=[{"function": {"name": "test_tool"}, "result": "test result"}],
        raw_response={"content": "Member 1 response"},
        usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    )
    member2_response = ModelResponse(
        content="Member 2 response",
        role=Role.ASSISTANT,
        raw_response={"content": "Member 2 response"},
        usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    )
    
    mock_team.leader.aprocess.return_value = leader_response
    mock_team.members["member1"].aprocess.return_value = member1_response
    mock_team.members["member2"].aprocess.return_value = member2_response
    
    # Set input
    input_channel = team_node.get_input_channel("input")
    input_channel.set("Test input")
    
    # Execute
    result = await team_node._execute()
    
    # Verify
    assert team_node.status == NodeStatus.COMPLETED
    assert result["output"] == "Member 2 response"
    assert "leader" in result["member_outputs"]
    assert "member1" in result["member_outputs"]
    assert "member2" in result["member_outputs"]
    
    # Check channels
    leader_output = team_node.get_output_channel("leader_output")
    assert leader_output.get() == "Leader response"
    
    member1_output = team_node.get_output_channel("member1_output")
    assert member1_output.get() == "Member 1 response"
    
    member2_output = team_node.get_output_channel("member2_output")
    assert member2_output.get() == "Member 2 response"
    
    member1_delegations = team_node.get_output_channel("member1_delegations")
    assert len(member1_delegations.get_all()) == 1

@pytest.mark.asyncio
async def test_team_node_mode_switching(team_node):
    """Test switching between atomic and expanded modes"""
    # Start in atomic mode
    assert team_node.mode == TeamMode.ATOMIC
    assert "leader_output" not in team_node._output_channels
    
    # Switch to expanded mode
    team_node.mode = TeamMode.EXPANDED
    assert team_node.mode == TeamMode.EXPANDED
    assert "leader_output" in team_node._output_channels
    assert "member1_output" in team_node._output_channels
    assert "member2_output" in team_node._output_channels
    
    # Switch back to atomic mode
    team_node.mode = TeamMode.ATOMIC
    assert team_node.mode == TeamMode.ATOMIC
    assert "leader_output" not in team_node._output_channels
    assert "member1_output" not in team_node._output_channels
    assert "member2_output" not in team_node._output_channels

@pytest.mark.asyncio
async def test_team_node_pause_resume_expanded(team_node, mock_team):
    """Test pausing and resuming in expanded mode"""
    # Switch to expanded mode
    team_node.mode = TeamMode.EXPANDED
    
    # Setup mock responses
    leader_response = ModelResponse(
        content="Leader response",
        role=Role.ASSISTANT,
        raw_response={"content": "Leader response"},
        usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    )
    member1_response = ModelResponse(
        content="Member 1 response",
        role=Role.ASSISTANT,
        raw_response={"content": "Member 1 response"},
        usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    )
    
    mock_team.leader.aprocess.return_value = leader_response
    mock_team.members["member1"].aprocess.return_value = member1_response
    
    # Set input
    input_channel = team_node.get_input_channel("input")
    input_channel.set("Test input")
    
    # Start execution and pause during member1
    def side_effect(*args, **kwargs):
        team_node._update_status(NodeStatus.PAUSED)
        return member1_response
    mock_team.members["member1"].aprocess.side_effect = side_effect
    
    # Execute
    result = await team_node._execute()
    
    # Verify paused state
    assert team_node.status == NodeStatus.PAUSED
    assert team_node._metadata.custom_data["paused_at_member"] == "member1"
    assert result is None
    
    # Resume execution
    mock_team.members["member1"].aprocess.side_effect = None
    result = await team_node._execute()
    
    # Verify completed state
    assert team_node.status == NodeStatus.COMPLETED
    assert result is not None
    assert "member1" in result["member_outputs"]

@pytest.mark.asyncio
async def test_team_node_checkpoint_restore(team_node, mock_team, graph_state):
    """Test checkpointing and restoring team node state"""
    # Switch to expanded mode and process some input
    team_node.mode = TeamMode.EXPANDED
    
    # Setup mock responses
    leader_response = ModelResponse(
        content="Leader response",
        role=Role.ASSISTANT,
        raw_response={"content": "Leader response"},
        usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    )
    mock_team.leader.aprocess.return_value = leader_response
    
    # Set input and execute
    input_channel = team_node.get_input_channel("input")
    input_channel.set("Test input")
    await team_node._execute()
    
    # Create checkpoint
    checkpoint = team_node.checkpoint()
    
    # Create new team node
    new_team_node = TeamNode(graph_state, mock_team)
    
    # Restore from checkpoint
    new_team_node.restore(checkpoint)
    
    # Verify restored state
    assert new_team_node.mode == TeamMode.EXPANDED
    assert "leader_output" in new_team_node._output_channels
    assert new_team_node._team.name == mock_team.name
    assert new_team_node._team.leader.name == mock_team.leader.name
    
    # Verify channels were restored
    leader_output = new_team_node.get_output_channel("leader_output")
    assert leader_output.get() == "Leader response"
