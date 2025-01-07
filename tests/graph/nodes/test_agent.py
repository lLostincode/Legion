import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List, Sequence, Type
from pydantic import BaseModel

from legion.graph.state import GraphState
from legion.graph.nodes.agent import AgentNode
from legion.agents.base import Agent
from legion.interface.schemas import Message, ModelResponse, Role, SystemPrompt, TokenUsage, ProviderConfig
from legion.interface.tools import BaseTool
from legion.interface.base import LLMInterface

class MockLLMInterface(LLMInterface):
    """Mock LLM interface for testing"""
    
    def __init__(self):
        super().__init__(ProviderConfig(api_key="test"), debug=False)
    
    def _setup_client(self) -> None:
        """Mock client setup"""
        pass
    
    async def _asetup_client(self) -> None:
        """Mock async client setup"""
        pass
    
    def _format_messages(self, messages: List[Message]) -> Any:
        """Mock message formatting"""
        return messages
    
    def _extract_tool_calls(self, response: Any) -> Optional[List[Dict[str, Any]]]:
        """Mock tool call extraction"""
        return response.get("tool_calls")
    
    def _extract_content(self, response: Any) -> str:
        """Mock content extraction"""
        return response.get("content", "")
    
    def _extract_usage(self, response: Any) -> TokenUsage:
        """Mock usage extraction"""
        return TokenUsage(
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20
        )
    
    def _get_chat_completion(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Mock chat completion"""
        last_message = messages[-1]
        return ModelResponse(
            content=f"Processed: {last_message.content}",
            raw_response={},
            usage=TokenUsage(
                prompt_tokens=10,
                completion_tokens=10,
                total_tokens=20
            )
        )
    
    def _get_tool_completion(
        self,
        messages: List[Message],
        model: str,
        tools: Sequence[BaseTool],
        temperature: float,
        max_tokens: Optional[int] = None,
        format_json: bool = False,
        json_schema: Optional[Type[BaseModel]] = None
    ) -> ModelResponse:
        """Mock tool completion"""
        last_message = messages[-1]
        return ModelResponse(
            content=f"Processed: {last_message.content}",
            raw_response={},
            tool_calls=[
                {
                    "id": "1",
                    "function": {
                        "name": "test_tool",
                        "arguments": '{"arg": "test"}'
                    },
                    "result": "tool_result"
                }
            ] if tools else None,
            usage=TokenUsage(
                prompt_tokens=10,
                completion_tokens=10,
                total_tokens=20
            )
        )
    
    def _get_json_completion(
        self,
        messages: List[Message],
        model: str,
        schema: Type[BaseModel],
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Mock JSON completion"""
        last_message = messages[-1]
        return ModelResponse(
            content='{"result": "test"}',
            raw_response={},
            usage=TokenUsage(
                prompt_tokens=10,
                completion_tokens=10,
                total_tokens=20
            )
        )
    
    async def _aget_chat_completion(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Mock async chat completion"""
        return self._get_chat_completion(messages, model, temperature, max_tokens)
    
    async def _aget_tool_completion(
        self,
        messages: List[Message],
        model: str,
        tools: Sequence[BaseTool],
        temperature: float,
        max_tokens: Optional[int] = None,
        format_json: bool = False,
        json_schema: Optional[Type[BaseModel]] = None
    ) -> ModelResponse:
        """Mock async tool completion"""
        return self._get_tool_completion(
            messages, model, tools, temperature, max_tokens,
            format_json, json_schema
        )
    
    async def _aget_json_completion(
        self,
        messages: List[Message],
        model: str,
        schema: Type[BaseModel],
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Mock async JSON completion"""
        return self._get_json_completion(messages, model, schema, temperature, max_tokens)

class MockToolParams(BaseModel):
    """Parameters for mock tool"""
    arg: str

class MockTool(BaseTool):
    """Mock tool for testing"""
    
    def __init__(self):
        super().__init__(
            name="test_tool",
            description="Test tool",
            parameters=MockToolParams
        )
    
    def run(self, arg: str) -> str:
        """Mock execution"""
        return "tool_result"

@pytest.fixture
def graph_state():
    """Create graph state fixture"""
    return GraphState()

@pytest.fixture
def mock_agent():
    """Create mock agent fixture"""
    agent = Agent(
        name="test_agent",
        model="openai:gpt-4",  # Use a valid provider
        system_prompt=SystemPrompt(static_prompt="Test prompt")
    )
    # Replace LLM interface with mock
    agent.llm = MockLLMInterface()
    return agent

@pytest.fixture
def agent_node(graph_state, mock_agent):
    """Create agent node fixture"""
    return AgentNode(
        graph_state=graph_state,
        agent=mock_agent
    )

@pytest.mark.asyncio
async def test_agent_node_initialization(agent_node):
    """Test agent node initialization"""
    assert agent_node.agent.name == "test_agent"
    assert "input" in agent_node.list_input_channels()
    assert "output" in agent_node.list_output_channels()
    assert "tool_results" in agent_node.list_output_channels()
    assert "memory" in agent_node.list_output_channels()

@pytest.mark.asyncio
async def test_agent_node_execution(agent_node):
    """Test agent node execution"""
    # Set input
    input_channel = agent_node.get_input_channel("input")
    input_channel.set("test input")
    
    # Execute
    result = await agent_node.execute()
    
    # Check output channels
    output_channel = agent_node.get_output_channel("output")
    assert output_channel.get() == "Processed: test input"
    
    # Check memory channel
    memory_channel = agent_node.get_output_channel("memory")
    memory_state = memory_channel.get()
    assert isinstance(memory_state, dict)
    assert "messages" in memory_state
    assert "last_updated" in memory_state
    
    # Check execution result
    assert result["output"] == "Processed: test input"
    assert "tool_results" in result
    assert "memory" in result

@pytest.mark.asyncio
async def test_agent_node_with_tools(agent_node, mock_agent):
    """Test agent node with tools"""
    # Add tool to agent
    mock_agent.tools = [MockTool()]
    
    # Set input
    input_channel = agent_node.get_input_channel("input")
    input_channel.set("test input")
    
    # Execute
    result = await agent_node.execute()
    
    # Check tool results channel
    tool_results = agent_node.get_output_channel("tool_results")
    assert len(tool_results.get_all()) == 1
    assert "test_tool: tool_result" in tool_results.get_all()
    
    # Check execution result
    assert result["tool_results"][0]["name"] == "test_tool"
    assert result["tool_results"][0]["result"] == "tool_result"

@pytest.mark.asyncio
async def test_agent_node_checkpointing(agent_node, mock_agent):
    """Test agent node checkpointing"""
    # Add some state
    input_channel = agent_node.get_input_channel("input")
    input_channel.set("test input")
    await agent_node.execute()
    
    # Create checkpoint
    checkpoint = agent_node.checkpoint()
    
    # Verify checkpoint contents
    assert "agent_metadata" in checkpoint
    metadata = checkpoint["agent_metadata"]
    assert metadata["name"] == "test_agent"
    assert metadata["model"] == "openai:gpt-4"
    assert len(metadata["memory"]) > 0
    
    # Create new node and restore
    new_node = AgentNode(
        graph_state=agent_node._graph_state,
        agent=Agent(
            name="new_agent",
            model="openai:gpt-4"
        )
    )
    # Replace LLM interface with mock
    new_node.agent.llm = MockLLMInterface()
    new_node.restore(checkpoint)
    
    # Verify restored state
    assert new_node.agent.name == "test_agent"
    assert len(new_node.agent.memory.messages) == len(agent_node.agent.memory.messages)

@pytest.mark.asyncio
async def test_agent_node_empty_input(agent_node):
    """Test agent node with empty input"""
    # Execute without setting input
    result = await agent_node.execute()
    assert result is None
    
    # Check output channel is empty
    output_channel = agent_node.get_output_channel("output")
    assert output_channel.get() is None
