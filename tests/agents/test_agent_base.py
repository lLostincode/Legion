import os
import pytest
from dotenv import load_dotenv
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from legion.interface.schemas import Message, Role, ModelResponse, SystemPrompt, SystemPromptSection
from legion.interface.tools import BaseTool
from legion.agents.base import Agent
from legion.errors.exceptions import ToolError
from legion.memory.providers.memory import ConversationMemory

import sys
# Load environment variables
load_dotenv()

# Test schemas
class PersonInfo(BaseModel):
    name: str
    age: int
    occupation: str

class SimpleToolParams(BaseModel):
    message: str

class SimpleTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="simple_tool",
            description="A simple test tool",
            parameters=SimpleToolParams
        )
    
    def run(self, message: str) -> str:
        """Implement the sync run method"""
        return f"Tool response: {message}"
    
    async def arun(self, message: str) -> str:
        """Implement the async run method"""
        return self.run(message)

@pytest.fixture
def agent():
    return Agent(
        name="test_agent",
        model="gpt-4o-mini",
        temperature=0,
        system_prompt="You are a helpful assistant."
    )

@pytest.fixture
def agent_with_tools(agent):
    agent.tools = [SimpleTool()]
    return agent

def test_agent_initialization():
    # Test basic initialization
    agent = Agent(name="test", model="gpt-4o-mini")
    assert agent.name == "test"
    assert agent.model == "gpt-4o-mini"
    assert agent._provider_name == "openai"
    assert agent.temperature == 0.7  # default value
    assert agent.max_tokens is None
    assert isinstance(agent.system_prompt, SystemPrompt)
    assert len(agent.memory.messages) == 1  # system prompt message
    assert agent.memory.messages[0].role == Role.SYSTEM
    assert agent.memory.messages[0].content == ""  # Initially empty

    # Test with provider prefix
    agent = Agent(name="test", model="openai:gpt-4o-mini")
    assert agent._provider_name == "openai"
    assert agent.model == "gpt-4o-mini"

    # Test with custom system prompt
    custom_prompt = "Custom system prompt"
    agent = Agent(name="test", model="gpt-4o-mini", system_prompt=custom_prompt)
    assert agent.system_prompt.render() == custom_prompt
    assert agent.memory.messages[0].role == Role.SYSTEM
    assert agent.memory.messages[0].content == ""  # Initially empty

    # Test with SystemPrompt object
    system_prompt = SystemPrompt(sections=[
        SystemPromptSection(content="Section 1"),
        SystemPromptSection(content="Section 2")
    ])
    agent = Agent(name="test", model="gpt-4o-mini", system_prompt=system_prompt)
    assert agent.system_prompt == system_prompt
    assert agent.memory.messages[0].content == ""  # Initially empty

def test_agent_properties(agent):
    # Test full_model_name property
    assert agent.full_model_name == "openai:gpt-4o-mini"

    # Test tools property
    assert isinstance(agent.tools, list)
    assert len(agent.tools) == 0

    # Test tools setter
    tool = SimpleTool()
    agent.tools = [tool]
    assert len(agent.tools) == 1
    assert agent.tools[0] == tool

def test_message_creation(agent):
    # Test string message
    msg = agent._create_message("Hello")
    assert isinstance(msg, Message)
    assert msg.role == Role.USER
    assert msg.content == "Hello"

    # Test Message object
    original = Message(role=Role.SYSTEM, content="Test")
    msg = agent._create_message(original)
    assert msg == original

    # Test dict message
    msg = agent._create_message({
        "role": "user",
        "content": "Test",
        "name": "test_user",
        "tool_call_id": "123",
        "tool_calls": [{"id": "1"}]
    })
    assert isinstance(msg, Message)
    assert msg.role == Role.USER
    assert msg.content == "Test"
    assert msg.name == "test_user"
    assert msg.tool_call_id == "123"
    assert msg.tool_calls == [{"id": "1"}]

    # Test invalid message format
    with pytest.raises(ValueError):
        agent._create_message(123)

def test_enhanced_prompt(agent_with_tools):
    # Test prompt enhancement with tools
    enhanced = agent_with_tools._build_enhanced_prompt()
    assert isinstance(enhanced, str)
    assert "simple_tool" in enhanced
    assert "A simple test tool" in enhanced

def test_basic_completion(agent):
    # Test basic message completion
    response = agent.process("Say 'Hello, World!'")
    assert isinstance(response, ModelResponse)
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    assert response.usage is not None
    assert response.tool_calls is None

    # Verify memory was updated
    assert len(agent.memory.messages) == 3  # system + user + assistant
    assert agent.memory.messages[-2].role == Role.USER
    assert agent.memory.messages[-1].role == Role.ASSISTANT

@pytest.mark.asyncio
async def test_async_completion(agent):
    # Test async message completion
    response = await agent.aprocess("Say 'Hello, World!'")
    assert isinstance(response, ModelResponse)
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    assert response.usage is not None
    assert response.tool_calls is None

def test_tool_completion(agent_with_tools):
    # Test completion with tool usage
    response = agent_with_tools.process("Use the simple tool to say hello")
    assert isinstance(response, ModelResponse)
    assert response.tool_calls is not None
    assert len(response.tool_calls) > 0
    assert response.tool_calls[0]["function"]["name"] == "simple_tool"

    # Verify tool response was added to memory
    tool_messages = [m for m in agent_with_tools.memory.messages if m.role == Role.TOOL]
    assert len(tool_messages) > 0
    assert "Tool response" in tool_messages[0].content

@pytest.mark.asyncio
async def test_async_tool_completion(agent_with_tools):
    # Test async completion with tool usage
    response = await agent_with_tools.aprocess("Use the simple tool to say hello")
    assert isinstance(response, ModelResponse)
    assert response.tool_calls is not None
    assert len(response.tool_calls) > 0
    assert response.tool_calls[0]["function"]["name"] == "simple_tool"

def test_json_completion(agent):
    # Test completion with JSON schema
    response = agent.process(
        "Give me information about a person named John who is 30 and works as a developer",
        response_schema=PersonInfo
    )
    assert isinstance(response, ModelResponse)
    data = PersonInfo.model_validate_json(response.content)
    assert isinstance(data.name, str)
    assert isinstance(data.age, int)
    assert isinstance(data.occupation, str)

def test_tool_and_json_completion(agent_with_tools):
    # Test completion with both tool usage and JSON schema
    response = agent_with_tools.process(
        "Use the simple tool to say hello, then format the response as a person's info",
        response_schema=PersonInfo
    )
    assert isinstance(response, ModelResponse)
    assert response.tool_calls is not None
    assert len(response.tool_calls) > 0
    
    # Verify JSON response
    data = PersonInfo.model_validate_json(response.content)
    assert isinstance(data.name, str)
    assert isinstance(data.age, int)
    assert isinstance(data.occupation, str)

def test_memory_management(agent):
    # Test memory initialization
    assert len(agent.memory.messages) == 1  # system prompt
    
    # Test message addition
    agent.process("Hello")
    assert len(agent.memory.messages) == 3  # system + user + assistant
    
    # Test memory reset by creating new memory instance
    agent._memory = ConversationMemory()
    agent._memory.add_message(Message(
        role=Role.SYSTEM,
        content=agent.system_prompt.render()
    ))
    assert len(agent.memory.messages) == 1  # only system prompt remains
    assert agent.memory.messages[0].role == Role.SYSTEM

@pytest.mark.asyncio
async def test_memory_provider_integration(agent):
    # This would require mocking a memory provider
    # For now, verify that memory provider properties work
    assert agent.memory_provider is None
    assert isinstance(agent._memory, ConversationMemory)

def test_debug_mode():
    # Test agent with debug mode enabled
    agent = Agent(name="test", model="gpt-4o-mini", debug=True)
    assert agent.debug is True
    
    # Process a message (output will be visible in test logs)
    response = agent.process("Hello")
    assert isinstance(response, ModelResponse)

if __name__ == "__main__":
    # Configure pytest arguments
    args = [
        __file__,
        "-v",
        "-p", "no:warnings",
        "--tb=short",
        "--asyncio-mode=auto"
    ]
    
    # Run tests
    sys.exit(pytest.main(args))