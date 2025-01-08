"""Tests for agent-specific events"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from legion.monitoring.events.agent import (
    AgentDecisionEvent,
    AgentErrorEvent,
    AgentEvent,
    AgentMemoryEvent,
    AgentProcessingEvent,
    AgentResponseEvent,
    AgentStartEvent,
    AgentStateChangeEvent,
    AgentToolUseEvent,
)
from legion.monitoring.events.base import Event


def test_agent_start_event():
    """Test agent start event creation"""
    event = AgentStartEvent(
        component_id="test_agent",
        system_prompt="Test prompt",
        provider_name="openai",
        model_name="gpt-4o-mini"
    )

    assert event.provider_name == "openai"
    assert event.model_name == "gpt-4o-mini"
    assert event.metadata["system_prompt"] == "Test prompt"

def test_agent_processing_event():
    """Test agent processing event creation"""
    event = AgentProcessingEvent(
        component_id="test_agent",
        input_message="Test input",
        tokens_used=100,
        cost=0.002,
        provider_name="openai",
        model_name="gpt-4o-mini"
    )

    assert event.tokens_used == 100
    assert event.cost == 0.002
    assert event.provider_name == "openai"
    assert event.model_name == "gpt-4o-mini"
    assert event.metadata["input_message"] == "Test input"

def test_agent_decision_event():
    """Test agent decision event creation"""
    event = AgentDecisionEvent(
        component_id="test_agent",
        decision_type="tool_selection",
        options=["tool1", "tool2"],
        selected_option="tool1",
        tokens_used=50,
        cost=0.001,
        provider_name="openai",
        model_name="gpt-4o-mini"
    )

    assert event.tokens_used == 50
    assert event.cost == 0.001
    assert event.provider_name == "openai"
    assert event.model_name == "gpt-4o-mini"
    assert event.metadata["decision_type"] == "tool_selection"
    assert event.metadata["options"] == ["tool1", "tool2"]
    assert event.metadata["selected_option"] == "tool1"

def test_agent_tool_use_event():
    """Test agent tool use event creation"""
    event = AgentToolUseEvent(
        component_id="test_agent",
        tool_name="test_tool",
        tool_input={"param": "value"},
        tool_output={"result": "success"},
        tokens_used=75,
        cost=0.0015,
        provider_name="openai",
        model_name="gpt-4o-mini"
    )

    assert event.tokens_used == 75
    assert event.cost == 0.0015
    assert event.provider_name == "openai"
    assert event.model_name == "gpt-4o-mini"
    assert event.metadata["tool_name"] == "test_tool"
    assert event.metadata["tool_input"] == {"param": "value"}
    assert event.metadata["tool_output"] == {"result": "success"}

def test_agent_memory_event():
    """Test agent memory event creation"""
    event = AgentMemoryEvent(
        component_id="test_agent",
        operation="store",
        memory_key="test_key",
        content={"data": "test"},
        tokens_used=30,
        cost=0.0006,
        provider_name="openai",
        model_name="gpt-4o-mini"
    )

    assert event.tokens_used == 30
    assert event.cost == 0.0006
    assert event.provider_name == "openai"
    assert event.model_name == "gpt-4o-mini"
    assert event.metadata["operation"] == "store"
    assert event.metadata["memory_key"] == "test_key"
    assert event.metadata["content"] == {"data": "test"}

def test_agent_response_event():
    """Test agent response event creation"""
    event = AgentResponseEvent(
        component_id="test_agent",
        response="Test response",
        tokens_used=150,
        cost=0.003,
        provider_name="openai",
        model_name="gpt-4o-mini"
    )

    assert event.tokens_used == 150
    assert event.cost == 0.003
    assert event.provider_name == "openai"
    assert event.model_name == "gpt-4o-mini"
    assert event.metadata["response"] == "Test response"

def test_agent_error_event():
    """Test agent error event creation"""
    event = AgentErrorEvent(
        component_id="test_agent",
        error_type="ValueError",
        error_message="Test error",
        traceback="test traceback",
        tokens_used=25,
        cost=0.0005,
        provider_name="openai",
        model_name="gpt-4o-mini"
    )

    assert event.tokens_used == 25
    assert event.cost == 0.0005
    assert event.provider_name == "openai"
    assert event.model_name == "gpt-4o-mini"
    assert event.metadata["error_type"] == "ValueError"
    assert event.metadata["error_message"] == "Test error"
    assert event.metadata["traceback"] == "test traceback"

def test_agent_state_change_event():
    """Test agent state change event creation"""
    old_state = {"value": 1}
    new_state = {"value": 2}

    event = AgentStateChangeEvent(
        component_id="test_agent",
        old_state=old_state,
        new_state=new_state,
        change_type="value_update",
        tokens_used=40,
        cost=0.0008,
        provider_name="openai",
        model_name="gpt-4o-mini"
    )

    assert event.tokens_used == 40
    assert event.cost == 0.0008
    assert event.provider_name == "openai"
    assert event.model_name == "gpt-4o-mini"
    assert event.state_before == old_state
    assert event.state_after == new_state
    assert event.metadata["change_type"] == "value_update"

def test_event_inheritance():
    """Test event inheritance chain and common attributes"""
    event = AgentStartEvent(
        component_id="test_agent",
        system_prompt="Test prompt"
    )

    assert isinstance(event, AgentEvent)
    assert isinstance(event, Event)
    assert hasattr(event, "id")
    assert hasattr(event, "timestamp")
    assert hasattr(event, "metadata")

def test_event_timing():
    """Test event timing and duration tracking"""
    start_time = datetime.now(timezone.utc)
    event = AgentProcessingEvent(
        component_id="test_agent",
        input_message="Test",
        duration_ms=100.0
    )

    assert event.timestamp >= start_time
    assert event.duration_ms == 100.0

def test_event_parent_child_relationship():
    """Test parent-child relationship between events"""
    parent_id = uuid4()
    event = AgentResponseEvent(
        component_id="test_agent",
        response="Test",
        parent_event_id=parent_id
    )

    assert event.parent_event_id == parent_id
    assert parent_id in event.trace_path

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
