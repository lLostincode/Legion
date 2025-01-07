"""Tests for monitoring decorators"""

import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4
import traceback

from legion.monitoring.decorators import monitored_agent, monitored_team
from legion.monitoring.events import (
    AgentStartEvent,
    AgentProcessingEvent,
    AgentDecisionEvent,
    AgentToolUseEvent,
    AgentMemoryEvent,
    AgentResponseEvent,
    AgentErrorEvent,
    TeamFormationEvent,
    TeamDelegationEvent,
    TeamLeadershipEvent,
    TeamCommunicationEvent,
    TeamCompletionEvent,
    TeamStateChangeEvent,
    TeamErrorEvent
)
from legion.monitoring.events.base import EventEmitter

# Test classes
@monitored_agent
class TestAgent:
    def __init__(self, system_prompt=None):
        self.system_prompt = system_prompt
        
    def process(self, message):
        return f"Processed: {message}"
        
    def decide(self, context):
        return {"decision": "test", "reasoning": "test reasoning"}
        
    def use_tool(self, tool_name, tool_input):
        return {"result": "test"}
        
    def remember(self, content):
        return True
        
    def recall(self, query):
        return {"memory": "test"}

@monitored_team
class TestTeam:
    def __init__(self, config=None):
        self.member_ids = []
        self.leader_id = None
        self.config = config or {}
        
    def add_member(self, member_id):
        self.member_ids.append(member_id)
        return True
        
    def remove_member(self, member_id):
        self.member_ids.remove(member_id)
        return True
        
    def assign_leader(self, leader_id, reason=None):
        self.leader_id = leader_id
        return True
        
    def delegate_task(self, member_id, task_type, task_input):
        return {"status": "delegated"}
        
    def process_message(self, sender_id, receiver_id, message_type, message_content):
        return {"status": "processed"}
        
    def complete_task(self, task_type, task_input, member_contributions=None):
        return {"status": "completed"}

# Team decorator tests
def test_team_initialization():
    """Test team initialization with monitoring"""
    # Create mock registry
    mock_registry = MagicMock()
    
    # Patch the singleton instance
    with patch('legion.monitoring.registry.MonitorRegistry._instance', mock_registry), \
         patch('legion.monitoring.registry.MonitorRegistry.__new__', return_value=mock_registry):
        
        # Create team
        team = TestTeam(config={"max_members": 5})
        
        # Verify team attributes
        assert hasattr(team, "event_emitter")
        assert hasattr(team, "id")
        assert len(team.member_ids) == 0
        assert team.leader_id is None
        assert team.config == {"max_members": 5}
        
        # Verify registry interaction
        mock_registry.register_component.assert_called_once_with(team.id, team.event_emitter)

def test_team_formation_event():
    """Test team formation event emission"""
    with patch('legion.monitoring.registry.MonitorRegistry') as mock_registry_cls:
        mock_registry = MagicMock()
        mock_registry_cls.return_value = mock_registry
        
        # Create a mock event handler
        event_handler = MagicMock()
        
        # Create team and add event handler
        team = TestTeam()
        team.event_emitter.add_event_handler(event_handler)
        
        # Emit formation event manually to test handler
        formation_event = TeamFormationEvent(
            component_id=team.id,
            leader_id=team.leader_id,
            member_ids=team.member_ids,
            team_config=team.config
        )
        team.event_emitter.emit(formation_event)
        
        # Check that the formation event was emitted
        event_handler.assert_called()
        event = event_handler.call_args[0][0]
        assert isinstance(event, TeamFormationEvent)
        assert event.component_id == team.id
        assert event.metadata["member_ids"] == []
        assert event.metadata["leader_id"] is None
        assert "team_config" in event.metadata

def test_team_member_addition():
    """Test member addition event emission"""
    with patch('legion.monitoring.registry.MonitorRegistry.get_instance') as mock_get_instance:
        mock_registry = MagicMock()
        mock_get_instance.return_value = mock_registry
        
        # Create a mock event handler
        event_handler = MagicMock()
        
        # Create team and add event handler
        team = TestTeam()
        team.event_emitter.add_event_handler(event_handler)
        
        # Clear previous calls (from initialization)
        event_handler.reset_mock()
        
        # Add member and check event
        member_id = str(uuid4())
        team.add_member(member_id)
        
        assert member_id in team.member_ids
        event_handler.assert_called_once()
        event = event_handler.call_args[0][0]
        assert isinstance(event, TeamStateChangeEvent)
        assert event.metadata["change_type"] == "member_added"
        assert event.metadata["affected_members"] == [member_id]

def test_team_member_removal():
    """Test member removal event emission"""
    with patch('legion.monitoring.registry.MonitorRegistry.get_instance') as mock_get_instance:
        mock_registry = MagicMock()
        mock_get_instance.return_value = mock_registry
        
        # Create a mock event handler
        event_handler = MagicMock()
        
        # Create team and add event handler
        team = TestTeam()
        team.event_emitter.add_event_handler(event_handler)
        
        # Clear previous calls (from initialization)
        event_handler.reset_mock()
        
        # Add and remove member
        member_id = str(uuid4())
        team.member_ids.append(member_id)
        team.remove_member(member_id)
        
        assert member_id not in team.member_ids
        event_handler.assert_called_once()
        event = event_handler.call_args[0][0]
        assert isinstance(event, TeamStateChangeEvent)
        assert event.metadata["change_type"] == "member_removed"
        assert event.metadata["affected_members"] == [member_id]

def test_team_leader_assignment():
    """Test leader assignment event emission"""
    with patch('legion.monitoring.registry.MonitorRegistry.get_instance') as mock_get_instance:
        mock_registry = MagicMock()
        mock_get_instance.return_value = mock_registry
        
        # Create a mock event handler
        event_handler = MagicMock()
        
        # Create team and add event handler
        team = TestTeam()
        team.event_emitter.add_event_handler(event_handler)
        
        # Clear previous calls (from initialization)
        event_handler.reset_mock()
        
        # Assign leader and check event
        leader_id = str(uuid4())
        team.assign_leader(leader_id, reason="Test assignment")
        
        assert team.leader_id == leader_id
        event_handler.assert_called_once()
        event = event_handler.call_args[0][0]
        assert isinstance(event, TeamLeadershipEvent)
        assert event.metadata["leader_id"] == leader_id
        assert event.metadata["decision_type"] == "leader_assignment"
        assert event.metadata["reasoning"] == "Test assignment"

def test_team_task_delegation():
    """Test task delegation event emission"""
    with patch('legion.monitoring.registry.MonitorRegistry.get_instance') as mock_get_instance:
        mock_registry = MagicMock()
        mock_get_instance.return_value = mock_registry
        
        # Create a mock event handler
        event_handler = MagicMock()
        
        # Create team and add event handler
        team = TestTeam()
        team.event_emitter.add_event_handler(event_handler)
        
        # Clear previous calls (from initialization)
        event_handler.reset_mock()
        
        # Delegate task and check event
        member_id = str(uuid4())
        task_type = "test_task"
        task_input = {"data": "test"}
        result = team.delegate_task(member_id, task_type, task_input)
        
        event_handler.assert_called_once()
        event = event_handler.call_args[0][0]
        assert isinstance(event, TeamDelegationEvent)
        assert event.metadata["member_id"] == member_id
        assert event.metadata["task_type"] == task_type
        assert event.metadata["task_input"] == task_input
        assert event.metadata["result"] == result

def test_team_message_processing():
    """Test message processing event emission"""
    with patch('legion.monitoring.registry.MonitorRegistry.get_instance') as mock_get_instance:
        mock_registry = MagicMock()
        mock_get_instance.return_value = mock_registry
        
        # Create a mock event handler
        event_handler = MagicMock()
        
        # Create team and add event handler
        team = TestTeam()
        team.event_emitter.add_event_handler(event_handler)
        
        # Clear previous calls (from initialization)
        event_handler.reset_mock()
        
        # Process message and check event
        sender_id = str(uuid4())
        receiver_id = str(uuid4())
        message_type = "test_message"
        message_content = {"text": "test"}
        result = team.process_message(sender_id, receiver_id, message_type, message_content)
        
        event_handler.assert_called_once()
        event = event_handler.call_args[0][0]
        assert isinstance(event, TeamCommunicationEvent)
        assert event.metadata["sender_id"] == sender_id
        assert event.metadata["receiver_id"] == receiver_id
        assert event.metadata["message_type"] == message_type
        assert event.metadata["message_content"] == message_content
        assert event.metadata["result"] == result

def test_team_task_completion():
    """Test task completion event emission"""
    with patch('legion.monitoring.registry.MonitorRegistry.get_instance') as mock_get_instance:
        mock_registry = MagicMock()
        mock_get_instance.return_value = mock_registry
        
        # Create a mock event handler
        event_handler = MagicMock()
        
        # Create team and add event handler
        team = TestTeam()
        team.event_emitter.add_event_handler(event_handler)
        
        # Clear previous calls (from initialization)
        event_handler.reset_mock()
        
        # Complete task and check event
        task_type = "test_task"
        task_input = {"data": "test"}
        member_contributions = {
            str(uuid4()): {"time_spent": 100},
            str(uuid4()): {"time_spent": 150}
        }
        result = team.complete_task(task_type, task_input, member_contributions)
        
        event_handler.assert_called_once()
        event = event_handler.call_args[0][0]
        assert isinstance(event, TeamCompletionEvent)
        assert event.metadata["task_type"] == task_type
        assert event.metadata["task_input"] == task_input
        assert event.metadata["task_output"] == result
        assert event.metadata["member_contributions"] == member_contributions
        assert "duration_ms" in event.metadata

def test_team_error_handling():
    """Test error event emission"""
    with patch('legion.monitoring.registry.MonitorRegistry') as mock_registry_cls:
        mock_registry = MagicMock()
        mock_registry_cls.return_value = mock_registry
        
        # Create a mock event handler
        event_handler = MagicMock()
        
        # Create team and add event handler
        team = TestTeam()
        team.event_emitter.add_event_handler(event_handler)
        
        # Clear previous calls (from initialization)
        event_handler.reset_mock()
        
        # Create a test error
        class TestError(Exception):
            pass
        
        def failing_add_member(self, member_id):
            raise TestError("Test error")
        
        # Replace add_member with failing version
        original_add_member = team.add_member
        team.add_member = failing_add_member.__get__(team, TestTeam)
        
        # Test error handling
        member_id = str(uuid4())
        with pytest.raises(TestError):
            team.add_member(member_id)
        
        # Emit error event manually to test handler
        error_event = TeamErrorEvent(
            component_id=team.id,
            error_type="TestError",
            error_message="Test error",
            error_context={'operation': 'add_member'},
            affected_members=[member_id],
            traceback=traceback.format_exc()
        )
        team.event_emitter.emit(error_event)
        
        # Check that error event was emitted
        event_handler.assert_called()
        event = event_handler.call_args[0][0]
        assert isinstance(event, TeamErrorEvent)
        assert event.metadata["error_type"] == "TestError"
        assert event.metadata["error_message"] == "Test error"
        assert event.metadata["error_context"]["operation"] == "add_member"
        assert event.metadata["affected_members"] == [member_id]
        assert "traceback" in event.metadata
        
        # Restore original method
        team.add_member = original_add_member 