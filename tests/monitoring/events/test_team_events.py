"""Tests for team-specific event types"""

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from legion.monitoring.events.base import Event, EventType, EventCategory, EventSeverity
from legion.monitoring.events.team import (
    TeamEvent,
    TeamFormationEvent,
    TeamDelegationEvent,
    TeamLeadershipEvent,
    TeamCommunicationEvent,
    TeamCompletionEvent,
    TeamPerformanceEvent,
    TeamStateChangeEvent,
    TeamErrorEvent
)

def test_base_team_event():
    """Test base TeamEvent initialization"""
    event = TeamEvent(
        component_id="test_team",
        category=EventCategory.EXECUTION
    )
    
    assert event.event_type == EventType.TEAM
    assert event.component_id == "test_team"
    assert event.category == EventCategory.EXECUTION
    assert event.severity == EventSeverity.INFO
    assert event.id is not None
    assert event.timestamp is not None
    assert isinstance(event.metadata, dict)

def test_team_formation_event():
    """Test TeamFormationEvent initialization and metadata"""
    leader_id = str(uuid4())
    member_ids = [str(uuid4()) for _ in range(3)]
    team_config = {"max_members": 5, "specialization": "test"}
    
    event = TeamFormationEvent(
        component_id="test_team",
        leader_id=leader_id,
        member_ids=member_ids,
        team_config=team_config
    )
    
    assert event.event_type == EventType.TEAM
    assert event.category == EventCategory.EXECUTION
    assert event.metadata["leader_id"] == leader_id
    assert event.metadata["member_ids"] == member_ids
    assert event.metadata["team_config"] == team_config

def test_team_delegation_event():
    """Test TeamDelegationEvent initialization and metadata"""
    leader_id = str(uuid4())
    member_id = str(uuid4())
    task_input = {"data": "test"}
    delegation_context = {"priority": "high"}
    
    event = TeamDelegationEvent(
        component_id="test_team",
        leader_id=leader_id,
        member_id=member_id,
        task_type="analysis",
        task_input=task_input,
        delegation_context=delegation_context
    )
    
    assert event.event_type == EventType.TEAM
    assert event.category == EventCategory.EXECUTION
    assert event.metadata["leader_id"] == leader_id
    assert event.metadata["member_id"] == member_id
    assert event.metadata["task_type"] == "analysis"
    assert event.metadata["task_input"] == task_input
    assert event.metadata["delegation_context"] == delegation_context

def test_team_leadership_event():
    """Test TeamLeadershipEvent initialization and metadata"""
    leader_id = str(uuid4())
    affected_members = [str(uuid4()) for _ in range(2)]
    decision_context = {"workload": "high"}
    
    event = TeamLeadershipEvent(
        component_id="test_team",
        leader_id=leader_id,
        decision_type="resource_allocation",
        decision_context=decision_context,
        reasoning="Optimize team performance",
        affected_members=affected_members
    )
    
    assert event.event_type == EventType.TEAM
    assert event.category == EventCategory.EXECUTION
    assert event.metadata["leader_id"] == leader_id
    assert event.metadata["decision_type"] == "resource_allocation"
    assert event.metadata["decision_context"] == decision_context
    assert event.metadata["reasoning"] == "Optimize team performance"
    assert event.metadata["affected_members"] == affected_members

def test_team_communication_event():
    """Test TeamCommunicationEvent initialization and metadata"""
    sender_id = str(uuid4())
    receiver_id = str(uuid4())
    message_content = {"status": "in_progress"}
    context = {"thread_id": "123"}
    
    event = TeamCommunicationEvent(
        component_id="test_team",
        sender_id=sender_id,
        receiver_id=receiver_id,
        message_type="status_update",
        message_content=message_content,
        context=context
    )
    
    assert event.event_type == EventType.TEAM
    assert event.category == EventCategory.EXECUTION
    assert event.metadata["sender_id"] == sender_id
    assert event.metadata["receiver_id"] == receiver_id
    assert event.metadata["message_type"] == "status_update"
    assert event.metadata["message_content"] == message_content
    assert event.metadata["context"] == context

def test_team_completion_event():
    """Test TeamCompletionEvent initialization and metadata"""
    task_input = {"data": "test"}
    task_output = {"result": "success"}
    member_contributions = {
        str(uuid4()): {"time_spent": 100},
        str(uuid4()): {"time_spent": 150}
    }
    
    event = TeamCompletionEvent(
        component_id="test_team",
        task_type="analysis",
        task_input=task_input,
        task_output=task_output,
        duration_ms=250.0,
        member_contributions=member_contributions
    )
    
    assert event.event_type == EventType.TEAM
    assert event.category == EventCategory.EXECUTION
    assert event.metadata["task_type"] == "analysis"
    assert event.metadata["task_input"] == task_input
    assert event.metadata["task_output"] == task_output
    assert event.metadata["duration_ms"] == 250.0
    assert event.metadata["member_contributions"] == member_contributions

def test_team_performance_event():
    """Test TeamPerformanceEvent initialization and metadata"""
    metric_context = {"window_start": "2023-01-01T00:00:00Z"}
    
    event = TeamPerformanceEvent(
        component_id="test_team",
        metric_type="delegation_success_rate",
        metric_value=0.95,
        metric_context=metric_context,
        time_window_ms=3600000.0
    )
    
    assert event.event_type == EventType.TEAM
    assert event.category == EventCategory.EXECUTION
    assert event.metadata["metric_type"] == "delegation_success_rate"
    assert event.metadata["metric_value"] == 0.95
    assert event.metadata["metric_context"] == metric_context
    assert event.metadata["time_window_ms"] == 3600000.0

def test_team_state_change_event():
    """Test TeamStateChangeEvent initialization and metadata"""
    old_state = {"members": ["id1", "id2"]}
    new_state = {"members": ["id1", "id2", "id3"]}
    affected_members = ["id3"]
    
    event = TeamStateChangeEvent(
        component_id="test_team",
        change_type="member_added",
        old_state=old_state,
        new_state=new_state,
        change_reason="Team expansion",
        affected_members=affected_members
    )
    
    assert event.event_type == EventType.TEAM
    assert event.category == EventCategory.EXECUTION
    assert event.metadata["change_type"] == "member_added"
    assert event.metadata["old_state"] == old_state
    assert event.metadata["new_state"] == new_state
    assert event.metadata["change_reason"] == "Team expansion"
    assert event.metadata["affected_members"] == affected_members

def test_team_error_event():
    """Test TeamErrorEvent initialization and metadata"""
    error_context = {"operation": "delegation"}
    affected_members = [str(uuid4()), str(uuid4())]
    
    event = TeamErrorEvent(
        component_id="test_team",
        error_type="DelegationError",
        error_message="Member unavailable",
        error_context=error_context,
        affected_members=affected_members,
        traceback="Test traceback"
    )
    
    assert event.event_type == EventType.TEAM
    assert event.category == EventCategory.ERROR
    assert event.severity == EventSeverity.ERROR
    assert event.metadata["error_type"] == "DelegationError"
    assert event.metadata["error_message"] == "Member unavailable"
    assert event.metadata["error_context"] == error_context
    assert event.metadata["affected_members"] == affected_members
    assert event.metadata["traceback"] == "Test traceback"

def test_event_inheritance():
    """Test event inheritance chain and common attributes"""
    event = TeamFormationEvent(
        component_id="test_team",
        leader_id=str(uuid4()),
        member_ids=[str(uuid4())],
        team_config={}
    )
    
    assert isinstance(event, TeamEvent)
    assert isinstance(event, Event)
    assert hasattr(event, "id")
    assert hasattr(event, "timestamp")
    assert hasattr(event, "metadata")

def test_event_timing():
    """Test event timing and duration tracking"""
    start_time = datetime.now(timezone.utc)
    event = TeamCompletionEvent(
        component_id="test_team",
        task_type="test",
        task_input={},
        task_output={},
        duration_ms=100.0,
        member_contributions={}
    )
    
    assert event.timestamp >= start_time
    assert event.metadata["duration_ms"] == 100.0

def test_event_parent_child_relationship():
    """Test parent-child relationship between events"""
    parent_id = uuid4()
    event = TeamDelegationEvent(
        component_id="test_team",
        leader_id=str(uuid4()),
        member_id=str(uuid4()),
        task_type="test",
        task_input={},
        delegation_context={},
        parent_event_id=parent_id
    )
    
    assert event.parent_event_id == parent_id
    assert parent_id in event.trace_path 