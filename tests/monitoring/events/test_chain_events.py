"""Tests for chain-specific event types"""

from datetime import datetime, timezone
from uuid import uuid4

from legion.monitoring.events.base import Event, EventCategory, EventSeverity, EventType
from legion.monitoring.events.chain import (
    ChainBottleneckEvent,
    ChainCompletionEvent,
    ChainErrorEvent,
    ChainEvent,
    ChainStartEvent,
    ChainStateChangeEvent,
    ChainStepEvent,
    ChainTransformEvent,
)


def test_base_chain_event():
    """Test base ChainEvent initialization"""
    event = ChainEvent(
        component_id="test_chain",
        category=EventCategory.EXECUTION
    )

    assert event.event_type == EventType.CHAIN
    assert event.component_id == "test_chain"
    assert event.category == EventCategory.EXECUTION
    assert event.severity == EventSeverity.INFO
    assert event.id is not None
    assert event.timestamp is not None
    assert isinstance(event.metadata, dict)

def test_chain_start_event():
    """Test ChainStartEvent initialization and metadata"""
    event = ChainStartEvent(
        component_id="test_chain",
        input_message="Test input",
        member_count=3
    )

    assert event.event_type == EventType.CHAIN
    assert event.category == EventCategory.EXECUTION
    assert event.metadata["input_message"] == "Test input"
    assert event.metadata["member_count"] == 3

def test_chain_step_event():
    """Test ChainStepEvent initialization and metadata"""
    event = ChainStepEvent(
        component_id="test_chain",
        step_name="step_1",
        step_index=0,
        input_message="Test input",
        is_final_step=False
    )

    assert event.event_type == EventType.CHAIN
    assert event.category == EventCategory.EXECUTION
    assert event.metadata["step_name"] == "step_1"
    assert event.metadata["step_index"] == 0
    assert event.metadata["input_message"] == "Test input"
    assert event.metadata["is_final_step"] is False

def test_chain_transform_event():
    """Test ChainTransformEvent initialization and metadata"""
    event = ChainTransformEvent(
        component_id="test_chain",
        step_name="step_1",
        step_index=0,
        input_message="Test input",
        output_message="Test output",
        transformation_time_ms=100.0
    )

    assert event.event_type == EventType.CHAIN
    assert event.category == EventCategory.EXECUTION
    assert event.metadata["step_name"] == "step_1"
    assert event.metadata["step_index"] == 0
    assert event.metadata["input_message"] == "Test input"
    assert event.metadata["output_message"] == "Test output"
    assert event.metadata["transformation_time_ms"] == 100.0

def test_chain_completion_event():
    """Test ChainCompletionEvent initialization and metadata"""
    step_times = {
        "step_1": 100.0,
        "step_2": 200.0,
        "step_3": 150.0
    }

    event = ChainCompletionEvent(
        component_id="test_chain",
        input_message="Test input",
        output_message="Test output",
        total_time_ms=450.0,
        step_times=step_times
    )

    assert event.event_type == EventType.CHAIN
    assert event.category == EventCategory.EXECUTION
    assert event.metadata["input_message"] == "Test input"
    assert event.metadata["output_message"] == "Test output"
    assert event.metadata["total_time_ms"] == 450.0
    assert event.metadata["step_times"] == step_times

def test_chain_error_event():
    """Test ChainErrorEvent initialization and metadata"""
    event = ChainErrorEvent(
        component_id="test_chain",
        error_type="ValueError",
        error_message="Test error message",
        step_name="step_1",
        step_index=0,
        traceback="Test traceback"
    )

    assert event.event_type == EventType.CHAIN
    assert event.category == EventCategory.ERROR
    assert event.severity == EventSeverity.ERROR
    assert event.metadata["error_type"] == "ValueError"
    assert event.metadata["error_message"] == "Test error message"
    assert event.metadata["step_name"] == "step_1"
    assert event.metadata["step_index"] == 0
    assert event.metadata["traceback"] == "Test traceback"

def test_chain_state_change_event():
    """Test ChainStateChangeEvent initialization and metadata"""
    old_state = {"member_count": 2}
    new_state = {"member_count": 3}

    event = ChainStateChangeEvent(
        component_id="test_chain",
        change_type="member_added",
        old_state=old_state,
        new_state=new_state,
        change_reason="Added new member"
    )

    assert event.event_type == EventType.CHAIN
    assert event.category == EventCategory.EXECUTION
    assert event.metadata["change_type"] == "member_added"
    assert event.metadata["old_state"] == old_state
    assert event.metadata["new_state"] == new_state
    assert event.metadata["change_reason"] == "Added new member"

def test_chain_bottleneck_event():
    """Test ChainBottleneckEvent initialization and metadata"""
    event = ChainBottleneckEvent(
        component_id="test_chain",
        step_name="step_1",
        step_index=0,
        processing_time_ms=1000.0,
        average_time_ms=200.0,
        threshold_ms=500.0
    )

    assert event.event_type == EventType.CHAIN
    assert event.category == EventCategory.EXECUTION
    assert event.severity == EventSeverity.WARNING
    assert event.metadata["step_name"] == "step_1"
    assert event.metadata["step_index"] == 0
    assert event.metadata["processing_time_ms"] == 1000.0
    assert event.metadata["average_time_ms"] == 200.0
    assert event.metadata["threshold_ms"] == 500.0
    assert event.metadata["slowdown_factor"] == 5.0  # 1000/200

def test_event_inheritance():
    """Test event inheritance chain and common attributes"""
    event = ChainStartEvent(
        component_id="test_chain",
        input_message="Test input",
        member_count=3
    )

    assert isinstance(event, ChainEvent)
    assert isinstance(event, Event)
    assert hasattr(event, "id")
    assert hasattr(event, "timestamp")
    assert hasattr(event, "metadata")

def test_event_timing():
    """Test event timing and duration tracking"""
    start_time = datetime.now(timezone.utc)
    event = ChainTransformEvent(
        component_id="test_chain",
        step_name="step_1",
        step_index=0,
        input_message="Test input",
        output_message="Test output",
        transformation_time_ms=100.0
    )

    assert event.timestamp >= start_time
    assert event.metadata["transformation_time_ms"] == 100.0

def test_event_parent_child_relationship():
    """Test parent-child relationship between events"""
    parent_id = uuid4()
    event = ChainStepEvent(
        component_id="test_chain",
        step_name="step_1",
        step_index=0,
        input_message="Test input",
        is_final_step=False,
        parent_event_id=parent_id
    )

    assert event.parent_event_id == parent_id
    assert parent_id in event.trace_path
