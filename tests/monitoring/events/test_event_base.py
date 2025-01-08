import os
import platform
import sys
import threading
import time
from datetime import datetime
from uuid import UUID

import pytest

from legion.monitoring.events.base import (
    Event,
    EventCategory,
    EventEmitter,
    EventSeverity,
    EventType,
)


def test_event_creation():
    """Test basic event creation and field initialization"""
    event = Event(
        event_type=EventType.AGENT,
        component_id="test_agent",
        category=EventCategory.EXECUTION
    )

    assert isinstance(event.id, UUID)
    assert isinstance(event.timestamp, datetime)
    assert event.event_type == EventType.AGENT
    assert event.component_id == "test_agent"
    assert event.category == EventCategory.EXECUTION
    assert event.severity == EventSeverity.INFO  # Default severity
    assert event.root_event_id == event.id  # Root event is self for new events
    assert not event.parent_event_id
    assert not event.trace_path
    assert isinstance(event.metadata, dict)
    assert event.metadata == {}

def test_resource_utilization_fields():
    """Test resource utilization tracking fields"""
    event = Event(
        event_type=EventType.AGENT,
        component_id="test_agent",
        category=EventCategory.EXECUTION,
        tokens_used=100,
        cost=0.002,
        provider_name="openai",
        model_name="gpt-4o-mini"
    )

    assert event.tokens_used == 100
    assert event.cost == 0.002
    assert event.provider_name == "openai"
    assert event.model_name == "gpt-4o-mini"

def test_system_metrics_fields():
    """Test system metrics tracking fields"""
    event = Event(
        event_type=EventType.AGENT,
        component_id="test_agent",
        category=EventCategory.EXECUTION,
        system_cpu_percent=45.5,
        system_memory_percent=60.0,
        system_disk_usage_bytes=1024*1024,
        system_network_bytes_sent=512,
        system_network_bytes_received=1024
    )

    assert event.system_cpu_percent == 45.5
    assert event.system_memory_percent == 60.0
    assert event.system_disk_usage_bytes == 1024*1024
    assert event.system_network_bytes_sent == 512
    assert event.system_network_bytes_received == 1024

def test_execution_context_fields():
    """Test execution context tracking fields"""
    event = Event(
        event_type=EventType.AGENT,
        component_id="test_agent",
        category=EventCategory.EXECUTION,
        thread_id=threading.get_ident(),
        process_id=os.getpid(),
        host_name=platform.node(),
        python_version=sys.version
    )

    assert isinstance(event.thread_id, int)
    assert isinstance(event.process_id, int)
    assert isinstance(event.host_name, str)
    assert isinstance(event.python_version, str)

def test_dependencies_and_relationships():
    """Test dependencies and relationships tracking"""
    event = Event(
        event_type=EventType.AGENT,
        component_id="test_agent",
        category=EventCategory.EXECUTION,
        dependencies=["dep1", "dep2"],
        related_components=["comp1", "comp2"]
    )

    assert event.dependencies == ["dep1", "dep2"]
    assert event.related_components == ["comp1", "comp2"]

def test_state_tracking():
    """Test state tracking fields"""
    event = Event(
        event_type=EventType.AGENT,
        component_id="test_agent",
        category=EventCategory.EXECUTION,
        state_before={"value": 1},
        state_after={"value": 2},
        state_diff={"value": {"old": 1, "new": 2}}
    )

    assert event.state_before == {"value": 1}
    assert event.state_after == {"value": 2}
    assert event.state_diff == {"value": {"old": 1, "new": 2}}

def test_event_inheritance():
    """Test event inheritance and trace path construction"""
    parent_event = Event(
        event_type=EventType.TEAM,
        component_id="test_team",
        category=EventCategory.EXECUTION
    )

    child_event = Event(
        event_type=EventType.AGENT,
        component_id="test_agent",
        category=EventCategory.EXECUTION,
        parent_event_id=parent_event.id
    )

    assert child_event.parent_event_id == parent_event.id
    assert child_event.root_event_id == parent_event.id
    assert child_event.trace_path == [parent_event.id]


class TestEventEmitter:
    """Test suite for EventEmitter functionality"""

    class MockComponent(EventEmitter):
        """Mock component for testing event emission"""

        def __init__(self):
            super().__init__()
            self.component_id = "test_component"

        def do_something(self):
            """Test method that emits an event"""
            with self.event_span(
                event_type=EventType.SYSTEM,
                component_id=self.component_id,
                category=EventCategory.EXECUTION
            ):
                time.sleep(0.1)  # Simulate some work

    def test_event_handler_registration(self):
        """Test adding and removing event handlers"""
        emitter = self.MockComponent()
        events = []

        def handler(event):
            events.append(event)

        # Add handler
        emitter.add_event_handler(handler)
        assert handler in emitter._event_handlers

        # Remove handler
        emitter.remove_event_handler(handler)
        assert handler not in emitter._event_handlers

    def test_event_emission(self):
        """Test event emission and handling"""
        emitter = self.MockComponent()
        events = []

        def handler(event):
            events.append(event)

        emitter.add_event_handler(handler)
        emitter.do_something()

        assert len(events) == 1
        event = events[0]
        assert event.event_type == EventType.SYSTEM
        assert event.component_id == "test_component"
        assert event.category == EventCategory.EXECUTION
        assert event.duration_ms is not None
        assert event.duration_ms >= 100  # At least 100ms due to sleep

    def test_event_span_nesting(self):
        """Test nested event spans"""
        emitter = self.MockComponent()
        events = []

        def handler(event):
            events.append(event)

        emitter.add_event_handler(handler)

        with emitter.event_span(
            event_type=EventType.SYSTEM,
            component_id="parent",
            category=EventCategory.EXECUTION
        ):
            with emitter.event_span(
                event_type=EventType.SYSTEM,
                component_id="child",
                category=EventCategory.EXECUTION
            ):
                pass

        assert len(events) == 2
        child_event, parent_event = events  # Events are emitted in reverse order

        assert child_event.parent_event_id == parent_event.id
        assert child_event.root_event_id == parent_event.id
        assert child_event.trace_path == [parent_event.id]

    def test_monitoring_toggle(self):
        """Test enabling/disabling monitoring"""
        emitter = self.MockComponent()
        events = []

        def handler(event):
            events.append(event)

        emitter.add_event_handler(handler)

        # Disable monitoring
        emitter.disable_monitoring()
        emitter.do_something()
        assert len(events) == 0

        # Enable monitoring
        emitter.enable_monitoring()
        emitter.do_something()
        assert len(events) == 1

    def test_error_handling(self):
        """Test error handling in event handlers"""
        emitter = self.MockComponent()
        events = []

        def good_handler(event):
            events.append(event)

        def bad_handler(event):
            raise ValueError("Test error")

        emitter.add_event_handler(good_handler)
        emitter.add_event_handler(bad_handler)

        # Should not raise exception
        emitter.do_something()

        # Good handler should still receive event
        assert len(events) == 1

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
