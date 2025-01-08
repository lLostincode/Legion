import re

import pytest

from legion.monitoring.events.base import Event, EventCategory, EventSeverity, EventType
from legion.monitoring.monitors import Monitor, MonitorConfig


class TestMonitor:
    """Test suite for Monitor base class"""

    def test_default_config(self):
        """Test monitor initialization with default config"""
        monitor = Monitor()
        assert not monitor.is_active
        assert monitor._event_count == 0
        assert monitor._error_count == 0

        # Check default config values
        assert monitor.config.event_types == set(EventType)
        assert monitor.config.categories == set(EventCategory)
        assert monitor.config.min_severity == EventSeverity.DEBUG
        assert monitor.config.sample_rate == 1.0
        assert not monitor.config.component_patterns
        assert not monitor.config.excluded_component_patterns

    def test_custom_config(self):
        """Test monitor initialization with custom config"""
        config = MonitorConfig(
            event_types={EventType.AGENT},
            categories={EventCategory.EXECUTION},
            min_severity=EventSeverity.WARNING,
            component_patterns={re.compile(r"agent-.*")},
            excluded_component_patterns={re.compile(r".*-test")},
            sample_rate=0.5
        )

        monitor = Monitor(config)
        assert monitor.config == config

    def test_event_type_filtering(self):
        """Test event filtering by type"""
        config = MonitorConfig(event_types={EventType.AGENT})
        monitor = Monitor(config)

        # Should process agent events
        event = Event(
            event_type=EventType.AGENT,
            component_id="test",
            category=EventCategory.EXECUTION
        )
        assert monitor.should_process_event(event)

        # Should not process system events
        event = Event(
            event_type=EventType.SYSTEM,
            component_id="test",
            category=EventCategory.EXECUTION
        )
        assert not monitor.should_process_event(event)

    def test_category_filtering(self):
        """Test event filtering by category"""
        config = MonitorConfig(categories={EventCategory.EXECUTION})
        monitor = Monitor(config)

        # Should process execution events
        event = Event(
            event_type=EventType.AGENT,
            component_id="test",
            category=EventCategory.EXECUTION
        )
        assert monitor.should_process_event(event)

        # Should not process memory events
        event = Event(
            event_type=EventType.AGENT,
            component_id="test",
            category=EventCategory.MEMORY
        )
        assert not monitor.should_process_event(event)

    def test_severity_filtering(self):
        """Test event filtering by severity"""
        config = MonitorConfig(min_severity=EventSeverity.WARNING)
        monitor = Monitor(config)

        # Should process error events
        event = Event(
            event_type=EventType.AGENT,
            component_id="test",
            category=EventCategory.EXECUTION,
            severity=EventSeverity.ERROR
        )
        assert monitor.should_process_event(event)

        # Should not process info events
        event = Event(
            event_type=EventType.AGENT,
            component_id="test",
            category=EventCategory.EXECUTION,
            severity=EventSeverity.INFO
        )
        assert not monitor.should_process_event(event)

    def test_component_pattern_filtering(self):
        """Test event filtering by component patterns"""
        config = MonitorConfig(
            component_patterns={re.compile(r"agent-.*")},
            excluded_component_patterns={re.compile(r".*-test")}
        )
        monitor = Monitor(config)

        # Should process matching components
        event = Event(
            event_type=EventType.AGENT,
            component_id="agent-1",
            category=EventCategory.EXECUTION
        )
        assert monitor.should_process_event(event)

        # Should not process non-matching components
        event = Event(
            event_type=EventType.AGENT,
            component_id="system-1",
            category=EventCategory.EXECUTION
        )
        assert not monitor.should_process_event(event)

        # Should not process excluded components
        event = Event(
            event_type=EventType.AGENT,
            component_id="agent-test",
            category=EventCategory.EXECUTION
        )
        assert not monitor.should_process_event(event)

    def test_monitor_lifecycle(self):
        """Test monitor start/stop functionality"""
        monitor = Monitor()
        assert not monitor.is_active

        monitor.start()
        assert monitor.is_active

        monitor.stop()
        assert not monitor.is_active

    def test_event_processing(self):
        """Test event processing flow"""
        class TestMonitor(Monitor):
            def __init__(self):
                super().__init__()
                self.processed_events = []

            def _process_event_impl(self, event):
                self.processed_events.append(event)

        monitor = TestMonitor()
        monitor.start()

        # Should process event when active
        event = Event(
            event_type=EventType.AGENT,
            component_id="test",
            category=EventCategory.EXECUTION
        )
        monitor.process_event(event)
        assert len(monitor.processed_events) == 1
        assert monitor._event_count == 1

        # Should not process when inactive
        monitor.stop()
        monitor.process_event(event)
        assert len(monitor.processed_events) == 1
        assert monitor._event_count == 1

    def test_error_handling(self):
        """Test error handling during event processing"""
        class ErrorMonitor(Monitor):
            def _process_event_impl(self, event):
                raise ValueError("Test error")

        monitor = ErrorMonitor()
        monitor.start()

        event = Event(
            event_type=EventType.AGENT,
            component_id="test",
            category=EventCategory.EXECUTION
        )

        with pytest.raises(ValueError):
            monitor.process_event(event)

        assert monitor._error_count == 1
        assert monitor._event_count == 0

    def test_stats(self):
        """Test monitor statistics"""
        monitor = Monitor()
        monitor.start()

        assert monitor.stats == {
            "is_active": True,
            "event_count": 0,
            "error_count": 0
        }

        # Process some events
        event = Event(
            event_type=EventType.AGENT,
            component_id="test",
            category=EventCategory.EXECUTION
        )

        class TestMonitor(Monitor):
            def _process_event_impl(self, event):
                pass

        monitor = TestMonitor()
        monitor.start()

        monitor.process_event(event)
        monitor.process_event(event)

        assert monitor.stats == {
            "is_active": True,
            "event_count": 2,
            "error_count": 0
        }
