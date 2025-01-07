"""Tests for the monitor registry"""

import pytest
from datetime import datetime, timedelta, timezone
import logging
import tempfile
from pathlib import Path

from legion.monitoring.registry import MonitorRegistry, EventRegistry
from legion.monitoring.monitors import Monitor, MonitorConfig
from legion.monitoring.events.base import Event, EventType, EventCategory, EventEmitter
from legion.monitoring.storage import StorageType, StorageConfig

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestMonitor(Monitor):
    """Test monitor implementation"""
    def __init__(self, config=None):
        super().__init__(config)
        self.processed_events = []
        
    def _process_event_impl(self, event):
        logger.debug(f"Processing event in TestMonitor: {event}")
        self.processed_events.append(event)
        
    def should_process_event(self, event: Event) -> bool:
        result = super().should_process_event(event)
        logger.debug(f"Should process event? {result} for {event}")
        return result
        
    def get_stats(self) -> dict:
        """Get monitor statistics"""
        return {
            "event_count": len(self.processed_events)
        }

class TestEventEmitter(EventEmitter):
    """Test event emitter implementation"""
    def __init__(self):
        super().__init__()
        
    def emit_event(self, event: Event):
        logger.debug(f"Emitting event: {event}")
        logger.debug(f"Handlers: {self._event_handlers}")
        super().emit_event(event)

class TestEventRegistry:
    """Test suite for EventRegistry"""
    
    @pytest.fixture
    def registry(self):
        """Create a fresh event registry for each test"""
        return EventRegistry()
        
    @pytest.fixture
    def sample_event(self):
        """Create a sample event for testing"""
        return Event(
            event_type=EventType.AGENT,
            component_id="test",
            category=EventCategory.EXECUTION
        )
        
    def test_memory_storage(self, registry, sample_event):
        """Test memory storage backend"""
        registry.record_event(sample_event)
        events = registry.get_events()
        assert len(events) == 1
        assert events[0].id == sample_event.id
        
    def test_sqlite_storage(self, tmp_path, sample_event):
        """Test SQLite storage backend"""
        db_path = tmp_path / "test.db"
        registry = EventRegistry(
            storage_type=StorageType.SQLITE,
            db_path=str(db_path)
        )
        
        registry.record_event(sample_event)
        events = registry.get_events()
        assert len(events) == 1
        assert events[0].id == sample_event.id
        
        # Test persistence
        registry2 = EventRegistry(
            storage_type=StorageType.SQLITE,
            db_path=str(db_path)
        )
        events = registry2.get_events()
        assert len(events) == 1
        assert events[0].id == sample_event.id
        
    def test_storage_config(self, tmp_path):
        """Test storage configuration"""
        config = StorageConfig(
            retention_days=1,
            max_events=10
        )
        
        registry = EventRegistry(
            storage_type=StorageType.MEMORY,
            storage_config=config
        )
        
        # Add more events than max
        for i in range(20):
            event = Event(
                event_type=EventType.AGENT,
                component_id=f"test_{i}",
                category=EventCategory.EXECUTION
            )
            registry.record_event(event)
            
        events = registry.get_events()
        assert len(events) == 10
        assert events[-1].component_id == "test_19"
        
    def test_event_filtering(self, registry):
        """Test event filtering"""
        # Create events with different types
        agent_event = Event(
            event_type=EventType.AGENT,
            component_id="test",
            category=EventCategory.EXECUTION
        )
        
        system_event = Event(
            event_type=EventType.SYSTEM,
            component_id="test",
            category=EventCategory.EXECUTION
        )
        
        registry.record_event(agent_event)
        registry.record_event(system_event)
        
        # Filter by type
        agent_events = registry.get_events(event_types=[agent_event])
        assert len(agent_events) == 1
        assert agent_events[0].id == agent_event.id
        
        # Filter by time
        now = datetime.now(timezone.utc)
        past = now - timedelta(hours=1)
        future = now + timedelta(hours=1)
        
        events = registry.get_events(start_time=past, end_time=future)
        assert len(events) == 2
        
        events = registry.get_events(start_time=future)
        assert len(events) == 0

class TestMonitorRegistry:
    """Test suite for MonitorRegistry"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Reset registry before each test"""
        MonitorRegistry.reset()
        
    def test_singleton(self):
        """Test registry is a singleton"""
        registry1 = MonitorRegistry()
        registry2 = MonitorRegistry()
        assert registry1 is registry2
        
    def test_monitor_registration(self):
        """Test monitor registration and retrieval"""
        registry = MonitorRegistry()
        
        # Register a monitor
        monitor = registry.register_monitor("test", TestMonitor)
        assert isinstance(monitor, TestMonitor)
        assert registry.get_monitor("test") is monitor
        assert "test" in registry.list_monitors()
        
        # Try registering duplicate
        with pytest.raises(ValueError):
            registry.register_monitor("test", TestMonitor)
            
        # Try getting non-existent monitor
        with pytest.raises(KeyError):
            registry.get_monitor("nonexistent")
            
    def test_monitor_unregistration(self):
        """Test monitor unregistration"""
        registry = MonitorRegistry()
        
        # Register and unregister
        monitor = registry.register_monitor("test", TestMonitor)
        assert "test" in registry.list_monitors()
        
        registry.unregister_monitor("test")
        assert "test" not in registry.list_monitors()
        assert not monitor.is_active
        
        # Try unregistering non-existent monitor
        with pytest.raises(KeyError):
            registry.unregister_monitor("nonexistent")
            
    def test_monitor_configuration(self):
        """Test monitor configuration"""
        registry = MonitorRegistry()
        
        config = MonitorConfig(
            event_types={EventType.AGENT},
            categories={EventCategory.EXECUTION}
        )
        
        monitor = registry.register_monitor("test", TestMonitor, config)
        assert monitor.config == config
        
    def test_emitter_registration(self):
        """Test event emitter registration"""
        registry = MonitorRegistry()
        emitter = TestEventEmitter()
        
        # Register monitor with configuration
        config = MonitorConfig(
            event_types={EventType.AGENT},
            categories={EventCategory.EXECUTION}
        )
        monitor = registry.register_monitor("test", TestMonitor, config)
        monitor.start()
        
        logger.debug("Registering emitter")
        registry.register_emitter(emitter)
        
        # Emit event
        event = Event(
            event_type=EventType.AGENT,
            component_id="test",
            category=EventCategory.EXECUTION
        )
        logger.debug("Emitting event")
        emitter.emit_event(event)
        
        assert len(monitor.processed_events) == 1
        assert monitor.processed_events[0] is event
        
        # Unregister emitter
        registry.unregister_emitter(emitter)
        emitter.emit_event(event)
        assert len(monitor.processed_events) == 1  # No new events
        
    def test_monitor_lifecycle(self):
        """Test monitor start/stop functionality"""
        registry = MonitorRegistry()
        
        # Register monitors
        monitor1 = registry.register_monitor("test1", TestMonitor)
        monitor2 = registry.register_monitor("test2", TestMonitor)
        
        # Start all
        registry.start_all()
        assert monitor1.is_active
        assert monitor2.is_active
        
        # Stop all
        registry.stop_all()
        assert not monitor1.is_active
        assert not monitor2.is_active
        
    def test_event_routing(self):
        """Test event routing to monitors"""
        registry = MonitorRegistry()
        
        # Register monitors with different configs
        config1 = MonitorConfig(event_types={EventType.AGENT})
        config2 = MonitorConfig(event_types={EventType.SYSTEM})
        
        monitor1 = registry.register_monitor("test1", TestMonitor, config1)
        monitor2 = registry.register_monitor("test2", TestMonitor, config2)
        
        monitor1.start()
        monitor2.start()
        
        # Create events
        agent_event = Event(
            event_type=EventType.AGENT,
            component_id="test",
            category=EventCategory.EXECUTION
        )
        
        system_event = Event(
            event_type=EventType.SYSTEM,
            component_id="test",
            category=EventCategory.EXECUTION
        )
        
        # Route events
        registry._route_event(agent_event)
        registry._route_event(system_event)
        
        # Check routing
        assert len(monitor1.processed_events) == 1
        assert monitor1.processed_events[0] is agent_event
        
        assert len(monitor2.processed_events) == 1
        assert monitor2.processed_events[0] is system_event
        
        # Check event storage
        events = registry._event_registry.get_events()
        assert len(events) == 2
        
    def test_error_handling(self):
        """Test error handling during event routing"""
        registry = MonitorRegistry()
        
        # Create monitor that raises an error
        class ErrorMonitor(Monitor):
            def _process_event_impl(self, event):
                raise ValueError("Test error")
                
        monitor = registry.register_monitor("test", ErrorMonitor)
        monitor.start()
        
        # Route event - should not raise
        event = Event(
            event_type=EventType.AGENT,
            component_id="test",
            category=EventCategory.EXECUTION
        )
        registry._route_event(event)
        
        assert monitor._error_count == 1
        
        # Event should still be stored
        events = registry._event_registry.get_events()
        assert len(events) == 1
        assert events[0].id == event.id
        
    def test_stats(self):
        """Test monitor statistics"""
        registry = MonitorRegistry()
        
        # Register monitors
        monitor1 = registry.register_monitor("test1", TestMonitor)
        monitor2 = registry.register_monitor("test2", TestMonitor)
        
        monitor1.start()
        monitor2.start()
        
        # Process some events
        event = Event(
            event_type=EventType.AGENT,
            component_id="test",
            category=EventCategory.EXECUTION
        )
        
        registry._route_event(event)
        registry._route_event(event)
        
        # Check stats
        stats = registry.get_stats()
        assert "test1" in stats
        assert "test2" in stats
        
        assert stats["test1"]["event_count"] == 2
        assert stats["test2"]["event_count"] == 2 