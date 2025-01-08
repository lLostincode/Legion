"""Tests for memory storage backend"""

import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from legion.monitoring.events.base import Event, EventCategory, EventType
from legion.monitoring.storage.config import StorageConfig
from legion.monitoring.storage.memory import MemoryStorageBackend


@pytest.fixture
def storage():
    """Create a fresh storage backend for each test"""
    return MemoryStorageBackend()

@pytest.fixture
def sample_event():
    """Create a sample event for testing"""
    return Event(
        event_type=EventType.AGENT,
        component_id="test",
        category=EventCategory.EXECUTION
    )

def test_store_and_retrieve(storage, sample_event):
    """Test basic event storage and retrieval"""
    storage.store_event(sample_event)
    events = storage.get_events()
    assert len(events) == 1
    assert events[0].id == sample_event.id

def test_clear(storage, sample_event):
    """Test clearing all events"""
    storage.store_event(sample_event)
    storage.clear()
    assert len(storage.get_events()) == 0

def test_event_type_filtering(storage):
    """Test filtering events by type"""
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

    storage.store_event(agent_event)
    storage.store_event(system_event)

    agent_events = storage.get_events(event_types=[agent_event])
    assert len(agent_events) == 1
    assert agent_events[0].id == agent_event.id

def test_time_filtering(storage, sample_event):
    """Test filtering events by time range"""
    now = datetime.now(timezone.utc)
    past = now - timedelta(hours=1)
    future = now + timedelta(hours=1)

    storage.store_event(sample_event)

    # Events after past
    events = storage.get_events(start_time=past)
    assert len(events) == 1

    # Events before future
    events = storage.get_events(end_time=future)
    assert len(events) == 1

    # Events in range
    events = storage.get_events(start_time=past, end_time=future)
    assert len(events) == 1

    # Events before past
    events = storage.get_events(end_time=past)
    assert len(events) == 0

    # Events after future
    events = storage.get_events(start_time=future)
    assert len(events) == 0

def test_cleanup(storage):
    """Test cleanup of old events"""
    old_event = Event(
        event_type=EventType.AGENT,
        component_id="test",
        category=EventCategory.EXECUTION
    )
    # Manually set timestamp to simulate old event
    old_event.timestamp = datetime.now(timezone.utc) - timedelta(days=10)

    new_event = Event(
        event_type=EventType.AGENT,
        component_id="test",
        category=EventCategory.EXECUTION
    )

    storage.store_event(old_event)
    storage.store_event(new_event)

    # Cleanup events older than 5 days
    storage.cleanup(retention_days=5)

    events = storage.get_events()
    assert len(events) == 1
    assert events[0].id == new_event.id

def test_thread_safety(storage):
    """Test thread safety of storage operations"""
    def store_events():
        for _ in range(100):
            event = Event(
                event_type=EventType.AGENT,
                component_id="test",
                category=EventCategory.EXECUTION
            )
            storage.store_event(event)
            time.sleep(0.001)  # Force thread switching

    threads = [
        threading.Thread(target=store_events)
        for _ in range(5)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # 5 threads * 100 events each = 500 total
    assert len(storage.get_events()) == 500

def test_max_events(storage):
    """Test max events configuration"""
    config = StorageConfig(max_events=10)
    storage = MemoryStorageBackend(config)

    # Store more events than the limit
    for i in range(20):
        event = Event(
            event_type=EventType.AGENT,
            component_id=f"test_{i}",
            category=EventCategory.EXECUTION
        )
        storage.store_event(event)

    events = storage.get_events()
    assert len(events) == 10
    # Should keep most recent events
    assert events[-1].component_id == "test_19"
    assert events[0].component_id == "test_10"

def test_automatic_cleanup():
    """Test automatic cleanup based on configuration"""
    config = StorageConfig(
        retention_days=1,
        cleanup_interval=1  # Run every minute
    )
    storage = MemoryStorageBackend(config)

    # Create old event
    old_event = Event(
        event_type=EventType.AGENT,
        component_id="test_old",
        category=EventCategory.EXECUTION
    )
    old_event.timestamp = datetime.now(timezone.utc) - timedelta(days=2)

    # Create new event
    new_event = Event(
        event_type=EventType.AGENT,
        component_id="test_new",
        category=EventCategory.EXECUTION
    )

    storage.store_event(old_event)
    storage.store_event(new_event)

    # Perform immediate cleanup
    storage.cleanup(retention_days=1)

    events = storage.get_events()
    assert len(events) == 1
    assert events[0].id == new_event.id
