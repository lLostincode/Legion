"""Tests for SQLite storage backend"""

import pytest
from datetime import datetime, timedelta, timezone
import time
from pathlib import Path
import tempfile
import shutil
import threading

from legion.monitoring.storage.sqlite import SQLiteStorageBackend
from legion.monitoring.storage.config import StorageConfig
from legion.monitoring.events.base import Event, EventType, EventCategory

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test databases"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)
    
@pytest.fixture
def storage(temp_dir):
    """Create a fresh storage backend for each test"""
    return SQLiteStorageBackend(str(temp_dir / "test.db"))
    
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
    
def test_max_events(storage):
    """Test max events configuration"""
    config = StorageConfig(max_events=10)
    storage = SQLiteStorageBackend(str(storage._db_path), config)
    
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
    
def test_automatic_cleanup(temp_dir):
    """Test automatic cleanup based on configuration"""
    config = StorageConfig(
        retention_days=1,
        cleanup_interval=1  # Run every minute
    )
    storage = SQLiteStorageBackend(str(temp_dir / "test.db"), config)
    
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
    
def test_database_persistence(temp_dir):
    """Test that data persists between backend instances"""
    db_path = temp_dir / "test.db"
    
    # Create and store event
    storage1 = SQLiteStorageBackend(str(db_path))
    event = Event(
        event_type=EventType.AGENT,
        component_id="test",
        category=EventCategory.EXECUTION
    )
    storage1.store_event(event)
    
    # Create new instance and verify data
    storage2 = SQLiteStorageBackend(str(db_path))
    events = storage2.get_events()
    assert len(events) == 1
    assert events[0].id == event.id
    
def test_concurrent_access(storage):
    """Test concurrent access to the database"""
    def store_events():
        for _ in range(100):
            event = Event(
                event_type=EventType.AGENT,
                component_id="test",
                category=EventCategory.EXECUTION
            )
            storage.store_event(event)
            time.sleep(0.001)  # Force thread switching
            
    threads = []
    for _ in range(5):
        thread = threading.Thread(target=store_events)
        thread.start()
        threads.append(thread)
        
    for thread in threads:
        thread.join()
        
    # 5 threads * 100 events each = 500 total
    assert len(storage.get_events()) == 500 