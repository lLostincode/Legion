"""Tests for storage backend base class"""


import pytest

from legion.monitoring.events.base import Event, EventCategory, EventType
from legion.monitoring.storage.base import StorageBackend


def test_storage_backend_is_abstract():
    """Test that StorageBackend cannot be instantiated directly"""
    with pytest.raises(TypeError):
        StorageBackend()

def test_concrete_implementation_required():
    """Test that concrete implementations must implement all abstract methods"""

    class IncompleteStorage(StorageBackend):
        """Storage implementation missing required methods"""

        pass

    with pytest.raises(TypeError):
        IncompleteStorage()

def test_minimal_implementation():
    """Test that a complete implementation can be instantiated"""

    class MinimalStorage(StorageBackend):
        """Minimal storage implementation"""

        def store_event(self, event):
            pass

        def get_events(self, event_types=None, start_time=None, end_time=None):
            return []

        def clear(self):
            pass

        def cleanup(self, retention_days):
            pass

    # Should not raise
    storage = MinimalStorage()

    # Test method signatures
    event = Event(
        event_type=EventType.AGENT,
        component_id="test",
        category=EventCategory.EXECUTION
    )

    storage.store_event(event)
    assert storage.get_events() == []
    storage.clear()
    storage.cleanup(7)
