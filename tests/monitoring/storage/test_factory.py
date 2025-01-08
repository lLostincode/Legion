"""Tests for storage factory"""

from typing import Optional

import pytest

from legion.monitoring.storage.base import StorageBackend
from legion.monitoring.storage.config import StorageConfig
from legion.monitoring.storage.factory import StorageFactory, StorageType
from legion.monitoring.storage.memory import MemoryStorageBackend
from legion.monitoring.storage.sqlite import SQLiteStorageBackend


def test_create_memory_backend():
    """Test creating memory backend"""
    backend = StorageFactory.create(StorageType.MEMORY)
    assert isinstance(backend, MemoryStorageBackend)

def test_create_sqlite_backend(tmp_path):
    """Test creating SQLite backend"""
    backend = StorageFactory.create(
        StorageType.SQLITE,
        db_path=str(tmp_path / "test.db")
    )
    assert isinstance(backend, SQLiteStorageBackend)

def test_create_with_config():
    """Test creating backend with configuration"""
    config = StorageConfig(retention_days=7)
    backend = StorageFactory.create(StorageType.MEMORY, config=config)
    assert backend._config == config

def test_create_invalid_type():
    """Test creating backend with invalid type"""
    with pytest.raises(ValueError):
        StorageFactory.create("invalid")

def test_register_custom_backend():
    """Test registering custom backend"""
    class CustomBackend(StorageBackend):
        def __init__(self, config: Optional[StorageConfig] = None):
            pass

        def store_event(self, event):
            pass

        def get_events(self, event_types=None, start_time=None, end_time=None):
            return []

        def clear(self):
            pass

        def cleanup(self, retention_days=None):
            pass

    StorageFactory.register_backend("custom", CustomBackend)

    # Create instance of custom backend
    backend = StorageFactory.create("custom")
    assert isinstance(backend, CustomBackend)

def test_register_duplicate_backend():
    """Test registering duplicate backend type"""
    class DuplicateBackend(StorageBackend):
        def __init__(self, config: Optional[StorageConfig] = None):
            pass

        def store_event(self, event):
            pass

        def get_events(self, event_types=None, start_time=None, end_time=None):
            return []

        def clear(self):
            pass

        def cleanup(self, retention_days=None):
            pass

    with pytest.raises(ValueError):
        StorageFactory.register_backend(StorageType.MEMORY.value, DuplicateBackend)

def test_backend_kwargs():
    """Test passing additional arguments to backend"""
    class CustomBackend(StorageBackend):
        def __init__(self, config: Optional[StorageConfig] = None, custom_arg: str = None):
            self.custom_arg = custom_arg

        def store_event(self, event):
            pass

        def get_events(self, event_types=None, start_time=None, end_time=None):
            return []

        def clear(self):
            pass

        def cleanup(self, retention_days=None):
            pass

    StorageFactory.register_backend("custom_with_args", CustomBackend)

    backend = StorageFactory.create(
        "custom_with_args",
        custom_arg="test"
    )
    assert backend.custom_arg == "test"
