"""Tests for storage configuration"""

import pytest
from pydantic import ValidationError

from legion.monitoring.storage.config import StorageConfig


def test_default_config():
    """Test default configuration values"""
    config = StorageConfig()
    assert config.retention_days == 30
    assert config.cleanup_interval == 60
    assert config.max_events is None

def test_custom_config():
    """Test custom configuration values"""
    config = StorageConfig(
        retention_days=7,
        cleanup_interval=30,
        max_events=1000
    )
    assert config.retention_days == 7
    assert config.cleanup_interval == 30
    assert config.max_events == 1000

def test_validation():
    """Test configuration validation"""
    # Test invalid retention days
    with pytest.raises(ValidationError):
        StorageConfig(retention_days=0)

    with pytest.raises(ValidationError):
        StorageConfig(retention_days=-1)

    # Test invalid cleanup interval
    with pytest.raises(ValidationError):
        StorageConfig(cleanup_interval=0)

    with pytest.raises(ValidationError):
        StorageConfig(cleanup_interval=-1)

    # Test invalid max events
    with pytest.raises(ValidationError):
        StorageConfig(max_events=0)

    with pytest.raises(ValidationError):
        StorageConfig(max_events=-1)

def test_model_dump():
    """Test configuration serialization"""
    config = StorageConfig(
        retention_days=7,
        cleanup_interval=30,
        max_events=1000
    )

    data = config.model_dump()
    assert data == {
        "retention_days": 7,
        "cleanup_interval": 30,
        "max_events": 1000
    }
