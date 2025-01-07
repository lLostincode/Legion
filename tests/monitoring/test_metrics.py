"""Tests for system metrics collection"""

import pytest
import time
import threading
import os
import platform
import sys
from unittest.mock import patch, MagicMock

from legion.monitoring.metrics import SystemMetricsCollector, MetricsContext
from legion.monitoring.events.base import Event, EventType, EventCategory

def test_execution_context():
    """Test execution context collection"""
    collector = SystemMetricsCollector()
    context = collector.get_execution_context()
    
    assert context["thread_id"] == threading.get_ident()
    assert context["process_id"] == os.getpid()
    assert context["host_name"] == platform.node()
    assert context["python_version"] == sys.version

def test_system_metrics():
    """Test system metrics collection"""
    collector = SystemMetricsCollector()
    
    # Sleep to ensure some system activity
    time.sleep(0.1)
    
    metrics = collector.get_system_metrics()
    
    assert "system_cpu_percent" in metrics
    assert "system_memory_percent" in metrics
    assert "system_disk_usage_bytes" in metrics
    assert "system_network_bytes_sent" in metrics
    assert "system_network_bytes_received" in metrics
    
    assert isinstance(metrics["system_cpu_percent"], float)
    assert isinstance(metrics["system_memory_percent"], float)
    assert isinstance(metrics["system_disk_usage_bytes"], int)
    assert isinstance(metrics["system_network_bytes_sent"], int)
    assert isinstance(metrics["system_network_bytes_received"], int)

def test_process_metrics():
    """Test process metrics collection"""
    collector = SystemMetricsCollector()
    metrics = collector.get_process_metrics()
    
    assert "memory_usage_bytes" in metrics
    assert "cpu_usage_percent" in metrics
    
    assert isinstance(metrics["memory_usage_bytes"], int)
    assert isinstance(metrics["cpu_usage_percent"], float)

def test_metrics_context():
    """Test metrics context manager"""
    event = Event(
        event_type=EventType.AGENT,
        component_id="test_agent",
        category=EventCategory.EXECUTION
    )
    
    with MetricsContext(event):
        # Simulate some work
        time.sleep(0.1)
    
    # Check that metrics were collected
    assert event.duration_ms >= 100  # At least 100ms
    assert event.thread_id == threading.get_ident()
    assert event.process_id == os.getpid()
    assert event.host_name == platform.node()
    assert event.python_version == sys.version
    assert isinstance(event.system_cpu_percent, float)
    assert isinstance(event.system_memory_percent, float)
    assert isinstance(event.system_disk_usage_bytes, int)
    assert isinstance(event.system_network_bytes_sent, int)
    assert isinstance(event.system_network_bytes_received, int)
    assert isinstance(event.memory_usage_bytes, int)
    assert isinstance(event.cpu_usage_percent, float)

def test_metrics_context_no_event():
    """Test metrics context manager with no event"""
    # Should not raise any exceptions
    with MetricsContext():
        time.sleep(0.1)

def test_metrics_context_exception():
    """Test metrics context manager with exception"""
    event = Event(
        event_type=EventType.AGENT,
        component_id="test_agent",
        category=EventCategory.EXECUTION
    )
    
    with pytest.raises(ValueError):
        with MetricsContext(event):
            raise ValueError("Test error")
    
    # Metrics should still be collected
    assert event.duration_ms is not None
    assert event.thread_id is not None
    assert event.process_id is not None
    assert event.host_name is not None
    assert event.python_version is not None
    assert event.system_cpu_percent is not None
    assert event.system_memory_percent is not None
    assert event.system_disk_usage_bytes is not None
    assert event.system_network_bytes_sent is not None
    assert event.system_network_bytes_received is not None
    assert event.memory_usage_bytes is not None
    assert event.cpu_usage_percent is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 