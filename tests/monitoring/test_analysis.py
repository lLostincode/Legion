"""Tests for the analysis engine."""

import pytest
from datetime import datetime, timedelta, UTC
from typing import List
import statistics

from legion.monitoring.analysis import AnalysisPipeline, AnalysisPattern
from legion.monitoring.events.base import Event, EventType, EventCategory
from legion.monitoring.registry import EventRegistry

class MockEvent(Event):
    """Mock event for testing."""
    
    def __init__(self, **kwargs):
        """Initialize mock event with required fields."""
        super().__init__(
            event_type=EventType.SYSTEM,
            component_id="test_component",
            category=EventCategory.EXECUTION
        )
        for key, value in kwargs.items():
            setattr(self, key, value)

@pytest.fixture
def registry():
    """Create a test event registry."""
    return EventRegistry()

@pytest.fixture
def pipeline(registry):
    """Create a test analysis pipeline."""
    return AnalysisPipeline(registry)

def test_analysis_pattern_to_dict():
    """Test AnalysisPattern.to_dict()."""
    now = datetime.now(UTC)
    events = [MockEvent(event_id="test1"), MockEvent(event_id="test2")]
    pattern = AnalysisPattern(
        pattern_type="test",
        confidence=0.9,
        events=events,
        metadata={"key": "value"},
        detected_at=now
    )
    
    result = pattern.to_dict()
    assert result["pattern_type"] == "test"
    assert result["confidence"] == 0.9
    assert result["event_ids"] == ["test1", "test2"]
    assert result["metadata"] == {"key": "value"}
    assert result["detected_at"] == now.isoformat()

def test_analyze_events_empty_registry(pipeline):
    """Test analyzing events with empty registry."""
    patterns = pipeline.analyze_events()
    assert len(patterns) == 0

def test_analyze_duration_patterns(registry, pipeline):
    """Test duration pattern analysis."""
    # Create events with varying durations
    events = [
        MockEvent(duration_ms=100),
        MockEvent(duration_ms=110),
        MockEvent(duration_ms=90),
        MockEvent(duration_ms=500)  # Outlier
    ]
    
    for event in events:
        registry.record_event(event)
    
    patterns = pipeline.analyze_events()
    duration_patterns = [p for p in patterns if p.pattern_type == "long_running_operations"]
    
    assert len(duration_patterns) == 1
    pattern = duration_patterns[0]
    assert len(pattern.events) == 1
    assert pattern.events[0].duration_ms == 500
    assert pattern.confidence == 0.8
    assert "average_duration_ms" in pattern.metadata
    assert "max_duration_ms" in pattern.metadata
    assert "threshold_ms" in pattern.metadata

def test_analyze_error_patterns(registry, pipeline):
    """Test error pattern analysis."""
    class TestError(Exception):
        pass
    
    # Create events with errors
    events = [
        MockEvent(error=TestError("error1")),
        MockEvent(error=TestError("error2")),
        MockEvent(error=ValueError("different error"))
    ]
    
    for event in events:
        registry.record_event(event)
    
    patterns = pipeline.analyze_events()
    error_patterns = [p for p in patterns if p.pattern_type == "repeated_errors"]
    
    assert len(error_patterns) == 1
    pattern = error_patterns[0]
    assert len(pattern.events) == 2  # Only TestError events
    assert pattern.confidence == 0.9
    assert pattern.metadata["error_type"] == "TestError"
    assert pattern.metadata["count"] == 2

def test_analyze_resource_patterns(registry, pipeline):
    """Test resource pattern analysis."""
    # Create events with memory usage
    events = [
        MockEvent(memory_usage_bytes=1000),
        MockEvent(memory_usage_bytes=1100),
        MockEvent(memory_usage_bytes=900),
        MockEvent(memory_usage_bytes=2000)  # High usage
    ]
    
    for event in events:
        registry.record_event(event)
    
    patterns = pipeline.analyze_events()
    memory_patterns = [p for p in patterns if p.pattern_type == "high_memory_usage"]
    
    assert len(memory_patterns) == 1
    pattern = memory_patterns[0]
    assert len(pattern.events) == 1
    assert pattern.events[0].memory_usage_bytes == 2000
    assert pattern.confidence == 0.7
    assert "average_bytes" in pattern.metadata
    assert "max_bytes" in pattern.metadata
    assert "threshold_bytes" in pattern.metadata

def test_analyze_events_with_time_window(registry, pipeline):
    """Test analyzing events with time window."""
    now = datetime.now(UTC)
    old_event = MockEvent(duration_ms=100)
    old_event.timestamp = now - timedelta(hours=2)
    
    new_event = MockEvent(duration_ms=500)
    new_event.timestamp = now - timedelta(minutes=30)
    
    registry.record_event(old_event)
    registry.record_event(new_event)
    
    # Analyze last hour
    patterns = pipeline.analyze_events(time_window=timedelta(hours=1))
    
    assert len(patterns) == 1  # Only new_event should trigger pattern
    pattern = patterns[0]
    assert len(pattern.events) == 1
    assert pattern.events[0].duration_ms == 500

def test_analyze_events_with_event_types(registry, pipeline):
    """Test analyzing events with specific event types."""
    class TestEvent1(MockEvent):
        pass
        
    class TestEvent2(MockEvent):
        pass
    
    events = [
        TestEvent1(duration_ms=500),
        TestEvent2(duration_ms=100)
    ]
    
    for event in events:
        registry.record_event(event)
    
    # Only analyze TestEvent1
    patterns = pipeline.analyze_events(event_types=[TestEvent1])
    
    assert len(patterns) == 1
    pattern = patterns[0]
    assert len(pattern.events) == 1
    assert isinstance(pattern.events[0], TestEvent1)

def test_analyze_single_event_memory_threshold(registry, pipeline):
    """Test memory threshold for single events."""
    # Test event below threshold
    event1 = MockEvent(memory_usage_bytes=1024 * 1024 * 50)  # 50MB
    registry.record_event(event1)
    patterns = pipeline.analyze_events()
    assert len(patterns) == 0
    
    registry.clear()
    
    # Test event above threshold
    event2 = MockEvent(memory_usage_bytes=1024 * 1024 * 150)  # 150MB
    registry.record_event(event2)
    patterns = pipeline.analyze_events()
    assert len(patterns) == 1
    assert patterns[0].pattern_type == "high_memory_usage"
    assert patterns[0].confidence == 0.7

def test_analyze_mixed_patterns(registry, pipeline):
    """Test detection of multiple pattern types in the same event set."""
    events = [
        MockEvent(
            duration_ms=1000,  # Much higher duration to ensure pattern detection
            memory_usage_bytes=1024 * 1024 * 150,  # Above 100MB threshold
            error=ValueError("test error")
        ),
        MockEvent(duration_ms=100),
        MockEvent(memory_usage_bytes=1024 * 1024 * 50)
    ]
    
    for event in events:
        registry.record_event(event)
    
    patterns = pipeline.analyze_events()
    pattern_types = {p.pattern_type for p in patterns}
    
    assert "long_running_operations" in pattern_types
    assert "high_memory_usage" in pattern_types
    assert "error" in pattern_types
    
    # Verify pattern details
    for pattern in patterns:
        if pattern.pattern_type == "long_running_operations":
            assert pattern.events[0].duration_ms == 1000
        elif pattern.pattern_type == "high_memory_usage":
            assert pattern.events[0].memory_usage_bytes == 1024 * 1024 * 150
        elif pattern.pattern_type == "error":
            assert isinstance(pattern.events[0].error, ValueError)

def test_analyze_statistical_thresholds(registry, pipeline):
    """Test statistical threshold calculations."""
    # Create events with a clear statistical pattern
    base_memory = 1024 * 1024  # 1MB
    events = [
        MockEvent(memory_usage_bytes=base_memory * i) for i in range(1, 6)
    ]
    
    for event in events:
        registry.record_event(event)
    
    patterns = pipeline.analyze_events()
    memory_patterns = [p for p in patterns if p.pattern_type == "high_memory_usage"]
    
    assert len(memory_patterns) == 1
    pattern = memory_patterns[0]
    
    # Verify statistical calculations
    memory_values = [e.memory_usage_bytes for e in events]
    avg_memory = statistics.mean(memory_values)
    std_dev = statistics.stdev(memory_values)
    expected_threshold = min(avg_memory + (1.5 * std_dev), avg_memory * 1.5)
    
    assert pattern.metadata["average_bytes"] == avg_memory
    assert pattern.metadata["std_dev_bytes"] == std_dev
    assert pattern.metadata["threshold_bytes"] == expected_threshold
    assert all(e.memory_usage_bytes > expected_threshold for e in pattern.events)

def test_analyze_error_pattern_transitions(registry, pipeline):
    """Test transition between single and repeated error patterns."""
    class TestError(Exception):
        pass
    
    # Start with a single error
    event1 = MockEvent(error=TestError("error1"))
    registry.record_event(event1)
    
    patterns = pipeline.analyze_events()
    assert len(patterns) == 1
    assert patterns[0].pattern_type == "error"
    assert patterns[0].confidence == 0.7
    
    # Add another error of the same type
    event2 = MockEvent(error=TestError("error2"))
    registry.record_event(event2)
    
    patterns = pipeline.analyze_events()
    error_patterns = [p for p in patterns if p.pattern_type == "repeated_errors"]
    assert len(error_patterns) == 1
    assert error_patterns[0].confidence == 0.9
    assert len(error_patterns[0].events) == 2

def test_analyze_duration_edge_cases(registry, pipeline):
    """Test duration analysis edge cases."""
    # Test with very small durations
    events = [
        MockEvent(duration_ms=0.1),
        MockEvent(duration_ms=0.2),
        MockEvent(duration_ms=0.15),
        MockEvent(duration_ms=1.0)  # 10x larger
    ]
    
    for event in events:
        registry.record_event(event)
    
    patterns = pipeline.analyze_events()
    duration_patterns = [p for p in patterns if p.pattern_type == "long_running_operations"]
    
    assert len(duration_patterns) == 1
    assert len(duration_patterns[0].events) == 1
    assert duration_patterns[0].events[0].duration_ms == 1.0 