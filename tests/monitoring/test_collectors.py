"""Tests for collector implementations"""

import pytest
from datetime import datetime, timedelta, timezone
from legion.monitoring.collectors import MemoryCollector, MemoryCollectorConfig, EventQuery, MetricSummary
from legion.monitoring.events.base import Event, EventType, EventCategory, EventSeverity

def approx_equal(a: float, b: float, epsilon: float = 1e-10) -> bool:
    """Compare two floats with a small epsilon to handle floating point precision"""
    return abs(a - b) < epsilon

class TestMemoryCollector:
    """Test suite for MemoryCollector"""
    
    def test_default_config(self):
        """Test collector initialization with default config"""
        collector = MemoryCollector()
        assert collector.config.max_events == 10000
        assert collector.config.retention_period is None
        assert len(collector.get_events()) == 0
        
    def test_custom_config(self):
        """Test collector initialization with custom config"""
        config = MemoryCollectorConfig(
            max_events=100,
            retention_period=timedelta(hours=1)
        )
        collector = MemoryCollector(config)
        assert collector.config.max_events == 100
        assert collector.config.retention_period == timedelta(hours=1)
        
    def test_event_storage(self):
        """Test basic event storage functionality"""
        collector = MemoryCollector()
        collector.start()
        
        # Create and process events
        events = [
            Event(
                event_type=EventType.AGENT,
                component_id=f"test-{i}",
                category=EventCategory.EXECUTION,
                timestamp=datetime.now(timezone.utc)
            )
            for i in range(5)
        ]
        
        for event in events:
            collector.process_event(event)
            
        # Verify events are stored
        stored = collector.get_events()
        assert len(stored) == 5
        assert stored == list(reversed(events))  # Should be newest first
        
    def test_circular_buffer(self):
        """Test circular buffer behavior"""
        config = MemoryCollectorConfig(max_events=3)
        collector = MemoryCollector(config)
        collector.start()
        
        # Add more events than buffer size
        events = [
            Event(
                event_type=EventType.AGENT,
                component_id=f"test-{i}",
                category=EventCategory.EXECUTION,
                timestamp=datetime.now(timezone.utc)
            )
            for i in range(5)
        ]
        
        for event in events:
            collector.process_event(event)
            
        # Should only keep latest 3 events
        stored = collector.get_events()
        assert len(stored) == 3
        assert stored == list(reversed(events[-3:]))
        
    def test_retention_period(self):
        """Test event retention period"""
        config = MemoryCollectorConfig(
            retention_period=timedelta(seconds=1)
        )
        collector = MemoryCollector(config)
        collector.start()
        
        now = datetime.now(timezone.utc)
        
        # Add old event
        old_event = Event(
            event_type=EventType.AGENT,
            component_id="old",
            category=EventCategory.EXECUTION,
            timestamp=now - timedelta(seconds=2)
        )
        collector.process_event(old_event)
        
        # Add new event
        new_event = Event(
            event_type=EventType.AGENT,
            component_id="new",
            category=EventCategory.EXECUTION,
            timestamp=now
        )
        collector.process_event(new_event)
        
        # Should only have new event
        stored = collector.get_events()
        assert len(stored) == 1
        assert stored[0].component_id == "new"
        
    def test_event_limit(self):
        """Test event limit parameter"""
        collector = MemoryCollector()
        collector.start()
        
        # Add events
        events = [
            Event(
                event_type=EventType.AGENT,
                component_id=f"test-{i}",
                category=EventCategory.EXECUTION,
                timestamp=datetime.now(timezone.utc)
            )
            for i in range(5)
        ]
        
        for event in events:
            collector.process_event(event)
            
        # Get limited events
        limited = collector.get_events(limit=3)
        assert len(limited) == 3
        assert limited == list(reversed(events[-3:]))
        
    def test_collector_lifecycle(self):
        """Test collector start/stop behavior"""
        collector = MemoryCollector()
        
        # Should not process events when stopped
        event = Event(
            event_type=EventType.AGENT,
            component_id="test",
            category=EventCategory.EXECUTION,
            timestamp=datetime.now(timezone.utc)
        )
        collector.process_event(event)
        assert len(collector.get_events()) == 0
        
        # Should process events when started
        collector.start()
        collector.process_event(event)
        assert len(collector.get_events()) == 1
        
        # Should stop processing when stopped
        collector.stop()
        collector.process_event(event)
        assert len(collector.get_events()) == 1
        
    def test_stats(self):
        """Test collector statistics"""
        config = MemoryCollectorConfig(max_events=100)
        collector = MemoryCollector(config)
        collector.start()
        
        # Add some events
        for i in range(5):
            event = Event(
                event_type=EventType.AGENT,
                component_id=f"test-{i}",
                category=EventCategory.EXECUTION,
                timestamp=datetime.now(timezone.utc)
            )
            collector.process_event(event)
            
        stats = collector.stats
        assert stats["stored_events"] == 5
        assert stats["buffer_capacity"] == 100
        assert stats["buffer_usage"] == 0.05
        assert stats["uptime"] is not None
        
    def test_query_builder(self):
        """Test query builder functionality"""
        collector = MemoryCollector()
        collector.start()
        
        now = datetime.now(timezone.utc)
        
        # Add test events
        events = [
            # Agent events
            Event(
                event_type=EventType.AGENT,
                component_id="agent-1",
                category=EventCategory.EXECUTION,
                severity=EventSeverity.INFO,
                timestamp=now
            ),
            Event(
                event_type=EventType.AGENT,
                component_id="agent-2",
                category=EventCategory.MEMORY,
                severity=EventSeverity.WARNING,
                timestamp=now
            ),
            # System events
            Event(
                event_type=EventType.SYSTEM,
                component_id="system-1",
                category=EventCategory.EXECUTION,
                severity=EventSeverity.ERROR,
                timestamp=now
            )
        ]
        
        for event in events:
            collector.process_event(event)
            
        # Test type filtering
        query = collector.query().of_types(EventType.AGENT)
        results = collector.find_events(query)
        assert len(results) == 2
        assert all(e.event_type == EventType.AGENT for e in results)
        
        # Test category filtering
        query = collector.query().in_categories(EventCategory.EXECUTION)
        results = collector.find_events(query)
        assert len(results) == 2
        assert all(e.category == EventCategory.EXECUTION for e in results)
        
        # Test severity filtering
        query = collector.query().min_severity_level(EventSeverity.WARNING)
        results = collector.find_events(query)
        assert len(results) == 2
        assert all(e.severity in (EventSeverity.WARNING, EventSeverity.ERROR) for e in results)
        
        # Test component pattern filtering
        query = collector.query().matching_components(r"agent-.*")
        results = collector.find_events(query)
        assert len(results) == 2
        assert all(e.component_id.startswith("agent-") for e in results)
        
        # Test combined filters
        query = (collector.query()
                .of_types(EventType.AGENT)
                .in_categories(EventCategory.EXECUTION)
                .min_severity_level(EventSeverity.INFO)
                .matching_components(r"agent-.*"))
        results = collector.find_events(query)
        assert len(results) == 1
        assert results[0].component_id == "agent-1"
        
    def test_time_range_query(self):
        """Test time range filtering"""
        collector = MemoryCollector()
        collector.start()
        
        now = datetime.now(timezone.utc)
        
        # Add events with different timestamps
        events = [
            Event(
                event_type=EventType.AGENT,
                component_id="old",
                category=EventCategory.EXECUTION,
                timestamp=now - timedelta(hours=2)
            ),
            Event(
                event_type=EventType.AGENT,
                component_id="mid",
                category=EventCategory.EXECUTION,
                timestamp=now - timedelta(hours=1)
            ),
            Event(
                event_type=EventType.AGENT,
                component_id="new",
                category=EventCategory.EXECUTION,
                timestamp=now
            )
        ]
        
        for event in events:
            collector.process_event(event)
            
        # Query last hour
        query = collector.query().in_time_range(
            now - timedelta(hours=1),
            now
        )
        results = collector.find_events(query)
        assert len(results) == 2
        assert set(e.component_id for e in results) == {"mid", "new"}
        
    def test_custom_filter_query(self):
        """Test custom filter predicates"""
        collector = MemoryCollector()
        collector.start()
        
        now = datetime.now(timezone.utc)
        
        # Add test events
        events = [
            Event(
                event_type=EventType.AGENT,
                component_id="test-1",
                category=EventCategory.EXECUTION,
                metadata={"value": 10},
                timestamp=now
            ),
            Event(
                event_type=EventType.AGENT,
                component_id="test-2",
                category=EventCategory.EXECUTION,
                metadata={"value": 20},
                timestamp=now
            ),
            Event(
                event_type=EventType.AGENT,
                component_id="test-3",
                category=EventCategory.EXECUTION,
                metadata={"value": 30},
                timestamp=now
            )
        ]
        
        for event in events:
            collector.process_event(event)
            
        # Query with custom filter
        query = collector.query().custom_filter(
            lambda e: e.metadata.get("value", 0) > 15
        )
        results = collector.find_events(query)
        assert len(results) == 2
        assert all(e.metadata["value"] > 15 for e in results)
        
    def test_token_usage_metrics(self):
        """Test token usage metric calculations"""
        collector = MemoryCollector()
        collector.start()
        
        now = datetime.now(timezone.utc)
        
        # Add test events
        events = [
            Event(
                event_type=EventType.AGENT,
                component_id="agent-1",
                category=EventCategory.COST,
                metadata={"token_count": 100},
                timestamp=now
            ),
            Event(
                event_type=EventType.AGENT,
                component_id="agent-1",
                category=EventCategory.COST,
                metadata={"token_count": 200},
                timestamp=now
            ),
            Event(
                event_type=EventType.AGENT,
                component_id="agent-2",
                category=EventCategory.COST,
                metadata={"token_count": 150},
                timestamp=now
            )
        ]
        
        for event in events:
            collector.process_event(event)
            
        usage = collector.get_token_usage()
        
        # Check agent-1 metrics
        agent1_metrics = usage["agent-1"]
        assert agent1_metrics.count == 2
        assert agent1_metrics.min == 100
        assert agent1_metrics.max == 200
        assert agent1_metrics.mean == 150
        assert agent1_metrics.median == 150
        
        # Check agent-2 metrics
        agent2_metrics = usage["agent-2"]
        assert agent2_metrics.count == 1
        assert agent2_metrics.min == 150
        assert agent2_metrics.max == 150
        assert agent2_metrics.mean == 150
        
    def test_response_time_metrics(self):
        """Test response time metric calculations"""
        collector = MemoryCollector()
        collector.start()
        
        now = datetime.now(timezone.utc)
        
        # Add test events
        events = [
            Event(
                event_type=EventType.AGENT,
                component_id="agent-1",
                category=EventCategory.EXECUTION,
                duration_ms=100,
                timestamp=now
            ),
            Event(
                event_type=EventType.AGENT,
                component_id="agent-1",
                category=EventCategory.EXECUTION,
                duration_ms=200,
                timestamp=now
            ),
            Event(
                event_type=EventType.AGENT,
                component_id="agent-2",
                category=EventCategory.EXECUTION,
                duration_ms=150,
                timestamp=now
            )
        ]
        
        for event in events:
            collector.process_event(event)
            
        times = collector.get_response_times()
        
        # Check agent-1 metrics
        agent1_metrics = times["agent-1"]
        assert agent1_metrics.count == 2
        assert agent1_metrics.min == 100
        assert agent1_metrics.max == 200
        assert agent1_metrics.mean == 150
        assert agent1_metrics.median == 150
        
        # Check agent-2 metrics
        agent2_metrics = times["agent-2"]
        assert agent2_metrics.count == 1
        assert agent2_metrics.min == 150
        assert agent2_metrics.max == 150
        assert agent2_metrics.mean == 150
        
    def test_error_rate_metrics(self):
        """Test error rate calculations"""
        collector = MemoryCollector()
        collector.start()
        
        now = datetime.now(timezone.utc)
        
        # Add test events
        events = [
            # agent-1: 1 error out of 2 events (50% error rate)
            Event(
                event_type=EventType.AGENT,
                component_id="agent-1",
                category=EventCategory.EXECUTION,
                severity=EventSeverity.ERROR,
                timestamp=now
            ),
            Event(
                event_type=EventType.AGENT,
                component_id="agent-1",
                category=EventCategory.EXECUTION,
                severity=EventSeverity.INFO,
                timestamp=now
            ),
            # agent-2: 0 errors out of 1 event (0% error rate)
            Event(
                event_type=EventType.AGENT,
                component_id="agent-2",
                category=EventCategory.EXECUTION,
                severity=EventSeverity.INFO,
                timestamp=now
            )
        ]
        
        for event in events:
            collector.process_event(event)
            
        error_rates = collector.get_error_rates()
        
        assert error_rates["agent-1"] == 0.5  # 50% error rate
        assert error_rates["agent-2"] == 0.0  # 0% error rate
        
    def test_resource_usage_metrics(self):
        """Test resource usage metric calculations"""
        collector = MemoryCollector()
        collector.start()
        
        now = datetime.now(timezone.utc)
        
        # Add test events
        events = [
            Event(
                event_type=EventType.AGENT,
                component_id="agent-1",
                category=EventCategory.EXECUTION,
                memory_usage_bytes=100 * 1024 * 1024,  # 100 MB
                cpu_usage_percent=50.0,
                timestamp=now
            ),
            Event(
                event_type=EventType.AGENT,
                component_id="agent-1",
                category=EventCategory.EXECUTION,
                memory_usage_bytes=200 * 1024 * 1024,  # 200 MB
                cpu_usage_percent=75.0,
                timestamp=now
            ),
            Event(
                event_type=EventType.AGENT,
                component_id="agent-2",
                category=EventCategory.EXECUTION,
                memory_usage_bytes=150 * 1024 * 1024,  # 150 MB
                cpu_usage_percent=60.0,
                timestamp=now
            )
        ]
        
        for event in events:
            collector.process_event(event)
            
        usage = collector.get_resource_usage()
        
        # Check agent-1 metrics
        agent1_metrics = usage["agent-1"]
        
        # Memory metrics
        assert agent1_metrics["memory_mb"].count == 2
        assert agent1_metrics["memory_mb"].min == 100
        assert agent1_metrics["memory_mb"].max == 200
        assert agent1_metrics["memory_mb"].mean == 150
        
        # CPU metrics
        assert agent1_metrics["cpu_percent"].count == 2
        assert agent1_metrics["cpu_percent"].min == 50.0
        assert agent1_metrics["cpu_percent"].max == 75.0
        assert agent1_metrics["cpu_percent"].mean == 62.5
        
        # Check agent-2 metrics
        agent2_metrics = usage["agent-2"]
        
        # Memory metrics
        assert agent2_metrics["memory_mb"].count == 1
        assert agent2_metrics["memory_mb"].min == 150
        assert agent2_metrics["memory_mb"].max == 150
        
        # CPU metrics
        assert agent2_metrics["cpu_percent"].count == 1
        assert agent2_metrics["cpu_percent"].min == 60.0
        assert agent2_metrics["cpu_percent"].max == 60.0
        
    def test_cost_analysis_metrics(self):
        """Test cost analysis calculations"""
        collector = MemoryCollector()
        collector.start()
        
        now = datetime.now(timezone.utc)
        
        # Add test events
        events = [
            Event(
                event_type=EventType.AGENT,
                component_id="agent-1",
                category=EventCategory.COST,
                metadata={"cost_usd": 0.10},
                timestamp=now
            ),
            Event(
                event_type=EventType.AGENT,
                component_id="agent-1",
                category=EventCategory.COST,
                metadata={"cost_usd": 0.20},
                timestamp=now
            ),
            Event(
                event_type=EventType.AGENT,
                component_id="agent-2",
                category=EventCategory.COST,
                metadata={"cost_usd": 0.15},
                timestamp=now
            )
        ]
        
        for event in events:
            collector.process_event(event)
            
        costs = collector.get_cost_analysis()
        
        assert approx_equal(costs["agent-1"], 0.30)  # Sum of costs
        assert approx_equal(costs["agent-2"], 0.15) 