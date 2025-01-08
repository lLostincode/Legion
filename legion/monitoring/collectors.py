"""Memory-based event collector implementation"""

import re
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Set

from .events.base import Event, EventCategory, EventSeverity, EventType
from .monitors import Monitor, MonitorConfig


@dataclass
class MemoryCollectorConfig(MonitorConfig):
    """Configuration for MemoryCollector"""

    max_events: int = 10000  # Maximum number of events to store
    retention_period: Optional[timedelta] = None  # How long to retain events
    metric_window: timedelta = field(default_factory=lambda: timedelta(minutes=5))  # Window for metric calculations

class EventQuery:
    """Builder class for constructing event queries"""

    def __init__(self):
        self.time_range: Optional[tuple[datetime, datetime]] = None
        self.event_types: Optional[Set[EventType]] = None
        self.categories: Optional[Set[EventCategory]] = None
        self.min_severity: Optional[EventSeverity] = None
        self.component_patterns: Set[re.Pattern] = set()
        self.custom_filters: List[Callable[[Event], bool]] = []

    def in_time_range(self, start: datetime, end: datetime) -> "EventQuery":
        """Filter events within a time range

        Args:
        ----
            start: Start time (inclusive)
            end: End time (inclusive)

        Returns:
        -------
            self for chaining

        """
        # Ensure timezone-aware datetimes
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        self.time_range = (start, end)
        return self

    def of_types(self, *types: EventType) -> "EventQuery":
        """Filter events by type

        Args:
        ----
            *types: Event types to include

        Returns:
        -------
            self for chaining

        """
        self.event_types = set(types)
        return self

    def in_categories(self, *categories: EventCategory) -> "EventQuery":
        """Filter events by category

        Args:
        ----
            *categories: Categories to include

        Returns:
        -------
            self for chaining

        """
        self.categories = set(categories)
        return self

    def min_severity_level(self, severity: EventSeverity) -> "EventQuery":
        """Filter events by minimum severity

        Args:
        ----
            severity: Minimum severity level

        Returns:
        -------
            self for chaining

        """
        self.min_severity = severity
        return self

    def matching_components(self, *patterns: str) -> "EventQuery":
        """Filter events by component ID patterns

        Args:
        ----
            *patterns: Regex patterns to match component IDs

        Returns:
        -------
            self for chaining

        """
        self.component_patterns.update(re.compile(p) for p in patterns)
        return self

    def custom_filter(self, predicate: Callable[[Event], bool]) -> "EventQuery":
        """Add a custom filter predicate

        Args:
        ----
            predicate: Function that takes an event and returns True to include it

        Returns:
        -------
            self for chaining

        """
        self.custom_filters.append(predicate)
        return self

    def matches(self, event: Event) -> bool:
        """Check if an event matches this query

        Args:
        ----
            event: Event to check

        Returns:
        -------
            True if the event matches all criteria

        """
        if self.time_range:
            start, end = self.time_range
            if not (start <= event.timestamp <= end):
                return False

        if self.event_types and event.event_type not in self.event_types:
            return False

        if self.categories and event.category not in self.categories:
            return False

        if self.min_severity:
            severity_values = {
                EventSeverity.DEBUG: 0,
                EventSeverity.INFO: 1,
                EventSeverity.WARNING: 2,
                EventSeverity.ERROR: 3
            }
            if severity_values[event.severity] < severity_values[self.min_severity]:
                return False

        if self.component_patterns:
            if not any(p.match(event.component_id) for p in self.component_patterns):
                return False

        return all(f(event) for f in self.custom_filters)

@dataclass
class MetricSummary:
    """Summary of metric calculations"""

    count: int = 0
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    p95: Optional[float] = None

    @classmethod
    def from_values(cls, values: List[float]) -> "MetricSummary":
        """Create a summary from a list of values

        Args:
        ----
            values: List of metric values

        Returns:
        -------
            MetricSummary instance

        """
        if not values:
            return cls(count=0)

        return cls(
            count=len(values),
            min=min(values),
            max=max(values),
            mean=statistics.mean(values),
            median=statistics.median(values),
            p95=statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values)
        )

class MemoryCollector(Monitor):
    """In-memory event collector with circular buffer storage

    This collector stores events in memory using a circular buffer,
    providing immediate access for analysis and debugging.
    """

    def __init__(self, config: Optional[MemoryCollectorConfig] = None):
        """Initialize the collector

        Args:
        ----
            config: Optional configuration. If not provided, uses defaults.

        """
        super().__init__(config or MemoryCollectorConfig())
        self.config: MemoryCollectorConfig = self.config  # Type hint for IDE support
        self._events: Deque[Event] = deque(maxlen=self.config.max_events)
        self._start_time: Optional[datetime] = None

    def start(self):
        """Start the collector"""
        super().start()
        self._start_time = datetime.now(timezone.utc)

    def stop(self):
        """Stop the collector"""
        super().stop()
        self._start_time = None

    def _process_event_impl(self, event: Event):
        """Process and store an event

        Args:
        ----
            event: The event to process

        """
        if self.config.retention_period:
            self._cleanup_old_events()

        self._events.append(event)

    def _cleanup_old_events(self):
        """Remove events older than the retention period"""
        if not self.config.retention_period or not self._events:
            return

        cutoff = datetime.now(timezone.utc) - self.config.retention_period
        while self._events and self._events[0].timestamp < cutoff:
            self._events.popleft()

    def get_events(self, limit: Optional[int] = None) -> List[Event]:
        """Get stored events

        Args:
        ----
            limit: Optional maximum number of events to return

        Returns:
        -------
            List of events, newest first

        """
        events = list(self._events)
        events.reverse()  # Newest first
        return events[:limit] if limit else events

    def query(self) -> EventQuery:
        """Create a new query builder

        Returns
        -------
            A new EventQuery instance for building queries

        """
        return EventQuery()

    def find_events(self, query: EventQuery, limit: Optional[int] = None) -> List[Event]:
        """Find events matching a query

        Args:
        ----
            query: The query to execute
            limit: Optional maximum number of events to return

        Returns:
        -------
            List of matching events, newest first

        """
        matching = [e for e in self._events if query.matches(e)]
        matching.reverse()  # Newest first
        return matching[:limit] if limit else matching

    def get_token_usage(self, window: Optional[timedelta] = None) -> Dict[str, MetricSummary]:
        """Get token usage statistics by component

        Args:
        ----
            window: Optional time window, defaults to config.metric_window

        Returns:
        -------
            Dict mapping component IDs to token usage summaries

        """
        window = window or self.config.metric_window
        cutoff = datetime.now(timezone.utc) - window

        usage_by_component: Dict[str, List[float]] = defaultdict(list)

        for event in self._events:
            if event.timestamp < cutoff:
                continue

            if event.category == EventCategory.COST:
                tokens = event.metadata.get("token_count")
                if tokens is not None:
                    usage_by_component[event.component_id].append(float(tokens))

        return {
            component: MetricSummary.from_values(values)
            for component, values in usage_by_component.items()
        }

    def get_response_times(self, window: Optional[timedelta] = None) -> Dict[str, MetricSummary]:
        """Get response time statistics by component

        Args:
        ----
            window: Optional time window, defaults to config.metric_window

        Returns:
        -------
            Dict mapping component IDs to response time summaries (in ms)

        """
        window = window or self.config.metric_window
        cutoff = datetime.now(timezone.utc) - window

        times_by_component: Dict[str, List[float]] = defaultdict(list)

        for event in self._events:
            if event.timestamp < cutoff:
                continue

            if event.duration_ms is not None:
                times_by_component[event.component_id].append(event.duration_ms)

        return {
            component: MetricSummary.from_values(values)
            for component, values in times_by_component.items()
        }

    def get_error_rates(self, window: Optional[timedelta] = None) -> Dict[str, float]:
        """Get error rates by component

        Args:
        ----
            window: Optional time window, defaults to config.metric_window

        Returns:
        -------
            Dict mapping component IDs to error rates (0-1)

        """
        window = window or self.config.metric_window
        cutoff = datetime.now(timezone.utc) - window

        total_by_component: Dict[str, int] = defaultdict(int)
        errors_by_component: Dict[str, int] = defaultdict(int)

        for event in self._events:
            if event.timestamp < cutoff:
                continue

            total_by_component[event.component_id] += 1
            if event.severity == EventSeverity.ERROR:
                errors_by_component[event.component_id] += 1

        return {
            component: errors_by_component[component] / total
            for component, total in total_by_component.items()
            if total > 0
        }

    def get_resource_usage(self, window: Optional[timedelta] = None) -> Dict[str, Dict[str, MetricSummary]]:
        """Get resource usage statistics by component

        Args:
        ----
            window: Optional time window, defaults to config.metric_window

        Returns:
        -------
            Dict mapping component IDs to resource usage summaries

        """
        window = window or self.config.metric_window
        cutoff = datetime.now(timezone.utc) - window

        memory_by_component: Dict[str, List[float]] = defaultdict(list)
        cpu_by_component: Dict[str, List[float]] = defaultdict(list)

        for event in self._events:
            if event.timestamp < cutoff:
                continue

            if event.memory_usage_bytes is not None:
                memory_by_component[event.component_id].append(event.memory_usage_bytes / 1024 / 1024)  # Convert to MB

            if event.cpu_usage_percent is not None:
                cpu_by_component[event.component_id].append(event.cpu_usage_percent)

        return {
            component: {
                "memory_mb": MetricSummary.from_values(memory_by_component[component]),
                "cpu_percent": MetricSummary.from_values(cpu_by_component[component])
            }
            for component in set(memory_by_component.keys()) | set(cpu_by_component.keys())
        }

    def get_cost_analysis(self, window: Optional[timedelta] = None) -> Dict[str, float]:
        """Get total costs by component

        Args:
        ----
            window: Optional time window, defaults to config.metric_window

        Returns:
        -------
            Dict mapping component IDs to total costs

        """
        window = window or self.config.metric_window
        cutoff = datetime.now(timezone.utc) - window

        costs_by_component: Dict[str, float] = defaultdict(float)

        for event in self._events:
            if event.timestamp < cutoff:
                continue

            if event.category == EventCategory.COST:
                cost = event.metadata.get("cost_usd")
                if cost is not None:
                    costs_by_component[event.component_id] += float(cost)

        return dict(costs_by_component)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get collector statistics

        Returns
        -------
            Dict containing collector stats

        """
        base_stats = super().stats

        # Add metric summaries
        window = timedelta(minutes=1)  # Use 1-minute window for stats
        return {
            **base_stats,
            "stored_events": len(self._events),
            "buffer_capacity": self.config.max_events,
            "buffer_usage": len(self._events) / self.config.max_events if self.config.max_events else 0,
            "uptime": str(datetime.now(timezone.utc) - self._start_time) if self._start_time else None,
            "metrics": {
                "token_usage": self.get_token_usage(window),
                "response_times": self.get_response_times(window),
                "error_rates": self.get_error_rates(window),
                "resource_usage": self.get_resource_usage(window),
                "costs": self.get_cost_analysis(window)
            }
        }
