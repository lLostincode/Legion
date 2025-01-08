from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Pattern, Set

from .events.base import Event, EventCategory, EventSeverity, EventType


@dataclass
class MonitorConfig:
    """Configuration for a monitor instance"""

    # Event type filtering
    event_types: Set[EventType] = field(default_factory=lambda: {e for e in EventType})
    categories: Set[EventCategory] = field(default_factory=lambda: {c for c in EventCategory})
    min_severity: EventSeverity = EventSeverity.DEBUG

    # Component filtering
    component_patterns: Set[Pattern] = field(default_factory=set)
    excluded_component_patterns: Set[Pattern] = field(default_factory=set)

    # Sampling
    sample_rate: float = 1.0  # 1.0 = 100% sampling

    # Resource limits
    max_events_per_second: Optional[int] = None
    max_memory_mb: Optional[int] = None

class Monitor:
    """Base class for all monitors in Legion"""

    def __init__(self, config: Optional[MonitorConfig] = None):
        """Initialize a new monitor

        Args:
        ----
            config: Optional monitor configuration. If not provided, uses defaults

        """
        self.config = config or MonitorConfig()
        self.is_active = False
        self._event_count = 0
        self._error_count = 0

    def should_process_event(self, event: Event) -> bool:
        """Determine if an event should be processed based on configuration

        Args:
        ----
            event: The event to check

        Returns:
        -------
            bool: True if the event should be processed

        """
        # Check event type and category
        if event.event_type not in self.config.event_types:
            return False
        if event.category not in self.config.categories:
            return False

        # Check severity - compare enum values
        severity_values = {
            EventSeverity.DEBUG: 0,
            EventSeverity.INFO: 1,
            EventSeverity.WARNING: 2,
            EventSeverity.ERROR: 3
        }
        if severity_values[event.severity] < severity_values[self.config.min_severity]:
            return False

        # Check component patterns
        if self.config.component_patterns:
            if not any(p.match(event.component_id) for p in self.config.component_patterns):
                return False

        if any(p.match(event.component_id) for p in self.config.excluded_component_patterns):
            return False

        return True

    def process_event(self, event: Event):
        """Process an event

        Args:
        ----
            event: Event to process

        """
        if not self.is_active:
            return

        if not self.should_process_event(event):
            return

        try:
            self._process_event_impl(event)
            self._event_count += 1
        except Exception:
            self._error_count += 1
            raise

    def _process_event_impl(self, event: Event):
        """Implementation of event processing

        Args:
        ----
            event: Event to process

        This should be overridden by subclasses

        """
        raise NotImplementedError()

    def start(self):
        """Start the monitor"""
        self.is_active = True

    def stop(self):
        """Stop the monitor"""
        self.is_active = False

    @property
    def stats(self) -> Dict[str, Any]:
        """Get monitor statistics

        Returns
        -------
            Dict containing monitor stats

        """
        return {
            "is_active": self.is_active,
            "event_count": self._event_count,
            "error_count": self._error_count
        }
