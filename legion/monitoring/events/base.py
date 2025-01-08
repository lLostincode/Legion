"""Base event types and emitter for Legion monitoring system"""

import logging
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID, uuid4

from ..metrics import MetricsContext

logger = logging.getLogger(__name__)

class EventType(str, Enum):
    """Type of event in the system"""

    AGENT = "agent"
    TEAM = "team"
    SYSTEM = "system"
    TOOL = "tool"
    CHAIN = "chain"  # Chain-specific events


class EventCategory(str, Enum):
    """Category of the event"""

    EXECUTION = "execution"
    MEMORY = "memory"
    COST = "cost"
    ERROR = "error"


class EventSeverity(str, Enum):
    """Severity level of the event"""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Event:
    """Base event class for Legion monitoring system"""

    # Required fields
    event_type: EventType
    component_id: str
    category: EventCategory

    # Optional fields with defaults
    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    severity: EventSeverity = EventSeverity.INFO
    parent_event_id: Optional[UUID] = None
    root_event_id: Optional[UUID] = None
    trace_path: List[UUID] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    duration_ms: Optional[float] = None
    memory_usage_bytes: Optional[int] = None
    cpu_usage_percent: Optional[float] = None

    # Resource utilization
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    provider_name: Optional[str] = None
    model_name: Optional[str] = None

    # System metrics
    system_cpu_percent: Optional[float] = None
    system_memory_percent: Optional[float] = None
    system_disk_usage_bytes: Optional[int] = None
    system_network_bytes_sent: Optional[int] = None
    system_network_bytes_received: Optional[int] = None

    # Execution context
    thread_id: Optional[int] = None
    process_id: Optional[int] = None
    host_name: Optional[str] = None
    python_version: Optional[str] = None

    # Dependencies and relationships
    dependencies: List[str] = field(default_factory=list)
    related_components: List[str] = field(default_factory=list)

    # State information
    state_before: Optional[Dict[str, Any]] = None
    state_after: Optional[Dict[str, Any]] = None
    state_diff: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize derived fields after instance creation"""
        if not self.root_event_id and not self.parent_event_id:
            self.root_event_id = self.id
        elif not self.root_event_id and self.parent_event_id:
            self.root_event_id = self.parent_event_id

        if self.parent_event_id:
            self.trace_path.append(self.parent_event_id)


class EventEmitter:
    """Mixin class providing event emission capabilities"""

    def __init__(self):
        """Initialize the event emitter"""
        self._event_handlers = weakref.WeakSet()
        self._monitoring_enabled = True
        self._current_event: Optional[Event] = None

    def add_event_handler(self, handler: Callable[[Event], None]):
        """Add an event handler

        Args:
        ----
            handler: Callable that takes an Event as argument

        """
        self._event_handlers.add(handler)

    def remove_event_handler(self, handler: Callable[[Event], None]):
        """Remove an event handler

        Args:
        ----
            handler: Handler to remove

        """
        self._event_handlers.discard(handler)

    def emit_event(self, event: Event):
        """Emit an event to all registered handlers

        Args:
        ----
            event: Event to emit

        """
        if not self._monitoring_enabled:
            return

        # Collect metrics before emitting
        with MetricsContext(event):
            for handler in self._event_handlers:
                try:
                    handler(event)
                except Exception as e:
                    # Log error but continue processing other handlers
                    logger.error(f"Error in event handler: {e}")

    def emit(self, event: Event):
        """Emit an event to all registered handlers (alias for emit_event)

        Args:
        ----
            event: Event to emit

        """
        self.emit_event(event)

    @contextmanager
    def event_span(self, event_type: EventType, component_id: str, category: EventCategory):
        """Context manager for tracking event spans with timing

        Args:
        ----
            event_type: Type of event
            component_id: ID of the component emitting the event
            category: Category of event

        Yields:
        ------
            Event: The created event

        """
        event = Event(
            event_type=event_type,
            component_id=component_id,
            category=category,
            parent_event_id=self._current_event.id if self._current_event else None
        )

        old_event = self._current_event
        self._current_event = event

        # Use metrics context to track the span
        with MetricsContext(event):
            try:
                yield event
            finally:
                self.emit_event(event)
                self._current_event = old_event

    def enable_monitoring(self):
        """Enable event emission"""
        self._monitoring_enabled = True

    def disable_monitoring(self):
        """Disable event emission"""
        self._monitoring_enabled = False
