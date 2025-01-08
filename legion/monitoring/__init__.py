"""Legion monitoring system for observability and telemetry"""

from .events.base import Event, EventCategory, EventEmitter, EventSeverity, EventType

__all__ = [
    "Event",
    "EventEmitter",
    "EventType",
    "EventCategory",
    "EventSeverity"
]
