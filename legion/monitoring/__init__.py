"""Legion monitoring system for observability and telemetry"""

from .events.base import (
    Event,
    EventEmitter,
    EventType,
    EventCategory,
    EventSeverity
)

__all__ = [
    'Event',
    'EventEmitter',
    'EventType',
    'EventCategory',
    'EventSeverity'
] 