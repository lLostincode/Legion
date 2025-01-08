"""Memory storage backend implementation"""

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from ..events.base import Event
from .base import StorageBackend
from .config import StorageConfig

logger = logging.getLogger(__name__)

def _ensure_timezone(dt: datetime) -> datetime:
    """Ensure a datetime has a timezone

    Args:
    ----
        dt: The datetime to check

    Returns:
    -------
        The datetime with UTC timezone if it was naive

    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

class MemoryStorageBackend(StorageBackend):
    """In-memory implementation of event storage"""

    def __init__(self, config: Optional[StorageConfig] = None):
        """Initialize the storage backend

        Args:
        ----
            config: Optional storage configuration

        """
        self._events: List[Event] = []
        self._lock = threading.Lock()
        self._config = config or StorageConfig()
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        """Start the background cleanup task"""
        def cleanup_loop():
            while not self._stop_cleanup.is_set():
                try:
                    # Wait first to allow initial events to be stored
                    self._stop_cleanup.wait(self._config.cleanup_interval * 60)
                    if self._stop_cleanup.is_set():
                        break

                    # Perform cleanup
                    self.cleanup(self._config.retention_days)

                    # Check max events
                    if self._config.max_events:
                        with self._lock:
                            if len(self._events) > self._config.max_events:
                                self._events = self._events[-self._config.max_events:]
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")

        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def store_event(self, event: Event) -> None:
        """Store an event in memory

        Args:
        ----
            event: The event to store

        """
        with self._lock:
            self._events.append(event)

            # Check max events limit
            if self._config.max_events and len(self._events) > self._config.max_events:
                self._events.pop(0)  # Remove oldest event

    def get_events(self,
                  event_types: Optional[List[Event]] = None,
                  start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None) -> List[Event]:
        """Get events matching the specified criteria

        Args:
        ----
            event_types: Optional list of events to filter by type
            start_time: Optional start time to filter by
            end_time: Optional end time to filter by

        Returns:
        -------
            List of matching events

        """
        with self._lock:
            events = self._events.copy()

        if start_time:
            start_time = _ensure_timezone(start_time)
            events = [e for e in events if _ensure_timezone(e.timestamp) >= start_time]

        if end_time:
            end_time = _ensure_timezone(end_time)
            events = [e for e in events if _ensure_timezone(e.timestamp) <= end_time]

        if event_types:
            # Filter by event type value
            type_values = [e.event_type.value for e in event_types]
            events = [e for e in events if e.event_type.value in type_values]

        return events

    def clear(self) -> None:
        """Clear all events from memory"""
        with self._lock:
            self._events.clear()

    def cleanup(self, retention_days: Optional[int] = None) -> None:
        """Remove events older than the retention period

        Args:
        ----
            retention_days: Number of days to retain events for. If not specified,
                          uses the configured value.

        """
        retention_days = retention_days or self._config.retention_days
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)

        with self._lock:
            self._events = [e for e in self._events if _ensure_timezone(e.timestamp) >= cutoff]

    def __del__(self):
        """Cleanup when the backend is destroyed"""
        if self._cleanup_thread:
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=1)
