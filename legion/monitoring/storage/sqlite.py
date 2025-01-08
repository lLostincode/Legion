"""SQLite storage backend implementation"""

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional
from uuid import UUID

from ..events.base import Event, EventCategory, EventSeverity, EventType
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

def _serialize_event(event: Event) -> dict:
    """Serialize an event to a dictionary

    Args:
    ----
        event: The event to serialize

    Returns:
    -------
        Dictionary representation of the event

    """
    return {
        "id": str(event.id),
        "timestamp": _ensure_timezone(event.timestamp).isoformat(),
        "event_type": event.event_type.value,
        "component_id": event.component_id,
        "category": event.category.value,
        "severity": event.severity.value,
        "parent_event_id": str(event.parent_event_id) if event.parent_event_id else None,
        "root_event_id": str(event.root_event_id) if event.root_event_id else None,
        "trace_path": event.trace_path,
        "metadata": event.metadata,
        "duration_ms": event.duration_ms,
        "memory_usage_bytes": event.memory_usage_bytes,
        "cpu_usage_percent": event.cpu_usage_percent,
        "tokens_used": event.tokens_used,
        "cost": event.cost,
        "provider_name": event.provider_name,
        "model_name": event.model_name,
        "system_cpu_percent": event.system_cpu_percent,
        "system_memory_percent": event.system_memory_percent,
        "system_disk_usage_bytes": event.system_disk_usage_bytes,
        "system_network_bytes_sent": event.system_network_bytes_sent,
        "system_network_bytes_received": event.system_network_bytes_received,
        "thread_id": event.thread_id,
        "process_id": event.process_id,
        "host_name": event.host_name,
        "python_version": event.python_version,
        "dependencies": event.dependencies,
        "related_components": event.related_components,
        "state_before": event.state_before,
        "state_after": event.state_after,
        "state_diff": event.state_diff
    }

def _deserialize_event(data: dict) -> Event:
    """Deserialize an event from a dictionary

    Args:
    ----
        data: Dictionary representation of the event

    Returns:
    -------
        The deserialized event

    """
    return Event(
        event_type=EventType(data["event_type"]),
        component_id=data["component_id"],
        category=EventCategory(data["category"]),
        id=UUID(data["id"]),
        timestamp=datetime.fromisoformat(data["timestamp"]),
        severity=EventSeverity(data["severity"]),
        parent_event_id=UUID(data["parent_event_id"]) if data["parent_event_id"] else None,
        root_event_id=UUID(data["root_event_id"]) if data["root_event_id"] else None,
        trace_path=data["trace_path"],
        metadata=data["metadata"],
        duration_ms=data["duration_ms"],
        memory_usage_bytes=data["memory_usage_bytes"],
        cpu_usage_percent=data["cpu_usage_percent"],
        tokens_used=data["tokens_used"],
        cost=data["cost"],
        provider_name=data["provider_name"],
        model_name=data["model_name"],
        system_cpu_percent=data["system_cpu_percent"],
        system_memory_percent=data["system_memory_percent"],
        system_disk_usage_bytes=data["system_disk_usage_bytes"],
        system_network_bytes_sent=data["system_network_bytes_sent"],
        system_network_bytes_received=data["system_network_bytes_received"],
        thread_id=data["thread_id"],
        process_id=data["process_id"],
        host_name=data["host_name"],
        python_version=data["python_version"],
        dependencies=data["dependencies"],
        related_components=data["related_components"],
        state_before=data["state_before"],
        state_after=data["state_after"],
        state_diff=data["state_diff"]
    )

class SQLiteStorageBackend(StorageBackend):
    """SQLite implementation of event storage"""

    def __init__(self,
                 db_path: str = "events.db",
                 config: Optional[StorageConfig] = None):
        """Initialize the storage backend

        Args:
        ----
            db_path: Path to the SQLite database file
            config: Optional storage configuration

        """
        self._db_path = Path(db_path)
        self._config = config or StorageConfig()
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()

        # Ensure directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()
        self._start_cleanup_task()

    def _init_db(self):
        """Initialize the database schema"""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME NOT NULL,
                    event_type TEXT NOT NULL,
                    component_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    data TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON events(event_type)")

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
                        with sqlite3.connect(self._db_path) as conn:
                            # Get total count
                            count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
                            if count > self._config.max_events:
                                # Delete oldest events
                                conn.execute("""
                                    DELETE FROM events
                                    WHERE id IN (
                                        SELECT id FROM events
                                        ORDER BY timestamp ASC
                                        LIMIT ?
                                    )
                                """, (count - self._config.max_events,))
                                conn.commit()
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")

        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def store_event(self, event: Event) -> None:
        """Store an event in the database

        Args:
        ----
            event: The event to store

        """
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                INSERT INTO events (id, timestamp, event_type, component_id, category, data)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                str(event.id),
                _ensure_timezone(event.timestamp).isoformat(),
                event.event_type.value,
                event.component_id,
                event.category.value,
                json.dumps(_serialize_event(event))
            ))
            conn.commit()

            # Check max events limit
            if self._config.max_events:
                count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
                if count > self._config.max_events:
                    # Delete oldest event
                    conn.execute("""
                        DELETE FROM events
                        WHERE id IN (
                            SELECT id FROM events
                            ORDER BY timestamp ASC
                            LIMIT 1
                        )
                    """)
                    conn.commit()

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
        query = "SELECT data FROM events WHERE 1=1"
        params = []

        if event_types:
            # Filter by event type value
            type_values = [e.event_type.value for e in event_types]
            placeholders = ",".join("?" * len(type_values))
            query += f" AND event_type IN ({placeholders})"
            params.extend(type_values)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(_ensure_timezone(start_time).isoformat())

        if end_time:
            query += " AND timestamp <= ?"
            params.append(_ensure_timezone(end_time).isoformat())

        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(query, params).fetchall()

        return [_deserialize_event(json.loads(row[0])) for row in rows]

    def clear(self) -> None:
        """Clear all events from the database"""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM events")
            conn.commit()

    def cleanup(self, retention_days: Optional[int] = None) -> None:
        """Remove events older than the retention period

        Args:
        ----
            retention_days: Number of days to retain events for. If not specified,
                          uses the configured value.

        """
        retention_days = retention_days or self._config.retention_days
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)

        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM events WHERE timestamp < ?", (cutoff.isoformat(),))
            conn.commit()

    def __del__(self):
        """Cleanup when the backend is destroyed"""
        if self._cleanup_thread:
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=1)
