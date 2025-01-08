from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Type

from ..events.base import Event


class StorageBackend(ABC):
    """Abstract base class for event storage backends"""

    @abstractmethod
    def store_event(self, event: Event) -> None:
        """Store an event in the backend

        Args:
        ----
            event: The event to store

        """
        pass

    @abstractmethod
    def get_events(self,
                  event_types: Optional[List[Type[Event]]] = None,
                  start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None) -> List[Event]:
        """Retrieve events matching the specified criteria

        Args:
        ----
            event_types: Optional list of event types to filter by
            start_time: Optional start time to filter by
            end_time: Optional end time to filter by

        Returns:
        -------
            List of matching events

        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all events from storage"""
        pass

    @abstractmethod
    def cleanup(self, retention_days: int) -> None:
        """Clean up old events based on retention policy

        Args:
        ----
            retention_days: Number of days to retain events for

        """
        pass
