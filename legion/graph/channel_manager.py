from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type

from .channels import Channel


class ChannelManager:
    """Manages channel lifecycle, registration, and monitoring"""

    def __init__(self):
        """Initialize channel manager"""
        self._channels: Dict[str, Channel] = {}
        self._type_registry: Dict[str, Type[Channel]] = {}
        self._performance_metrics: Dict[str, Dict[str, Any]] = {}
        self._error_handlers: Dict[str, Callable[[Exception], None]] = {}
        self._debug_mode: bool = False

    def register_channel_type(self, name: str, channel_type: Type[Channel]) -> None:
        """Register a new channel type

        Args:
        ----
            name: Name of the channel type
            channel_type: Channel class to register

        """
        if name in self._type_registry:
            raise ValueError(f"Channel type '{name}' already registered")
        self._type_registry[name] = channel_type

    def create_channel(
        self,
        channel_type: str,
        channel_id: Optional[str] = None,
        **kwargs
    ) -> Channel:
        """Create a new channel instance

        Args:
        ----
            channel_type: Type of channel to create
            channel_id: Optional channel ID (generated if not provided)
            **kwargs: Additional arguments for channel creation

        Returns:
        -------
            Created channel instance

        """
        if channel_type not in self._type_registry:
            raise ValueError(f"Unknown channel type '{channel_type}'")

        channel_class = self._type_registry[channel_type]
        channel = channel_class(**kwargs)

        if channel_id:
            channel._id = channel_id

        self._channels[channel.id] = channel
        self._performance_metrics[channel.id] = {
            "created_at": datetime.now(),
            "update_count": 0,
            "last_update": None,
            "error_count": 0
        }

        return channel

    def get_channel(self, channel_id: str) -> Optional[Channel]:
        """Get channel by ID

        Args:
        ----
            channel_id: ID of channel to get

        Returns:
        -------
            Channel if found, None otherwise

        """
        return self._channels.get(channel_id)

    def delete_channel(self, channel_id: str) -> None:
        """Delete a channel

        Args:
        ----
            channel_id: ID of channel to delete

        """
        if channel_id in self._channels:
            self._channels.pop(channel_id)
            self._performance_metrics.pop(channel_id)

    def register_error_handler(
        self,
        channel_id: str,
        handler: Callable[[Exception], None]
    ) -> None:
        """Register an error handler for a channel

        Args:
        ----
            channel_id: ID of channel to handle errors for
            handler: Error handler function

        """
        if channel_id not in self._channels:
            raise ValueError(f"Unknown channel '{channel_id}'")
        self._error_handlers[channel_id] = handler

    def set_debug_mode(self, enabled: bool) -> None:
        """Enable or disable debug mode

        Args:
        ----
            enabled: Whether to enable debug mode

        """
        self._debug_mode = enabled

    def update_metrics(self, channel_id: str, error: Optional[Exception] = None) -> None:
        """Update performance metrics for a channel

        Args:
        ----
            channel_id: ID of channel to update metrics for
            error: Optional error that occurred

        """
        if channel_id not in self._performance_metrics:
            return

        metrics = self._performance_metrics[channel_id]
        metrics["update_count"] += 1
        metrics["last_update"] = datetime.now()

        if error:
            metrics["error_count"] += 1
            if channel_id in self._error_handlers:
                try:
                    self._error_handlers[channel_id](error)
                except Exception as e:
                    if self._debug_mode:
                        print(f"Error in error handler for channel {channel_id}: {e}")

    def get_metrics(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a channel

        Args:
        ----
            channel_id: ID of channel to get metrics for

        Returns:
        -------
            Metrics dictionary if found, None otherwise

        """
        return self._performance_metrics.get(channel_id)

    def get_registered_types(self) -> List[str]:
        """Get list of registered channel types

        Returns
        -------
            List of registered channel type names

        """
        return list(self._type_registry.keys())

    def get_active_channels(self) -> List[str]:
        """Get list of active channel IDs

        Returns
        -------
            List of active channel IDs

        """
        return list(self._channels.keys())

    def clear(self) -> None:
        """Clear all channels and metrics"""
        self._channels.clear()
        self._performance_metrics.clear()
        self._error_handlers.clear()
