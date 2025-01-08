from abc import ABC, abstractmethod
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    get_args,
    get_origin,
)
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")

class ChannelMetadata(BaseModel):
    """Metadata for a channel"""

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    version: int = 0
    type_hint: Optional[str] = None

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

class Channel(Generic[T], ABC):
    """Base class for all channels"""

    def __init__(self, type_hint: Optional[Type[T]] = None, validate_type: bool = True):
        """Initialize channel

        Args:
        ----
            type_hint: Optional type hint for values
            validate_type: Whether to validate types at runtime

        """
        self._type_hint = type_hint
        self._validate_type = validate_type
        self._metadata = ChannelMetadata(
            type_hint=type_hint.__name__ if type_hint else None
        )
        self._id = str(uuid4())

    @property
    def id(self) -> str:
        """Get channel ID"""
        return self._id

    @property
    def metadata(self) -> ChannelMetadata:
        """Get channel metadata"""
        return self._metadata

    def _validate_value_type(self, value: Any) -> None:
        """Validate value type"""
        if not self._validate_type or not self._type_hint:
            return

        if value is None:
            return

        # Handle Any type
        if self._type_hint is Any:
            return

        # Handle Pydantic models
        if isinstance(self._type_hint, type) and issubclass(self._type_hint, BaseModel):
            if not isinstance(value, self._type_hint):
                try:
                    if isinstance(value, dict):
                        self._type_hint.model_validate(value)
                        return
                except Exception:
                    raise TypeError(f"Value must be of type {self._type_hint.__name__}")
                raise TypeError(f"Value must be of type {self._type_hint.__name__}")
            return

        # Handle generic types
        origin = get_origin(self._type_hint)
        if origin:
            args = get_args(self._type_hint)
            if not isinstance(value, origin):
                raise TypeError(f"Value must be of type {origin.__name__}")
            # Validate generic type arguments if possible
            if args and hasattr(value, "__iter__"):
                for item in value:
                    for arg in args:
                        if not isinstance(item, arg):
                            raise TypeError(f"Invalid item type {type(item).__name__}, expected {arg.__name__}")
            return

        # Handle basic types
        if not isinstance(value, self._type_hint):
            raise TypeError(f"Value must be of type {self._type_hint.__name__}")

    def _update_metadata(self) -> None:
        """Update metadata after value change"""
        self._metadata.updated_at = datetime.now()
        self._metadata.version += 1

    @abstractmethod
    def get(self) -> T:
        """Get current value"""
        pass

    @abstractmethod
    def set(self, value: T) -> None:
        """Set new value"""
        pass

    @abstractmethod
    def checkpoint(self) -> Dict[str, Any]:
        """Create a checkpoint of current state"""
        pass

    @abstractmethod
    def restore(self, checkpoint: Dict[str, Any]) -> None:
        """Restore from checkpoint"""
        pass

class LastValue(Channel[T]):
    """Channel that stores only the last value"""

    def __init__(self, type_hint: Optional[Type[T]] = None):
        super().__init__(type_hint)
        self._value: Optional[T] = None

    def get(self) -> Optional[T]:
        """Get the last value"""
        return self._value

    def set(self, value: T) -> None:
        """Set the value"""
        self._validate_value_type(value)
        self._value = value
        self._update_metadata()

    def clear(self) -> None:
        """Clear the stored value"""
        self._value = None
        self._update_metadata()

    def checkpoint(self) -> Dict[str, Any]:
        """Create checkpoint"""
        return {
            "metadata": self._metadata.model_dump(),
            "value": self._value
        }

    def restore(self, checkpoint: Dict[str, Any]) -> None:
        """Restore from checkpoint"""
        self._metadata = ChannelMetadata(**checkpoint["metadata"])
        self._value = checkpoint["value"]

class ValueSequence(Channel[T]):
    """Channel that maintains an ordered sequence of values"""

    def __init__(self, type_hint: Optional[Type[T]] = None, max_size: Optional[int] = None):
        super().__init__(type_hint)
        self._values: List[T] = []
        self._max_size = max_size

    def get(self) -> List[T]:
        return self._values.copy()

    def get_all(self) -> List[T]:
        """Get all values in sequence"""
        return self._values.copy()

    def append(self, value: T) -> None:
        self._validate_value_type(value)
        self._values.append(value)
        if self._max_size and len(self._values) > self._max_size:
            self._values.pop(0)
        self._update_metadata()

    def set(self, values: List[T]) -> None:
        for value in values:
            self._validate_value_type(value)
        self._values = values[-self._max_size:] if self._max_size else values.copy()
        self._update_metadata()

    def checkpoint(self) -> Dict[str, Any]:
        return {
            "metadata": self._metadata.model_dump(),
            "values": self._values,
            "max_size": self._max_size
        }

    def restore(self, checkpoint: Dict[str, Any]) -> None:
        self._metadata = ChannelMetadata(**checkpoint["metadata"])
        self._values = checkpoint["values"]
        self._max_size = checkpoint["max_size"]

class SharedState(Channel[Dict[str, Any]]):
    """Channel that maintains a shared dictionary state"""

    def __init__(self, type_hint: Optional[Type[Dict[str, Any]]] = None):
        super().__init__(dict)  # Use dict instead of Dict
        self._state: Dict[str, Any] = {}

    def get(self) -> Dict[str, Any]:
        return self._state.copy()

    def set(self, state: Dict[str, Any]) -> None:
        self._validate_value_type(state)
        self._state = state.copy()
        self._update_metadata()

    def update(self, updates: Dict[str, Any]) -> None:
        self._state.update(updates)
        self._update_metadata()

    def checkpoint(self) -> Dict[str, Any]:
        return {
            "metadata": self._metadata.model_dump(),
            "state": self._state
        }

    def restore(self, checkpoint: Dict[str, Any]) -> None:
        self._metadata = ChannelMetadata(**checkpoint["metadata"])
        self._state = checkpoint["state"]

class MessageChannel(Channel[T]):
    """FIFO message queue channel with capacity management and batch operations"""

    def __init__(self, type_hint: Optional[Type[T]] = None, capacity: Optional[int] = None):
        super().__init__(type_hint)
        self._messages: List[T] = []
        self._capacity = capacity

    def get(self) -> List[T]:
        """Get all messages without removing them"""
        return self._messages.copy()

    def set(self, messages: List[T]) -> None:
        """Replace all messages with new ones"""
        for message in messages:
            self._validate_value_type(message)

        if self._capacity and len(messages) > self._capacity:
            raise ValueError(f"Message count {len(messages)} exceeds channel capacity {self._capacity}")

        self._messages = messages.copy()
        self._update_metadata()

    def push(self, message: T) -> bool:
        """Push a single message to the queue. Returns True if successful."""
        self._validate_value_type(message)

        if self._capacity and len(self._messages) >= self._capacity:
            return False

        self._messages.append(message)
        self._update_metadata()
        return True

    def push_batch(self, messages: List[T]) -> int:
        """Push multiple messages. Returns number of messages successfully pushed."""
        remaining_capacity = float("inf") if self._capacity is None else self._capacity - len(self._messages)
        messages_to_add = messages[:remaining_capacity]

        for message in messages_to_add:
            self._validate_value_type(message)

        self._messages.extend(messages_to_add)
        self._update_metadata()
        return len(messages_to_add)

    def pop(self) -> Optional[T]:
        """Remove and return the oldest message"""
        if not self._messages:
            return None
        message = self._messages.pop(0)
        self._update_metadata()
        return message

    def pop_batch(self, max_count: Optional[int] = None) -> List[T]:
        """Remove and return multiple messages"""
        if not self._messages:
            return []

        count = min(max_count or len(self._messages), len(self._messages))
        messages = self._messages[:count]
        self._messages = self._messages[count:]
        self._update_metadata()
        return messages

    def clear(self) -> None:
        """Remove all messages"""
        self._messages.clear()
        self._update_metadata()

    @property
    def is_full(self) -> bool:
        """Check if channel has reached capacity"""
        return bool(self._capacity and len(self._messages) >= self._capacity)

    @property
    def available_capacity(self) -> Optional[int]:
        """Get remaining capacity. Returns None if unlimited."""
        if self._capacity is None:
            return None
        return max(0, self._capacity - len(self._messages))

    def checkpoint(self) -> Dict[str, Any]:
        return {
            "metadata": self._metadata.model_dump(),
            "messages": self._messages,
            "capacity": self._capacity
        }

    def restore(self, checkpoint: Dict[str, Any]) -> None:
        self._metadata = ChannelMetadata(**checkpoint["metadata"])
        self._messages = checkpoint["messages"]
        self._capacity = checkpoint["capacity"]

class BarrierChannel(Channel[bool]):
    """Channel that acts as a synchronization barrier for multiple contributors"""

    def __init__(self, contributor_count: int, timeout: Optional[float] = None):
        """Initialize barrier channel

        Args:
        ----
            contributor_count: Number of contributors required to trigger the barrier
            timeout: Optional timeout in seconds. None means no timeout.

        """
        super().__init__(bool)
        if contributor_count < 1:
            raise ValueError("Contributor count must be at least 1")

        self._contributor_count = contributor_count
        self._timeout = timeout
        self._current_contributors: Set[str] = set()
        self._triggered = False
        self._last_reset = datetime.now()

    def contribute(self, contributor_id: str) -> bool:
        """Add a contribution to the barrier. Returns True if barrier is triggered."""
        if self._triggered:
            return True

        if self._timeout:
            if (datetime.now() - self._last_reset).total_seconds() > self._timeout:
                self.reset()

        self._current_contributors.add(contributor_id)
        if len(self._current_contributors) >= self._contributor_count:
            self._triggered = True
            self._update_metadata()
            return True

        return False

    def is_triggered(self) -> bool:
        """Check if barrier is triggered"""
        if self._timeout:
            if (datetime.now() - self._last_reset).total_seconds() > self._timeout:
                self.reset()
        return self._triggered

    def reset(self) -> None:
        """Reset the barrier to initial state"""
        self._current_contributors.clear()
        self._triggered = False
        self._last_reset = datetime.now()
        self._update_metadata()

    def get(self) -> bool:
        """Get barrier state"""
        return self.is_triggered()

    def set(self, value: bool) -> None:
        """Set barrier state directly (mainly for restoration)"""
        self._triggered = value
        if not value:
            self._current_contributors.clear()
        self._update_metadata()

    @property
    def remaining_contributors(self) -> int:
        """Get number of remaining contributors needed"""
        return max(0, self._contributor_count - len(self._current_contributors))

    @property
    def current_contributors(self) -> Set[str]:
        """Get set of current contributor IDs"""
        return self._current_contributors.copy()

    def checkpoint(self) -> Dict[str, Any]:
        return {
            "metadata": self._metadata.model_dump(),
            "contributor_count": self._contributor_count,
            "timeout": self._timeout,
            "current_contributors": list(self._current_contributors),
            "triggered": self._triggered,
            "last_reset": self._last_reset.isoformat()
        }

    def restore(self, checkpoint: Dict[str, Any]) -> None:
        self._metadata = ChannelMetadata(**checkpoint["metadata"])
        self._contributor_count = checkpoint["contributor_count"]
        self._timeout = checkpoint["timeout"]
        self._current_contributors = set(checkpoint["current_contributors"])
        self._triggered = checkpoint["triggered"]
        self._last_reset = datetime.fromisoformat(checkpoint["last_reset"])

class BroadcastChannel(Channel[T]):
    """Channel for one-to-many communication with subscription management and history tracking"""

    def __init__(self, type_hint: Optional[Type[T]] = None, history_size: Optional[int] = None):
        """Initialize broadcast channel

        Args:
        ----
            type_hint: Optional type hint for values
            history_size: Optional maximum number of messages to keep in history

        """
        super().__init__(type_hint)
        self._subscribers: Set[str] = set()
        self._history: List[T] = []
        self._history_size = history_size
        self._current_value: Optional[T] = None

    def subscribe(self, subscriber_id: str) -> None:
        """Add a subscriber to the channel"""
        self._subscribers.add(subscriber_id)
        self._update_metadata()

    def unsubscribe(self, subscriber_id: str) -> None:
        """Remove a subscriber from the channel"""
        self._subscribers.discard(subscriber_id)
        self._update_metadata()

    def broadcast(self, value: T) -> None:
        """Broadcast a value to all subscribers"""
        self._validate_value_type(value)
        self._current_value = value

        if self._history_size is not None:
            self._history.append(value)
            if len(self._history) > self._history_size:
                self._history.pop(0)
        else:
            self._history.append(value)

        self._update_metadata()

    def get(self) -> Optional[T]:
        """Get current value"""
        return self._current_value

    def set(self, value: T) -> None:
        """Set current value (broadcasts to subscribers)"""
        self.broadcast(value)

    @property
    def history(self) -> List[T]:
        """Get message history"""
        return self._history.copy()

    @property
    def subscribers(self) -> Set[str]:
        """Get current subscribers"""
        return self._subscribers.copy()

    @property
    def subscriber_count(self) -> int:
        """Get number of subscribers"""
        return len(self._subscribers)

    def clear_history(self) -> None:
        """Clear message history"""
        self._history.clear()
        self._update_metadata()

    def checkpoint(self) -> Dict[str, Any]:
        return {
            "metadata": self._metadata.model_dump(),
            "subscribers": list(self._subscribers),
            "history": self._history,
            "history_size": self._history_size,
            "current_value": self._current_value
        }

    def restore(self, checkpoint: Dict[str, Any]) -> None:
        self._metadata = ChannelMetadata(**checkpoint["metadata"])
        self._subscribers = set(checkpoint["subscribers"])
        self._history = checkpoint["history"]
        self._history_size = checkpoint["history_size"]
        self._current_value = checkpoint["current_value"]

class AggregatorChannel(Channel[T]):
    """Channel for many-to-one aggregation with custom reduction operations"""

    def __init__(
        self,
        type_hint: Optional[Type[T]] = None,
        reducer: Optional[Callable[[List[T]], T]] = None,
        window_size: Optional[int] = None
    ):
        """Initialize aggregator channel

        Args:
        ----
            type_hint: Optional type hint for values
            reducer: Optional custom reduction function. If None, uses last value
            window_size: Optional window size for aggregation. None means unlimited

        """
        super().__init__(type_hint)
        self._values: List[T] = []
        self._window_size = window_size
        self._reducer = reducer or (lambda x: x[-1] if x else None)  # Default to last value
        self._current_result: Optional[T] = None

    def contribute(self, value: T) -> None:
        """Add a value to be aggregated"""
        self._validate_value_type(value)

        if self._window_size is not None:
            if len(self._values) >= self._window_size:
                self._values.pop(0)

        self._values.append(value)
        self._update_result()
        self._update_metadata()

    def _update_result(self) -> None:
        """Update aggregation result using reducer"""
        if not self._values:
            self._current_result = None
            return

        try:
            result = self._reducer(self._values.copy())
            if result is not None:  # Allow reducer to return None
                self._validate_value_type(result)
            self._current_result = result
        except Exception:
            # If reduction fails, use last value as fallback
            self._current_result = self._values[-1]

    def get(self) -> Optional[T]:
        """Get current aggregated result"""
        return self._current_result

    def set(self, value: T) -> None:
        """Set a single value (clears window and sets as only value)"""
        self._validate_value_type(value)
        self._values = [value]
        self._update_result()
        self._update_metadata()

    def clear(self) -> None:
        """Clear all values"""
        self._values.clear()
        self._current_result = None
        self._update_metadata()

    @property
    def window(self) -> List[T]:
        """Get current window of values"""
        return self._values.copy()

    @property
    def window_size(self) -> Optional[int]:
        """Get configured window size"""
        return self._window_size

    def checkpoint(self) -> Dict[str, Any]:
        return {
            "metadata": self._metadata.model_dump(),
            "values": self._values,
            "window_size": self._window_size,
            "current_result": self._current_result
        }

    def restore(self, checkpoint: Dict[str, Any]) -> None:
        self._metadata = ChannelMetadata(**checkpoint["metadata"])
        self._values = checkpoint["values"]
        self._window_size = checkpoint["window_size"]
        self._current_result = checkpoint["current_result"]
        # Note: reducer is not serialized/restored as it's a function

class SharedMemory(Channel[T]):
    """Channel that provides shared memory access with type safety"""

    def __init__(self, type_hint: Type[T]):
        """Initialize shared memory channel

        Args:
        ----
            type_hint: Type hint for channel values

        """
        super().__init__(type_hint)
        self._value: Optional[T] = None

    def set(self, value: T) -> None:
        """Set shared memory value

        Args:
        ----
            value: Value to set

        """
        self._validate_value_type(value)
        self._value = value
        self._update_metadata()

    def get(self) -> Optional[T]:
        """Get shared memory value

        Returns
        -------
            Current value if set, None otherwise

        """
        return self._value

    def clear(self) -> None:
        """Clear shared memory value"""
        self._value = None
        self._update_metadata()

    def checkpoint(self) -> Dict[str, Any]:
        """Create checkpoint of channel state

        Returns
        -------
            Channel state

        """
        return {
            "type": "shared_memory",
            "type_hint": self._type_hint.__name__,
            "value": self._value
        }

    def restore(self, checkpoint: Dict[str, Any]) -> None:
        """Restore channel state from checkpoint

        Args:
        ----
            checkpoint: Channel state

        """
        if checkpoint["type"] != "shared_memory":
            raise ValueError("Invalid checkpoint type")
        self._value = checkpoint["value"]
        self._update_metadata()

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
