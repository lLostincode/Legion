"""Monitor registry for managing and coordinating monitors"""

import logging
import weakref
from datetime import datetime
from threading import Lock
from typing import Callable, Dict, List, Optional, Set, Type

from .events.base import Event, EventEmitter
from .monitors import Monitor, MonitorConfig
from .storage import StorageBackend, StorageConfig, StorageFactory, StorageType

logger = logging.getLogger(__name__)

class EventRegistry:
    """Registry for storing and querying events"""

    def __init__(self,
                 storage_type: StorageType = StorageType.MEMORY,
                 storage_config: Optional[StorageConfig] = None,
                 **storage_kwargs):
        """Initialize the event registry

        Args:
        ----
            storage_type: Type of storage backend to use
            storage_config: Optional storage configuration
            **storage_kwargs: Additional arguments for storage backend

        """
        self._storage = StorageFactory.create(
            storage_type,
            config=storage_config,
            **storage_kwargs
        )

    def record_event(self, event: Event):
        """Record an event in the registry

        Args:
        ----
            event: The event to record

        """
        self._storage.store_event(event)

    def get_events(self,
                  event_types: Optional[List[Type[Event]]] = None,
                  start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None) -> List[Event]:
        """Get events matching the specified criteria

        Args:
        ----
            event_types: Optional list of event types to filter by
            start_time: Optional start time to filter by
            end_time: Optional end time to filter by

        Returns:
        -------
            List of matching events

        """
        return self._storage.get_events(
            event_types=event_types,
            start_time=start_time,
            end_time=end_time
        )

    def clear(self):
        """Clear all events from the registry"""
        self._storage.clear()

    @property
    def storage(self) -> StorageBackend:
        """Get the underlying storage backend

        Returns
        -------
            The storage backend instance

        """
        return self._storage

class MonitorRegistry:
    """Central registry for managing monitors and routing events

    This is a singleton class that provides global access to monitoring functionality.
    It handles monitor lifecycle, event routing, and configuration management.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        """Ensure only one instance exists"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self):
        """Initialize the registry"""
        if not hasattr(self, "_initialized"):
            self._init_state()

    def _init_state(self):
        """Initialize or reset registry state"""
        logger.debug("Initializing registry state")
        self._monitors: Dict[str, Monitor] = {}
        self._monitor_configs: Dict[str, MonitorConfig] = {}
        self._emitters: Set[EventEmitter] = weakref.WeakSet()
        self._event_handlers: Dict[EventEmitter, Callable] = {}
        self._event_registry = EventRegistry()
        self._initialized = True

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of the registry

        Returns
        -------
            MonitorRegistry: The singleton instance

        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the registry state (for testing)"""
        logger.debug("Resetting registry")
        if cls._instance is not None:
            cls._instance._init_state()

    def register_component(self, component_id: str, component: EventEmitter):
        """Register a component with the registry

        Args:
        ----
            component_id: Unique identifier for the component
            component: The component to register

        """
        logger.debug(f"Registering component {component_id}")
        self._emitters.add(component)
        component.add_event_handler(self.process_event)

    def unregister_component(self, component_id: str, component: EventEmitter):
        """Unregister a component from the registry

        Args:
        ----
            component_id: Unique identifier for the component
            component: The component to unregister

        """
        logger.debug(f"Unregistering component {component_id}")
        self._emitters.discard(component)
        component.remove_event_handler(self.process_event)

    def process_event(self, event: Event):
        """Process an event from a registered component

        Args:
        ----
            event: The event to process

        """
        logger.debug(f"Processing event {event.id} from {event.component_id}")
        self._event_registry.record_event(event)
        for monitor in self._monitors.values():
            if monitor.should_process_event(event):
                monitor.process_event(event)

    def register_monitor(self, name: str, monitor_cls: Type[Monitor], config: Optional[MonitorConfig] = None) -> Monitor:
        """Register a new monitor

        Args:
        ----
            name: Unique name for the monitor
            monitor_cls: Monitor class to instantiate
            config: Optional configuration for the monitor

        Returns:
        -------
            The created monitor instance

        Raises:
        ------
            ValueError: If a monitor with the given name already exists

        """
        logger.debug(f"Registering monitor '{name}' with config: {config}")
        if name in self._monitors:
            raise ValueError(f"Monitor '{name}' already exists")

        monitor = monitor_cls(config)
        self._monitors[name] = monitor
        if config:
            self._monitor_configs[name] = config

        return monitor

    def unregister_monitor(self, name: str):
        """Unregister a monitor

        Args:
        ----
            name: Name of the monitor to unregister

        Raises:
        ------
            KeyError: If no monitor exists with the given name

        """
        logger.debug(f"Unregistering monitor '{name}'")
        if name not in self._monitors:
            raise KeyError(f"No monitor found with name '{name}'")

        monitor = self._monitors[name]
        monitor.stop()
        del self._monitors[name]
        self._monitor_configs.pop(name, None)

    def get_monitor(self, name: str) -> Monitor:
        """Get a monitor by name

        Args:
        ----
            name: Name of the monitor to get

        Returns:
        -------
            The requested monitor

        Raises:
        ------
            KeyError: If no monitor exists with the given name

        """
        if name not in self._monitors:
            raise KeyError(f"No monitor found with name '{name}'")
        return self._monitors[name]

    def list_monitors(self) -> List[str]:
        """Get a list of all registered monitor names

        Returns
        -------
            List of monitor names

        """
        return list(self._monitors.keys())

    def register_emitter(self, emitter: EventEmitter):
        """Register an event emitter

        Args:
        ----
            emitter: The emitter to register

        """
        logger.debug(f"Registering emitter: {emitter}")
        self._emitters.add(emitter)

        # Create a strong reference to the handler to prevent it from being garbage collected
        handler = self._route_event
        self._event_handlers[emitter] = handler
        emitter.add_event_handler(handler)

        logger.debug(f"Added route_event handler to emitter. Handlers: {emitter._event_handlers}")

    def unregister_emitter(self, emitter: EventEmitter):
        """Unregister an event emitter

        Args:
        ----
            emitter: The emitter to unregister

        """
        logger.debug(f"Unregistering emitter: {emitter}")
        self._emitters.discard(emitter)

        # Remove the handler and its reference
        if emitter in self._event_handlers:
            handler = self._event_handlers[emitter]
            emitter.remove_event_handler(handler)
            del self._event_handlers[emitter]

    def _route_event(self, event: Event):
        """Route an event to interested monitors

        Args:
        ----
            event: The event to route

        """
        self._event_registry.record_event(event)
        for monitor in self._monitors.values():
            if monitor.should_process_event(event):
                try:
                    monitor.process_event(event)
                except Exception as e:
                    # Log error but continue processing with other monitors
                    logger.error(f"Error in monitor {monitor.__class__.__name__}: {e}")

    def start_all(self):
        """Start all registered monitors"""
        for monitor in self._monitors.values():
            monitor.start()

    def stop_all(self):
        """Stop all registered monitors"""
        for monitor in self._monitors.values():
            monitor.stop()

    def get_stats(self) -> Dict[str, Dict]:
        """Get statistics for all monitors

        Returns
        -------
            Dictionary mapping monitor names to their statistics

        """
        return {
            name: monitor.get_stats()
            for name, monitor in self._monitors.items()
        }
