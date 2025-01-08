from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
from uuid import uuid4

from .channels import Channel, MessageChannel, SharedMemory
from .nodes.base import NodeBase
from .state import GraphState


class ComponentType(str, Enum):
    """Types of components that can be coordinated"""

    AGENT = "agent"
    BLOCK = "block"
    CHAIN = "chain"
    TEAM = "team"
    GRAPH = "graph"

@dataclass
class ComponentMetadata:
    """Metadata for a component instance"""

    component_id: str
    component_type: ComponentType
    node: NodeBase
    channels: Dict[str, Channel]
    state_scope: str  # Added state scope field

class ComponentCoordinator:
    """Coordinates interaction between different component types in the graph"""

    def __init__(self, graph_state: GraphState):
        """Initialize component coordinator

        Args:
        ----
            graph_state: Graph state instance

        """
        self._graph_state = graph_state
        self._components: Dict[str, ComponentMetadata] = {}
        self._type_registry: Dict[ComponentType, Dict[str, ComponentMetadata]] = {
            t: {} for t in ComponentType
        }

        # Channels for component coordination
        self._event_channel = MessageChannel[str](str)  # For component lifecycle events
        self._error_channel = MessageChannel[Exception](Exception)  # For component errors

        # State scoping
        self._state_scopes: Dict[str, SharedMemory] = {}
        self._scope_hierarchy: Dict[str, List[str]] = {}  # parent -> children mapping

    def register_component(self,
                         node: NodeBase,
                         component_type: ComponentType,
                         channels: Optional[Dict[str, Channel]] = None,
                         parent_scope: Optional[str] = None) -> str:
        """Register a component with the coordinator

        Args:
        ----
            node: Node instance
            component_type: Type of component
            channels: Optional channel mapping
            parent_scope: Optional parent state scope

        Returns:
        -------
            Component ID

        """
        component_id = str(uuid4())
        state_scope = f"scope_{component_id}"

        # Create state scope
        self._state_scopes[state_scope] = SharedMemory(Dict)
        if parent_scope:
            if parent_scope not in self._scope_hierarchy:
                self._scope_hierarchy[parent_scope] = []
            self._scope_hierarchy[parent_scope].append(state_scope)

        metadata = ComponentMetadata(
            component_id=component_id,
            component_type=component_type,
            node=node,
            channels=channels or {},
            state_scope=state_scope
        )

        self._components[component_id] = metadata
        self._type_registry[component_type][component_id] = metadata
        self._event_channel.push(f"component_registered:{component_id}")

        return component_id

    def unregister_component(self, component_id: str) -> None:
        """Unregister a component from the coordinator

        Args:
        ----
            component_id: ID of component to unregister

        """
        if component_id not in self._components:
            return

        metadata = self._components[component_id]

        # Find and unregister child components first
        if metadata.state_scope in self._scope_hierarchy:
            child_scopes = self._scope_hierarchy[metadata.state_scope]
            # Find components with these child scopes
            child_components = [
                cid for cid, comp in self._components.items()
                if comp.state_scope in child_scopes
            ]
            for child_id in child_components:
                self.unregister_component(child_id)

        # Clean up channels
        for channel in metadata.channels:
            if channel in self._channels:
                del self._channels[channel]

        # Clean up state scope and child scopes
        if metadata.state_scope in self._state_scopes:
            # Recursively clean up child scopes
            self._cleanup_scope_hierarchy(metadata.state_scope)
            del self._state_scopes[metadata.state_scope]

        # Clean up component
        del self._components[component_id]

        # Clean up type registry
        if metadata.component_type in self._type_registry:
            self._type_registry[metadata.component_type].pop(component_id)
            if not self._type_registry[metadata.component_type]:
                del self._type_registry[metadata.component_type]

        # Push unregistration event
        self._event_channel.push(f"component_unregistered:{component_id}")

    def _cleanup_scope_hierarchy(self, scope: str) -> None:
        """Recursively clean up a state scope and its children

        Args:
        ----
            scope: State scope to clean up

        """
        if scope in self._scope_hierarchy:
            child_scopes = self._scope_hierarchy[scope]
            for child_scope in child_scopes:
                self._cleanup_scope_hierarchy(child_scope)
                if child_scope in self._state_scopes:
                    del self._state_scopes[child_scope]
            del self._scope_hierarchy[scope]

    def get_state(self, component_id: str) -> Optional[Dict]:
        """Get component state

        Args:
        ----
            component_id: Component ID

        Returns:
        -------
            Component state if found

        """
        if component_id not in self._components:
            return None

        metadata = self._components[component_id]
        return self._state_scopes[metadata.state_scope].get()

    def set_state(self, component_id: str, state: Dict) -> None:
        """Set component state

        Args:
        ----
            component_id: Component ID
            state: State to set

        """
        if component_id not in self._components:
            raise ValueError(f"Unknown component {component_id}")

        metadata = self._components[component_id]
        self._state_scopes[metadata.state_scope].set(state)

    def get_parent_state(self, component_id: str) -> Optional[Dict]:
        """Get parent component state

        Args:
        ----
            component_id: Component ID

        Returns:
        -------
            Parent state if found

        """
        if component_id not in self._components:
            return None

        metadata = self._components[component_id]
        state_scope = metadata.state_scope

        # Find parent scope
        for parent_scope, children in self._scope_hierarchy.items():
            if state_scope in children:
                return self._state_scopes[parent_scope].get()

        return None

    def get_component(self, component_id: str) -> Optional[ComponentMetadata]:
        """Get component metadata

        Args:
        ----
            component_id: Component ID

        Returns:
        -------
            Component metadata if found

        """
        return self._components.get(component_id)

    def get_components_by_type(self, component_type: ComponentType) -> List[ComponentMetadata]:
        """Get all components of a specific type

        Args:
        ----
            component_type: Component type

        Returns:
        -------
            List of component metadata

        """
        return list(self._type_registry[component_type].values())

    def add_channel(self, component_id: str, channel_id: str, channel: Channel) -> None:
        """Add a channel to a component

        Args:
        ----
            component_id: Component ID
            channel_id: Channel ID
            channel: Channel instance

        """
        if component_id not in self._components:
            raise ValueError(f"Unknown component {component_id}")

        self._components[component_id].channels[channel_id] = channel

    def get_channel(self, component_id: str, channel_id: str) -> Optional[Channel]:
        """Get a channel from a component

        Args:
        ----
            component_id: Component ID
            channel_id: Channel ID

        Returns:
        -------
            Channel if found

        """
        if component_id not in self._components:
            return None

        return self._components[component_id].channels.get(channel_id)

    def report_error(self, component_id: str, error: Exception) -> None:
        """Report an error from a component

        Args:
        ----
            component_id: Component ID
            error: Error that occurred

        """
        if component_id not in self._components:
            raise ValueError(f"Unknown component {component_id}")

        self._components[component_id]
        self._error_channel.push(error)
        self._event_channel.push(f"component_error:{component_id}")

    @property
    def errors(self) -> List[Exception]:
        """Get all reported errors"""
        return self._error_channel.get()

    @property
    def events(self) -> List[str]:
        """Get all component events"""
        return self._event_channel.get()
