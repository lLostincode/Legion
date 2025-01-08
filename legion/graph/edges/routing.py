from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, Optional, TypeVar

from pydantic import BaseModel, Field

from ..nodes.base import NodeBase
from ..state import GraphState

T = TypeVar("T")

class ConditionMetadata(BaseModel):
    """Metadata for routing conditions"""

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    version: int = 0
    condition_type: str = Field(default="")
    error: Optional[str] = None

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to handle datetime serialization"""
        data = super().model_dump(**kwargs)
        data["created_at"] = data["created_at"].isoformat()
        data["updated_at"] = data["updated_at"].isoformat()
        return data

class RoutingCondition(ABC):
    """Base class for routing conditions"""

    def __init__(self, graph_state: GraphState):
        """Initialize condition

        Args:
        ----
            graph_state: Graph state manager

        """
        self._metadata = ConditionMetadata(
            condition_type=self.__class__.__name__
        )
        self._graph_state = graph_state

    @property
    def metadata(self) -> ConditionMetadata:
        """Get condition metadata"""
        return self._metadata

    def _update_metadata(self) -> None:
        """Update metadata after state change"""
        self._metadata.updated_at = datetime.now()
        self._metadata.version += 1

    @abstractmethod
    async def evaluate(self, source_node: NodeBase, **kwargs) -> bool:
        """Evaluate the condition

        Args:
        ----
            source_node: Source node
            **kwargs: Additional evaluation arguments

        Returns:
        -------
            True if condition is met, False otherwise

        """
        pass

    def checkpoint(self) -> Dict[str, Any]:
        """Create a checkpoint of current state"""
        return {
            "metadata": self._metadata.model_dump()
        }

    def restore(self, checkpoint: Dict[str, Any]) -> None:
        """Restore from checkpoint"""
        self._metadata = ConditionMetadata(**checkpoint["metadata"])

class StateCondition(RoutingCondition):
    """Condition based on graph state"""

    def __init__(
        self,
        graph_state: GraphState,
        state_key: str,
        predicate: Callable[[Any], bool]
    ):
        """Initialize condition

        Args:
        ----
            graph_state: Graph state manager
            state_key: Key to check in graph state
            predicate: Function that takes state value and returns bool

        """
        super().__init__(graph_state)
        self._state_key = state_key
        self._predicate = predicate

    async def evaluate(self, source_node: NodeBase, **kwargs) -> bool:
        """Evaluate state condition"""
        state_data = self._graph_state.get_global_state()
        state_value = state_data.get(self._state_key)
        return self._predicate(state_value)

    def checkpoint(self) -> Dict[str, Any]:
        """Create checkpoint"""
        checkpoint = super().checkpoint()
        checkpoint["state_key"] = self._state_key
        return checkpoint

    def restore(self, checkpoint: Dict[str, Any]) -> None:
        """Restore from checkpoint"""
        super().restore(checkpoint)
        self._state_key = checkpoint["state_key"]

class ChannelCondition(RoutingCondition):
    """Condition based on channel value"""

    def __init__(
        self,
        graph_state: GraphState,
        channel_name: str,
        predicate: Callable[[Any], bool]
    ):
        """Initialize condition

        Args:
        ----
            graph_state: Graph state manager
            channel_name: Name of channel to check
            predicate: Function that takes channel value and returns bool

        """
        super().__init__(graph_state)
        self._channel_name = channel_name
        self._predicate = predicate

    async def evaluate(self, source_node: NodeBase, **kwargs) -> bool:
        """Evaluate channel condition"""
        channel = source_node.get_output_channel(self._channel_name)
        if not channel:
            return False
        value = channel.get()
        return self._predicate(value)

    def checkpoint(self) -> Dict[str, Any]:
        """Create checkpoint"""
        checkpoint = super().checkpoint()
        checkpoint["channel_name"] = self._channel_name
        return checkpoint

    def restore(self, checkpoint: Dict[str, Any]) -> None:
        """Restore from checkpoint"""
        super().restore(checkpoint)
        self._channel_name = checkpoint["channel_name"]

class CustomCondition(RoutingCondition):
    """Custom condition with user-defined evaluation"""

    def __init__(
        self,
        graph_state: GraphState,
        evaluator: Callable[[NodeBase, GraphState, Dict[str, Any]], bool]
    ):
        """Initialize condition

        Args:
        ----
            graph_state: Graph state manager
            evaluator: Custom evaluation function

        """
        super().__init__(graph_state)
        self._evaluator = evaluator

    async def evaluate(self, source_node: NodeBase, **kwargs) -> bool:
        """Evaluate custom condition"""
        return self._evaluator(source_node, self._graph_state, kwargs)
