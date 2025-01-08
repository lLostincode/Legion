from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from ..channels import Channel
from ..state import GraphState

T = TypeVar("T")

class NodeStatus(str, Enum):
    """Node execution status"""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class NodeMetadata(BaseModel):
    """Node metadata"""

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    version: int = 0
    node_id: str = Field(default_factory=lambda: str(uuid4()))
    node_type: str = Field(default="")
    status: NodeStatus = Field(default=NodeStatus.IDLE)
    error: Optional[str] = None
    execution_count: int = 0
    last_execution: Optional[datetime] = None
    custom_data: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

class ExecutionContext(BaseModel):
    """Context for node execution"""

    started_at: datetime = Field(default_factory=datetime.now)
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

class NodeBase(ABC):
    """Base class for all graph nodes"""

    def __init__(self, graph_state: GraphState):
        self._metadata = NodeMetadata(
            node_type=self.__class__.__name__
        )
        self._graph_state = graph_state
        self._input_channels: Dict[str, Channel] = {}
        self._output_channels: Dict[str, Channel] = {}
        self._execution_history: List[ExecutionContext] = []

    @property
    def metadata(self) -> NodeMetadata:
        """Get node metadata"""
        return self._metadata

    @property
    def node_id(self) -> str:
        """Get node ID"""
        return self._metadata.node_id

    @property
    def status(self) -> NodeStatus:
        """Get node status"""
        return self._metadata.status

    def _update_metadata(self) -> None:
        """Update metadata after state change"""
        self._metadata.updated_at = datetime.now()
        self._metadata.version += 1

    def _update_status(self, status: NodeStatus, error: Optional[str] = None) -> None:
        """Update node status"""
        self._metadata.status = status
        self._metadata.error = error
        self._update_metadata()

    def create_input_channel(
        self,
        name: str,
        channel_type: Type[Channel[T]],
        type_hint: Optional[Type[T]] = None,
        **kwargs
    ) -> Channel[T]:
        """Create an input channel"""
        if name in self._input_channels:
            raise ValueError(f"Input channel '{name}' already exists")

        channel = self._graph_state.create_channel(
            channel_type,
            f"{self.node_id}_in_{name}",
            type_hint=type_hint,
            **kwargs
        )
        self._input_channels[name] = channel
        return channel

    def create_output_channel(
        self,
        name: str,
        channel_type: Type[Channel[T]],
        type_hint: Optional[Type[T]] = None,
        **kwargs
    ) -> Channel[T]:
        """Create an output channel"""
        if name in self._output_channels:
            raise ValueError(f"Output channel '{name}' already exists")

        channel = self._graph_state.create_channel(
            channel_type,
            f"{self.node_id}_out_{name}",
            type_hint=type_hint,
            **kwargs
        )
        self._output_channels[name] = channel
        return channel

    def get_input_channel(self, name: str) -> Optional[Channel]:
        """Get input channel by name"""
        return self._input_channels.get(name)

    def get_output_channel(self, name: str) -> Optional[Channel]:
        """Get output channel by name"""
        return self._output_channels.get(name)

    def list_input_channels(self) -> List[str]:
        """List all input channel names"""
        return list(self._input_channels.keys())

    def list_output_channels(self) -> List[str]:
        """List all output channel names"""
        return list(self._output_channels.keys())

    async def execute(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Execute node with given inputs"""
        context = ExecutionContext(inputs=kwargs)
        self._metadata.execution_count += 1
        self._metadata.last_execution = datetime.now()

        try:
            self._update_status(NodeStatus.RUNNING)
            outputs = await self._execute(**kwargs)
            context.outputs = outputs or {}
            self._update_status(NodeStatus.COMPLETED)
            return outputs
        except Exception as e:
            error_msg = str(e)
            context.error = error_msg
            self._update_status(NodeStatus.FAILED, error_msg)
            raise
        finally:
            self._execution_history.append(context)

    def get_execution_history(self) -> List[ExecutionContext]:
        """Get execution history"""
        return self._execution_history.copy()

    def clear_execution_history(self) -> None:
        """Clear execution history"""
        self._execution_history.clear()

    def checkpoint(self) -> Dict[str, Any]:
        """Create a checkpoint of current state"""
        return {
            "metadata": self._metadata.model_dump(),
            "execution_history": [
                context.model_dump()
                for context in self._execution_history
            ],
            "input_channels": {
                name: channel.id
                for name, channel in self._input_channels.items()
            },
            "output_channels": {
                name: channel.id
                for name, channel in self._output_channels.items()
            }
        }

    def restore(self, checkpoint: Dict[str, Any]) -> None:
        """Restore from checkpoint"""
        self._metadata = NodeMetadata(**checkpoint["metadata"])

        self._execution_history = [
            ExecutionContext(**context)
            for context in checkpoint["execution_history"]
        ]

        # Channels are restored through GraphState
        self._input_channels = {
            name: self._graph_state.get_channel(channel_id)
            for name, channel_id in checkpoint["input_channels"].items()
        }

        self._output_channels = {
            name: self._graph_state.get_channel(channel_id)
            for name, channel_id in checkpoint["output_channels"].items()
        }

    @abstractmethod
    async def _execute(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Execute node implementation

        Args:
        ----
            **kwargs: Execution inputs

        Returns:
        -------
            Optional dictionary of execution outputs

        """
        pass
