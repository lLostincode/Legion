"""Chain-specific event types for Legion monitoring system"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from ...interface.schemas import Message
from .base import Event, EventCategory, EventSeverity, EventType


@dataclass
class ChainEvent(Event):
    """Base class for chain-specific events"""

    def __init__(self, component_id: str, category: EventCategory, **kwargs):
        super().__init__(
            event_type=EventType.CHAIN,
            component_id=component_id,
            category=category,
            **kwargs
        )

@dataclass
class ChainStartEvent(ChainEvent):
    """Emitted when a chain starts processing"""

    def __init__(
        self,
        component_id: str,
        input_message: Union[str, Message],
        member_count: int,
        **kwargs
    ):
        super().__init__(
            component_id=component_id,
            category=EventCategory.EXECUTION,
            metadata={
                "input_message": str(input_message),
                "member_count": member_count
            },
            **kwargs
        )

@dataclass
class ChainStepEvent(ChainEvent):
    """Emitted when a chain step starts processing"""

    def __init__(
        self,
        component_id: str,
        step_name: str,
        step_index: int,
        input_message: Union[str, Message],
        is_final_step: bool,
        **kwargs
    ):
        super().__init__(
            component_id=component_id,
            category=EventCategory.EXECUTION,
            metadata={
                "step_name": step_name,
                "step_index": step_index,
                "input_message": str(input_message),
                "is_final_step": is_final_step
            },
            **kwargs
        )

@dataclass
class ChainTransformEvent(ChainEvent):
    """Emitted when data is transformed between chain steps"""

    def __init__(
        self,
        component_id: str,
        step_name: str,
        step_index: int,
        input_message: Union[str, Message],
        output_message: Union[str, Message],
        transformation_time_ms: float,
        **kwargs
    ):
        super().__init__(
            component_id=component_id,
            category=EventCategory.EXECUTION,
            metadata={
                "step_name": step_name,
                "step_index": step_index,
                "input_message": str(input_message),
                "output_message": str(output_message),
                "transformation_time_ms": transformation_time_ms
            },
            **kwargs
        )

@dataclass
class ChainCompletionEvent(ChainEvent):
    """Emitted when a chain completes processing"""

    def __init__(
        self,
        component_id: str,
        input_message: Union[str, Message],
        output_message: Union[str, Message],
        total_time_ms: float,
        step_times: Dict[str, float],
        **kwargs
    ):
        super().__init__(
            component_id=component_id,
            category=EventCategory.EXECUTION,
            metadata={
                "input_message": str(input_message),
                "output_message": str(output_message),
                "total_time_ms": total_time_ms,
                "step_times": step_times
            },
            **kwargs
        )

@dataclass
class ChainErrorEvent(ChainEvent):
    """Emitted when a chain encounters an error"""

    def __init__(
        self,
        component_id: str,
        error_type: str,
        error_message: str,
        step_name: Optional[str] = None,
        step_index: Optional[int] = None,
        traceback: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            component_id=component_id,
            category=EventCategory.ERROR,
            severity=EventSeverity.ERROR,
            metadata={
                "error_type": error_type,
                "error_message": error_message,
                "step_name": step_name,
                "step_index": step_index,
                "traceback": traceback
            },
            **kwargs
        )

@dataclass
class ChainStateChangeEvent(ChainEvent):
    """Emitted when chain configuration or state changes"""

    def __init__(
        self,
        component_id: str,
        change_type: str,
        old_state: Dict[str, Any],
        new_state: Dict[str, Any],
        change_reason: str,
        **kwargs
    ):
        super().__init__(
            component_id=component_id,
            category=EventCategory.EXECUTION,
            metadata={
                "change_type": change_type,
                "old_state": old_state,
                "new_state": new_state,
                "change_reason": change_reason
            },
            **kwargs
        )

@dataclass
class ChainBottleneckEvent(ChainEvent):
    """Emitted when a potential bottleneck is detected in the chain"""

    def __init__(
        self,
        component_id: str,
        step_name: str,
        step_index: int,
        processing_time_ms: float,
        average_time_ms: float,
        threshold_ms: float,
        **kwargs
    ):
        super().__init__(
            component_id=component_id,
            category=EventCategory.EXECUTION,
            severity=EventSeverity.WARNING,
            metadata={
                "step_name": step_name,
                "step_index": step_index,
                "processing_time_ms": processing_time_ms,
                "average_time_ms": average_time_ms,
                "threshold_ms": threshold_ms,
                "slowdown_factor": processing_time_ms / average_time_ms
            },
            **kwargs
        )
