"""Agent-specific event types for Legion monitoring system"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import Event, EventCategory, EventSeverity, EventType


@dataclass
class AgentEvent(Event):
    """Base class for agent-specific events"""

    def __init__(self, component_id: str, category: EventCategory, **kwargs):
        super().__init__(
            event_type=EventType.AGENT,
            component_id=component_id,
            category=category,
            **kwargs
        )

@dataclass
class AgentStartEvent(AgentEvent):
    """Emitted when an agent starts up"""

    def __init__(
        self,
        component_id: str,
        system_prompt: str,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            component_id=component_id,
            category=EventCategory.EXECUTION,
            provider_name=provider_name,
            model_name=model_name,
            metadata={
                "system_prompt": system_prompt
            },
            **kwargs
        )

@dataclass
class AgentProcessingEvent(AgentEvent):
    """Emitted during agent message processing"""

    def __init__(
        self,
        component_id: str,
        input_message: str,
        tokens_used: Optional[int] = None,
        cost: Optional[float] = None,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            component_id=component_id,
            category=EventCategory.EXECUTION,
            tokens_used=tokens_used,
            cost=cost,
            provider_name=provider_name,
            model_name=model_name,
            metadata={
                "input_message": input_message
            },
            **kwargs
        )

@dataclass
class AgentDecisionEvent(AgentEvent):
    """Emitted when an agent makes a decision"""

    def __init__(
        self,
        component_id: str,
        decision_type: str,
        options: List[str],
        selected_option: str,
        tokens_used: Optional[int] = None,
        cost: Optional[float] = None,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            component_id=component_id,
            category=EventCategory.EXECUTION,
            tokens_used=tokens_used,
            cost=cost,
            provider_name=provider_name,
            model_name=model_name,
            metadata={
                "decision_type": decision_type,
                "options": options,
                "selected_option": selected_option
            },
            **kwargs
        )

@dataclass
class AgentToolUseEvent(AgentEvent):
    """Emitted when an agent uses a tool"""

    def __init__(
        self,
        component_id: str,
        tool_name: str,
        tool_input: Any,
        tool_output: Optional[Any] = None,
        tokens_used: Optional[int] = None,
        cost: Optional[float] = None,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            component_id=component_id,
            category=EventCategory.EXECUTION,
            tokens_used=tokens_used,
            cost=cost,
            provider_name=provider_name,
            model_name=model_name,
            metadata={
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_output": tool_output
            },
            **kwargs
        )

@dataclass
class AgentMemoryEvent(AgentEvent):
    """Emitted when an agent interacts with memory"""

    def __init__(
        self,
        component_id: str,
        operation: str,
        memory_key: str,
        content: Any,
        tokens_used: Optional[int] = None,
        cost: Optional[float] = None,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            component_id=component_id,
            category=EventCategory.MEMORY,
            tokens_used=tokens_used,
            cost=cost,
            provider_name=provider_name,
            model_name=model_name,
            metadata={
                "operation": operation,
                "memory_key": memory_key,
                "content": content
            },
            **kwargs
        )

@dataclass
class AgentResponseEvent(AgentEvent):
    """Emitted when an agent generates a response"""

    def __init__(
        self,
        component_id: str,
        response: str,
        tokens_used: Optional[int] = None,
        cost: Optional[float] = None,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            component_id=component_id,
            category=EventCategory.EXECUTION,
            tokens_used=tokens_used,
            cost=cost,
            provider_name=provider_name,
            model_name=model_name,
            metadata={
                "response": response
            },
            **kwargs
        )

@dataclass
class AgentErrorEvent(AgentEvent):
    """Emitted when an agent encounters an error"""

    def __init__(
        self,
        component_id: str,
        error_type: str,
        error_message: str,
        traceback: Optional[str] = None,
        tokens_used: Optional[int] = None,
        cost: Optional[float] = None,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            component_id=component_id,
            category=EventCategory.ERROR,
            severity=EventSeverity.ERROR,
            tokens_used=tokens_used,
            cost=cost,
            provider_name=provider_name,
            model_name=model_name,
            metadata={
                "error_type": error_type,
                "error_message": error_message,
                "traceback": traceback
            },
            **kwargs
        )

@dataclass
class AgentStateChangeEvent(AgentEvent):
    """Emitted when an agent's state changes"""

    def __init__(
        self,
        component_id: str,
        old_state: Dict[str, Any],
        new_state: Dict[str, Any],
        change_type: str,
        tokens_used: Optional[int] = None,
        cost: Optional[float] = None,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            component_id=component_id,
            category=EventCategory.EXECUTION,
            tokens_used=tokens_used,
            cost=cost,
            provider_name=provider_name,
            model_name=model_name,
            state_before=old_state,
            state_after=new_state,
            metadata={
                "change_type": change_type
            },
            **kwargs
        )
