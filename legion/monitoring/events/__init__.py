"""Event types for Legion monitoring system"""

from .agent import (
    AgentDecisionEvent,
    AgentErrorEvent,
    AgentEvent,
    AgentMemoryEvent,
    AgentProcessingEvent,
    AgentResponseEvent,
    AgentStartEvent,
    AgentStateChangeEvent,
    AgentToolUseEvent,
)
from .base import Event, EventCategory, EventEmitter, EventSeverity, EventType
from .chain import (
    ChainBottleneckEvent,
    ChainCompletionEvent,
    ChainErrorEvent,
    ChainEvent,
    ChainStartEvent,
    ChainStateChangeEvent,
    ChainStepEvent,
    ChainTransformEvent,
)
from .team import (
    TeamCommunicationEvent,
    TeamCompletionEvent,
    TeamDelegationEvent,
    TeamErrorEvent,
    TeamEvent,
    TeamFormationEvent,
    TeamLeadershipEvent,
    TeamPerformanceEvent,
    TeamStateChangeEvent,
)

__all__ = [
    # Base types
    "Event",
    "EventType",
    "EventCategory",
    "EventSeverity",
    "EventEmitter",

    # Agent events
    "AgentEvent",
    "AgentStartEvent",
    "AgentProcessingEvent",
    "AgentDecisionEvent",
    "AgentToolUseEvent",
    "AgentMemoryEvent",
    "AgentResponseEvent",
    "AgentErrorEvent",
    "AgentStateChangeEvent",

    # Team events
    "TeamEvent",
    "TeamFormationEvent",
    "TeamDelegationEvent",
    "TeamLeadershipEvent",
    "TeamCommunicationEvent",
    "TeamCompletionEvent",
    "TeamPerformanceEvent",
    "TeamStateChangeEvent",
    "TeamErrorEvent",

    # Chain events
    "ChainEvent",
    "ChainStartEvent",
    "ChainStepEvent",
    "ChainTransformEvent",
    "ChainCompletionEvent",
    "ChainErrorEvent",
    "ChainStateChangeEvent",
    "ChainBottleneckEvent"
]
