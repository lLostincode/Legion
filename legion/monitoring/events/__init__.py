"""Event types for Legion monitoring system"""

from .base import (
    Event,
    EventType,
    EventCategory,
    EventSeverity,
    EventEmitter
)

from .agent import (
    AgentEvent,
    AgentStartEvent,
    AgentProcessingEvent,
    AgentDecisionEvent,
    AgentToolUseEvent,
    AgentMemoryEvent,
    AgentResponseEvent,
    AgentErrorEvent,
    AgentStateChangeEvent
)

from .team import (
    TeamEvent,
    TeamFormationEvent,
    TeamDelegationEvent,
    TeamLeadershipEvent,
    TeamCommunicationEvent,
    TeamCompletionEvent,
    TeamPerformanceEvent,
    TeamStateChangeEvent,
    TeamErrorEvent
)

from .chain import (
    ChainEvent,
    ChainStartEvent,
    ChainStepEvent,
    ChainTransformEvent,
    ChainCompletionEvent,
    ChainErrorEvent,
    ChainStateChangeEvent,
    ChainBottleneckEvent
)

__all__ = [
    # Base types
    'Event',
    'EventType',
    'EventCategory',
    'EventSeverity',
    'EventEmitter',
    
    # Agent events
    'AgentEvent',
    'AgentStartEvent',
    'AgentProcessingEvent',
    'AgentDecisionEvent',
    'AgentToolUseEvent',
    'AgentMemoryEvent',
    'AgentResponseEvent',
    'AgentErrorEvent',
    'AgentStateChangeEvent',
    
    # Team events
    'TeamEvent',
    'TeamFormationEvent',
    'TeamDelegationEvent',
    'TeamLeadershipEvent',
    'TeamCommunicationEvent',
    'TeamCompletionEvent',
    'TeamPerformanceEvent',
    'TeamStateChangeEvent',
    'TeamErrorEvent',
    
    # Chain events
    'ChainEvent',
    'ChainStartEvent',
    'ChainStepEvent',
    'ChainTransformEvent',
    'ChainCompletionEvent',
    'ChainErrorEvent',
    'ChainStateChangeEvent',
    'ChainBottleneckEvent'
] 