"""Team-specific event types for Legion monitoring system"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import Event, EventCategory, EventSeverity, EventType


@dataclass
class TeamEvent(Event):
    """Base class for team-specific events"""

    def __init__(self, component_id: str, category: EventCategory, **kwargs):
        super().__init__(
            event_type=EventType.TEAM,
            component_id=component_id,
            category=category,
            **kwargs
        )

@dataclass
class TeamFormationEvent(TeamEvent):
    """Emitted when a team is created or modified"""

    def __init__(
        self,
        component_id: str,
        leader_id: str,
        member_ids: List[str],
        team_config: Dict[str, Any],
        **kwargs
    ):
        super().__init__(
            component_id=component_id,
            category=EventCategory.EXECUTION,
            metadata={
                "leader_id": leader_id,
                "member_ids": member_ids,
                "team_config": team_config
            },
            **kwargs
        )

@dataclass
class TeamDelegationEvent(TeamEvent):
    """Emitted when a task is delegated to a team member"""

    def __init__(
        self,
        component_id: str,
        leader_id: str,
        member_id: str,
        task_type: str,
        task_input: Dict[str, Any],
        delegation_context: Dict[str, Any],
        **kwargs
    ):
        super().__init__(
            component_id=component_id,
            category=EventCategory.EXECUTION,
            metadata={
                "leader_id": leader_id,
                "member_id": member_id,
                "task_type": task_type,
                "task_input": task_input,
                "delegation_context": delegation_context
            },
            **kwargs
        )

@dataclass
class TeamLeadershipEvent(TeamEvent):
    """Emitted when a leader makes a strategic decision"""

    def __init__(
        self,
        component_id: str,
        leader_id: str,
        decision_type: str,
        decision_context: Dict[str, Any],
        reasoning: str,
        affected_members: List[str],
        **kwargs
    ):
        super().__init__(
            component_id=component_id,
            category=EventCategory.EXECUTION,
            metadata={
                "leader_id": leader_id,
                "decision_type": decision_type,
                "decision_context": decision_context,
                "reasoning": reasoning,
                "affected_members": affected_members
            },
            **kwargs
        )

@dataclass
class TeamCommunicationEvent(TeamEvent):
    """Emitted when team members communicate"""

    def __init__(
        self,
        component_id: str,
        sender_id: str,
        receiver_id: str,
        message_type: str,
        message_content: Any,
        context: Dict[str, Any],
        **kwargs
    ):
        super().__init__(
            component_id=component_id,
            category=EventCategory.EXECUTION,
            metadata={
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "message_type": message_type,
                "message_content": message_content,
                "context": context
            },
            **kwargs
        )

@dataclass
class TeamCompletionEvent(TeamEvent):
    """Emitted when a team task is completed"""

    def __init__(
        self,
        component_id: str,
        task_type: str,
        task_input: Dict[str, Any],
        task_output: Any,
        duration_ms: float,
        member_contributions: Dict[str, Any],
        **kwargs
    ):
        super().__init__(
            component_id=component_id,
            category=EventCategory.EXECUTION,
            metadata={
                "task_type": task_type,
                "task_input": task_input,
                "task_output": task_output,
                "duration_ms": duration_ms,
                "member_contributions": member_contributions
            },
            **kwargs
        )

@dataclass
class TeamPerformanceEvent(TeamEvent):
    """Emitted to track team performance metrics"""

    def __init__(
        self,
        component_id: str,
        metric_type: str,
        metric_value: float,
        metric_context: Dict[str, Any],
        time_window_ms: Optional[float] = None,
        **kwargs
    ):
        super().__init__(
            component_id=component_id,
            category=EventCategory.EXECUTION,
            metadata={
                "metric_type": metric_type,
                "metric_value": metric_value,
                "metric_context": metric_context,
                "time_window_ms": time_window_ms
            },
            **kwargs
        )

@dataclass
class TeamStateChangeEvent(TeamEvent):
    """Emitted when team structure or state changes"""

    def __init__(
        self,
        component_id: str,
        change_type: str,
        old_state: Dict[str, Any],
        new_state: Dict[str, Any],
        change_reason: str,
        affected_members: List[str],
        **kwargs
    ):
        super().__init__(
            component_id=component_id,
            category=EventCategory.EXECUTION,
            metadata={
                "change_type": change_type,
                "old_state": old_state,
                "new_state": new_state,
                "change_reason": change_reason,
                "affected_members": affected_members
            },
            **kwargs
        )

@dataclass
class TeamErrorEvent(TeamEvent):
    """Emitted when team coordination fails"""

    def __init__(
        self,
        component_id: str,
        error_type: str,
        error_message: str,
        error_context: Dict[str, Any],
        affected_members: List[str],
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
                "error_context": error_context,
                "affected_members": affected_members,
                "traceback": traceback
            },
            **kwargs
        )
