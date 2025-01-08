"""Decorators for adding monitoring capabilities to Legion components"""

import time
import traceback
from typing import Any, Type, TypeVar
from uuid import uuid4

from legion.monitoring.events import (
    AgentDecisionEvent,
    AgentErrorEvent,
    AgentMemoryEvent,
    AgentProcessingEvent,
    AgentResponseEvent,
    # Agent events
    AgentStartEvent,
    AgentToolUseEvent,
    # Base types
    EventEmitter,
    TeamCommunicationEvent,
    TeamCompletionEvent,
    TeamDelegationEvent,
    TeamErrorEvent,
    TeamFormationEvent,
    TeamLeadershipEvent,
    TeamStateChangeEvent,
)
from legion.monitoring.registry import MonitorRegistry

T = TypeVar("T")

def monitored_agent(cls: Type[T]) -> Type[T]:
    """Decorator that adds monitoring capabilities to an agent class"""
    # Store original methods
    original_init = getattr(cls, "__init__", None)
    original_process = getattr(cls, "process", None)
    original_decide = getattr(cls, "decide", None)
    original_use_tool = getattr(cls, "use_tool", None)
    original_remember = getattr(cls, "remember", None)
    original_recall = getattr(cls, "recall", None)

    def __monitored_init__(self: Any, *args: Any, **kwargs: Any) -> None:
        """Monitored initialization"""
        # Initialize EventEmitter
        self.event_emitter = EventEmitter()
        self.id = str(uuid4())

        # Register with MonitorRegistry
        MonitorRegistry.get_instance().register_component(self.id, self.event_emitter)

        # Call original init
        if original_init:
            original_init(self, *args, **kwargs)

        # Emit start event
        self.event_emitter.emit(AgentStartEvent(
            component_id=self.id,
            system_prompt=getattr(self, "system_prompt", None),
            config=kwargs.get("config", {})
        ))

    def __monitored_process__(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Monitored message processing"""
        start_time = time.time()

        try:
            # Emit processing event
            processing_event = AgentProcessingEvent(
                component_id=self.id,
                input_message=args[0] if args else None,
                context=kwargs
            )
            self.event_emitter.emit(processing_event)

            # Process message
            result = original_process(self, *args, **kwargs) if original_process else None

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Emit response event
            self.event_emitter.emit(AgentResponseEvent(
                component_id=self.id,
                input_message=args[0] if args else None,
                output_message=result,
                duration_ms=duration_ms,
                parent_event_id=processing_event.id
            ))

            return result

        except Exception as e:
            # Emit error event
            self.event_emitter.emit(AgentErrorEvent(
                component_id=self.id,
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc()
            ))
            raise

    def __monitored_decide__(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Monitored decision making"""
        try:
            # Emit decision event
            decision_event = AgentDecisionEvent(
                component_id=self.id,
                decision_type=kwargs.get("decision_type", "unknown"),
                context=kwargs,
                options=kwargs.get("options", [])
            )
            self.event_emitter.emit(decision_event)

            # Make decision
            result = original_decide(self, *args, **kwargs) if original_decide else None

            # Update decision event with result
            decision_event.metadata["decision"] = result

            return result

        except Exception as e:
            # Emit error event
            self.event_emitter.emit(AgentErrorEvent(
                component_id=self.id,
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc()
            ))
            raise

    def __monitored_use_tool__(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Monitored tool usage"""
        start_time = time.time()

        try:
            # Emit tool use event
            tool_event = AgentToolUseEvent(
                component_id=self.id,
                tool_name=args[0] if args else kwargs.get("tool_name"),
                tool_input=args[1] if len(args) > 1 else kwargs.get("tool_input"),
                context=kwargs
            )
            self.event_emitter.emit(tool_event)

            # Use tool
            result = original_use_tool(self, *args, **kwargs) if original_use_tool else None

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Update tool event with result
            tool_event.metadata["output"] = result
            tool_event.metadata["duration_ms"] = duration_ms

            return result

        except Exception as e:
            # Emit error event
            self.event_emitter.emit(AgentErrorEvent(
                component_id=self.id,
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc()
            ))
            raise

    def __monitored_remember__(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Monitored memory storage"""
        try:
            # Emit memory event
            memory_event = AgentMemoryEvent(
                component_id=self.id,
                operation="store",
                content=args[0] if args else kwargs.get("content"),
                context=kwargs
            )
            self.event_emitter.emit(memory_event)

            # Store memory
            result = original_remember(self, *args, **kwargs) if original_remember else None

            # Update memory event with result
            memory_event.metadata["result"] = result

            return result

        except Exception as e:
            # Emit error event
            self.event_emitter.emit(AgentErrorEvent(
                component_id=self.id,
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc()
            ))
            raise

    def __monitored_recall__(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Monitored memory retrieval"""
        try:
            # Emit memory event
            memory_event = AgentMemoryEvent(
                component_id=self.id,
                operation="retrieve",
                query=args[0] if args else kwargs.get("query"),
                context=kwargs
            )
            self.event_emitter.emit(memory_event)

            # Retrieve memory
            result = original_recall(self, *args, **kwargs) if original_recall else None

            # Update memory event with result
            memory_event.metadata["result"] = result

            return result

        except Exception as e:
            # Emit error event
            self.event_emitter.emit(AgentErrorEvent(
                component_id=self.id,
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc()
            ))
            raise

    # Replace methods with monitored versions
    setattr(cls, "__init__", __monitored_init__)
    if original_process:
        setattr(cls, "process", __monitored_process__)
    if original_decide:
        setattr(cls, "decide", __monitored_decide__)
    if original_use_tool:
        setattr(cls, "use_tool", __monitored_use_tool__)
    if original_remember:
        setattr(cls, "remember", __monitored_remember__)
    if original_recall:
        setattr(cls, "recall", __monitored_recall__)

    return cls

def monitored_team(cls: Type[T]) -> Type[T]:
    """Decorator that adds monitoring capabilities to a team class"""
    # Store original methods
    original_init = getattr(cls, "__init__", None)
    original_add_member = getattr(cls, "add_member", None)
    original_remove_member = getattr(cls, "remove_member", None)
    original_assign_leader = getattr(cls, "assign_leader", None)
    original_delegate_task = getattr(cls, "delegate_task", None)
    original_process_message = getattr(cls, "process_message", None)
    original_complete_task = getattr(cls, "complete_task", None)

    def __monitored_init__(self: Any, *args: Any, **kwargs: Any) -> None:
        """Monitored initialization"""
        # Initialize EventEmitter and ID first
        self.event_emitter = EventEmitter()
        self.id = str(uuid4())

        # Initialize default attributes
        self.member_ids = []
        self.leader_id = None
        self.config = kwargs.get("config", {})

        # Call original init to allow overriding defaults
        if original_init:
            original_init(self, *args, **kwargs)

        # Register with MonitorRegistry
        registry = MonitorRegistry()
        registry.register_component(self.id, self.event_emitter)

        # Emit formation event after registration
        formation_event = TeamFormationEvent(
            component_id=self.id,
            leader_id=self.leader_id,
            member_ids=self.member_ids,
            team_config=self.config
        )
        self.event_emitter.emit(formation_event)

    def __monitored_add_member__(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Monitored member addition"""
        try:
            old_state = {
                "members": getattr(self, "member_ids", []).copy()
            }

            # Add member
            result = original_add_member(self, *args, **kwargs) if original_add_member else None

            new_state = {
                "members": getattr(self, "member_ids", []).copy()
            }

            # Emit state change event
            event = TeamStateChangeEvent(
                component_id=self.id,
                change_type="member_added",
                old_state=old_state,
                new_state=new_state,
                change_reason="Member addition",
                affected_members=[args[0]] if args else [kwargs.get("member_id")]
            )
            self.event_emitter.emit(event)

            return result

        except Exception as e:
            # Emit error event
            error_event = TeamErrorEvent(
                component_id=self.id,
                error_type=type(e).__name__,
                error_message=str(e),
                error_context={"operation": "add_member"},
                affected_members=[args[0]] if args else [kwargs.get("member_id")],
                traceback=traceback.format_exc()
            )
            self.event_emitter.emit(error_event)
            raise

    def __monitored_remove_member__(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Monitored member removal"""
        try:
            old_state = {
                "members": getattr(self, "member_ids", []).copy()
            }

            # Remove member
            result = original_remove_member(self, *args, **kwargs) if original_remove_member else None

            new_state = {
                "members": getattr(self, "member_ids", []).copy()
            }

            # Emit state change event
            self.event_emitter.emit(TeamStateChangeEvent(
                component_id=self.id,
                change_type="member_removed",
                old_state=old_state,
                new_state=new_state,
                change_reason="Member removal",
                affected_members=[args[0]] if args else [kwargs.get("member_id")]
            ))

            return result

        except Exception as e:
            # Emit error event
            self.event_emitter.emit(TeamErrorEvent(
                component_id=self.id,
                error_type=type(e).__name__,
                error_message=str(e),
                error_context={"operation": "remove_member"},
                affected_members=[args[0]] if args else [kwargs.get("member_id")],
                traceback=traceback.format_exc()
            ))
            raise

    def __monitored_assign_leader__(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Monitored leader assignment"""
        try:
            old_leader = getattr(self, "leader_id", None)

            # Assign leader
            result = original_assign_leader(self, *args, **kwargs) if original_assign_leader else None

            new_leader = getattr(self, "leader_id", None)

            # Emit leadership event
            self.event_emitter.emit(TeamLeadershipEvent(
                component_id=self.id,
                leader_id=new_leader,
                decision_type="leader_assignment",
                decision_context={"old_leader": old_leader},
                reasoning=kwargs.get("reason", "Leadership change"),
                affected_members=[old_leader, new_leader] if old_leader else [new_leader]
            ))

            return result

        except Exception as e:
            # Emit error event
            self.event_emitter.emit(TeamErrorEvent(
                component_id=self.id,
                error_type=type(e).__name__,
                error_message=str(e),
                error_context={"operation": "assign_leader"},
                affected_members=[args[0]] if args else [kwargs.get("leader_id")],
                traceback=traceback.format_exc()
            ))
            raise

    def __monitored_delegate_task__(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Monitored task delegation"""
        try:
            # Emit delegation event
            delegation_event = TeamDelegationEvent(
                component_id=self.id,
                leader_id=getattr(self, "leader_id", None),
                member_id=args[0] if args else kwargs.get("member_id"),
                task_type=args[1] if len(args) > 1 else kwargs.get("task_type"),
                task_input=args[2] if len(args) > 2 else kwargs.get("task_input"),
                delegation_context=kwargs
            )
            self.event_emitter.emit(delegation_event)

            # Delegate task
            result = original_delegate_task(self, *args, **kwargs) if original_delegate_task else None

            # Update delegation event with result
            delegation_event.metadata["result"] = result

            return result

        except Exception as e:
            # Emit error event
            self.event_emitter.emit(TeamErrorEvent(
                component_id=self.id,
                error_type=type(e).__name__,
                error_message=str(e),
                error_context={"operation": "delegate_task"},
                affected_members=[args[0]] if args else [kwargs.get("member_id")],
                traceback=traceback.format_exc()
            ))
            raise

    def __monitored_process_message__(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Monitored message processing"""
        try:
            # Emit communication event
            communication_event = TeamCommunicationEvent(
                component_id=self.id,
                sender_id=args[0] if args else kwargs.get("sender_id"),
                receiver_id=args[1] if len(args) > 1 else kwargs.get("receiver_id"),
                message_type=args[2] if len(args) > 2 else kwargs.get("message_type"),
                message_content=args[3] if len(args) > 3 else kwargs.get("message_content"),
                context=kwargs
            )
            self.event_emitter.emit(communication_event)

            # Process message
            result = original_process_message(self, *args, **kwargs) if original_process_message else None

            # Update communication event with result
            communication_event.metadata["result"] = result

            return result

        except Exception as e:
            # Emit error event
            self.event_emitter.emit(TeamErrorEvent(
                component_id=self.id,
                error_type=type(e).__name__,
                error_message=str(e),
                error_context={"operation": "process_message"},
                affected_members=[args[0], args[1]] if len(args) > 1 else [kwargs.get("sender_id"), kwargs.get("receiver_id")],
                traceback=traceback.format_exc()
            ))
            raise

    def __monitored_complete_task__(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Monitored task completion"""
        try:
            start_time = time.time()

            # Complete task
            result = original_complete_task(self, *args, **kwargs) if original_complete_task else None

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Get task parameters
            task_type = args[0] if args else kwargs.get("task_type")
            task_input = args[1] if len(args) > 1 else kwargs.get("task_input")
            member_contributions = args[2] if len(args) > 2 else kwargs.get("member_contributions", {})

            # Emit completion event
            self.event_emitter.emit(TeamCompletionEvent(
                component_id=self.id,
                task_type=task_type,
                task_input=task_input,
                task_output=result,
                duration_ms=duration_ms,
                member_contributions=member_contributions
            ))

            return result

        except Exception as e:
            # Emit error event
            self.event_emitter.emit(TeamErrorEvent(
                component_id=self.id,
                error_type=type(e).__name__,
                error_message=str(e),
                error_context={"operation": "complete_task"},
                affected_members=getattr(self, "member_ids", []),
                traceback=traceback.format_exc()
            ))
            raise

    # Replace methods with monitored versions
    setattr(cls, "__init__", __monitored_init__)
    if original_add_member:
        setattr(cls, "add_member", __monitored_add_member__)
    if original_remove_member:
        setattr(cls, "remove_member", __monitored_remove_member__)
    if original_assign_leader:
        setattr(cls, "assign_leader", __monitored_assign_leader__)
    if original_delegate_task:
        setattr(cls, "delegate_task", __monitored_delegate_task__)
    if original_process_message:
        setattr(cls, "process_message", __monitored_process_message__)
    if original_complete_task:
        setattr(cls, "complete_task", __monitored_complete_task__)

    return cls
