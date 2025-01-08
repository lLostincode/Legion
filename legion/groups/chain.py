import time
import traceback
from asyncio.log import logger
from typing import Any, Dict, List, Optional, OrderedDict, Type, Union

from pydantic import BaseModel
from rich import print as rprint

from legion.interface.schemas import Message, ModelResponse, Role, SystemPrompt, SystemPromptSection
from legion.monitoring.events.base import EventEmitter
from legion.monitoring.events.chain import (
    ChainBottleneckEvent,
    ChainCompletionEvent,
    ChainErrorEvent,
    ChainStartEvent,
    ChainStateChangeEvent,
    ChainStepEvent,
    ChainTransformEvent,
)

from ..agents.base import Agent
from ..blocks.base import FunctionalBlock
from .base import BaseGroup, GroupMetadata

# Chain-specific system prompts
CHAIN_PROMPTS = {
    "chain": """You are part of a processing chain. Your role is to process incoming content and improve/transform it based on your expertise.
    Your output will be passed to the next agent in the chain, so focus on your specific transformation and trust that other agents will handle their parts.

    Your position: {position}"""
}

class Chain(BaseGroup, EventEmitter):
    """Sequential processing chain where each member's output becomes the next member's input.

    Example:
    -------
        chain = Chain(
            name="data_pipeline",
            members=[
                data_cleaner,    # First processes raw data
                data_analyzer,   # Then analyzes cleaned data
                report_writer    # Finally creates report from analysis
            ]
        )

        # Basic usage
        result = chain.process("Here's the raw data: ...")

        # With JSON schema for final output
        result = chain.process("Here's the raw data: ...", response_schema=OutputSchema)

    """

    def __init__(
        self,
        name: str,
        members: List[Union[Agent, BaseGroup, FunctionalBlock]],
        debug: bool = False,
        verbose: bool = False,
        bottleneck_threshold_ms: float = 5000.0
    ):
        """Initialize chain

        Args:
        ----
            name: Chain name
            members: Ordered list of chain members
            debug: Enable debug mode
            verbose: Enable verbose logging

        """
        BaseGroup.__init__(self, name=name, debug=debug, verbose=verbose)
        EventEmitter.__init__(self)

        if len(members) < 2:
            raise ValueError("Chain must have at least 2 members")

        # Convert list to OrderedDict with auto-generated names
        self._members = OrderedDict()
        for i, member in enumerate(members):
            if isinstance(member, (Agent, BaseGroup, FunctionalBlock)):
                self._members[f"step_{i+1}"] = member
            elif isinstance(member, type):
                if hasattr(member, "__agent_decorator__"):
                    self._members[f"step_{i+1}"] = member()
                elif hasattr(member, "__block_decorator__"):
                    self._members[f"step_{i+1}"] = member()
                elif hasattr(member, "__chain_decorator__"):
                    self._members[f"step_{i+1}"] = member()
                else:
                    raise TypeError(f"Invalid member type: {type(member)}. Member must be an Agent, BaseGroup, FunctionalBlock, or a decorated class.")
            else:
                raise TypeError(f"Invalid member type: {type(member)}. Member must be an Agent, BaseGroup, FunctionalBlock, or a decorated class.")

        # Update metadata for chain type
        self.metadata.group_type = "chain"
        self._setup_members()

        # Initialize performance tracking
        self._step_times: Dict[str, float] = {}
        self._step_averages: Dict[str, float] = {}
        self._bottleneck_threshold_ms = bottleneck_threshold_ms

    @property
    def members(self) -> OrderedDict:
        """Get chain members"""
        return self._members

    def add_member(self, name: str, member: Union[Agent, BaseGroup, FunctionalBlock]) -> None:
        """Add a new member to the chain with validation"""
        # Check for circular references before adding
        if hasattr(member, "metadata"):
            self._check_circular_reference(member)

        # Get old state for event
        old_state = {
            "member_count": len(self._members),
            "member_names": list(self._members.keys())
        }

        self._members[name] = member
        # Set up the new member
        self._setup_member(name, member, len(self._members) - 1)

        # Emit state change event
        new_state = {
            "member_count": len(self._members),
            "member_names": list(self._members.keys())
        }
        self.emit_event(ChainStateChangeEvent(
            component_id=self.name,
            change_type="member_added",
            old_state=old_state,
            new_state=new_state,
            change_reason=f"Added new member: {name}"
        ))

    def _setup_member(self, step_name: str, member: Union[Agent, BaseGroup, FunctionalBlock], position_idx: int) -> None:
        """Configure a single member with chain context"""
        position = (
            "first" if position_idx == 0
            else "last" if position_idx == len(self.members)-1
            else f"position {position_idx+1}"
        )

        if isinstance(member, Agent):
            # Get existing sections or create empty list
            existing_sections = member.system_prompt.sections if member.system_prompt else []

            # Add chain-specific section
            chain_section = SystemPromptSection(
                content=CHAIN_PROMPTS["chain"].format(position=position),
                is_dynamic=False
            )

            # Create new system prompt with all sections
            member.system_prompt = SystemPrompt(
                sections=existing_sections + [chain_section]
            )

        elif isinstance(member, BaseGroup):
            member.set_parent(self)
            member.metadata = GroupMetadata(
                group_type="chain_member",
                depth=member.metadata.depth,
                path=member.metadata.path,
                position=position
            )

            member.debug = self.debug
            member.verbose = self.verbose

    def _setup_members(self):
        """Configure members with chain context"""
        for i, (step_name, member) in enumerate(self.members.items()):
            self._setup_member(step_name, member, i)

    def _log_step_output(self, member_name: str, content: str, is_final: bool = False) -> None:
        """Log the output of a chain step with proper formatting"""
        if not self.verbose:
            return

        self._log_message(
            f"\nOutput from {member_name}:",
            color="yellow"
        )
        print("=" * 40)
        print(content.strip())
        print("=" * 40)

        if not is_final:
            self._log_message(
                "↓ Passing output to next member ↓",
                color="blue"
            )

    def _get_member_name(self, member: Union[Agent, BaseGroup, FunctionalBlock]) -> str:
        """Get the name of a member, handling blocks that don't have a name attribute"""
        if isinstance(member, FunctionalBlock):
            return member.metadata.name
        return member.name

    def _check_bottleneck(self, step_name: str, step_index: int, processing_time_ms: float) -> None:
        """Check for potential bottlenecks in processing time"""
        # Update running average
        if step_name not in self._step_averages:
            self._step_averages[step_name] = processing_time_ms
        else:
            # Simple moving average
            self._step_averages[step_name] = (
                0.7 * self._step_averages[step_name] +
                0.3 * processing_time_ms
            )

        # Check for bottleneck
        if processing_time_ms > self._bottleneck_threshold_ms:
            self.emit_event(ChainBottleneckEvent(
                component_id=self.name,
                step_name=step_name,
                step_index=step_index + 1,  # Convert to 1-based indexing for events
                processing_time_ms=processing_time_ms,
                average_time_ms=self._step_averages[step_name],
                threshold_ms=self._bottleneck_threshold_ms
            ))

    async def process(
        self,
        message: Union[str, Message],
        response_schema: Optional[Type[BaseModel]] = None,
        parent_context: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Process message through chain of agents/groups"""
        # Convert initial message
        current_message = message if isinstance(message, Message) else Message(
            role=Role.USER,
            content=message
        )

        if self.verbose:
            self._log_message(f"\n⛓️  Chain: {self.name}", color="cyan")
            self._log_message("\n📊 Chain Structure:")
            self.print_hierarchy()
            self._log_message(f"📨 Input: {current_message.content}", color="blue")

        # Emit chain start event
        chain_start_time = time.time()
        self.emit_event(ChainStartEvent(
            component_id=self.name,
            input_message=current_message,
            member_count=len(self._members)
        ))

        for i, (step_name, member) in enumerate(self._members.items()):
            if self.verbose:
                member_name = self._get_member_name(member)
                self._log_message(
                    f"\n🔄 Step {i+1}: {member_name} "
                    f"({'nested ' + member.metadata.group_type if isinstance(member, BaseGroup) else 'agent' if isinstance(member, Agent) else 'block'})",
                    color="yellow"
                )

            # Emit step start event
            is_final = i == len(self._members) - 1
            self.emit_event(ChainStepEvent(
                component_id=self.name,
                step_name=step_name,
                step_index=i,
                input_message=current_message,
                is_final_step=is_final
            ))

            try:
                step_start_time = time.time()

                # Process based on member type
                if isinstance(member, BaseGroup):
                    response = await member.aprocess(
                        current_message,
                        response_schema=response_schema if is_final else None,
                        parent_context=parent_context
                    )
                elif isinstance(member, FunctionalBlock):
                    block_response = await member(current_message.content)
                    response = Message(
                        role=Role.USER,
                        content=str(block_response)
                    )
                else:  # Agent
                    response = await member.aprocess(
                        current_message,
                        response_schema=response_schema if is_final else None
                    )

                # Convert ModelResponse to Message if needed
                if isinstance(response, ModelResponse):
                    current_message = Message(
                        role=Role.USER,
                        content=response.content,
                        tool_calls=response.tool_calls
                    )
                else:
                    current_message = response

                # Track step timing
                step_time_ms = (time.time() - step_start_time) * 1000
                self._step_times[step_name] = step_time_ms
                self._check_bottleneck(step_name, i, step_time_ms)

                # Log output
                member_name = self._get_member_name(member)
                self._log_step_output(member_name, current_message.content, is_final)

                # Emit transform event
                self.emit_event(ChainTransformEvent(
                    component_id=self.name,
                    step_name=step_name,
                    step_index=i,
                    input_message=current_message,
                    output_message=current_message,
                    transformation_time_ms=step_time_ms
                ))

            except Exception as e:
                self.emit_event(ChainErrorEvent(
                    component_id=self.name,
                    step_name=step_name,
                    step_index=i,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    traceback=traceback.format_exc()
                ))
                raise

        # Calculate total time
        chain_time_ms = (time.time() - chain_start_time) * 1000

        # Emit completion event
        self.emit_event(ChainCompletionEvent(
            component_id=self.name,
            input_message=message,
            output_message=current_message,
            total_time_ms=chain_time_ms,
            step_times=self._step_times.copy()
        ))

        return current_message

    async def aprocess(
        self,
        message: Union[str, Message],
        response_schema: Optional[Type[BaseModel]] = None,
        parent_context: Optional[Dict[str, Any]] = None
    ) -> ModelResponse:
        """Process message through chain of agents/groups asynchronously"""
        # Convert initial message
        current_message = message if isinstance(message, Message) else Message(
            role=Role.USER,
            content=str(message) if isinstance(message, (str, BaseModel)) else message.model_dump_json()
        )

        if self.verbose:
            self._log_message(f"\n⛓️  Chain: {self.name}", color="cyan")
            self._log_message("\n📊 Chain Structure:")
            self.print_hierarchy()
            self._log_message(f"📨 Input: {current_message.content}", color="blue")

        # Emit chain start event
        chain_start_time = time.time()
        self.emit_event(ChainStartEvent(
            component_id=self.name,
            input_message=current_message,
            member_count=len(self._members)
        ))

        for i, (step_name, member) in enumerate(self._members.items()):
            if self.verbose:
                member_name = self._get_member_name(member)
                self._log_message(
                    f"\n🔄 Step {i+1}: {member_name} "
                    f"({'nested ' + member.metadata.group_type if isinstance(member, BaseGroup) else 'agent' if isinstance(member, Agent) else 'block'})",
                    color="yellow"
                )

            # Emit step start event
            is_final = i == len(self._members) - 1
            self.emit_event(ChainStepEvent(
                component_id=self.name,
                step_name=step_name,
                step_index=i,
                input_message=current_message,
                is_final_step=is_final
            ))

            try:
                step_start_time = time.time()

                # Process based on member type
                if isinstance(member, BaseGroup):
                    response = await member.aprocess(
                        current_message,
                        response_schema=response_schema if is_final else None,
                        parent_context=parent_context
                    )
                elif isinstance(member, FunctionalBlock):
                    # Handle block input conversion
                    if hasattr(member.metadata, "input_schema"):
                        try:
                            # Try to parse JSON if content is a string
                            if isinstance(current_message.content, str):
                                try:
                                    input_data = member.metadata.input_schema.model_validate_json(current_message.content)
                                except:
                                    # If not JSON, try direct validation
                                    input_data = member.metadata.input_schema(text=current_message.content)
                            else:
                                input_data = current_message.content
                        except Exception as e:
                            logger.error(f"Failed to convert input for block {member.metadata.name}: {e}")
                            raise
                    else:
                        input_data = current_message.content

                    # Execute block
                    block_response = await member(input_data)

                    # Handle block output conversion
                    if isinstance(block_response, BaseModel):
                        response = Message(
                            role=Role.USER,
                            content=block_response.model_dump_json()
                        )
                    else:
                        response = Message(
                            role=Role.USER,
                            content=str(block_response)
                        )
                else:  # Agent
                    response = await member.aprocess(
                        current_message,
                        response_schema=response_schema if is_final else None
                    )

                # Calculate step time
                step_time_ms = (time.time() - step_start_time) * 1000
                self._step_times[step_name] = step_time_ms

                # Emit transform event
                self.emit_event(ChainTransformEvent(
                    component_id=self.name,
                    step_name=step_name,
                    step_index=i,
                    input_message=current_message,
                    output_message=response.content,
                    transformation_time_ms=step_time_ms
                ))

                # Check for bottlenecks
                self._check_bottleneck(step_name, i, step_time_ms)

                # Log completion
                if self.verbose:
                    member_name = self._get_member_name(member)
                    self._log_message(f"✅ {member_name} complete", color="green")
                    if self.debug:
                        content = response.content[:200]
                        total_chars = len(response.content)
                        self._log_message(
                            f"Output: {content}...\n"
                            f"Total characters: {total_chars}",
                            color="blue"
                        )

                # Convert response for next member
                current_message = Message(
                    role=Role.USER if i < len(self._members) - 1 else response.role,
                    content=response.content
                )

            except Exception as e:
                self.emit_event(ChainErrorEvent(
                    component_id=self.name,
                    step_name=step_name,
                    step_index=i,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    traceback=traceback.format_exc()
                ))
                raise

        # Calculate total time
        chain_time_ms = (time.time() - chain_start_time) * 1000

        # Emit completion event
        self.emit_event(ChainCompletionEvent(
            component_id=self.name,
            input_message=message,
            output_message=current_message,
            total_time_ms=chain_time_ms,
            step_times=self._step_times.copy()
        ))

        return ModelResponse(
            content=current_message.content,
            raw_response={},
            usage=None,
            tool_calls=current_message.tool_calls,
            role=current_message.role
        )

    def print_hierarchy(self, indent: str = "") -> None:
        """Print chain in hierarchy"""
        rprint(f"{indent}[cyan]└──[/cyan] [bold]{self.name}[/bold] ([blue]Chain[/blue])")

        # Print metadata
        meta_indent = indent + "    "
        rprint(f"{meta_indent}[dim]• Depth: {self.metadata.depth}[/dim]")
        rprint(f"{meta_indent}[dim]• Path: {self.metadata.path}[/dim]")

        # Print members
        for member in self._members.values():
            if isinstance(member, (Agent, BaseGroup)):
                member.print_hierarchy(indent + "    ")
