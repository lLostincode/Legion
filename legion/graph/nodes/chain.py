from enum import Enum
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel

from ...groups.chain import Chain
from ...interface.schemas import Message, Role
from ..channels import LastValue
from ..state import GraphState
from .base import NodeBase, NodeStatus


class ChainMode(str, Enum):
    """Chain node operation mode"""

    ATOMIC = "atomic"  # Chain operates as a single node
    EXPANDED = "expanded"  # Chain expands into subgraph

class ChainNode(NodeBase):
    """Node wrapper for Legion chains"""

    def __init__(
        self,
        graph_state: GraphState,
        chain: Chain,
        mode: ChainMode = ChainMode.ATOMIC,
        input_channel_type: Optional[Type[Any]] = None,
        output_channel_type: Optional[Type[Any]] = None,
        response_schema: Optional[Type[BaseModel]] = None
    ):
        """Initialize chain node

        Args:
        ----
            graph_state: Graph state manager
            chain: Legion chain instance
            mode: Operation mode (atomic or expanded)
            input_channel_type: Optional type hint for input channel
            output_channel_type: Optional type hint for output channel
            response_schema: Optional pydantic model for response validation

        """
        super().__init__(graph_state)
        self._chain = chain
        self._mode = mode
        self._response_schema = response_schema

        # Create standard channels
        self.create_input_channel(
            "input",
            LastValue,
            type_hint=input_channel_type or str
        )
        self.create_output_channel(
            "output",
            LastValue,
            type_hint=output_channel_type or str
        )

        # Create member outputs channel
        self.create_output_channel(
            "member_outputs",
            LastValue,
            type_hint=dict
        )

        # Create expanded mode channels if needed
        if mode == ChainMode.EXPANDED:
            self._setup_expanded_mode()

    @property
    def mode(self) -> ChainMode:
        """Get current operation mode"""
        return self._mode

    @mode.setter
    def mode(self, value: ChainMode) -> None:
        """Set operation mode"""
        if value != self._mode:
            self._mode = value
            if value == ChainMode.EXPANDED:
                self._setup_expanded_mode()
            else:
                self._cleanup_expanded_mode()

    def _setup_expanded_mode(self) -> None:
        """Setup channels and state for expanded mode"""
        # Create channels for each chain member
        for name, member in self._chain.members.items():
            # Create intermediate channels
            self.create_output_channel(
                f"{name}_output",
                LastValue,
                type_hint=str
            )

    def _cleanup_expanded_mode(self) -> None:
        """Cleanup expanded mode resources"""
        # Remove member-specific channels
        channels_to_remove = []
        for name in self._output_channels:
            if name.endswith("_output") and name != "output":
                channels_to_remove.append(name)

        for name in channels_to_remove:
            del self._output_channels[name]

    @property
    def chain(self) -> Chain:
        """Get wrapped chain"""
        return self._chain

    async def _execute(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Execute chain with input from channels"""
        # Get input from channel
        input_channel = self.get_input_channel("input")
        if not input_channel:
            raise ValueError("Input channel not found")

        input_value = input_channel.get()
        if input_value is None:
            return None  # No input to process

        # Convert input to Message if needed
        if not isinstance(input_value, Message):
            input_value = Message(
                role="user",
                content=str(input_value)
            )

        try:
            # Set status to running
            self._update_status(NodeStatus.RUNNING)

            if self._mode == ChainMode.ATOMIC:
                # Process with chain as single unit
                response = await self._chain.aprocess(
                    input_value,
                    response_schema=self._response_schema
                )

                # Store response in output channel
                output_channel = self.get_output_channel("output")
                if output_channel:
                    output_channel.set(response.content)

                # Store member outputs
                member_outputs = self.get_output_channel("member_outputs")
                outputs = {}
                if member_outputs:
                    # Extract outputs from each member's memory
                    for name, member in self._chain.members.items():
                        # Get last assistant message from memory
                        assistant_messages = [
                            msg for msg in member.memory.messages
                            if msg.role == Role.ASSISTANT
                        ]
                        if assistant_messages:
                            outputs[name] = assistant_messages[-1].content
                        else:
                            outputs[name] = ""
                    member_outputs.set(outputs)

                return {
                    "output": response.content,
                    "member_outputs": outputs
                }

            else:  # EXPANDED mode
                current_message = input_value
                outputs = {}

                # Process through chain members sequentially
                for i, (name, member) in enumerate(self._chain.members.items()):
                    # Check if paused
                    if self.status == NodeStatus.PAUSED:
                        # Store current progress
                        self._metadata.custom_data["paused_at_member"] = i
                        return None

                    # Process with current member
                    response = await member.aprocess(
                        current_message,
                        response_schema=self._response_schema if name == list(self._chain.members.keys())[-1] else None
                    )

                    # Store member output
                    member_channel = self.get_output_channel(f"{name}_output")
                    if member_channel:
                        member_channel.set(response.content)
                    outputs[name] = response.content

                    # Update message for next member
                    current_message = Message(
                        role=Role.USER,
                        content=response.content
                    )

                # Store final output
                output_channel = self.get_output_channel("output")
                if output_channel:
                    output_channel.set(current_message.content)

                # Store all member outputs
                member_outputs = self.get_output_channel("member_outputs")
                if member_outputs:
                    member_outputs.set(outputs)

                return {
                    "output": current_message.content,
                    "member_outputs": outputs
                }

        except Exception as e:
            self._update_status(NodeStatus.FAILED, str(e))
            raise
        finally:
            if self.status != NodeStatus.PAUSED:
                self._update_status(NodeStatus.COMPLETED)

    def checkpoint(self) -> Dict[str, Any]:
        """Create checkpoint of node state"""
        checkpoint = super().checkpoint()
        # Add chain-specific state
        checkpoint.update({
            "chain_metadata": {
                "name": self._chain.name,
                "mode": self._mode,
                "members": {
                    name: {
                        "name": member.name,
                        "model": member.full_model_name,
                        "temperature": member.temperature,
                        "memory": [msg.model_dump() for msg in member.memory.messages]
                    }
                    for name, member in self._chain.members.items()
                }
            }
        })
        return checkpoint

    def restore(self, checkpoint: Dict[str, Any]) -> None:
        """Restore node state from checkpoint"""
        super().restore(checkpoint)
        # Restore chain metadata
        if "chain_metadata" in checkpoint:
            self._chain.name = checkpoint["chain_metadata"]["name"]
            self._mode = checkpoint["chain_metadata"].get("mode", ChainMode.ATOMIC)
            # Restore member state
            for name, member_data in checkpoint["chain_metadata"]["members"].items():
                if name in self._chain.members:
                    member = self._chain.members[name]
                    # Restore member configuration
                    member.name = member_data["name"]
                    member.temperature = member_data["temperature"]
                    # Restore memory
                    member.memory.messages = [
                        Message(**msg) for msg in member_data["memory"]
                    ]

    def _get_current_member_index(self) -> Optional[int]:
        """Get index of currently executing member"""
        if self.status != NodeStatus.RUNNING:
            return None

        # Check member output channels to find last completed
        member_names = list(self._chain.members.keys())
        for i in range(len(member_names)-1, -1, -1):
            member_channel = self.get_output_channel(f"{member_names[i]}_output")
            if member_channel and member_channel.get() is not None:
                return i + 1  # Return next member index
        return 0  # No members completed yet

    async def pause(self) -> None:
        """Pause chain execution"""
        if self.status != NodeStatus.RUNNING:
            return

        # Store current progress in metadata
        if self._mode == ChainMode.EXPANDED:
            current_member = self._get_current_member_index()
            if current_member is not None:
                self._metadata.custom_data["paused_at_member"] = current_member

        self._update_status(NodeStatus.PAUSED)

    async def resume(self) -> None:
        """Resume chain execution"""
        if self.status != NodeStatus.PAUSED:
            return

        self._update_status(NodeStatus.RUNNING)

        if self._mode == ChainMode.EXPANDED:
            # Get paused position
            paused_at = self._metadata.custom_data.get("paused_at_member", 0)

            # Get last output as input for resumption
            member_names = list(self._chain.members.keys())
            if paused_at > 0:
                prev_output = self.get_output_channel(f"{member_names[paused_at-1]}_output")
                if prev_output:
                    current_message = Message(
                        role=Role.USER,
                        content=prev_output.get()
                    )
            else:
                input_channel = self.get_input_channel("input")
                current_message = Message(
                    role=Role.USER,
                    content=input_channel.get()
                )

            # Resume processing from paused member
            outputs = {}
            for i in range(paused_at, len(member_names)):
                name = member_names[i]
                member = self._chain.members[name]

                # Process with current member
                response = await member.aprocess(
                    current_message,
                    response_schema=self._response_schema if i == len(member_names)-1 else None
                )

                # Store member output
                member_channel = self.get_output_channel(f"{name}_output")
                if member_channel:
                    member_channel.set(response.content)
                outputs[name] = response.content

                # Update message for next member
                current_message = Message(
                    role=Role.USER,
                    content=response.content
                )

            # Update final outputs
            output_channel = self.get_output_channel("output")
            if output_channel:
                output_channel.set(current_message.content)

            member_outputs = self.get_output_channel("member_outputs")
            if member_outputs:
                # Merge with existing outputs
                existing = member_outputs.get() or {}
                existing.update(outputs)
                member_outputs.set(existing)

        else:  # ATOMIC mode
            # Simply re-run the chain
            await self.execute()
