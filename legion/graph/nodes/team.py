from enum import Enum
from typing import Any, Dict, Optional

from legion.graph.channels import LastValue, ValueSequence
from legion.graph.nodes.base import NodeBase, NodeStatus
from legion.graph.state import GraphState
from legion.groups.team import Team
from legion.interface.schemas import Message


class TeamMode(str, Enum):
    """Team node operation mode"""

    ATOMIC = "atomic"  # Team operates as a single node
    EXPANDED = "expanded"  # Team expands into subgraph with member nodes

class TeamNode(NodeBase):
    """Node that wraps a Team for use in a graph"""

    def __init__(self, graph_state: GraphState, team: Team, mode: TeamMode = TeamMode.ATOMIC):
        super().__init__(graph_state)
        self._team = team
        self._mode = mode
        self._setup_channels()

    @property
    def mode(self) -> TeamMode:
        """Get the current operation mode"""
        return self._mode

    @mode.setter
    def mode(self, value: TeamMode):
        """Set the operation mode and update channels"""
        if value != self._mode:
            self._mode = value
            self._setup_channels()

    def _setup_channels(self):
        """Setup input and output channels based on mode"""
        # Store existing channel values
        existing_values = {}
        for name, channel in self._input_channels.items():
            if channel is not None:  # Skip None channels
                existing_values[f"in_{name}"] = channel.get()
                # Remove channel from graph state using the channel's key
                channel_key = f"{self._metadata.node_id}_in_{name}"
                self._graph_state._channels.pop(channel_key, None)

        for name, channel in self._output_channels.items():
            if channel is not None:  # Skip None channels
                existing_values[f"out_{name}"] = channel.get()
                # Remove channel from graph state using the channel's key
                channel_key = f"{self._metadata.node_id}_out_{name}"
                self._graph_state._channels.pop(channel_key, None)

        # Clear existing channels
        self._input_channels.clear()
        self._output_channels.clear()

        # Add base channels
        self.create_input_channel("input", LastValue, type_hint=str)
        self.create_output_channel("output", LastValue, type_hint=str)

        # Restore values
        if "in_input" in existing_values:
            self.get_input_channel("input").set(existing_values["in_input"])
        if "out_output" in existing_values:
            self.get_output_channel("output").set(existing_values["out_output"])

        if self._mode == TeamMode.ATOMIC:
            # Add delegation results channel for atomic mode
            self.create_output_channel("delegation_results", ValueSequence)  # Remove type hint for Any
            if "out_delegation_results" in existing_values:
                for item in existing_values["out_delegation_results"]:
                    self.get_output_channel("delegation_results").append(item)
        else:
            # Add member-specific channels for expanded mode
            self.create_output_channel("leader_output", LastValue, type_hint=str)
            if "out_leader_output" in existing_values:
                self.get_output_channel("leader_output").set(existing_values["out_leader_output"])

            for member_name in self._team.members:
                self.create_output_channel(f"{member_name}_output", LastValue, type_hint=str)
                self.create_output_channel(f"{member_name}_delegations", ValueSequence)  # Remove type hint for Any

                if f"out_{member_name}_output" in existing_values:
                    self.get_output_channel(f"{member_name}_output").set(existing_values[f"out_{member_name}_output"])
                if f"out_{member_name}_delegations" in existing_values:
                    for item in existing_values[f"out_{member_name}_delegations"]:
                        self.get_output_channel(f"{member_name}_delegations").append(item)

    async def _execute(self) -> Optional[Dict[str, Any]]:
        """Execute the team node based on current mode"""
        input_msg = self.get_input_channel("input").get()

        if self._mode == TeamMode.ATOMIC:
            return await self._execute_atomic(input_msg)
        else:
            return await self._execute_expanded(input_msg)

    async def _execute_atomic(self, input_msg: str) -> Dict[str, Any]:
        """Execute in atomic mode - process as single unit"""
        try:
            response = await self._team.aprocess(Message(content=input_msg))

            # Store outputs
            self.get_output_channel("output").set(response.content)
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    self.get_output_channel("delegation_results").append(tool_call)

            # Update status to completed
            self._metadata.status = NodeStatus.COMPLETED

            return {
                "output": response.content,
                "delegations": response.tool_calls or []
            }
        except Exception as e:
            # Update status to error
            self._metadata.status = NodeStatus.ERROR
            self._metadata.error = str(e)
            raise

    async def _execute_expanded(self, input_msg: str) -> Optional[Dict[str, Any]]:
        """Execute in expanded mode - process through each member"""
        try:
            # Check if resuming from pause
            current_member = None
            if self.status == NodeStatus.PAUSED:
                current_member = self._metadata.custom_data.get("paused_at_member")
                # Reset status when resuming
                self._metadata.status = NodeStatus.RUNNING

            # Process through leader first if not resuming
            member_outputs = {}
            last_output = None
            if not current_member:
                leader_response = await self._team.leader.aprocess(Message(content=input_msg))
                leader_output = str(leader_response.content)
                self.get_output_channel("leader_output").set(leader_output)
                current_msg = leader_output
                member_outputs["leader"] = {
                    "output": leader_output,
                    "delegations": leader_response.tool_calls or []
                }
                last_output = leader_output
            else:
                current_msg = input_msg

            # Process through members
            for member_name, member in self._team.members.items():
                # Skip members before paused member when resuming
                if current_member and member_name != current_member:
                    continue

                # Process member
                member_response = await member.aprocess(Message(content=current_msg))
                last_output = str(member_response.content)

                # Store outputs
                self.get_output_channel(f"{member_name}_output").set(last_output)
                if member_response.tool_calls:
                    for tool_call in member_response.tool_calls:
                        self.get_output_channel(f"{member_name}_delegations").append(tool_call)

                member_outputs[member_name] = {
                    "output": last_output,
                    "delegations": member_response.tool_calls or []
                }

                # Update current message for next member
                current_msg = last_output

                # Handle pause if requested
                if self.status == NodeStatus.PAUSED:
                    self._metadata.custom_data["paused_at_member"] = member_name
                    return None

                # Check if this was the paused member
                if current_member and member_name == current_member:
                    current_member = None

            # Clear pause data and update status
            if "paused_at_member" in self._metadata.custom_data:
                del self._metadata.custom_data["paused_at_member"]
            self._metadata.status = NodeStatus.COMPLETED

            # Set final output
            self.get_output_channel("output").set(last_output)

            return {
                "output": last_output,
                "member_outputs": member_outputs
            }
        except Exception as e:
            # Update status to error
            self._metadata.status = NodeStatus.ERROR
            self._metadata.error = str(e)
            raise

    def checkpoint(self) -> Dict[str, Any]:
        """Create a checkpoint of the node's state"""
        checkpoint = super().checkpoint()

        # Store channel values
        channel_values = {}
        for name, channel in self._input_channels.items():
            if channel is not None:
                channel_values[f"in_{name}"] = channel.get()

        for name, channel in self._output_channels.items():
            if channel is not None:
                if isinstance(channel, ValueSequence):
                    channel_values[f"out_{name}"] = channel.get_all()
                else:
                    channel_values[f"out_{name}"] = channel.get()

        checkpoint.update({
            "mode": self._mode,
            "team": {
                "name": self._team.name,
                "leader": self._team.leader.name,
                "members": {name: member.name for name, member in self._team.members.items()}
            },
            "channel_values": channel_values
        })
        return checkpoint

    def restore(self, checkpoint: Dict[str, Any]) -> None:
        """Restore node state from checkpoint"""
        # Clear existing channels from graph state
        for name, channel in list(self._input_channels.items()):
            if channel is not None:
                channel_key = f"{self._metadata.node_id}_in_{name}"
                self._graph_state._channels.pop(channel_key, None)

        for name, channel in list(self._output_channels.items()):
            if channel is not None:
                channel_key = f"{self._metadata.node_id}_out_{name}"
                self._graph_state._channels.pop(channel_key, None)

        # Clear existing channels
        self._input_channels.clear()
        self._output_channels.clear()

        # Restore base state
        super().restore(checkpoint)

        # Restore team-specific state
        self._mode = TeamMode(checkpoint.get("mode", TeamMode.ATOMIC))

        # Ensure graph state channels are cleaned up before setting up new ones
        for key in list(self._graph_state._channels.keys()):
            if key.startswith(f"{self._metadata.node_id}_"):
                self._graph_state._channels.pop(key, None)

        # Setup channels after cleanup
        self._setup_channels()

        # Restore channel values
        channel_values = checkpoint.get("channel_values", {})
        for name, value in channel_values.items():
            if value is not None:
                if name.startswith("in_"):
                    channel_name = name[3:]  # Remove "in_" prefix
                    if channel_name in self._input_channels:
                        self._input_channels[channel_name].set(value)
                elif name.startswith("out_"):
                    channel_name = name[4:]  # Remove "out_" prefix
                    if channel_name in self._output_channels:
                        channel = self._output_channels[channel_name]
                        if isinstance(channel, ValueSequence) and isinstance(value, list):
                            for item in value:
                                channel.append(item)
                        else:
                            channel.set(value)

    def _update_status(self, status: NodeStatus) -> None:
        """Update node status"""
        self._metadata.status = status
        if status == NodeStatus.PAUSED:
            # Store current member for pause/resume
            for member_name in self._team.members:
                if member_name in self._output_channels:
                    self._metadata.custom_data["paused_at_member"] = member_name
                    break
