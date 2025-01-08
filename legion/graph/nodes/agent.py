from datetime import datetime
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel

from ...agents.base import Agent
from ...interface.schemas import Message
from ..channels import LastValue, ValueSequence
from ..state import GraphState
from .base import NodeBase


class AgentNode(NodeBase):
    """Node wrapper for Legion agents"""

    def __init__(
        self,
        graph_state: GraphState,
        agent: Agent,
        input_channel_type: Optional[Type[Any]] = None,
        output_channel_type: Optional[Type[Any]] = None,
        response_schema: Optional[Type[BaseModel]] = None
    ):
        """Initialize agent node

        Args:
        ----
            graph_state: Graph state manager
            agent: Legion agent instance
            input_channel_type: Optional type hint for input channel
            output_channel_type: Optional type hint for output channel
            response_schema: Optional pydantic model for response validation

        """
        super().__init__(graph_state)
        self._agent = agent
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

        # Create tool result channel
        self.create_output_channel(
            "tool_results",
            ValueSequence,
            type_hint=str
        )

        # Create memory channel for state persistence
        self.create_output_channel(
            "memory",
            LastValue,
            type_hint=dict
        )

    @property
    def agent(self) -> Agent:
        """Get wrapped agent"""
        return self._agent

    async def _execute(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Execute agent with input from channels"""
        # Get input from channel
        input_channel = self.get_input_channel("input")
        if not input_channel:
            raise ValueError("Input channel not found")

        input_value = input_channel.get()
        if input_value is None:
            return None  # No input to process

        # Process with agent
        response = await self._agent.aprocess(
            input_value,
            response_schema=self._response_schema
        )

        # Store response in output channel
        output_channel = self.get_output_channel("output")
        if output_channel:
            output_channel.set(response.content)

        # Store tool results if any
        tool_results_channel = self.get_output_channel("tool_results")
        if tool_results_channel and response.tool_calls:
            for tool_call in response.tool_calls:
                if "result" in tool_call:
                    tool_results_channel.append(
                        f"{tool_call['function']['name']}: {tool_call['result']}"
                    )

        # Store memory state
        memory_channel = self.get_output_channel("memory")
        if memory_channel:
            memory_state = {
                "messages": [msg.model_dump() for msg in self._agent.memory.messages],
                "last_updated": datetime.now().isoformat()
            }
            memory_channel.set(memory_state)

        return {
            "output": response.content,
            "tool_results": [
                {
                    "name": tc["function"]["name"],
                    "result": tc.get("result")
                }
                for tc in (response.tool_calls or [])
            ],
            "memory": memory_state if memory_channel else None
        }

    def checkpoint(self) -> Dict[str, Any]:
        """Create checkpoint of node state"""
        checkpoint = super().checkpoint()
        # Add agent-specific state
        checkpoint.update({
            "agent_metadata": {
                "name": self._agent.name,
                "model": self._agent.full_model_name,
                "temperature": self._agent.temperature,
                "memory": [msg.model_dump() for msg in self._agent.memory.messages],
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description
                    }
                    for tool in self._agent.tools
                ]
            }
        })
        return checkpoint

    def restore(self, checkpoint: Dict[str, Any]) -> None:
        """Restore node state from checkpoint"""
        super().restore(checkpoint)
        # Restore agent metadata
        if "agent_metadata" in checkpoint:
            metadata = checkpoint["agent_metadata"]
            self._agent.name = metadata["name"]
            self._agent.temperature = metadata["temperature"]
            # Restore memory
            self._agent.memory.messages = [
                Message(**msg) for msg in metadata["memory"]
            ]
            # Note: Tools are not restored as they may contain stateful objects
