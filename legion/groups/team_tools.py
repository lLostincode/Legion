from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field
from rich import print as rprint

from ..agents.base import Agent
from ..interface.tools import BaseTool

if TYPE_CHECKING:
    pass

class DelegationParameters(BaseModel):
    """Parameters for delegation tool"""

    member: str = Field(..., description="Name of team member to delegate to")
    task: str = Field(..., description="Task description for the team member")

class DelegationTool(BaseTool):
    """Tool for delegating tasks to team members"""

    def __init__(self, members: Dict[str, Agent], leader: Optional[Agent] = None, verbose: bool = False):
        """Initialize delegation tool"""
        self.members = members
        self.leader = leader
        self.verbose = verbose
        # Track full conversation history
        self._conversation_history: List[Dict[str, Any]] = []

        # Build member capabilities description
        member_desc = self._build_capabilities_description(members)

        super().__init__(
            name="delegate",
            description=f"Delegate a task to a specific team member. Each member has specific capabilities:\n\n{member_desc}\n\nWhen delegating, choose the most appropriate member based on their tools and expertise.",
            parameters=DelegationParameters
        )

    def _build_capabilities_description(self, members: Dict[str, Agent]) -> str:
        """Build detailed description of each member's capabilities"""
        descriptions = []

        for name, member in members.items():
            if isinstance(member, Agent):
                # Get member's specialty from first line of system prompt
                specialty = member.system_prompt.render().split("\n")[0] if member.system_prompt else "Specialist"

                # List member's tools if any
                tool_descriptions = []
                if hasattr(member, "tools") and member.tools:
                    for tool in member.tools:
                        tool_descriptions.append(f"    - {tool.name}: {tool.description}")

                member_desc = [f"- {name} ({specialty})"]
                if tool_descriptions:
                    member_desc.append("  Available tools:")
                    member_desc.extend(tool_descriptions)

                descriptions.append("\n".join(member_desc))
            elif hasattr(member, "members") and member.metadata.group_type == "chain":
                descriptions.append(
                    f"- {name} (Chain): Sequential processing pipeline with {len(member.members)} steps\n"
                    f"  Steps: {' -> '.join(member.members[step].name for step in member.members)}"
                )
            elif hasattr(member, "leader") and member.metadata.group_type == "team":
                descriptions.append(
                    f"- {name} (Team): Hierarchical team led by {member.leader.name}\n"
                    f"  Members: {', '.join(member.members.keys())}"
                )

        return "\n".join(descriptions)

    def _format_conversation_history(self, member: str) -> str:
        """Format conversation history for a specific member"""
        history = []

        # Format the full conversation history chronologically
        for entry in self._conversation_history:
            if entry["type"] == "user_request":
                history.append(f"User Request: {entry['content']}")
            elif entry["type"] == "leader_response":
                history.append(f"Leader Response: {entry['content']}")
            elif entry["type"] == "delegation":
                if entry["member"] == member:  # Only show this member's direct interactions
                    history.append(f"Previous Task to Me: {entry['task']}")
                    if entry.get("response"):
                        history.append(f"My Previous Response: {entry['response']}")
                else:  # Show other members' results as context
                    history.append(f"Task Delegated to {entry['member']}: {entry['task']}")
                    if entry.get("response"):
                        history.append(f"Response from {entry['member']}: {entry['response']}")

        if not history:
            return ""

        return "\nConversation History:\n" + "\n\n".join(history)

    def add_user_message(self, content: str):
        """Add a user message to the conversation history"""
        self._conversation_history.append({
            "type": "user_request",
            "content": content
        })

    def add_leader_response(self, content: str):
        """Add a leader response to the conversation history"""
        self._conversation_history.append({
            "type": "leader_response",
            "content": content
        })

    def run(self, member: str, task: str, context: Optional[Dict] = None) -> str:
        """Execute delegation with context"""
        if member not in self.members:
            raise ValueError(f"Unknown team member: {member}")

        target = self.members[member]

        # Record this delegation
        delegation_entry = {
            "type": "delegation",
            "member": member,
            "task": task
        }
        self._conversation_history.append(delegation_entry)

        # Format task with conversation history
        history = self._format_conversation_history(member)
        formatted_task = f"{task}\n\n{history}" if history else task

        if self.verbose:
            self._log_message(f"\n📤 Delegating to: {member}")
            self._log_message(f"Task: {formatted_task}")

        # Process the task
        try:
            result = target.process(formatted_task)

            # Record the response
            delegation_entry["response"] = result.content

            if self.verbose:
                self._log_message(f"\n📥 Response from {member}:", color="green")
                self._log_message(result.content)

            return result

        except Exception as e:
            if self.verbose:
                self._log_message(f"\n❌ Delegation failed: {str(e)}", color="red")
            raise

    def _log_message(self, message: str, color: str = None) -> None:
        """Internal method for consistent logging"""
        if self.verbose:
            if color:
                rprint(f"[{color}]{message}[/{color}]")
            else:
                rprint(message)
