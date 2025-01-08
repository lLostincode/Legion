from typing import Any, Dict, List, Union

from rich import print as rprint

from legion.agents.base import Agent
from legion.interface.schemas import Message, ModelResponse, Role, SystemPromptSection
from legion.memory.providers.memory import InMemoryProvider

from .team_tools import DelegationTool


class Team:
    """A team of agents with a leader and members"""

    def __init__(self, name: str, leader: Agent, members: Dict[str, Agent], verbose: bool = True):
        """Initialize team

        Args:
        ----
            name: Team name
            leader: Leader agent
            members: Dictionary of member agents
            verbose: Whether to show delegation logs

        """
        self.name = name
        self.leader = leader
        self.members = members
        self.verbose = verbose

        # Initialize state tracking
        self._delegation_history: List[Dict[str, Any]] = []
        self._last_context: Dict[str, Any] = {
            "last_message": None,
            "message_type": None,
            "source": None
        }

        # Configure leader's system prompt to include member information
        member_info = "\n".join([
            f"- {name}: {member.__doc__ or 'No description'}"
            for name, member in members.items()
        ])
        leader_prompt = f"""You are the leader of a team with the following members:
{member_info}

Important Guidelines for Delegation:
1. Each team member is completely isolated and has no knowledge of:
   - Your interactions with other team members
   - Previous delegations or conversations
   - The overall context of the task
2. When delegating, always include:
   - All relevant context needed for the task
   - Any previous results or information they need
   - Clear expectations and requirements
3. Use the delegate tool to:
   - Assign tasks to team members
   - Wait for their response
   - Incorporate their response into your final answer

Available team members as tools: {list(members.keys())}"""

        if hasattr(leader, "system_prompt"):
            leader.system_prompt.sections.append(SystemPromptSection(
                content=leader_prompt,
                is_dynamic=False
            ))

        # Add delegation tool to leader with memory provider
        delegation_tool = DelegationTool(
            members=members,
            leader=leader,
            verbose=verbose
        )
        if not hasattr(leader, "_tools"):
            leader._tools = []
        leader._tools.append(delegation_tool)

        # Ensure leader has memory provider
        if hasattr(leader, "_memory_provider") and leader._memory_provider is None:
            leader._memory_provider = InMemoryProvider()

    def _record_delegation(self, source: str, target: str, task: str):
        """Record a delegation from one agent to another"""
        if self.verbose:
            rprint("\n[bold blue]🔄 Delegation:[/bold blue]")
            rprint(f"[blue]From:[/blue] {source}")
            rprint(f"[blue]To:[/blue] {target}")
            rprint(f"[blue]Task:[/blue] {task}")

        self._delegation_history.append({
            "source": source,
            "target": target,
            "task": task,
            "timestamp": None  # Could add timestamp if needed
        })

    def _get_context_for_member(self, member_name: str) -> Dict[str, Any]:
        """Get context for a specific member"""
        # Get relevant delegations
        delegations = [
            d for d in self._delegation_history
            if d["target"] == member_name
        ]

        return {
            "last_message": self._last_context["last_message"],
            "message_type": self._last_context["message_type"],
            "source": self._last_context["source"],
            "delegations": delegations
        }

    def _create_message(self, message: Union[str, Message]) -> Message:
        """Convert input to Message object if needed"""
        if isinstance(message, Message):
            return message
        return Message(role=Role.USER, content=str(message))

    async def aprocess(self, message: Union[str, Message]) -> ModelResponse:
        """Process a message through the team"""
        # Convert message to Message object if needed
        message_obj = self._create_message(message)

        if self.verbose:
            rprint("\n[bold green]📥 Team Input:[/bold green]")
            rprint(f"[green]{message_obj.content}[/green]")

        # Get the delegation tool
        delegation_tool = next(
            (tool for tool in self.leader._tools if tool.name == "delegate"),
            None
        )
        if delegation_tool:
            # Add user message to history
            delegation_tool.add_user_message(message_obj.content)

        # Process through leader first
        if self.verbose:
            rprint("\n[bold yellow]👤 Leader Processing...[/bold yellow]")

        leader_response = await self.leader.aprocess(message_obj)

        # Add leader's response to history
        if delegation_tool and leader_response.content:
            delegation_tool.add_leader_response(leader_response.content)

        # Check for delegations in leader response
        if leader_response.tool_calls:
            for tool_call in leader_response.tool_calls:
                member_name = tool_call["function"]["name"]

                # Record delegation
                self._record_delegation(
                    source=self.leader.name,
                    target=member_name,
                    task=tool_call["function"]["arguments"]
                )

                # Get target member
                if member_name in self.members:
                    member = self.members[member_name]

                    # Get context for member
                    context = self._get_context_for_member(member_name)

                    # Process through member
                    member_message = Message(
                        role=Role.USER,
                        content=tool_call["function"]["arguments"],
                        context=context
                    )

                    if self.verbose:
                        rprint(f"\n[bold cyan]👥 Member {member_name} Processing...[/bold cyan]")

                    member_response = await member.aprocess(member_message)

                    if self.verbose:
                        rprint("\n[bold magenta]📤 Member Response:[/bold magenta]")
                        rprint(f"[magenta]{member_response.content}[/magenta]")

                    # Store result
                    tool_call["result"] = member_response.content

                    # Update leader with member's response
                    leader_update = Message(
                        role=Role.ASSISTANT,
                        content=f"Task delegated to {member_name} has been completed. Response: {member_response.content}",
                        context={
                            "delegation_result": True,
                            "member": member_name,
                            "task": tool_call["function"]["arguments"],
                            "response": member_response.content
                        }
                    )

                    if self.verbose:
                        rprint("\n[bold yellow]👤 Leader Processing Member Response...[/bold yellow]")

                    update_response = await self.leader.aprocess(leader_update)

                    # Update leader response content but keep tool calls
                    if not leader_response.content:
                        leader_response.content = update_response.content
                    else:
                        leader_response.content = update_response.content.strip()

        if self.verbose:
            rprint("\n[bold green]📤 Final Team Response:[/bold green]")
            rprint(f"[green]{leader_response.content}[/green]")

        return leader_response

    def process(self, message: Union[str, Message]) -> ModelResponse:
        """Synchronous version of aprocess"""
        raise NotImplementedError("Teams only support async processing")
