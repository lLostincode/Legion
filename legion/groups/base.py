from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel
from rich import print as rprint

from ..agents.base import Agent
from ..errors import LegionError
from ..interface.schemas import Message, ModelResponse
from ..memory.base import MemoryProvider


class GroupMetadata(BaseModel):
    """Metadata for tracking group position and hierarchy"""

    position: Optional[str] = None  # first/middle/last/position_N
    depth: int = 0  # Nesting depth (0 for top-level)
    path: str = ""  # Full path in hierarchy (e.g., "root_team/processing_chain/analysis_team")
    group_type: str  # team/chain

class BaseGroup(ABC):
    """Base class for all agent groups"""

    # System prompts for all group types (except chain, which is in its own module)
    SYSTEM_PROMPTS = {
        "team_leader": """You are the leader of a team with access to various specialists through the 'delegate' tool. Your role is to:
        1. Understand incoming requests
        2. Break down complex tasks
        3. Delegate subtasks to appropriate team members
        4. Synthesize results into cohesive responses

        Available team members and their specialties:
        {member_descriptions}""",

        "team_member": """You are a specialist member of a team led by {leader_name}. When you receive tasks, they are delegations from your team leader.
        Focus on your specific expertise and provide clear, actionable responses.

        Your specialty: {specialty}""",

        "workshop_specialist": """You are a specialist in a collaborative workshop environment. You will be asked to:
        1. Evaluate your ability to contribute at different stages (1-10 scale)
        2. Process workshop states when selected
        3. Share relevant tool outputs with the team automatically

        Your specific expertise: {specialty}

        All your tools have been configured to optionally share their outputs with the team. Consider this when using tools.""",

        "workshop_synthesizer": """You are the synthesizer for a collaborative workshop. Your role is to:
        1. Review the complete workshop state
        2. Understand the original problem
        3. Synthesize all specialist contributions
        4. Create clear, coherent final responses

        Focus on creating outputs that effectively utilize all relevant workshop contributions."""
    }

    def __init__(
        self,
        name: str,
        memory_provider: Optional[MemoryProvider] = None,
        debug: bool = False,
        verbose: bool = False
    ):
        """Initialize group

        Args:
        ----
            name: Group name
            debug: Enable debug mode
            verbose: Enable verbose logging

        """
        self.name = name
        self.debug = debug
        self.verbose = verbose
        self.parent: Optional["BaseGroup"] = None
        self.metadata = GroupMetadata(
            group_type="base",
            path=name
        )
        self.memory_provider = memory_provider
        self._current_thread: Optional[str] = None

    def set_parent(self, parent: "BaseGroup") -> None:
        """Set parent group reference and update metadata"""
        self.parent = parent

        # Update metadata
        if parent.metadata.group_type == "chain":
            self.metadata.group_type = "chain_member"
        elif parent.metadata.group_type == "team":
            self.metadata.group_type = "team_member"

        # Recursively update depths and paths
        self._update_hierarchy_metadata()

        # Check for circular references
        parent._check_circular_reference(self)

    def _update_hierarchy_metadata(self) -> None:
        """Update depth and path based on position in hierarchy"""
        if not self.parent:
            self.metadata.depth = 0
            self.metadata.path = self.name
            return

        # Get parent's current state
        parent_depth = self.parent.metadata.depth
        parent_path = self.parent.metadata.path

        # Update own metadata
        self.metadata.depth = parent_depth + 1
        self.metadata.path = f"{parent_path}/{self.name}"

        # Recursively update all children
        if hasattr(self, "members"):
            for member in self.members.values():
                if hasattr(member, "_update_hierarchy_metadata"):
                    member._update_hierarchy_metadata()

    def get_root_group(self) -> "BaseGroup":
        """Get the top-level group in the hierarchy"""
        current = self
        while current.parent:
            current = current.parent
        return current

    def get_hierarchy_info(self) -> Dict[str, Any]:
        """Get information about group's position in hierarchy"""
        return {
            "name": self.name,
            "type": self.metadata.group_type,
            "depth": self.metadata.depth,
            "path": self.metadata.path,
            "position": self.metadata.position,
            "has_parent": bool(self.parent)
        }

    def _get_member_specialty(self, member: Union["Agent", "BaseGroup"]) -> str:
        """Extract member specialty from system prompt"""
        if isinstance(member, Agent):
            # Get first line of system prompt as specialty
            return member.system_prompt.render().split("\n")[0]
        else:
            return f"Group specializing in {member.name}"

    def _log_message(self, message: str, level: str = "info", color: str = None) -> None:
        """Internal method for consistent logging"""
        if self.verbose:
            if color:
                rprint(f"\n[{color}]{message}[/{color}]")
            else:
                rprint(f"\n{message}")

    def print_hierarchy(self, indent: str = "") -> None:
        """Print the group hierarchy with metadata"""
        # Get group type indicator
        if isinstance(self, Agent):
            type_str = "Agent"
        else:
            type_str = self.metadata.group_type.replace("_", " ").title()

        # Print current node with metadata
        rprint(f"{indent}[cyan]└──[/cyan] [bold]{self.name}[/bold] ([yellow]{type_str}[/yellow])")

        # Print metadata if it's a group
        if hasattr(self, "metadata"):
            meta_indent = indent + "    "
            rprint(f"{meta_indent}[dim]• Depth: {self.metadata.depth}[/dim]")
            rprint(f"{meta_indent}[dim]• Path: {self.metadata.path}[/dim]")

        # Print children for groups
        if hasattr(self, "members"):
            child_indent = indent + "    "
            for name, member in self.members.items():
                if isinstance(member, (Agent, BaseGroup)):
                    member.print_hierarchy(child_indent)

    @property
    @abstractmethod
    def members(self) -> Union[List["Agent"], Dict[str, "Agent"]]:
        """Get group members - to be implemented by subclasses"""
        pass

    @abstractmethod
    def process(self, message: Message) -> Message:
        """Process a message through the group - to be implemented by subclasses"""
        pass

    def _check_circular_reference(self, group: "BaseGroup", visited: Optional[set] = None) -> None:
        """Check for circular references in nested groups"""
        if visited is None:
            visited = set()

        # Check if current group would create a circular reference
        current = self
        while current:
            if current == group:
                raise LegionError(f"Circular reference detected: {group.name} would create a cycle")
            current = current.parent

        # Check members recursively
        if hasattr(group, "members"):
            for member in group.members.values():
                if isinstance(member, BaseGroup):
                    self._check_circular_reference(member, visited)

    async def _create_child_thread(
        self,
        member_name: str,
        parent_thread_id: str
    ) -> str:
        """Create a thread for a child member"""
        if not self.memory_provider:
            return parent_thread_id

        return await self.memory_provider.create_thread(
            entity_id=f"{self.name}/{member_name}",
            parent_thread_id=parent_thread_id
        )

    async def process(
        self,
        message: Union[str, Message],
        thread_id: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Process a message, optionally in a specific thread"""
        if thread_id and self.memory_provider:
            # Create or load thread
            self._current_thread = thread_id

            # Create child threads for members
            child_threads = {}
            for name, member in self.members.items():
                child_thread_id = await self._create_child_thread(
                    name, thread_id
                )
                child_threads[name] = child_thread_id

            # Process with child threads
            try:
                return await self._process_with_threads(
                    message, child_threads, **kwargs
                )
            finally:
                self._current_thread = None

        # Process without threads
        return await super().process(message, **kwargs)
