from functools import wraps
from typing import List, Optional

from ..agents.base import Agent
from ..blocks.base import FunctionalBlock
from ..interface.tools import BaseTool
from ..memory.base import MemoryProvider
from .base import BaseGroup
from .chain import Chain
from .team import Team


def team(cls=None, *, name: Optional[str] = None):
    """Decorator to create a team"""
    def decorator(cls):
        # Mark the class as team-decorated
        cls.__team_decorator__ = True

        # Collect members from class
        team_members = {}
        leader_instance = None
        memory_provider = None

        for attr_name, attr_value in cls.__dict__.items():
            # Skip special attributes
            if attr_name.startswith("__"):
                continue

            # Handle different types of members
            if isinstance(attr_value, (Agent, BaseGroup)):
                team_members[attr_name] = attr_value
            elif callable(attr_value) and hasattr(attr_value, "__leader_decorator__"):
                # Handle leader decorated class/function
                leader_instance = attr_value()
            elif isinstance(attr_value, type):
                # Handle nested class that might be an agent or chain
                if hasattr(attr_value, "__agent_decorator__"):
                    team_members[attr_name] = attr_value()
                elif hasattr(attr_value, "__chain_decorator__"):
                    team_members[attr_name] = attr_value()
            elif isinstance(attr_value, MemoryProvider):
                memory_provider = attr_value

        @wraps(cls)
        def wrapper(*args, **kwargs):
            if not leader_instance:
                raise ValueError(f"Team {name or cls.__name__} must have a leader (use @leader decorator)")

            # If we have a memory provider, set it for all members
            if memory_provider:
                if hasattr(leader_instance, "_memory_provider"):
                    leader_instance._memory_provider = memory_provider
                for member in team_members.values():
                    if hasattr(member, "_memory_provider"):
                        member._memory_provider = memory_provider

            return Team(
                name=name or cls.__name__,
                leader=leader_instance,
                members=team_members
            )

        return wrapper

    if cls is None:
        return decorator
    return decorator(cls)

def chain(cls=None, *, name: Optional[str] = None):
    """Decorator to create a processing chain"""
    def decorator(cls):
        # Mark the class as chain-decorated
        cls.__chain_decorator__ = True

        # Get members list from class
        if not hasattr(cls, "members"):
            raise ValueError(f"Chain {name or cls.__name__} must define a 'members' list")

        chain_members = []
        memory_provider = None

        # Process members list
        for member in cls.members:
            print(f"Processing member: {member}, type: {type(member)}")
            if isinstance(member, (Agent, BaseGroup, FunctionalBlock)):
                chain_members.append(member)
            elif isinstance(member, type):
                if hasattr(member, "__agent_decorator__"):
                    print(f"Found agent class: {member.__name__}")
                    chain_members.append(member())
                elif hasattr(member, "__block_decorator__"):
                    chain_members.append(member())
                elif hasattr(member, "__chain_decorator__"):
                    chain_members.append(member())
            elif isinstance(member, MemoryProvider):
                memory_provider = member
            else:
                # Assume it's a function decorated with @block
                chain_members.append(member)

        print(f"Final chain members: {chain_members}")

        @wraps(cls)
        def wrapper(*args, **kwargs):
            if not chain_members:
                raise ValueError(f"Chain {name or cls.__name__} has no members")

            chain_kwargs = {**kwargs}
            if memory_provider:
                chain_kwargs["memory_provider"] = memory_provider

                for member in chain_members:
                    if hasattr(member, "memory_provider") and not member.memory_provider:
                        member.memory_provider = memory_provider

            return Chain(
                name=name or cls.__name__,
                members=chain_members,
                **chain_kwargs
            )

        return wrapper

    if cls is None:
        return decorator
    return decorator(cls)

def leader(
    name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    model: str = "openai:gpt-4o-mini",
    temperature: float = 0.7,
    tools: Optional[List[BaseTool]] = None,
    memory_provider: Optional[MemoryProvider] = None
):
    """Decorator to create a team leader agent

    Example:
    -------
    @team
    class ResearchTeam:
        memory = InMemoryProvider()  # Optional memory provider

        @leader(model="openai:gpt-4")  # Automatically becomes team leader
        class Leader:
            '''Research team coordinator who delegates tasks.'''

    """
    def decorator(cls):
        # Collect tools from class methods
        class_tools = []
        if tools:
            class_tools.extend(tools)

        for attr_name, attr_value in cls.__dict__.items():
            if isinstance(attr_value, BaseTool):
                class_tools.append(attr_value)

        from ..agents.base import Agent

        @wraps(cls)
        def wrapper(*args, **kwargs):
            instance = cls()
            bound_tools = [
                tool.bind_to(instance) if hasattr(tool, "bind_to") else tool
                for tool in class_tools
            ]

            # Handle memory provider
            agent_kwargs = {**kwargs}
            if memory_provider:
                agent_kwargs["_memory_provider"] = memory_provider

            agent_instance = Agent(
                name=name or cls.__name__,
                system_prompt=system_prompt or cls.__doc__ or f"Team leader: {cls.__name__}",
                model=model,
                tools=bound_tools,
                temperature=temperature,
                **agent_kwargs
            )
            return agent_instance

        # Mark the wrapper with the necessary attributes
        wrapper.__agent_decorator__ = True
        wrapper.__leader_decorator__ = True
        wrapper.__wrapped__ = cls

        return wrapper

    return decorator
