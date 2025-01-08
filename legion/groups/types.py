"""Type definitions for group implementations"""
from typing import TYPE_CHECKING, Dict, List, TypeVar, Union

from ..agents.base import Agent
from ..blocks.base import FunctionalBlock
from .base import BaseGroup

if TYPE_CHECKING:
    from .chain import Chain
    from .team import Team

# Type definitions
ChainMember = Union[Agent, BaseGroup, FunctionalBlock]
AgentOrGroup = Union[Agent, BaseGroup]
MemberDict = Dict[str, AgentOrGroup]
MemberList = List[ChainMember]

# Type variable for group classes
GroupT = TypeVar("GroupT", bound=BaseGroup)

# Group type hints
ChainType = TypeVar("ChainType", bound="Chain")
TeamType = TypeVar("TeamType", bound="Team")
