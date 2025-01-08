from typing import Any, Optional, Type

from pydantic import BaseModel

from ...agents.base import Agent
from ...groups.chain import Chain
from ...groups.team import Team
from ..state import GraphState
from .agent import AgentNode
from .base import NodeBase
from .chain import ChainNode
from .team import TeamNode


class NodeDecorator:
    """Decorator for creating graph nodes"""

    def __init__(
        self,
        name: Optional[str] = None,
        input_channel_type: Optional[Type[Any]] = None,
        output_channel_type: Optional[Type[Any]] = None,
        response_schema: Optional[Type[BaseModel]] = None,
        **kwargs
    ):
        self.name = name
        self.input_channel_type = input_channel_type
        self.output_channel_type = output_channel_type
        self.response_schema = response_schema
        self.kwargs = kwargs

    def __call__(self, cls):
        # Get parent class configuration if it exists
        parent_config = {}
        for base in cls.__bases__:
            if hasattr(base, "__node_config__"):
                parent_config.update(base.__node_config__)

        # Create new configuration by merging parent config with current config
        current_config = {
            "name": self.name or cls.__name__,
            "input_channel_type": self.input_channel_type,
            "output_channel_type": self.output_channel_type,
            "response_schema": self.response_schema,
            **self.kwargs
        }

        # Remove None values from current config
        current_config = {k: v for k, v in current_config.items() if v is not None}

        # Merge configurations, giving precedence to current config
        merged_config = {**parent_config, **current_config}

        # Store node configuration on original class
        cls.__node_decorator__ = True
        cls.__node_config__ = merged_config

        # Create wrapper class that inherits from original class
        class NodeClass(cls):
            """Node wrapper class"""

            def __init__(self, *args, **kwargs):
                # Initialize original class
                cls.__init__(self, *args, **kwargs)

                # Store configuration
                self.__node_config__ = merged_config

                # Mark as node-decorated
                self.__node_decorator__ = True

            def create_node(self, graph_state: GraphState) -> NodeBase:
                """Create a graph node from this instance"""
                config = self.__node_config__

                # Determine node type based on instance type
                if isinstance(self, Agent):
                    return AgentNode(
                        graph_state=graph_state,
                        agent=self,
                        input_channel_type=config.get("input_channel_type"),
                        output_channel_type=config.get("output_channel_type"),
                        response_schema=config.get("response_schema")
                    )
                elif isinstance(self, Chain):
                    return ChainNode(
                        graph_state=graph_state,
                        chain=self,
                        input_channel_type=config.get("input_channel_type"),
                        output_channel_type=config.get("output_channel_type"),
                        response_schema=config.get("response_schema")
                    )
                elif isinstance(self, Team):
                    return TeamNode(
                        graph_state=graph_state,
                        team=self,
                        input_channel_type=config.get("input_channel_type"),
                        output_channel_type=config.get("output_channel_type")
                    )
                else:
                    raise ValueError(f"Unsupported node type: {type(self)}")

        # Copy class attributes
        for attr in ["__module__", "__name__", "__qualname__", "__doc__", "__annotations__"]:
            try:
                setattr(NodeClass, attr, getattr(cls, attr))
            except AttributeError:
                pass

        return NodeClass

# Create decorator function
def node(
    name: Optional[str] = None,
    input_channel_type: Optional[Type[Any]] = None,
    output_channel_type: Optional[Type[Any]] = None,
    response_schema: Optional[Type[BaseModel]] = None,
    **kwargs
):
    """Decorator for creating graph nodes

    Args:
    ----
        name: Optional node name
        input_channel_type: Optional type hint for input channel
        output_channel_type: Optional type hint for output channel
        response_schema: Optional pydantic model for response validation
        **kwargs: Additional node parameters

    Example:
    -------
        @node(
            input_channel_type=str,
            output_channel_type=AnalysisResult,
            response_schema=AnalysisResponse
        )
        @agent(model="gpt-4")
        class Analyzer:
            '''Analyzes input data.'''

    """
    if isinstance(name, type):
        # @node used without parameters
        return NodeDecorator()(name)
    # @node() used with parameters
    return NodeDecorator(
        name=name,
        input_channel_type=input_channel_type,
        output_channel_type=output_channel_type,
        response_schema=response_schema,
        **kwargs
    )
