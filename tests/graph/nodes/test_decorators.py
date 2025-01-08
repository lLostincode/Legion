from typing import Annotated, Any, Dict, List
from unittest.mock import patch

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from tests.utils import MockOpenAIProvider

# Load environment variables from .env file
load_dotenv()

from legion.agents.decorators import agent
from legion.graph.nodes.agent import AgentNode
from legion.graph.nodes.chain import ChainNode
from legion.graph.nodes.decorators import node
from legion.graph.nodes.team import TeamNode
from legion.graph.state import GraphState
from legion.groups.decorators import leader
from legion.interface.decorators import output_schema, tool


# Test schemas
@output_schema
class AnalysisResult(BaseModel):
    summary: str = Field(description="Analysis summary")
    key_points: List[str] = Field(description="Key findings")
    confidence: float = Field(description="Confidence score", ge=0, le=1)

@output_schema
class ProcessedData(BaseModel):
    data: Dict[str, Any] = Field(description="Processed data")
    metadata: Dict[str, str] = Field(description="Processing metadata")

# Test components
@agent(model="gpt-4-mini")
@node(
    input_channel_type=str,
    output_channel_type=AnalysisResult,
    response_schema=AnalysisResult
)
class Analyzer:
    """Analysis agent node."""

    @tool
    def analyze(
        self,
        text: Annotated[str, Field(description="Text to analyze")]
    ) -> AnalysisResult:
        """Analyze input text"""
        return AnalysisResult(
            summary=f"Analysis of: {text[:100]}...",
            key_points=["Finding 1", "Finding 2"],
            confidence=0.85
        )

class ProcessorMeta(type):
    """Metaclass for processor nodes"""

    def __new__(mcs, name, bases, attrs):
        # Create preprocessor and transformer instances
        if name == "Processor":
            # Create instances
            preprocessor = attrs["Preprocessor"]()
            transformer = attrs["Transformer"]()

            # Add instances to class attributes
            attrs["preprocessor"] = preprocessor
            attrs["transformer"] = transformer

            # Create chain members list
            attrs["_chain_members"] = [preprocessor, transformer]

            # Add create_node method
            def create_node(self, graph_state):
                return ChainNode(
                    graph_state=graph_state,
                    chain=self,
                    input_channel_type=str,
                    output_channel_type=ProcessedData,
                    response_schema=None
                )
            attrs["create_node"] = create_node

            # Create class
            cls = super().__new__(mcs, name, bases, attrs)

            # Mark as chain-decorated
            cls.__chain_decorator__ = True

            # Mark as node-decorated
            cls.__node_decorator__ = True

            # Store node configuration
            cls.__node_config__ = {
                "name": None,
                "input_channel_type": str,
                "output_channel_type": ProcessedData,
                "response_schema": None
            }

            return cls
        return super().__new__(mcs, name, bases, attrs)

class Processor(metaclass=ProcessorMeta):
    """Processing chain node."""

    @agent(model="gpt-4-mini")
    class Preprocessor:
        """Data preprocessor."""

        @tool
        def preprocess(
            self,
            data: Annotated[str, Field(description="Text to preprocess")]
        ) -> str:
            """Preprocess input data"""
            return f"Preprocessed: {data}"

    @agent(model="gpt-4-mini")
    class Transformer:
        """Data transformer."""

        @tool
        def transform(
            self,
            data: Annotated[str, Field(description="Text to transform")]
        ) -> ProcessedData:
            """Transform preprocessed data"""
            return ProcessedData(
                data={"result": data},
                metadata={"processor": "transformer"}
            )

class ReviewTeamMeta(type):
    """Metaclass for review team nodes"""

    def __new__(mcs, name, bases, attrs):
        # Create team member instances
        if name == "ReviewTeam":
            # Create instances
            coordinator = attrs["Coordinator"]()
            reviewer1 = attrs["Reviewer1"]()
            reviewer2 = attrs["Reviewer2"]()

            # Add instances to class attributes
            attrs["coordinator"] = coordinator
            attrs["reviewer1"] = reviewer1
            attrs["reviewer2"] = reviewer2

            # Create team members dict
            attrs["_team_members"] = {
                "reviewer1": reviewer1,
                "reviewer2": reviewer2
            }
            attrs["_team_leader"] = coordinator

            # Add create_node method
            def create_node(self, graph_state):
                return TeamNode(
                    graph_state=graph_state,
                    team=self
                )
            attrs["create_node"] = create_node

            # Create class
            cls = super().__new__(mcs, name, bases, attrs)

            # Mark as team-decorated
            cls.__team_decorator__ = True

            # Mark as node-decorated
            cls.__node_decorator__ = True

            # Store node configuration
            cls.__node_config__ = {
                "name": None,
                "input_channel_type": None,
                "output_channel_type": None,
                "response_schema": None,
                "tools": []
            }

            return cls
        return super().__new__(mcs, name, bases, attrs)

class ReviewTeam(metaclass=ReviewTeamMeta):
    """Review team node."""

    @leader(model="gpt-4-mini")
    class Coordinator:
        """Review coordinator."""

        @tool
        def coordinate(
            self,
            task: Annotated[str, Field(description="Task to coordinate")]
        ) -> str:
            """Coordinate team task"""
            return f"Coordinating: {task}"

    @agent(model="gpt-4-mini")
    class Reviewer1:
        """First reviewer."""

        @tool
        def review(
            self,
            content: Annotated[str, Field(description="Content to review")]
        ) -> str:
            """Review content"""
            return f"Review 1: {content}"

    @agent(model="gpt-4-mini")
    class Reviewer2:
        """Second reviewer."""

        @tool
        def review(
            self,
            content: Annotated[str, Field(description="Content to review")]
        ) -> str:
            """Review content"""
            return f"Review 2: {content}"

# Tests
@pytest.fixture
def graph_state():
    """Create graph state for testing"""
    return GraphState()

@patch("legion.providers.openai.OpenAIProvider", MockOpenAIProvider)
def test_agent_node_creation(graph_state):
    """Test agent node creation"""
    analyzer = Analyzer()
    assert hasattr(analyzer, "__node_decorator__")
    assert hasattr(analyzer, "__node_config__")

    # Create node
    node = analyzer.create_node(graph_state)
    assert isinstance(node, AgentNode)
    assert node.get_input_channel("input") is not None
    assert node.get_output_channel("output") is not None
    assert node.get_output_channel("tool_results") is not None

def test_chain_node_creation(graph_state):
    """Test chain node creation"""
    processor = Processor()
    assert hasattr(processor, "__node_decorator__")
    assert hasattr(processor, "__node_config__")

    # Create node
    node = processor.create_node(graph_state)
    assert isinstance(node, ChainNode)
    assert node.get_input_channel("input") is not None
    assert node.get_output_channel("output") is not None
    assert node.get_output_channel("member_outputs") is not None

def test_team_node_creation(graph_state):
    """Test team node creation"""
    team = ReviewTeam()
    assert hasattr(team, "__node_decorator__")
    assert hasattr(team, "__node_config__")

    # Create node
    node = team.create_node(graph_state)
    assert isinstance(node, TeamNode)
    assert node.get_input_channel("input") is not None
    assert node.get_output_channel("output") is not None

def test_node_configuration():
    """Test node configuration handling"""
    analyzer = Analyzer()
    config = analyzer.__node_config__

    assert config["input_channel_type"] == str
    assert config["output_channel_type"] == AnalysisResult
    assert config["response_schema"] == AnalysisResult

def test_node_inheritance():
    """Test node inheritance"""

    @agent(model="gpt-4-mini")
    @node(
        input_channel_type=str,
        tools=[]
    )
    class BaseAnalyzer:
        """Base analyzer."""

        pass

    @node(
        output_channel_type=AnalysisResult,
        tools=[]
    )
    class SpecializedAnalyzer(BaseAnalyzer):
        """Specialized analyzer."""

        pass

    base = BaseAnalyzer()
    specialized = SpecializedAnalyzer()

    # Base class should have input_channel_type set but not output_channel_type
    assert base.__node_config__["input_channel_type"] == str
    assert "output_channel_type" not in base.__node_config__

    # Specialized class should inherit input_channel_type and set its own output_channel_type
    assert specialized.__node_config__["input_channel_type"] == str
    assert specialized.__node_config__["output_channel_type"] == AnalysisResult

@patch("legion.providers.openai.OpenAIProvider", MockOpenAIProvider)
def test_decorator_with_params():
    """Test agent decorator with custom parameters"""
    # Your existing test code...

if __name__ == "__main__":
    pytest.main(["-v", __file__])
