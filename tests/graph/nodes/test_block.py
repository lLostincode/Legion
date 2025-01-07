import pytest
from typing import Dict, Any, List
from pydantic import BaseModel, ConfigDict

from legion.blocks.base import FunctionalBlock, BlockMetadata
from legion.graph.nodes.block import BlockNode
from legion.graph.state import GraphState

class TestInput(BaseModel):
    """Test input schema"""
    value: str
    
    model_config = ConfigDict(
        json_encoders=None
    )

class TestOutput(BaseModel):
    """Test output schema"""
    result: str
    metadata: Dict[str, Any]
    
    model_config = ConfigDict(
        json_encoders=None
    )

def block_func(data: TestInput) -> TestOutput:
    """Test block function"""
    return TestOutput(
        result=f"Processed: {data.value}",
        metadata={"source": "test"}
    )

@pytest.fixture
def graph_state():
    """Create a graph state instance"""
    return GraphState()

@pytest.fixture
def test_block():
    """Create a test block"""
    return FunctionalBlock(
        func=block_func,
        metadata=BlockMetadata(
            name="test_block",
            description="A test block",
            input_schema=TestInput,
            output_schema=TestOutput
        )
    )

@pytest.fixture
def block_node(graph_state, test_block):
    """Create a block node instance"""
    return BlockNode(
        graph_state=graph_state,
        block=test_block
    )

@pytest.mark.asyncio
async def test_block_node_initialization(block_node, test_block):
    """Test block node initialization"""
    assert block_node.block is test_block
    assert block_node.get_input_channel("input") is not None
    assert block_node.get_output_channel("output") is not None
    
    # Verify channel type hints
    input_channel = block_node.get_input_channel("input")
    output_channel = block_node.get_output_channel("output")
    assert input_channel._type_hint is TestInput
    assert output_channel._type_hint is TestOutput

@pytest.mark.asyncio
async def test_block_node_execution(block_node):
    """Test block node execution"""
    # Set input
    input_channel = block_node.get_input_channel("input")
    input_data = TestInput(value="test")
    input_channel.set(input_data)
    
    # Execute node
    result = await block_node.execute()
    
    # Verify output
    assert result is not None
    assert "output" in result
    output = result["output"]
    assert isinstance(output, TestOutput)
    assert output.result == "Processed: test"
    assert output.metadata["source"] == "test"
    
    # Check output channel
    output_channel = block_node.get_output_channel("output")
    channel_output = output_channel.get()
    assert isinstance(channel_output, TestOutput)
    assert channel_output.result == "Processed: test"

@pytest.mark.asyncio
async def test_block_node_no_input(block_node):
    """Test block node behavior with no input"""
    result = await block_node.execute()
    assert result is None

@pytest.mark.asyncio
async def test_block_node_checkpointing(block_node, test_block):
    """Test block node checkpointing"""
    # Set input and execute
    input_channel = block_node.get_input_channel("input")
    input_data = TestInput(value="test")
    input_channel.set(input_data)
    await block_node.execute()
    
    # Create checkpoint
    checkpoint = block_node.checkpoint()
    
    # Verify checkpoint contents
    assert "block_metadata" in checkpoint
    assert checkpoint["block_metadata"]["name"] == "test_block"
    assert checkpoint["block_validate"] == test_block.validate
    assert "validate_types" in checkpoint
    
    # Create new node and restore
    new_node = BlockNode(
        graph_state=block_node._graph_state,
        block=FunctionalBlock(
            func=block_func,
            metadata=BlockMetadata(
                name="test_block",
                description="A test block",
                input_schema=TestInput,
                output_schema=TestOutput
            )
        )
    )
    new_node.restore(checkpoint)
    
    # Verify restored state
    assert new_node.block.metadata.name == block_node.block.metadata.name
    assert new_node.block.validate == block_node.block.validate
    assert new_node._validate_types == block_node._validate_types

@pytest.mark.asyncio
async def test_block_node_custom_type_hints(graph_state, test_block):
    """Test block node with custom type hints"""
    # Create node with custom type hints
    node = BlockNode(
        graph_state=graph_state,
        block=test_block,
        input_channel_type=Dict[str, Any],
        output_channel_type=Dict[str, Any]
    )
    
    # Verify channel type hints
    input_channel = node.get_input_channel("input")
    output_channel = node.get_output_channel("output")
    assert input_channel._type_hint is Dict[str, Any]
    assert output_channel._type_hint is Dict[str, Any]

@pytest.mark.asyncio
async def test_block_node_type_validation(graph_state, test_block):
    """Test block node type validation"""
    node = BlockNode(
        graph_state=graph_state,
        block=test_block,
        validate_types=True
    )
    
    # Test valid input
    input_channel = node.get_input_channel("input")
    input_data = TestInput(value="test")
    input_channel.set(input_data)
    result = await node.execute()
    assert result is not None
    
    # Test invalid input type
    with pytest.raises(TypeError):
        input_channel.set({"invalid": "input"})
    
    # Test invalid input data
    with pytest.raises(TypeError):
        input_channel.set({"value": 123})  # Wrong type for value field

@pytest.mark.asyncio
async def test_block_node_validation_context(graph_state, test_block):
    """Test block node validation context"""
    node = BlockNode(
        graph_state=graph_state,
        block=test_block,
        input_channel_type=TestInput  # Use correct type
    )
    
    # Check validation warnings
    validation = await node.validate()
    assert len(validation.warnings) == 0  # No warnings with correct type
    
    # Check validation after error
    input_channel = node.get_input_channel("input")
    with pytest.raises(TypeError):
        input_channel.set({"invalid": "input"})  # Invalid input structure
    
    # Verify validation context
    validation = await node.validate()
    assert len(validation.warnings) == 0  # Still no warnings

@pytest.mark.asyncio
async def test_block_node_type_coercion(graph_state):
    """Test block node type coercion"""
    # Create a block with Dict input/output
    def dict_block(data: Dict[str, str]) -> Dict[str, str]:
        return {"result": data["value"]}
    
    block = FunctionalBlock(
        func=dict_block,
        metadata=BlockMetadata(
            name="dict_block",
            description="A dict block",
            input_schema=None,  # No schema for raw dict
            output_schema=None
        )
    )
    
    node = BlockNode(
        graph_state=graph_state,
        block=block,
        input_channel_type=Dict[str, str],
        validate_types=True
    )
    
    # Test with dict input
    input_channel = node.get_input_channel("input")
    input_data = {"value": "test"}
    input_channel.set(input_data)
    result = await node.execute()
    assert result is not None
    assert result["output"]["result"] == "test"
