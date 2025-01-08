from datetime import datetime
from typing import Any, Dict, Optional

import pytest

from legion.graph.channels import LastValue, ValueSequence
from legion.graph.nodes.base import ExecutionContext, NodeBase, NodeMetadata, NodeStatus
from legion.graph.state import GraphState


class TestNode(NodeBase):
    """Test node implementation"""

    def __init__(self, graph_state: GraphState, should_fail: bool = False):
        super().__init__(graph_state)
        self.should_fail = should_fail

        # Create standard channels
        self.create_input_channel("input", LastValue, type_hint=str)
        self.create_output_channel("output", LastValue, type_hint=str)

    async def _execute(self, **kwargs) -> Optional[Dict[str, Any]]:
        if self.should_fail:
            raise ValueError("Test failure")

        input_value = kwargs.get("input", "default")
        output_value = f"processed_{input_value}"

        input_channel = self.get_input_channel("input")
        output_channel = self.get_output_channel("output")

        input_channel.set(input_value)
        output_channel.set(output_value)

        return {"output": output_value}

@pytest.fixture
def graph_state():
    """Fixture for graph state"""
    return GraphState()

@pytest.fixture
def test_node(graph_state):
    """Fixture for test node"""
    return TestNode(graph_state)

@pytest.fixture
def failing_node(graph_state):
    """Fixture for failing test node"""
    return TestNode(graph_state, should_fail=True)

def test_node_metadata(test_node):
    """Test node metadata functionality"""
    metadata = test_node.metadata

    assert isinstance(metadata, NodeMetadata)
    assert isinstance(metadata.created_at, datetime)
    assert metadata.version == 0
    assert metadata.status == NodeStatus.IDLE
    assert metadata.error is None
    assert metadata.execution_count == 0
    assert metadata.last_execution is None

def test_node_channels(test_node):
    """Test node channel management"""
    # Test initial channels
    assert "input" in test_node.list_input_channels()
    assert "output" in test_node.list_output_channels()

    # Test channel creation
    sequence = test_node.create_input_channel(
        "sequence",
        ValueSequence,
        type_hint=int,
        max_size=3
    )
    assert isinstance(sequence, ValueSequence)
    assert "sequence" in test_node.list_input_channels()

    # Test duplicate channel creation
    with pytest.raises(ValueError):
        test_node.create_input_channel("input", LastValue)

    # Test channel retrieval
    assert test_node.get_input_channel("input") is not None
    assert test_node.get_output_channel("output") is not None
    assert test_node.get_input_channel("nonexistent") is None

@pytest.mark.asyncio
async def test_node_execution(test_node):
    """Test node execution"""
    # Test successful execution
    await test_node.execute(input="test")

    assert test_node.status == NodeStatus.COMPLETED
    assert test_node.metadata.execution_count == 1
    assert test_node.metadata.last_execution is not None

    # Verify channel values
    input_channel = test_node.get_input_channel("input")
    output_channel = test_node.get_output_channel("output")
    assert input_channel.get() == "test"
    assert output_channel.get() == "processed_test"

    # Verify execution history
    history = test_node.get_execution_history()
    assert len(history) == 1
    assert history[0].inputs == {"input": "test"}
    assert history[0].outputs == {"output": "processed_test"}
    assert history[0].error is None

@pytest.mark.asyncio
async def test_node_execution_failure(failing_node):
    """Test node execution failure"""
    # Test failed execution
    with pytest.raises(ValueError, match="Test failure"):
        await failing_node.execute(input="test")

    assert failing_node.status == NodeStatus.FAILED
    assert failing_node.metadata.error == "Test failure"
    assert failing_node.metadata.execution_count == 1

    # Verify execution history
    history = failing_node.get_execution_history()
    assert len(history) == 1
    assert history[0].error == "Test failure"

def test_execution_history(test_node):
    """Test execution history management"""
    # Create some execution history
    context1 = ExecutionContext(
        inputs={"test": 1},
        outputs={"result": 2}
    )
    context2 = ExecutionContext(
        inputs={"test": 3},
        outputs={"result": 4}
    )

    test_node._execution_history.extend([context1, context2])

    # Test history retrieval
    history = test_node.get_execution_history()
    assert len(history) == 2
    assert history[0].inputs == {"test": 1}
    assert history[1].outputs == {"result": 4}

    # Test history clearing
    test_node.clear_execution_history()
    assert len(test_node.get_execution_history()) == 0

def test_checkpointing(test_node):
    """Test node checkpointing"""
    # Setup some state
    test_node._metadata.status = NodeStatus.COMPLETED
    test_node._metadata.execution_count = 5

    context = ExecutionContext(
        inputs={"test": 1},
        outputs={"result": 2}
    )
    test_node._execution_history.append(context)

    # Create checkpoint
    checkpoint = test_node.checkpoint()

    # Create new node and restore
    new_node = TestNode(test_node._graph_state)
    new_node.restore(checkpoint)

    # Verify metadata was restored
    assert new_node.status == NodeStatus.COMPLETED
    assert new_node.metadata.execution_count == 5

    # Verify execution history was restored
    history = new_node.get_execution_history()
    assert len(history) == 1
    assert history[0].inputs == {"test": 1}
    assert history[0].outputs == {"result": 2}

    # Verify channels were restored
    assert "input" in new_node.list_input_channels()
    assert "output" in new_node.list_output_channels()

if __name__ == "__main__":
    pytest.main([__file__])
