import pytest
import logging
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import sys

from legion.graph.checkpointing import GraphCheckpointer, GraphCheckpoint
from legion.graph.state import GraphState
from legion.memory.base import MemoryProvider, ThreadState, MemoryDump
from legion.graph.channels import LastValue
import pytest_asyncio

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MockMemoryProvider(MemoryProvider):
    """Mock memory provider for testing"""
    
    def __init__(self):
        self._threads: Dict[str, Dict[str, Any]] = {}
        self._states: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.logger = logging.getLogger(__name__ + ".MockMemoryProvider")
    
    async def create_thread(
        self,
        entity_id: str,
        parent_thread_id: Optional[str] = None
    ) -> str:
        thread_id = f"thread_{len(self._threads)}"
        self._threads[thread_id] = {
            'entity_id': entity_id,
            'parent_thread_id': parent_thread_id
        }
        self.logger.debug(f"Created thread {thread_id} for entity {entity_id}")
        return thread_id
    
    async def save_state(
        self,
        entity_id: str,
        thread_id: str,
        state: Dict[str, Any]
    ) -> None:
        if thread_id not in self._states:
            self._states[thread_id] = {}
        self._states[thread_id][entity_id] = state
        self.logger.debug(f"Saved state for thread {thread_id}, entity {entity_id}: {state}")
    
    async def load_state(
        self,
        entity_id: str,
        thread_id: str
    ) -> Optional[Dict[str, Any]]:
        state = self._states.get(thread_id, {}).get(entity_id)
        self.logger.debug(f"Loading state for thread {thread_id}, entity {entity_id}: {state}")
        return state
    
    async def delete_thread(
        self,
        thread_id: str,
        recursive: bool = True
    ) -> None:
        if thread_id in self._threads:
            del self._threads[thread_id]
        if thread_id in self._states:
            del self._states[thread_id]
    
    async def list_threads(
        self,
        entity_id: Optional[str] = None
    ) -> List[ThreadState]:
        threads = []
        for thread_id, thread in self._threads.items():
            if not entity_id or thread['entity_id'] == entity_id:
                threads.append(ThreadState(
                    thread_id=thread_id,
                    entity_id=thread['entity_id'],
                    parent_thread_id=thread['parent_thread_id'],
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                ))
        return threads
    
    async def _create_memory_dump(self) -> MemoryDump:
        """Create a dump of the current memory state"""
        threads = {}
        for thread_id, thread in self._threads.items():
            threads[thread_id] = ThreadState(
                thread_id=thread_id,
                entity_id=thread['entity_id'],
                parent_thread_id=thread['parent_thread_id'],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        
        return MemoryDump(
            threads=threads,
            store=self._states
        )
    
    async def _restore_memory_dump(self, dump: MemoryDump) -> None:
        """Restore memory state from a dump"""
        self._threads.clear()
        self._states.clear()
        
        for thread_id, thread in dump.threads.items():
            self._threads[thread_id] = {
                'entity_id': thread.entity_id,
                'parent_thread_id': thread.parent_thread_id
            }
        
        self._states = dump.store

@pytest.fixture
def temp_checkpoint_file(tmp_path):
    """Fixture for temporary checkpoint file"""
    return tmp_path / "checkpoint.json"

@pytest.fixture
def memory_provider():
    """Fixture for mock memory provider"""
    return MockMemoryProvider()

@pytest.fixture
def graph_state():
    """Fixture for graph state with some test data"""
    state = GraphState()
    channel = state.create_channel(LastValue, "test", type_hint=str)
    channel.set("test_value")
    state.set_global_state({'key': 'value'})
    return state

def test_checkpoint_model():
    """Test GraphCheckpoint model"""
    checkpoint = GraphCheckpoint(
        state_data={'test': 'data'}
    )
    
    assert checkpoint.version == "1.0"
    assert isinstance(checkpoint.created_at, datetime)
    assert checkpoint.state_data == {'test': 'data'}

@pytest.mark.asyncio
async def test_file_checkpointing(graph_state, temp_checkpoint_file):
    """Test saving and loading checkpoints to/from file"""
    checkpointer = GraphCheckpointer()
    
    # Save checkpoint
    await checkpointer.save_checkpoint(graph_state, path=temp_checkpoint_file)
    
    # Verify file exists and contains valid JSON
    assert temp_checkpoint_file.exists()
    with temp_checkpoint_file.open('r') as f:
        data = json.load(f)
        assert 'version' in data
        assert 'state_data' in data
    
    # Load into new state
    new_state = GraphState()
    await checkpointer.load_checkpoint(new_state, path=temp_checkpoint_file)
    
    # Verify state was restored
    assert new_state.get_channel("test").get() == "test_value"
    assert new_state.get_global_state() == {'key': 'value'}

@pytest.mark.asyncio
async def test_memory_provider_checkpointing(graph_state, memory_provider):
    """Test saving and loading checkpoints using memory provider"""
    logger.info("Starting memory provider checkpointing test")
    checkpointer = GraphCheckpointer(memory_provider)
    
    original_graph_id = graph_state.graph_id
    
    # Create thread and save checkpoint
    thread_id = await memory_provider.create_thread(original_graph_id)
    logger.debug(f"Created thread {thread_id} for graph {original_graph_id}")
    
    checkpoint_data = graph_state.checkpoint()
    logger.debug(f"Graph state checkpoint data: {checkpoint_data}")
    
    await checkpointer.save_checkpoint(graph_state, thread_id=thread_id)
    logger.debug("Saved checkpoint")
    
    # Debug: Check memory provider state
    state = await memory_provider.load_state(original_graph_id, thread_id)
    logger.debug(f"Raw state in memory provider: {state}")
    
    # Load into new state
    new_state = GraphState()
    logger.debug("Created new graph state for restoration")
    
    try:
        await checkpointer.load_checkpoint(
            new_state, 
            thread_id=thread_id,
            graph_id=original_graph_id
        )
        logger.debug("Successfully loaded checkpoint")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {str(e)}", exc_info=True)
        raise
    
    # Verify state was restored
    channel_value = new_state.get_channel("test").get()
    global_state = new_state.get_global_state()
    logger.debug(f"Restored channel value: {channel_value}")
    logger.debug(f"Restored global state: {global_state}")
    
    assert channel_value == "test_value"
    assert global_state == {'key': 'value'}

@pytest.mark.asyncio
async def test_checkpoint_listing(graph_state, memory_provider):
    """Test listing available checkpoints"""
    checkpointer = GraphCheckpointer(memory_provider)
    
    # Create multiple checkpoints
    thread_id1 = await memory_provider.create_thread(graph_state.graph_id)
    thread_id2 = await memory_provider.create_thread(graph_state.graph_id)
    
    await checkpointer.save_checkpoint(graph_state, thread_id=thread_id1)
    await checkpointer.save_checkpoint(graph_state, thread_id=thread_id2)
    
    # List checkpoints
    checkpoints = await checkpointer.list_checkpoints(graph_state.graph_id)
    
    assert len(checkpoints) == 2
    assert thread_id1 in checkpoints
    assert thread_id2 in checkpoints
    assert all(isinstance(cp, GraphCheckpoint) for cp in checkpoints.values())

@pytest.mark.asyncio
async def test_checkpoint_deletion(graph_state, memory_provider):
    """Test checkpoint deletion"""
    checkpointer = GraphCheckpointer(memory_provider)
    
    # Create and save checkpoint
    thread_id = await memory_provider.create_thread(graph_state.graph_id)
    await checkpointer.save_checkpoint(graph_state, thread_id=thread_id)
    
    # Delete checkpoint
    await checkpointer.delete_checkpoint(thread_id)
    
    # Verify checkpoint was deleted
    checkpoints = await checkpointer.list_checkpoints(graph_state.graph_id)
    assert thread_id not in checkpoints

@pytest.mark.asyncio
async def test_checkpoint_not_found(graph_state):
    """Test error handling when checkpoint not found"""
    checkpointer = GraphCheckpointer()
    
    with pytest.raises(ValueError, match="No checkpoint found"):
        await checkpointer.load_checkpoint(
            graph_state,
            path=Path("nonexistent.json")
        )

@pytest.mark.asyncio
async def test_multiple_save_locations(graph_state, temp_checkpoint_file, memory_provider):
    """Test saving checkpoint to both file and memory provider"""
    logger.info("Starting multiple save locations test")
    checkpointer = GraphCheckpointer(memory_provider)
    
    original_graph_id = graph_state.graph_id
    thread_id = await memory_provider.create_thread(original_graph_id)
    logger.debug(f"Created thread {thread_id}")
    
    # Save to both locations
    await checkpointer.save_checkpoint(
        graph_state,
        path=temp_checkpoint_file,
        thread_id=thread_id
    )
    logger.debug(f"Saved checkpoint to file {temp_checkpoint_file} and thread {thread_id}")
    
    # Debug: Check file contents
    with temp_checkpoint_file.open('r') as f:
        file_data = json.load(f)
        logger.debug(f"File checkpoint data: {file_data}")
    
    # Debug: Check memory provider state
    mem_state = await memory_provider.load_state(original_graph_id, thread_id)
    logger.debug(f"Memory provider state: {mem_state}")
    
    # Verify file checkpoint
    new_state1 = GraphState()
    await checkpointer.load_checkpoint(new_state1, path=temp_checkpoint_file)
    logger.debug("Loaded file checkpoint")
    
    # Verify memory provider checkpoint
    new_state2 = GraphState()
    try:
        await checkpointer.load_checkpoint(
            new_state2, 
            thread_id=thread_id,
            graph_id=original_graph_id
        )
        logger.debug("Loaded memory provider checkpoint")
    except Exception as e:
        logger.error(f"Failed to load memory provider checkpoint: {str(e)}", exc_info=True)
        raise
    
    assert new_state1.get_channel("test").get() == "test_value"
    assert new_state2.get_channel("test").get() == "test_value"

if __name__ == "__main__":
    # Configure logging for main execution
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Run tests with pytest
    args = [
        __file__,
        "-v",
        "--tb=short",
        "-p", "no:warnings",
        "-p", "asyncio",
        "--log-cli-level=DEBUG",  # Show logs in pytest output
        "-s"  # Don't capture stdout/stderr
    ]
    sys.exit(pytest.main(args))