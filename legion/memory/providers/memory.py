import asyncio
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from legion.interface.schemas import Message

from ..base import MemoryDump, MemoryProvider, ThreadState


class ConversationMemory(BaseModel):
    """Stores conversation history and metadata"""

    messages: List[Message] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.now)

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation history"""
        self.messages.append(message)
        self.last_updated = datetime.now()

    def get_context_window(self, max_messages: Optional[int] = None) -> List[Message]:
        """Get recent message history, optionally limited"""
        if max_messages:
            return self.messages[-max_messages:]
        return self.messages

class InMemoryProvider(MemoryProvider):
    """Simple in-memory implementation of MemoryProvider"""

    def __init__(self):
        self._store: Dict[tuple[str, str], Dict[str, Any]] = {}
        self._threads: Dict[str, ThreadState] = {}
        self._lock = asyncio.Lock()

    async def create_thread(
        self,
        entity_id: str,
        parent_thread_id: Optional[str] = None
    ) -> str:
        """Create a new thread"""
        async with self._lock:
            thread_id = str(uuid4())
            thread_state = ThreadState(
                thread_id=thread_id,
                entity_id=entity_id,
                parent_thread_id=parent_thread_id,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            self._threads[thread_id] = thread_state
            return thread_id

    async def save_state(
        self,
        entity_id: str,
        thread_id: str,
        state: Dict[str, Any]
    ) -> None:
        """Save state to memory"""
        async with self._lock:
            if thread_id not in self._threads:
                raise ValueError(f"Thread {thread_id} does not exist")

            # Update thread state
            thread_state = self._threads[thread_id]
            thread_state.updated_at = datetime.now()

            # Update messages in ThreadState
            if "messages" in state:
                thread_state.messages = state["messages"]

            # Update metadata if present
            if "metadata" in state:
                thread_state.metadata.update(state["metadata"])

            # Save entity state
            key = (entity_id, thread_id)
            self._store[key] = deepcopy(state)

    async def load_state(
        self,
        entity_id: str,
        thread_id: str
    ) -> Optional[Dict[str, Any]]:
        """Load state from memory"""
        key = (entity_id, thread_id)
        state = self._store.get(key)
        return deepcopy(state) if state else None

    async def delete_thread(
        self,
        thread_id: str,
        recursive: bool = True
    ) -> None:
        """Delete a thread and optionally its children"""
        async with self._lock:
            if thread_id not in self._threads:
                return

            # Get all child threads if recursive
            to_delete = {thread_id}
            if recursive:
                child_threads = [
                    t.thread_id for t in self._threads.values()
                    if t.parent_thread_id == thread_id
                ]
                to_delete.update(child_threads)

            # Delete thread states and data
            for tid in to_delete:
                if tid in self._threads:
                    thread_state = self._threads[tid]
                    key = (thread_state.entity_id, tid)
                    self._store.pop(key, None)
                    del self._threads[tid]

    async def list_threads(
        self,
        entity_id: Optional[str] = None
    ) -> List[ThreadState]:
        """List all threads, optionally filtered by entity"""
        if entity_id:
            return [
                t for t in self._threads.values()
                if t.entity_id == entity_id
            ]
        return list(self._threads.values())

    async def _create_memory_dump(self) -> MemoryDump:
        """Create a dump of the current memory state"""
        async with self._lock:
            # Convert tuple keys to strings for JSON serialization
            store_dict = {
                f"{entity_id}:{thread_id}": state
                for (entity_id, thread_id), state in self._store.items()
            }

            return MemoryDump(
                threads={
                    thread_id: thread_state
                    for thread_id, thread_state in self._threads.items()
                },
                store=store_dict
            )

    async def _restore_memory_dump(self, dump: MemoryDump) -> None:
        """Restore memory state from a dump"""
        async with self._lock:
            # Clear current state
            self._threads.clear()
            self._store.clear()

            # Restore threads
            for thread_id, thread_state in dump.threads.items():
                self._threads[thread_id] = ThreadState(**thread_state.dict())

            # Restore store (convert string keys back to tuples)
            for key_str, state in dump.store.items():
                entity_id, thread_id = key_str.split(":", 1)
                self._store[(entity_id, thread_id)] = state
