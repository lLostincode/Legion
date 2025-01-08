import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field


class ThreadState(BaseModel):
    """State information for a thread"""

    thread_id: str
    entity_id: str
    parent_thread_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    messages: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    @property
    def age(self) -> float:
        """Get thread age in seconds"""
        return (datetime.now() - self.created_at).total_seconds()

    @property
    def message_count(self) -> int:
        """Get total message count"""
        return len(self.messages)

    @property
    def last_message(self) -> Optional[Dict[str, Any]]:
        """Get the last message in the thread"""
        return self.messages[-1] if self.messages else None

    @property
    def roles(self) -> Set[str]:
        """Get unique message roles in thread"""
        return {msg.get("role", "") for msg in self.messages if msg.get("role")}

class MemoryDump(BaseModel):
    """Container for memory provider state"""

    version: str = "1.0"
    created_at: datetime = Field(default_factory=datetime.now)
    threads: Dict[str, ThreadState]
    store: Dict[str, Dict[str, Any]]

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class MemoryProvider(ABC):
    """Abstract base class for memory providers"""

    @abstractmethod
    async def create_thread(
        self,
        entity_id: str,
        parent_thread_id: Optional[str] = None
    ) -> str:
        """Create a new thread and return its ID"""
        pass

    @abstractmethod
    async def save_state(
        self,
        entity_id: str,
        thread_id: str,
        state: Dict[str, Any]
    ) -> None:
        """Save state for an entity in a thread"""
        pass

    @abstractmethod
    async def load_state(
        self,
        entity_id: str,
        thread_id: str
    ) -> Optional[Dict[str, Any]]:
        """Load state for an entity in a thread"""
        pass

    @abstractmethod
    async def delete_thread(
        self,
        thread_id: str,
        recursive: bool = True
    ) -> None:
        """Delete a thread and optionally all child threads"""
        pass

    @abstractmethod
    async def list_threads(
        self,
        entity_id: Optional[str] = None
    ) -> List[ThreadState]:
        """List all threads, optionally filtered by entity"""
        pass

    async def dump_memory(self, path: Union[str, Path]) -> None:
        """Dump entire memory state to a file"""
        dump = await self._create_memory_dump()

        path = Path(path)
        with path.open("w") as f:
            json_str = dump.model_dump_json(indent=2)
            f.write(json_str)

    async def load_memory(self, path: Union[str, Path]) -> None:
        """Load memory state from a file"""
        path = Path(path)
        with path.open("r") as f:
            data = json.load(f)
            if "created_at" in data:
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            for thread_data in data.get("threads", {}).values():
                if "created_at" in thread_data:
                    thread_data["created_at"] = datetime.fromisoformat(thread_data["created_at"])
                if "updated_at" in thread_data:
                    thread_data["updated_at"] = datetime.fromisoformat(thread_data["updated_at"])

        dump = MemoryDump(**data)
        await self._restore_memory_dump(dump)

    @abstractmethod
    async def _create_memory_dump(self) -> MemoryDump:
        """Create a dump of the current memory state"""
        pass

    @abstractmethod
    async def _restore_memory_dump(self, dump: MemoryDump) -> None:
        """Restore memory state from a dump"""
        pass

    async def get_or_create_thread(self, entity_id: str) -> str:
        """Get the first available thread for an entity or create a new one"""
        # Try to get existing threads
        threads = await self.list_threads(entity_id)

        # Return first thread if exists, otherwise create new one
        if threads:
            return threads[0].thread_id

        # Create new thread if none exist
        return await self.create_thread(entity_id)

    # Thread Information Methods
    async def get_thread_count(self, entity_id: Optional[str] = None) -> int:
        """Get total number of threads for an entity or all threads"""
        threads = await self.list_threads(entity_id)
        return len(threads)

    async def get_thread(self, thread_id: str) -> Optional[ThreadState]:
        """Get a specific thread by ID"""
        threads = await self.list_threads()
        return next((t for t in threads if t.thread_id == thread_id), None)

    async def get_latest_thread(self, entity_id: Optional[str] = None) -> Optional[ThreadState]:
        """Get the most recently updated thread"""
        threads = await self.list_threads(entity_id)
        return max(threads, key=lambda t: t.updated_at) if threads else None

    async def get_oldest_thread(self, entity_id: Optional[str] = None) -> Optional[ThreadState]:
        """Get the oldest thread"""
        threads = await self.list_threads(entity_id)
        return min(threads, key=lambda t: t.created_at) if threads else None

    # Message Information Methods
    async def get_message_count(self, entity_id: Optional[str] = None) -> int:
        """Get total message count across all threads"""
        threads = await self.list_threads(entity_id)
        return sum(t.message_count for t in threads)

    async def get_last_message(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get the last message from a specific thread"""
        thread = await self.get_thread(thread_id)
        return thread.last_message if thread else None

    # Entity Information Methods
    async def get_entities(self) -> Set[str]:
        """Get all unique entity IDs"""
        threads = await self.list_threads()
        return {t.entity_id for t in threads}

    async def get_entity_stats(self, entity_id: str) -> Dict[str, Any]:
        """Get statistics for a specific entity"""
        threads = await self.list_threads(entity_id)
        if not threads:
            return {}

        return {
            "thread_count": len(threads),
            "total_messages": sum(t.message_count for t in threads),
            "oldest_thread": min(t.created_at for t in threads),
            "newest_thread": max(t.created_at for t in threads),
            "last_updated": max(t.updated_at for t in threads),
            "unique_roles": set().union(*(t.roles for t in threads))
        }

    # Search and Filter Methods
    async def find_threads_by_age(
        self,
        max_age_seconds: Optional[float] = None,
        min_age_seconds: Optional[float] = None
    ) -> List[ThreadState]:
        """Find threads within a specific age range"""
        threads = await self.list_threads()
        filtered = threads

        if max_age_seconds is not None:
            filtered = [t for t in filtered if t.age <= max_age_seconds]
        if min_age_seconds is not None:
            filtered = [t for t in filtered if t.age >= min_age_seconds]

        return filtered

    async def find_threads_by_message_count(
        self,
        min_messages: Optional[int] = None,
        max_messages: Optional[int] = None
    ) -> List[ThreadState]:
        """Find threads with message count in range"""
        threads = await self.list_threads()
        filtered = threads

        if min_messages is not None:
            filtered = [t for t in filtered if t.message_count >= min_messages]
        if max_messages is not None:
            filtered = [t for t in filtered if t.message_count <= max_messages]

        return filtered

    # Utility Methods
    async def cleanup_old_threads(self, max_age_seconds: float) -> int:
        """Delete threads older than specified age"""
        old_threads = await self.find_threads_by_age(min_age_seconds=max_age_seconds)
        for thread in old_threads:
            await self.delete_thread(thread.thread_id)
        return len(old_threads)
