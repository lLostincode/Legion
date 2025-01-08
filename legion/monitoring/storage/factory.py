"""Factory for creating storage backends"""

from enum import Enum
from typing import Dict, Optional, Type, Union

from .base import StorageBackend
from .config import StorageConfig
from .memory import MemoryStorageBackend
from .sqlite import SQLiteStorageBackend


class StorageType(Enum):
    """Available storage backend types"""

    MEMORY = "memory"
    SQLITE = "sqlite"

class StorageFactory:
    """Factory for creating storage backends

    This class manages the creation of storage backends and ensures
    consistent configuration across the application.
    """

    _backends: Dict[str, Type[StorageBackend]] = {
        StorageType.MEMORY.value: MemoryStorageBackend,
        StorageType.SQLITE.value: SQLiteStorageBackend
    }

    @classmethod
    def create(cls,
               storage_type: Union[StorageType, str] = StorageType.MEMORY,
               config: Optional[StorageConfig] = None,
               **kwargs) -> StorageBackend:
        """Create a new storage backend

        Args:
        ----
            storage_type: Type of storage backend to create
            config: Optional storage configuration
            **kwargs: Additional arguments passed to the backend constructor

        Returns:
        -------
            The created storage backend

        Raises:
        ------
            ValueError: If the storage type is not supported

        """
        if isinstance(storage_type, StorageType):
            storage_type = storage_type.value

        if storage_type not in cls._backends:
            raise ValueError(f"Unsupported storage type: {storage_type}")

        backend_cls = cls._backends[storage_type]
        return backend_cls(config=config, **kwargs)

    @classmethod
    def register_backend(cls, storage_type: str, backend_cls: Type[StorageBackend]):
        """Register a new storage backend type

        Args:
        ----
            storage_type: Unique identifier for the backend type
            backend_cls: Backend class to register

        Raises:
        ------
            ValueError: If the storage type is already registered

        """
        if storage_type in cls._backends:
            raise ValueError(f"Storage type already registered: {storage_type}")

        cls._backends[storage_type] = backend_cls
