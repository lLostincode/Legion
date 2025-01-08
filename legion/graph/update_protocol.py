from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional, TypeVar
from uuid import uuid4

T = TypeVar("T")

@dataclass
class UpdateOperation:
    """Represents a single update operation"""

    channel_id: str
    value: Any
    timestamp: datetime = datetime.now()
    operation_id: str = str(uuid4())

class UpdateTransaction:
    """Represents a group of atomic updates"""

    def __init__(self):
        """Initialize update transaction"""
        self.operations: List[UpdateOperation] = []
        self.started_at: datetime = datetime.now()
        self.completed_at: Optional[datetime] = None
        self.transaction_id: str = str(uuid4())
        self.is_committed: bool = False
        self.is_rolled_back: bool = False
        self.error: Optional[Exception] = None

    def add_operation(self, channel_id: str, value: Any) -> None:
        """Add an update operation to the transaction

        Args:
        ----
            channel_id: ID of channel to update
            value: New value to set

        """
        if self.is_committed or self.is_rolled_back:
            raise RuntimeError("Cannot modify completed transaction")
        self.operations.append(UpdateOperation(channel_id, value))

    def commit(self) -> None:
        """Mark transaction as committed"""
        if self.is_committed or self.is_rolled_back:
            raise RuntimeError("Transaction already completed")
        self.is_committed = True
        self.completed_at = datetime.now()

    def rollback(self, error: Optional[Exception] = None) -> None:
        """Mark transaction as rolled back

        Args:
        ----
            error: Optional error that caused rollback

        """
        if self.is_committed or self.is_rolled_back:
            raise RuntimeError("Transaction already completed")
        self.is_rolled_back = True
        self.completed_at = datetime.now()
        self.error = error

class UpdateProtocol:
    """Manages atomic updates and transactions for channels"""

    def __init__(self):
        """Initialize update protocol"""
        self._active_transactions: Dict[str, UpdateTransaction] = {}
        self._channel_versions: Dict[str, int] = {}
        self._channel_locks: Dict[str, Lock] = {}
        self._performance_metrics: Dict[str, Dict[str, Any]] = {}

    def _get_channel_lock(self, channel_id: str) -> Lock:
        """Get or create lock for channel

        Args:
        ----
            channel_id: ID of channel

        Returns:
        -------
            Lock for channel

        """
        if channel_id not in self._channel_locks:
            self._channel_locks[channel_id] = Lock()
        return self._channel_locks[channel_id]

    def _update_metrics(self, transaction: UpdateTransaction) -> None:
        """Update performance metrics for transaction

        Args:
        ----
            transaction: Completed transaction

        """
        duration = (transaction.completed_at - transaction.started_at).total_seconds()
        operation_count = len(transaction.operations)

        for operation in transaction.operations:
            channel_id = operation.channel_id
            if channel_id not in self._performance_metrics:
                self._performance_metrics[channel_id] = {
                    "update_count": 0,
                    "total_duration": 0.0,
                    "avg_duration": 0.0,
                    "error_count": 0
                }

            metrics = self._performance_metrics[channel_id]
            metrics["update_count"] += 1
            metrics["total_duration"] += duration / operation_count
            metrics["avg_duration"] = metrics["total_duration"] / metrics["update_count"]

            if transaction.error:
                metrics["error_count"] += 1

    def begin_transaction(self) -> str:
        """Begin a new transaction

        Returns
        -------
            Transaction ID

        """
        transaction = UpdateTransaction()
        self._active_transactions[transaction.transaction_id] = transaction
        return transaction.transaction_id

    def add_update(self, transaction_id: str, channel_id: str, value: Any) -> None:
        """Add an update to a transaction

        Args:
        ----
            transaction_id: ID of transaction
            channel_id: ID of channel to update
            value: New value to set

        """
        if transaction_id not in self._active_transactions:
            raise ValueError(f"Unknown transaction {transaction_id}")

        transaction = self._active_transactions[transaction_id]
        transaction.add_operation(channel_id, value)

    def commit_transaction(self, transaction_id: str) -> None:
        """Commit a transaction

        Args:
        ----
            transaction_id: ID of transaction to commit

        """
        if transaction_id not in self._active_transactions:
            raise ValueError(f"Unknown transaction {transaction_id}")

        transaction = self._active_transactions[transaction_id]

        # Acquire all locks in consistent order to prevent deadlocks
        channel_ids = sorted(set(op.channel_id for op in transaction.operations))
        locks = [self._get_channel_lock(channel_id) for channel_id in channel_ids]

        try:
            # Acquire all locks
            for lock in locks:
                lock.acquire()

            # Apply updates
            for operation in transaction.operations:
                self._channel_versions[operation.channel_id] = \
                    self._channel_versions.get(operation.channel_id, 0) + 1

            transaction.commit()

        except Exception as e:
            transaction.rollback(e)
            raise

        finally:
            # Release all locks in reverse order
            for lock in reversed(locks):
                lock.release()

            self._update_metrics(transaction)
            self._active_transactions.pop(transaction_id)

    def rollback_transaction(self, transaction_id: str, error: Optional[Exception] = None) -> None:
        """Rollback a transaction

        Args:
        ----
            transaction_id: ID of transaction to rollback
            error: Optional error that caused rollback

        """
        if transaction_id not in self._active_transactions:
            raise ValueError(f"Unknown transaction {transaction_id}")

        transaction = self._active_transactions[transaction_id]
        transaction.rollback(error)
        self._update_metrics(transaction)
        self._active_transactions.pop(transaction_id)

    def get_channel_version(self, channel_id: str) -> int:
        """Get current version of channel

        Args:
        ----
            channel_id: ID of channel

        Returns:
        -------
            Current version number

        """
        return self._channel_versions.get(channel_id, 0)

    def get_metrics(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for channel

        Args:
        ----
            channel_id: ID of channel

        Returns:
        -------
            Metrics dictionary if found, None otherwise

        """
        return self._performance_metrics.get(channel_id)

    def clear(self) -> None:
        """Clear all transactions and metrics"""
        self._active_transactions.clear()
        self._channel_versions.clear()
        self._channel_locks.clear()
        self._performance_metrics.clear()
