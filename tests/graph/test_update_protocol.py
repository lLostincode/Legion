import pytest
from datetime import datetime
from threading import Thread
from time import sleep

from legion.graph.update_protocol import UpdateProtocol, UpdateTransaction, UpdateOperation

class TestUpdateProtocol:
    """Test cases for UpdateProtocol"""
    
    def test_transaction_lifecycle(self):
        """Test transaction lifecycle"""
        protocol = UpdateProtocol()
        
        # Begin transaction
        transaction_id = protocol.begin_transaction()
        assert transaction_id in protocol._active_transactions
        
        # Add updates
        protocol.add_update(transaction_id, "channel1", "value1")
        protocol.add_update(transaction_id, "channel2", "value2")
        
        transaction = protocol._active_transactions[transaction_id]
        assert len(transaction.operations) == 2
        assert not transaction.is_committed
        assert not transaction.is_rolled_back
        
        # Commit transaction
        protocol.commit_transaction(transaction_id)
        assert transaction.is_committed
        assert transaction_id not in protocol._active_transactions
        
        # Verify channel versions
        assert protocol.get_channel_version("channel1") == 1
        assert protocol.get_channel_version("channel2") == 1
    
    def test_transaction_rollback(self):
        """Test transaction rollback"""
        protocol = UpdateProtocol()
        transaction_id = protocol.begin_transaction()
        
        # Add updates
        protocol.add_update(transaction_id, "channel1", "value1")
        protocol.add_update(transaction_id, "channel2", "value2")
        
        # Rollback transaction
        error = ValueError("Test error")
        protocol.rollback_transaction(transaction_id, error)
        
        transaction = protocol._active_transactions.get(transaction_id)
        assert transaction is None  # Transaction removed
        
        # Verify channel versions (should be unchanged)
        assert protocol.get_channel_version("channel1") == 0
        assert protocol.get_channel_version("channel2") == 0
    
    def test_concurrent_updates(self):
        """Test concurrent updates to different channels"""
        protocol = UpdateProtocol()
        
        def update_channel(channel_id: str, value: str):
            transaction_id = protocol.begin_transaction()
            protocol.add_update(transaction_id, channel_id, value)
            protocol.commit_transaction(transaction_id)
        
        # Create threads for concurrent updates
        thread1 = Thread(target=update_channel, args=("channel1", "value1"))
        thread2 = Thread(target=update_channel, args=("channel2", "value2"))
        
        # Start threads
        thread1.start()
        thread2.start()
        
        # Wait for completion
        thread1.join()
        thread2.join()
        
        # Verify updates
        assert protocol.get_channel_version("channel1") == 1
        assert protocol.get_channel_version("channel2") == 1
    
    def test_transaction_validation(self):
        """Test transaction validation"""
        protocol = UpdateProtocol()
        
        # Test unknown transaction
        with pytest.raises(ValueError):
            protocol.add_update("unknown", "channel1", "value1")
            
        with pytest.raises(ValueError):
            protocol.commit_transaction("unknown")
            
        with pytest.raises(ValueError):
            protocol.rollback_transaction("unknown")
        
        # Test completed transaction
        transaction_id = protocol.begin_transaction()
        protocol.add_update(transaction_id, "channel1", "value1")
        protocol.commit_transaction(transaction_id)
        
        with pytest.raises(ValueError):
            protocol.add_update(transaction_id, "channel1", "value2")
            
        with pytest.raises(ValueError):
            protocol.commit_transaction(transaction_id)
            
        with pytest.raises(ValueError):
            protocol.rollback_transaction(transaction_id)
    
    def test_performance_metrics(self):
        """Test performance metrics tracking"""
        protocol = UpdateProtocol()
        
        # Create and commit transaction
        transaction_id = protocol.begin_transaction()
        protocol.add_update(transaction_id, "channel1", "value1")
        protocol.commit_transaction(transaction_id)
        
        # Get metrics
        metrics = protocol.get_metrics("channel1")
        assert metrics is not None
        assert metrics['update_count'] == 1
        assert metrics['total_duration'] > 0
        assert metrics['avg_duration'] > 0
        assert metrics['error_count'] == 0
        
        # Test error tracking
        transaction_id = protocol.begin_transaction()
        protocol.add_update(transaction_id, "channel1", "value2")
        protocol.rollback_transaction(transaction_id, ValueError())
        
        metrics = protocol.get_metrics("channel1")
        assert metrics['error_count'] == 1
        
        # Test unknown channel
        assert protocol.get_metrics("unknown") is None
    
    def test_clear(self):
        """Test clearing protocol state"""
        protocol = UpdateProtocol()
        
        # Create and commit transaction
        transaction_id = protocol.begin_transaction()
        protocol.add_update(transaction_id, "channel1", "value1")
        protocol.commit_transaction(transaction_id)
        
        # Clear state
        protocol.clear()
        assert len(protocol._active_transactions) == 0
        assert len(protocol._channel_versions) == 0
        assert len(protocol._channel_locks) == 0
        assert len(protocol._performance_metrics) == 0
    
    def test_transaction_operations(self):
        """Test transaction operation management"""
        transaction = UpdateTransaction()
        
        # Add operations
        transaction.add_operation("channel1", "value1")
        transaction.add_operation("channel2", "value2")
        assert len(transaction.operations) == 2
        
        # Test operation properties
        operation = transaction.operations[0]
        assert isinstance(operation.timestamp, datetime)
        assert isinstance(operation.operation_id, str)
        
        # Test completed transaction
        transaction.commit()
        with pytest.raises(RuntimeError):
            transaction.add_operation("channel3", "value3")
            
        with pytest.raises(RuntimeError):
            transaction.commit()
            
        with pytest.raises(RuntimeError):
            transaction.rollback()
    
    def test_deadlock_prevention(self):
        """Test deadlock prevention with ordered locking"""
        protocol = UpdateProtocol()
        
        def update_channels(channel_ids: list, values: list):
            transaction_id = protocol.begin_transaction()
            for channel_id, value in zip(channel_ids, values):
                protocol.add_update(transaction_id, channel_id, value)
            protocol.commit_transaction(transaction_id)
        
        # Create threads that update channels in different orders
        thread1 = Thread(target=update_channels, args=(["channel1", "channel2"], ["value1", "value2"]))
        thread2 = Thread(target=update_channels, args=(["channel2", "channel1"], ["value3", "value4"]))
        
        # Start threads
        thread1.start()
        thread2.start()
        
        # Wait for completion
        thread1.join()
        thread2.join()
        
        # Both transactions should complete without deadlock
        assert protocol.get_channel_version("channel1") == 2
        assert protocol.get_channel_version("channel2") == 2 