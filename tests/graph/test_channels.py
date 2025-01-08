from datetime import datetime
from typing import List

import pytest

from legion.graph.channel_manager import ChannelManager
from legion.graph.channels import (
    AggregatorChannel,
    BarrierChannel,
    BroadcastChannel,
    ChannelMetadata,
    LastValue,
    MessageChannel,
    SharedMemory,
    SharedState,
    ValueSequence,
)


def test_channel_metadata():
    """Test channel metadata functionality"""
    channel = LastValue(type_hint=str)
    metadata = channel.metadata

    assert isinstance(metadata, ChannelMetadata)
    assert isinstance(metadata.created_at, datetime)
    assert metadata.version == 0
    assert metadata.type_hint == "str"

def test_last_value_channel():
    """Test LastValue channel functionality"""
    channel = LastValue(type_hint=str)

    # Test initial state
    assert channel.get() is None
    assert channel.metadata.version == 0

    # Test value setting
    channel.set("test")
    assert channel.get() == "test"
    assert channel.metadata.version == 1

    # Test type validation
    with pytest.raises(TypeError):
        channel.set(123)

    # Test checkpointing
    checkpoint = channel.checkpoint()
    assert isinstance(checkpoint, dict)
    assert checkpoint["value"] == "test"

    # Test restoration
    new_channel = LastValue(type_hint=str)
    new_channel.restore(checkpoint)
    assert new_channel.get() == "test"
    assert new_channel.metadata.version == checkpoint["metadata"]["version"]

def test_value_sequence_channel():
    """Test ValueSequence channel functionality"""
    channel = ValueSequence(type_hint=int, max_size=3)

    # Test initial state
    assert channel.get() == []
    assert channel.metadata.version == 0

    # Test append
    channel.append(1)
    channel.append(2)
    assert channel.get() == [1, 2]
    assert channel.metadata.version == 2

    # Test max size
    channel.append(3)
    channel.append(4)
    assert channel.get() == [2, 3, 4]  # First value should be dropped

    # Test bulk set
    channel.set([5, 6])
    assert channel.get() == [5, 6]

    # Test type validation
    with pytest.raises(TypeError):
        channel.append("invalid")

    # Test checkpointing
    checkpoint = channel.checkpoint()
    assert isinstance(checkpoint, dict)
    assert checkpoint["values"] == [5, 6]
    assert checkpoint["max_size"] == 3

    # Test restoration
    new_channel = ValueSequence(type_hint=int)
    new_channel.restore(checkpoint)
    assert new_channel.get() == [5, 6]

def test_shared_state_channel():
    """Test SharedState channel functionality"""
    channel = SharedState()

    # Test initial state
    assert channel.get() == {}
    assert channel.metadata.version == 0

    # Test state setting
    initial_state = {"key": "value"}
    channel.set(initial_state)
    assert channel.get() == initial_state
    assert channel.metadata.version == 1

    # Test state update
    channel.update({"new_key": "new_value"})
    assert channel.get() == {"key": "value", "new_key": "new_value"}
    assert channel.metadata.version == 2

    # Test type validation
    with pytest.raises(TypeError):
        channel.set([])  # Must be dict

    # Test checkpointing
    checkpoint = channel.checkpoint()
    assert isinstance(checkpoint, dict)
    assert checkpoint["state"] == {"key": "value", "new_key": "new_value"}

    # Test restoration
    new_channel = SharedState()
    new_channel.restore(checkpoint)
    assert new_channel.get() == {"key": "value", "new_key": "new_value"}

def test_channel_isolation():
    """Test that channels maintain proper isolation"""
    # Test LastValue isolation
    lv1 = LastValue(type_hint=str)
    lv2 = LastValue(type_hint=str)
    lv1.set("test")
    assert lv2.get() is None

    # Test ValueSequence isolation
    vs1 = ValueSequence(type_hint=int)
    vs2 = ValueSequence(type_hint=int)
    vs1.append(1)
    assert vs2.get() == []

    # Test SharedState isolation
    ss1 = SharedState()
    ss2 = SharedState()
    ss1.set({"key": "value"})
    assert ss2.get() == {}

class TestMessageChannel:
    def test_init(self):
        channel = MessageChannel[str](str, capacity=5)
        assert channel.available_capacity == 5
        assert not channel.is_full

        unlimited_channel = MessageChannel[int](int)
        assert unlimited_channel.available_capacity is None
        assert not unlimited_channel.is_full

    def test_push_pop(self):
        channel = MessageChannel[str](str, capacity=3)

        # Test pushing
        assert channel.push("msg1")
        assert channel.push("msg2")
        assert channel.push("msg3")
        assert not channel.push("msg4")  # Should fail, at capacity

        # Test popping
        assert channel.pop() == "msg1"
        assert channel.pop() == "msg2"
        assert channel.pop() == "msg3"
        assert channel.pop() is None

    def test_batch_operations(self):
        channel = MessageChannel[int](int, capacity=5)

        # Test batch push
        messages = [1, 2, 3, 4, 5, 6]
        assert channel.push_batch(messages) == 5  # Only 5 should be pushed
        assert len(channel.get()) == 5
        assert channel.is_full

        # Test batch pop
        popped = channel.pop_batch(3)
        assert popped == [1, 2, 3]
        assert len(channel.get()) == 2
        assert not channel.is_full

        # Test pushing after batch pop
        assert channel.push(4)
        assert len(channel.get()) == 3

    def test_clear(self):
        channel = MessageChannel[str](str)
        channel.push("msg1")
        channel.push("msg2")

        assert len(channel.get()) == 2
        channel.clear()
        assert len(channel.get()) == 0

    def test_type_validation(self):
        channel = MessageChannel[str](str)

        with pytest.raises(TypeError):
            channel.push(123)

        with pytest.raises(TypeError):
            channel.push_batch([123, "valid", 456])

    def test_checkpoint_restore(self):
        channel = MessageChannel[int](int, capacity=3)
        channel.push(1)
        channel.push(2)

        checkpoint = channel.checkpoint()

        new_channel = MessageChannel[int](int)
        new_channel.restore(checkpoint)

        assert new_channel.get() == [1, 2]
        assert new_channel.available_capacity == 1

class TestBarrierChannel:
    def test_init(self):
        # Test valid initialization
        channel = BarrierChannel(contributor_count=3)
        assert channel.remaining_contributors == 3
        assert not channel.is_triggered()
        assert len(channel.current_contributors) == 0

        # Test invalid initialization
        with pytest.raises(ValueError):
            BarrierChannel(contributor_count=0)

    def test_contribution(self):
        channel = BarrierChannel(contributor_count=2)

        # First contribution
        assert not channel.contribute("node1")
        assert channel.remaining_contributors == 1
        assert len(channel.current_contributors) == 1
        assert "node1" in channel.current_contributors

        # Same contributor again
        assert not channel.contribute("node1")
        assert channel.remaining_contributors == 1
        assert len(channel.current_contributors) == 1

        # Second contributor triggers
        assert channel.contribute("node2")
        assert channel.remaining_contributors == 0
        assert channel.is_triggered()
        assert len(channel.current_contributors) == 2

    def test_timeout(self):
        channel = BarrierChannel(contributor_count=2, timeout=0.1)

        # Add first contribution
        assert not channel.contribute("node1")
        assert channel.remaining_contributors == 1

        # Wait for timeout
        import time
        time.sleep(0.2)

        # Check timeout reset
        assert not channel.is_triggered()
        assert channel.remaining_contributors == 2
        assert len(channel.current_contributors) == 0

    def test_reset(self):
        channel = BarrierChannel(contributor_count=2)

        # Add contributions and trigger
        assert not channel.contribute("node1")
        assert channel.contribute("node2")
        assert channel.is_triggered()

        # Reset and verify state
        channel.reset()
        assert not channel.is_triggered()
        assert channel.remaining_contributors == 2
        assert len(channel.current_contributors) == 0

    def test_checkpoint_restore(self):
        channel = BarrierChannel(contributor_count=3, timeout=1.0)

        # Add some contributions
        channel.contribute("node1")
        channel.contribute("node2")

        # Create checkpoint
        checkpoint = channel.checkpoint()

        # Create new channel and restore
        new_channel = BarrierChannel(contributor_count=1)  # Different count
        new_channel.restore(checkpoint)

        # Verify restored state
        assert new_channel.remaining_contributors == 1
        assert len(new_channel.current_contributors) == 2
        assert "node1" in new_channel.current_contributors
        assert "node2" in new_channel.current_contributors
        assert new_channel._timeout == 1.0

class TestBroadcastChannel:
    def test_init(self):
        channel = BroadcastChannel[str](str, history_size=5)
        assert channel.subscriber_count == 0
        assert len(channel.history) == 0
        assert channel.get() is None

        unlimited_channel = BroadcastChannel[int](int)
        assert unlimited_channel.subscriber_count == 0
        assert len(unlimited_channel.history) == 0

    def test_subscription(self):
        channel = BroadcastChannel[str](str)

        # Test subscribe
        channel.subscribe("sub1")
        assert channel.subscriber_count == 1
        assert "sub1" in channel.subscribers

        # Test duplicate subscription
        channel.subscribe("sub1")
        assert channel.subscriber_count == 1

        # Test multiple subscribers
        channel.subscribe("sub2")
        assert channel.subscriber_count == 2
        assert "sub2" in channel.subscribers

        # Test unsubscribe
        channel.unsubscribe("sub1")
        assert channel.subscriber_count == 1
        assert "sub1" not in channel.subscribers
        assert "sub2" in channel.subscribers

        # Test unsubscribe non-existent
        channel.unsubscribe("non-existent")
        assert channel.subscriber_count == 1

    def test_broadcasting(self):
        channel = BroadcastChannel[str](str)

        # Test single broadcast
        channel.broadcast("msg1")
        assert channel.get() == "msg1"
        assert channel.history == ["msg1"]

        # Test multiple broadcasts
        channel.broadcast("msg2")
        channel.broadcast("msg3")
        assert channel.get() == "msg3"
        assert channel.history == ["msg1", "msg2", "msg3"]

    def test_history_limit(self):
        channel = BroadcastChannel[int](int, history_size=2)

        # Fill history
        channel.broadcast(1)
        channel.broadcast(2)
        assert channel.history == [1, 2]

        # Exceed limit
        channel.broadcast(3)
        assert channel.history == [2, 3]
        assert channel.get() == 3

    def test_clear_history(self):
        channel = BroadcastChannel[str](str)

        channel.broadcast("msg1")
        channel.broadcast("msg2")
        assert len(channel.history) == 2

        channel.clear_history()
        assert len(channel.history) == 0
        assert channel.get() == "msg2"  # Current value remains

    def test_type_validation(self):
        channel = BroadcastChannel[str](str)

        with pytest.raises(TypeError):
            channel.broadcast(123)

    def test_checkpoint_restore(self):
        channel = BroadcastChannel[int](int, history_size=3)

        # Setup state
        channel.subscribe("sub1")
        channel.subscribe("sub2")
        channel.broadcast(1)
        channel.broadcast(2)

        # Create checkpoint
        checkpoint = channel.checkpoint()

        # Create new channel and restore
        new_channel = BroadcastChannel[int](int)
        new_channel.restore(checkpoint)

        # Verify restored state
        assert new_channel.subscriber_count == 2
        assert "sub1" in new_channel.subscribers
        assert "sub2" in new_channel.subscribers
        assert new_channel.history == [1, 2]
        assert new_channel.get() == 2

class TestAggregatorChannel:
    def test_init(self):
        # Test with default reducer
        channel = AggregatorChannel[int](int)
        assert channel.window_size is None
        assert channel.get() is None
        assert len(channel.window) == 0

        # Test with custom reducer and window
        def sum_reducer(values: List[int]) -> int:
            return sum(values)

        channel = AggregatorChannel[int](int, reducer=sum_reducer, window_size=3)
        assert channel.window_size == 3
        assert channel.get() is None

    def test_default_reducer(self):
        channel = AggregatorChannel[str](str)

        # Test single value
        channel.contribute("first")
        assert channel.get() == "first"

        # Test multiple values
        channel.contribute("second")
        assert channel.get() == "second"  # Default reducer uses last value

    def test_custom_reducer(self):
        def avg_reducer(values: List[float]) -> float:
            return sum(values) / len(values)

        channel = AggregatorChannel[float](float, reducer=avg_reducer)

        channel.contribute(1.0)
        assert channel.get() == 1.0

        channel.contribute(2.0)
        assert channel.get() == 1.5

        channel.contribute(3.0)
        assert channel.get() == 2.0

    def test_window_management(self):
        def sum_reducer(values: List[int]) -> int:
            return sum(values)

        channel = AggregatorChannel[int](int, reducer=sum_reducer, window_size=2)

        # Fill window
        channel.contribute(1)
        assert channel.get() == 1
        assert channel.window == [1]

        channel.contribute(2)
        assert channel.get() == 3
        assert channel.window == [1, 2]

        # Exceed window size
        channel.contribute(3)
        assert channel.get() == 5  # 2 + 3
        assert channel.window == [2, 3]

    def test_reducer_error_handling(self):
        def faulty_reducer(values: List[int]) -> int:
            raise ValueError("Reducer error")

        channel = AggregatorChannel[int](int, reducer=faulty_reducer)

        # Should fallback to last value on reducer error
        channel.contribute(1)
        assert channel.get() == 1

        channel.contribute(2)
        assert channel.get() == 2

    def test_clear(self):
        channel = AggregatorChannel[int](int)

        channel.contribute(1)
        channel.contribute(2)
        assert len(channel.window) == 2

        channel.clear()
        assert len(channel.window) == 0
        assert channel.get() is None

    def test_set_value(self):
        def sum_reducer(values: List[int]) -> int:
            return sum(values)

        channel = AggregatorChannel[int](int, reducer=sum_reducer)

        # Add some values
        channel.contribute(1)
        channel.contribute(2)
        assert channel.get() == 3

        # Set single value
        channel.set(5)
        assert channel.get() == 5
        assert channel.window == [5]

    def test_type_validation(self):
        channel = AggregatorChannel[str](str)

        with pytest.raises(TypeError):
            channel.contribute(123)

        with pytest.raises(TypeError):
            channel.set(123)

    def test_checkpoint_restore(self):
        def sum_reducer(values: List[int]) -> int:
            return sum(values)

        channel = AggregatorChannel[int](int, reducer=sum_reducer, window_size=3)

        # Setup state
        channel.contribute(1)
        channel.contribute(2)

        # Create checkpoint
        checkpoint = channel.checkpoint()

        # Create new channel and restore
        new_channel = AggregatorChannel[int](int, reducer=sum_reducer)
        new_channel.restore(checkpoint)

        # Verify restored state
        assert new_channel.window == [1, 2]
        assert new_channel.window_size == 3
        assert new_channel.get() == 3  # Result recomputed with restored reducer

class TestChannelManager:
    """Test cases for ChannelManager"""

    def test_channel_type_registration(self):
        """Test channel type registration"""
        manager = ChannelManager()

        # Test registration
        manager.register_channel_type("last_value", LastValue)
        assert "last_value" in manager.get_registered_types()

        # Test duplicate registration
        with pytest.raises(ValueError):
            manager.register_channel_type("last_value", LastValue)

    def test_channel_creation(self):
        """Test channel creation"""
        manager = ChannelManager()
        manager.register_channel_type("last_value", LastValue)

        # Test creation with auto-generated ID
        channel1 = manager.create_channel("last_value", type_hint=str)
        assert channel1.id in manager.get_active_channels()

        # Test creation with custom ID
        channel2 = manager.create_channel("last_value", channel_id="custom_id", type_hint=int)
        assert channel2.id == "custom_id"
        assert channel2.id in manager.get_active_channels()

        # Test creation with unknown type
        with pytest.raises(ValueError):
            manager.create_channel("unknown_type")

    def test_channel_lifecycle(self):
        """Test channel lifecycle management"""
        manager = ChannelManager()
        manager.register_channel_type("last_value", LastValue)

        # Create channel
        channel = manager.create_channel("last_value")
        channel_id = channel.id

        # Get channel
        assert manager.get_channel(channel_id) == channel
        assert manager.get_channel("unknown") is None

        # Delete channel
        manager.delete_channel(channel_id)
        assert channel_id not in manager.get_active_channels()
        assert manager.get_channel(channel_id) is None

        # Delete non-existent channel (should not raise)
        manager.delete_channel("unknown")

    def test_error_handling(self):
        """Test error handling"""
        manager = ChannelManager()
        manager.register_channel_type("last_value", LastValue)
        channel = manager.create_channel("last_value")

        # Test error handler registration
        errors = []
        def error_handler(e: Exception):
            errors.append(e)

        manager.register_error_handler(channel.id, error_handler)

        # Test error handling
        error = ValueError("Test error")
        manager.update_metrics(channel.id, error)
        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)

        # Test registration for unknown channel
        with pytest.raises(ValueError):
            manager.register_error_handler("unknown", error_handler)

    def test_performance_metrics(self):
        """Test performance metrics tracking"""
        manager = ChannelManager()
        manager.register_channel_type("last_value", LastValue)
        channel = manager.create_channel("last_value")

        # Initial metrics
        metrics = manager.get_metrics(channel.id)
        assert metrics["update_count"] == 0
        assert metrics["error_count"] == 0
        assert metrics["last_update"] is None

        # Update metrics
        manager.update_metrics(channel.id)
        metrics = manager.get_metrics(channel.id)
        assert metrics["update_count"] == 1
        assert metrics["last_update"] is not None

        # Update with error
        manager.update_metrics(channel.id, ValueError())
        metrics = manager.get_metrics(channel.id)
        assert metrics["error_count"] == 1

        # Get metrics for unknown channel
        assert manager.get_metrics("unknown") is None

    def test_debug_mode(self):
        """Test debug mode"""
        manager = ChannelManager()
        manager.register_channel_type("last_value", LastValue)
        channel = manager.create_channel("last_value")

        def failing_handler(e: Exception):
            raise RuntimeError("Handler error")

        manager.register_error_handler(channel.id, failing_handler)

        # Test without debug mode
        manager.set_debug_mode(False)
        manager.update_metrics(channel.id, ValueError())  # Should not raise or print

        # Test with debug mode
        manager.set_debug_mode(True)
        manager.update_metrics(channel.id, ValueError())  # Should print error message

    def test_clear(self):
        """Test clearing all channels and metrics"""
        manager = ChannelManager()
        manager.register_channel_type("last_value", LastValue)

        # Create multiple channels
        channel1 = manager.create_channel("last_value")
        channel2 = manager.create_channel("last_value")

        def error_handler(e: Exception):
            pass

        manager.register_error_handler(channel1.id, error_handler)

        # Clear everything
        manager.clear()
        assert len(manager.get_active_channels()) == 0
        assert manager.get_metrics(channel1.id) is None
        assert manager.get_metrics(channel2.id) is None

class TestSharedMemory:
    """Test SharedMemory channel"""

    def test_init(self):
        """Test initialization"""
        channel = SharedMemory(str)
        assert channel.get() is None

    def test_set_get(self):
        """Test setting and getting values"""
        channel = SharedMemory(dict)
        value = {"key": "value"}

        channel.set(value)
        assert channel.get() == value

    def test_type_validation(self):
        """Test type validation"""
        channel = SharedMemory(int)

        with pytest.raises(TypeError):
            channel.set("invalid")

    def test_clear(self):
        """Test clearing value"""
        channel = SharedMemory(list)
        value = [1, 2, 3]

        channel.set(value)
        assert channel.get() == value

        channel.clear()
        assert channel.get() is None

    def test_checkpoint_restore(self):
        """Test checkpointing and restoring"""
        channel = SharedMemory(dict)
        value = {"key": "value"}

        channel.set(value)
        checkpoint = channel.checkpoint()

        new_channel = SharedMemory(dict)
        new_channel.restore(checkpoint)

        assert new_channel.get() == value

        # Test invalid checkpoint type
        with pytest.raises(ValueError):
            new_channel.restore({"type": "invalid"})
