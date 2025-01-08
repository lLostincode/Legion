
import pytest

from legion.graph.channel_manager import ChannelManager
from legion.graph.channels import LastValue


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
