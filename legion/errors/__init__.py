# Errors __init__.py
"""Custom exceptions for legion"""

from .exceptions import AgentError, LegionError, ProviderError


class LegionError(Exception):
    """Base exception class for Legion"""

    pass

class AgentError(LegionError):
    """Exception raised for errors in Agent operations"""

    pass

class ProviderError(LegionError):
    """Exception raised for errors in Provider operations"""

    pass

class ConfigError(LegionError):
    """Exception raised for configuration errors"""

    pass

class ValidationError(LegionError):
    """Exception raised for validation errors"""

    pass

class ToolError(LegionError):
    """Error raised when a tool fails to execute."""

    def __init__(self, message: str, tool_name: str = None):
        self.tool_name = tool_name
        if "Invalid parameters for tool" in message and not tool_name:
            # Already formatted message from tools.py
            super().__init__(f"Tool error: {message}")
        else:
            # Message needs formatting with tool name
            super().__init__(f"Tool error{f' in {tool_name}' if tool_name else ''}: {message}")

class ConfigurationError(LegionError):
    """Error raised when configuration is invalid."""

    pass

class NodeError(LegionError):
    """Error raised when a node fails."""

    def __init__(self, message: str, node_id: str = None):
        self.node_id = node_id
        super().__init__(f"Node error{f' in {node_id}' if node_id else ''}: {message}")

class StateError(LegionError):
    """Error raised when state operations fail."""

    pass

class ResourceError(LegionError):
    """Error raised when resource limits are exceeded."""

    pass

class NonRetryableError(LegionError):
    """Error that should not be retried."""

    pass

class FatalError(NonRetryableError):
    """Error that indicates a fatal condition."""

    pass
