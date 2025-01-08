"""Legion exceptions module."""

class LegionError(Exception):
    """Base exception for all Legion errors."""

    pass

class ExecutionError(LegionError):
    """Raised when there is an error during graph execution."""

    pass

class RetryableError(ExecutionError):
    """Base class for errors that can be retried."""

    def __init__(self, message: str, retry_count: int = 0, max_retries: int = 0):
        super().__init__(message)
        self.retry_count = retry_count
        self.max_retries = max_retries

    @property
    def can_retry(self) -> bool:
        """Check if error can be retried"""
        return self.retry_count < self.max_retries

class NonRetryableError(ExecutionError):
    """Base class for errors that cannot be retried."""

    pass

class RoutingError(LegionError):
    """Raised when there is an error in graph routing."""

    pass

class MetadataError(LegionError):
    """Raised when there is an error in metadata handling."""

    pass

class ValidationError(LegionError):
    """Raised when there is a validation error."""

    pass

class ConfigurationError(LegionError):
    """Raised when there is a configuration error."""

    pass

class GraphError(LegionError):
    """Raised when there is an error in graph structure or operation."""

    pass

class ResourceError(RetryableError):
    """Raised when resource limits are exceeded."""

    pass

class StateError(RetryableError):
    """Raised when there is an error in state management."""

    pass

class NodeError(RetryableError):
    """Raised when there is an error in node execution."""

    def __init__(self, message: str, node_id: str, retry_count: int = 0, max_retries: int = 0):
        super().__init__(message, retry_count, max_retries)
        self.node_id = node_id

class FatalError(NonRetryableError):
    """Raised when there is an unrecoverable error."""

    pass

class TimeoutError(RetryableError):
    """Raised when an operation times out."""

    pass
