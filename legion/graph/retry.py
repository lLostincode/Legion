"""Retry policy system for graph execution."""
import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar

from pydantic import BaseModel, Field

from ..exceptions import FatalError, NonRetryableError, RetryableError

T = TypeVar("T")

class RetryStrategy(str, Enum):
    """Retry strategy types"""

    IMMEDIATE = "immediate"  # Retry immediately
    LINEAR = "linear"        # Linear backoff
    EXPONENTIAL = "exponential"  # Exponential backoff

class RetryPolicy(BaseModel):
    """Configuration for retry behavior"""

    max_retries: int = Field(default=3, ge=0)
    strategy: RetryStrategy = Field(default=RetryStrategy.EXPONENTIAL)
    base_delay: float = Field(default=1.0, ge=0)  # Base delay in seconds
    max_delay: float = Field(default=60.0, ge=0)  # Max delay in seconds
    jitter: bool = Field(default=True)  # Add randomness to delay

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        if self.strategy == RetryStrategy.IMMEDIATE:
            delay = 0
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.base_delay * attempt
        else:  # EXPONENTIAL
            delay = self.base_delay * (2 ** (attempt - 1))

        # Apply max delay cap
        delay = min(delay, self.max_delay)

        # Add jitter if enabled (±10%)
        if self.jitter:
            import random
            jitter_range = delay * 0.1
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)

class RetryState(BaseModel):
    """State tracking for retries"""

    attempt: int = Field(default=0)
    last_error: Optional[str] = None
    last_attempt: Optional[datetime] = None
    next_attempt: Optional[datetime] = None

class RetryHandler:
    """Handles retry logic for operations"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self._logger = logger or logging.getLogger(__name__)
        self._states: Dict[str, RetryState] = {}

    def _get_state(self, operation_id: str) -> RetryState:
        """Get retry state for operation"""
        if operation_id not in self._states:
            self._states[operation_id] = RetryState()
        return self._states[operation_id]

    def clear_state(self, operation_id: str) -> None:
        """Clear retry state for operation"""
        if operation_id in self._states:
            del self._states[operation_id]

    async def execute_with_retry(
        self,
        operation_id: str,
        func: Callable[..., Awaitable[T]],
        policy: RetryPolicy,
        *args: Any,
        **kwargs: Any
    ) -> T:
        """Execute operation with retry logic

        Args:
        ----
            operation_id: Unique identifier for operation
            func: Async function to execute
            policy: Retry policy to use
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
        -------
            Result from successful execution

        Raises:
        ------
            NonRetryableError: If error cannot be retried
            FatalError: If max retries exceeded

        """
        state = self._get_state(operation_id)

        while True:
            try:
                # Attempt execution
                state.attempt += 1
                state.last_attempt = datetime.now()

                result = await func(*args, **kwargs)

                # Success - clear state and return
                self.clear_state(operation_id)
                return result

            except NonRetryableError:
                # Non-retryable - clear state and re-raise
                self.clear_state(operation_id)
                raise

            except RetryableError as e:
                # Update state and retry count
                state.last_error = str(e)

                # Update retry count in error if available
                if hasattr(e, "retry_count"):
                    e.retry_count = state.attempt - 1

                # Check if we can retry
                if state.attempt > policy.max_retries:
                    self.clear_state(operation_id)
                    raise FatalError(
                        f"Max retries ({policy.max_retries}) exceeded for {operation_id}: {e}"
                    )

                # Calculate delay
                delay = policy.calculate_delay(state.attempt)
                state.next_attempt = datetime.now() + timedelta(seconds=delay)

                # Log retry attempt
                self._logger.warning(
                    f"Retry {state.attempt}/{policy.max_retries} for {operation_id} "
                    f"in {delay:.2f}s: {e}"
                )

                # Wait before retry
                await asyncio.sleep(delay)

            except Exception as e:
                # Unexpected error - treat as non-retryable
                self.clear_state(operation_id)
                raise NonRetryableError(f"Unexpected error in {operation_id}: {e}") from e
