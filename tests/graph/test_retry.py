"""Tests for graph retry system."""
import pytest
import asyncio
from datetime import datetime, timedelta

from legion.exceptions import (
    RetryableError, NonRetryableError, FatalError,
    NodeError, StateError, ResourceError
)
from legion.graph.retry import (
    RetryStrategy, RetryPolicy, RetryState, RetryHandler
)

# Test fixtures
@pytest.fixture
def retry_handler():
    """Create retry handler for testing"""
    return RetryHandler()

@pytest.fixture
def retry_policy():
    """Create retry policy for testing"""
    return RetryPolicy(
        max_retries=3,
        strategy=RetryStrategy.IMMEDIATE,  # Use immediate for faster tests
        base_delay=0.1,
        max_delay=1.0,
        jitter=False  # Disable jitter for deterministic tests
    )

# Test retry policy
def test_retry_policy_validation():
    """Test retry policy validation"""
    # Valid policies
    RetryPolicy(max_retries=3, base_delay=1.0, max_delay=5.0)
    RetryPolicy(max_retries=0, base_delay=0.0, max_delay=0.0)
    
    # Invalid policies
    with pytest.raises(ValueError):
        RetryPolicy(max_retries=-1)
    with pytest.raises(ValueError):
        RetryPolicy(base_delay=-1.0)
    with pytest.raises(ValueError):
        RetryPolicy(max_delay=-1.0)

def test_retry_delay_calculation():
    """Test retry delay calculation"""
    policy = RetryPolicy(
        max_retries=3,
        strategy=RetryStrategy.EXPONENTIAL,
        base_delay=1.0,
        max_delay=10.0,
        jitter=False
    )
    
    # Test exponential backoff
    assert policy.calculate_delay(1) == 1.0  # 1 * 2^0
    assert policy.calculate_delay(2) == 2.0  # 1 * 2^1
    assert policy.calculate_delay(3) == 4.0  # 1 * 2^2
    assert policy.calculate_delay(4) == 8.0  # 1 * 2^3
    assert policy.calculate_delay(5) == 10.0  # Capped at max_delay
    
    # Test linear backoff
    policy.strategy = RetryStrategy.LINEAR
    assert policy.calculate_delay(1) == 1.0
    assert policy.calculate_delay(2) == 2.0
    assert policy.calculate_delay(3) == 3.0
    assert policy.calculate_delay(10) == 10.0  # Capped at max_delay
    
    # Test immediate
    policy.strategy = RetryStrategy.IMMEDIATE
    assert policy.calculate_delay(1) == 0.0
    assert policy.calculate_delay(2) == 0.0

def test_retry_delay_jitter():
    """Test retry delay jitter"""
    policy = RetryPolicy(
        strategy=RetryStrategy.LINEAR,
        base_delay=1.0,
        max_delay=10.0,
        jitter=True
    )
    
    # Get multiple delays and verify they're within jitter range
    delays = [policy.calculate_delay(2) for _ in range(10)]
    expected = 2.0  # Linear delay for attempt 2
    
    for delay in delays:
        assert expected * 0.9 <= delay <= expected * 1.1
    
    # Verify we got some variation
    assert len(set(delays)) > 1

# Test retry handler
@pytest.mark.asyncio
async def test_successful_execution(retry_handler, retry_policy):
    """Test successful execution without retries"""
    async def success():
        return "success"
    
    result = await retry_handler.execute_with_retry(
        "test_success",
        success,
        retry_policy
    )
    assert result == "success"

@pytest.mark.asyncio
async def test_retryable_error(retry_handler, retry_policy):
    """Test handling of retryable errors"""
    attempts = 0
    
    async def fail_twice():
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise StateError("Temporary failure", retry_count=attempts-1, max_retries=3)
        return "success"
    
    result = await retry_handler.execute_with_retry(
        "test_retry",
        fail_twice,
        retry_policy
    )
    assert result == "success"
    assert attempts == 3

@pytest.mark.asyncio
async def test_non_retryable_error(retry_handler, retry_policy):
    """Test handling of non-retryable errors"""
    async def non_retryable():
        raise NonRetryableError("Fatal error")
    
    with pytest.raises(NonRetryableError):
        await retry_handler.execute_with_retry(
            "test_non_retryable",
            non_retryable,
            retry_policy
        )

@pytest.mark.asyncio
async def test_max_retries_exceeded(retry_handler, retry_policy):
    """Test max retries exceeded"""
    async def always_fail():
        raise NodeError("Node failure", "test_node", retry_count=0, max_retries=3)
    
    with pytest.raises(FatalError) as exc_info:
        await retry_handler.execute_with_retry(
            "test_max_retries",
            always_fail,
            retry_policy
        )
    assert "Max retries (3) exceeded" in str(exc_info.value)

@pytest.mark.asyncio
async def test_retry_state_tracking(retry_handler, retry_policy):
    """Test retry state tracking"""
    operation_id = "test_state"
    attempts = 0
    
    async def fail_once():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ResourceError("Resource limit", retry_count=0, max_retries=3)
        return "success"
    
    # Execute with one failure
    await retry_handler.execute_with_retry(
        operation_id,
        fail_once,
        retry_policy
    )
    
    # Verify state was cleared after success
    assert operation_id not in retry_handler._states

@pytest.mark.asyncio
async def test_unexpected_error(retry_handler, retry_policy):
    """Test handling of unexpected errors"""
    async def unexpected():
        raise KeyError("Unexpected error")
    
    with pytest.raises(NonRetryableError) as exc_info:
        await retry_handler.execute_with_retry(
            "test_unexpected",
            unexpected,
            retry_policy
        )
    assert "Unexpected error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_multiple_operations(retry_handler, retry_policy):
    """Test handling multiple operations"""
    op1_attempts = 0
    op2_attempts = 0
    
    async def op1():
        nonlocal op1_attempts
        op1_attempts += 1
        if op1_attempts == 1:
            raise StateError("Op1 failure", retry_count=0, max_retries=3)
        return "op1_success"
    
    async def op2():
        nonlocal op2_attempts
        op2_attempts += 1
        if op2_attempts <= 2:
            raise NodeError("Op2 failure", "test_node", retry_count=op2_attempts-1, max_retries=3)
        return "op2_success"
    
    # Execute both operations concurrently
    results = await asyncio.gather(
        retry_handler.execute_with_retry("op1", op1, retry_policy),
        retry_handler.execute_with_retry("op2", op2, retry_policy)
    )
    
    assert results == ["op1_success", "op2_success"]
    assert op1_attempts == 2
    assert op2_attempts == 3 