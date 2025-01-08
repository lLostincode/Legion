from unittest.mock import AsyncMock, MagicMock

from legion.interface.schemas import ModelResponse, Role, TokenUsage


class MockOpenAIProvider:
    """Mock OpenAI provider for testing"""

    def __init__(self, *args, **kwargs):
        self.agenerate = AsyncMock(return_value=ModelResponse(
            content="Mock response",
            role=Role.ASSISTANT,
            raw_response={"content": "Mock response"},
            usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20)
        ))
        self.generate = MagicMock(return_value=ModelResponse(
            content="Mock response",
            role=Role.ASSISTANT,
            raw_response={"content": "Mock response"},
            usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20)
        ))
