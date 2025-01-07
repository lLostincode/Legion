"""
Legion: A provider-agnostic framework for building AI agent systems
"""

__version__ = "0.1.0"

# Core interfaces
from .interface.base import LLMInterface
from .interface.schemas import (
    Message,
    ModelResponse,
    TokenUsage,
    Role,
    ChatParameters,
    ProviderConfig,
    SystemPrompt,
    SystemPromptSection
)
from .interface.tools import BaseTool

# Core agent system
from .agents.base import Agent

# Provider management
from .providers import (
    get_provider,
    available_providers,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    GroqProvider,
    GeminiProvider
)

# Error types
from .errors.exceptions import (
    LegionError,
    AgentError,
    ProviderError,
    ToolError,
    JSONFormatError,
    InvalidSchemaError
)