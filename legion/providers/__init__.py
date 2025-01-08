"""Provider registry and factory functions"""

from typing import Dict, List, Optional, Type

from ..interface.base import LLMInterface
from ..interface.schemas import ProviderConfig
from .anthropic import AnthropicFactory
from .factory import ProviderFactory
from .gemini import GeminiFactory
from .groq import GroqFactory
from .ollama import OllamaFactory

# Import provider implementations
from .openai import OpenAIFactory

# Registry of provider factories
PROVIDER_FACTORIES: Dict[str, Type[ProviderFactory]] = {
    "openai": OpenAIFactory,
    "anthropic": AnthropicFactory,
    "groq": GroqFactory,
    "ollama": OllamaFactory,
    "gemini": GeminiFactory
}

def get_provider(
    name: str,
    config: Optional[ProviderConfig] = None,
    **kwargs
) -> LLMInterface:
    """Get an instance of a provider by name using the factory pattern

    Args:
    ----
        name: Name of the provider
        config: Provider configuration
        **kwargs: Additional provider-specific arguments

    Returns:
    -------
        Configured provider instance

    Raises:
    ------
        ValueError: If provider name is not recognized

    """
    if name not in PROVIDER_FACTORIES:
        raise ValueError(
            f"Unknown provider '{name}'. Available providers: {list(PROVIDER_FACTORIES.keys())}"
        )

    factory = PROVIDER_FACTORIES[name]()
    return factory.create_provider(config=config, **kwargs)

def available_providers() -> List[str]:
    """Get list of available provider names"""
    return list(PROVIDER_FACTORIES.keys())
