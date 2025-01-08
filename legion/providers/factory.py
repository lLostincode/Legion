"""Provider factory implementation"""

from abc import ABC, abstractmethod
from typing import Optional

from ..interface.base import LLMInterface
from ..interface.schemas import ProviderConfig


class ProviderFactory(ABC):
    """Abstract factory for creating LLM providers"""

    @abstractmethod
    def create_provider(self, config: Optional[ProviderConfig] = None, **kwargs) -> LLMInterface:
        """Create a new provider instance"""
        pass
