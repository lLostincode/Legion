import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set, Type

from pydantic import BaseModel

from ..errors import ToolError

# Set up logging
logger = logging.getLogger(__name__)

class BaseTool(ABC):
    """Base class for all tools"""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Type[BaseModel],
        injected_params: Optional[Set[str]] = None,
        injected_values: Optional[Dict[str, Any]] = None
    ):
        """Initialize tool"""
        self.name = name
        self.description = description
        self.parameters = parameters
        self.injected_params = injected_params or set()
        self._is_async = hasattr(self, "arun")
        self._injected_values = dict(injected_values or {})  # Make a copy
        logger.debug(f"BaseTool initialized with injected_values: {self._injected_values}")

    def get_schema(self) -> Dict[str, Any]:
        """Get OpenAI-compatible function schema"""
        # TODO: Ensure this is compatible with all providers
        schema = self.parameters.model_json_schema()

        # Filter out injected parameters from schema
        filtered_properties = {
            k: v for k, v in schema.get("properties", {}).items()
            if k not in self.injected_params
        }

        # Remove injected params from required list
        required = [
            param for param in schema.get("required", [])
            if param not in self.injected_params
        ]

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": filtered_properties,
                    "required": required,
                }
            }
        }

    def inject(self, **kwargs) -> "BaseTool":
        """Inject static parameter values"""
        # Only allow injecting declared injectable parameters
        for key in kwargs:
            if key not in self.injected_params:
                raise ValueError(f"Parameter '{key}' is not injectable")
        self._injected_values.update(kwargs)
        return self

    async def __call__(self, **kwargs) -> Any:
        """Make tool callable, validating parameters"""
        try:
            # Start with injected values
            all_kwargs = dict(self._injected_values)
            logger.debug(f"Starting with injected values: {all_kwargs}")

            # Update with provided kwargs, but don't override injected values
            for key, value in kwargs.items():
                if key not in self.injected_params or key not in all_kwargs:
                    all_kwargs[key] = value
            logger.debug(f"After merging with provided kwargs: {all_kwargs}")

            # Validate all parameters together
            try:
                validated = self.parameters(**all_kwargs)
                validated_dict = validated.model_dump()
                logger.debug("Validation successful")
            except Exception as e:
                logger.error(f"Validation failed: {str(e)}")
                raise ToolError(f"Invalid parameters for tool {self.name}: {str(e)}")

            if self._is_async:
                return await self.arun(**validated_dict)
            else:
                return await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.run(**validated_dict)
                )
        except Exception as e:
            if isinstance(e, (ValueError, ToolError)):
                raise
            raise ToolError(f"Error executing tool {self.name}: {str(e)}")

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Execute the tool with validated parameters (sync)"""
        pass

    async def arun(self, **kwargs) -> Any:
        """Execute the tool with validated parameters (async)"""
        # Default async implementation calls sync version
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.run(**kwargs)
        )

    def model_dump(self) -> Dict[str, Any]:
        """Serialize tool for provider APIs

        Returns format expected by OpenAI/compatible APIs:
        {
            "type": "function",
            "function": {
                "name": str,
                "description": str,
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }
        """
        return self.get_schema()
