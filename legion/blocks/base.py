import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Type, get_args, get_origin

from pydantic import BaseModel

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class BlockMetadata:
    """Metadata for functional blocks"""

    name: str
    description: str
    input_schema: Optional[Type[BaseModel]] = None
    output_schema: Optional[Type[BaseModel]] = None
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)

class BlockError(Exception):
    """Base exception for block-related errors"""

    pass

class ValidationError(BlockError):
    """Raised when block input/output validation fails"""

    def __init__(self, message: str, data: Any, schema: Optional[str] = None):
        self.data = data
        self.schema = schema
        super().__init__(message)

class FunctionalBlock:
    """A discrete processing unit that can be used in chains"""

    def __init__(
        self,
        func: Callable,
        metadata: BlockMetadata,
        validate: bool = True
    ):
        self.func = func
        self.metadata = metadata
        self.validate = validate
        self.is_async = inspect.iscoroutinefunction(func)

        # Store original function signature
        self.signature = inspect.signature(func)

        logger.debug(f"Initialized block {metadata.name} (async={self.is_async})")

    def _validate_input(self, data: Any) -> Any:
        """Validate input data against schema if present"""
        if not self.validate or not self.metadata.input_schema:
            return data

        try:
            # Handle generic types
            if get_origin(self.metadata.input_schema):
                origin = get_origin(self.metadata.input_schema)
                args = get_args(self.metadata.input_schema)
                if not isinstance(data, origin):
                    raise ValidationError(
                        f"Input must be of type {origin.__name__}",
                        data=data,
                        schema=str(self.metadata.input_schema)
                    )
                # Validate generic type arguments if possible
                if args and hasattr(data, "__iter__"):
                    for item in data:
                        for arg in args:
                            if not isinstance(item, arg):
                                raise ValidationError(
                                    f"Invalid item type {type(item).__name__}, expected {arg.__name__}",
                                    data=data,
                                    schema=str(self.metadata.input_schema)
                                )
                return data

            # Handle Pydantic models
            if isinstance(data, self.metadata.input_schema):
                return data

            if data is None:
                raise ValidationError(
                    "Input cannot be None",
                    data=data,
                    schema=self.metadata.input_schema.__name__
                )

            try:
                # Create model from raw data
                if isinstance(data, dict):
                    return self.metadata.input_schema.model_validate(data)
                return self.metadata.input_schema(**data if isinstance(data, dict) else {"value": data})
            except Exception as e:
                raise ValidationError(
                    f"Input validation failed: {str(e)}",
                    data=data,
                    schema=self.metadata.input_schema.__name__
                )

        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                f"Input validation failed: {str(e)}",
                data=data,
                schema=str(self.metadata.input_schema)
            )

    def _validate_output(self, data: Any) -> Any:
        """Validate output data against schema if present"""
        if not self.validate or not self.metadata.output_schema:
            return data

        try:
            if isinstance(data, self.metadata.output_schema):
                return data
            return self.metadata.output_schema.model_validate(data)
        except Exception as e:
            raise ValidationError(
                f"Output validation failed: {str(e)}",
                data=data,
                schema=self.metadata.output_schema.__name__
            )

    async def __call__(self, input_data: Any) -> Any:
        """Execute the block with validation"""
        logger.debug(f"Executing block {self.metadata.name}")

        try:
            # Validate input
            validated_input = self._validate_input(input_data)

            try:
                # Execute function based on type
                if self.is_async:
                    result = await self.func(validated_input)
                else:
                    # Run sync function in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda: self.func(validated_input)
                    )

                # Validate output
                validated_output = self._validate_output(result)
                logger.debug(f"Block {self.metadata.name} completed successfully")
                return validated_output

            except ValidationError:
                raise
            except Exception as e:
                logger.error(f"Block {self.metadata.name} failed: {str(e)}")
                raise BlockError(f"Block execution failed: {str(e)}") from e

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Block {self.metadata.name} failed: {str(e)}")
            raise BlockError(f"Block execution failed: {str(e)}") from e

    def __get__(self, obj, objtype=None):
        """Support descriptor protocol for instance binding"""
        if obj is None:
            return self

        # Create bound copy
        bound_block = FunctionalBlock(
            func=self.func.__get__(obj, objtype),
            metadata=self.metadata,
            validate=self.validate
        )

        return bound_block
