import inspect
from typing import List, Optional, Type

from pydantic import BaseModel

from .base import BlockMetadata, FunctionalBlock


def block(
    name: Optional[str] = None,
    description: Optional[str] = None,
    input_schema: Optional[Type[BaseModel]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    version: str = "1.0",
    tags: Optional[List[str]] = None,
    validate: bool = True
):
    """Decorator to create a functional block

    Example:
    -------
    @block(
        input_schema=InputModel,
        output_schema=OutputModel,
        tags=['preprocessing']
    )
    async def process_data(data: InputModel) -> OutputModel:
        ...

    """
    def decorator(func):
        # Extract function signature
        inspect.signature(func)

        # Get description from docstring if not provided
        block_description = description
        if not block_description and func.__doc__:
            block_description = func.__doc__.split("\n")[0].strip()
        block_description = block_description or f"Block: {func.__name__}"

        # Only create schemas if explicitly requested or if schemas are provided
        block_input_schema = input_schema
        block_output_schema = output_schema

        # Create metadata
        metadata = BlockMetadata(
            name=name or func.__name__,
            description=block_description,
            input_schema=block_input_schema,
            output_schema=block_output_schema,
            version=version,
            tags=tags or []
        )

        # Create and return block
        return FunctionalBlock(
            func=func,
            metadata=metadata,
            validate=validate
        )

    return decorator
