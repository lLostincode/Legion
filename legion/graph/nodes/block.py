from datetime import datetime
from typing import Any, Dict, List, Optional, Type, TypeVar, get_args, get_origin

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from ...blocks.base import FunctionalBlock
from ...blocks.base import ValidationError as BlockValidationError
from ..channels import LastValue
from ..state import GraphState
from .base import NodeBase

T = TypeVar("T")

class BlockValidationContext:
    """Context for block validation"""

    def __init__(self, node: "BlockNode"):
        self.node = node
        self.started_at = datetime.now()
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def add_error(self, error: str) -> None:
        """Add validation error"""
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        """Add validation warning"""
        self.warnings.append(warning)

    @property
    def is_valid(self) -> bool:
        """Check if validation passed"""
        return len(self.errors) == 0

class BlockNode(NodeBase):
    """Node wrapper for Legion functional blocks"""

    def __init__(
        self,
        graph_state: GraphState,
        block: FunctionalBlock,
        input_channel_type: Optional[Type[Any]] = None,
        output_channel_type: Optional[Type[Any]] = None,
        validate_types: bool = True
    ):
        """Initialize block node

        Args:
        ----
            graph_state: Graph state manager
            block: Legion functional block instance
            input_channel_type: Optional type hint for input channel
            output_channel_type: Optional type hint for output channel
            validate_types: Whether to validate types at runtime

        """
        super().__init__(graph_state)
        self._block = block
        self._validate_types = validate_types
        self._validation_context = BlockValidationContext(self)

        # Get type hints from block if not provided
        input_type = input_channel_type
        if not input_type and block.metadata.input_schema:
            input_type = block.metadata.input_schema

        output_type = output_channel_type
        if not output_type and block.metadata.output_schema:
            output_type = block.metadata.output_schema

        # Create standard channels
        self.create_input_channel(
            "input",
            LastValue,
            type_hint=input_type or Any
        )
        self.create_output_channel(
            "output",
            LastValue,
            type_hint=output_type or Any
        )

    @property
    def block(self) -> FunctionalBlock:
        """Get wrapped block"""
        return self._block

    def _validate_value_type(self, value: Any, expected_type: Type[T]) -> Optional[T]:
        """Validate and potentially coerce value to expected type"""
        if not self._validate_types or expected_type is Any:
            return value

        if value is None:
            return None

        # Handle Pydantic models first
        if isinstance(expected_type, type) and issubclass(expected_type, BaseModel):
            try:
                if isinstance(value, expected_type):
                    return value
                if isinstance(value, dict):
                    return expected_type.model_validate(value)
                raise TypeError(f"Cannot convert {type(value).__name__} to {expected_type.__name__}")
            except PydanticValidationError as e:
                raise TypeError(f"Validation failed: {str(e)}")

        # Handle generic types
        origin = get_origin(expected_type)
        if origin:
            args = get_args(expected_type)
            if not isinstance(value, origin):
                try:
                    # Try to convert basic types
                    return origin(value)
                except (TypeError, ValueError):
                    raise TypeError(f"Expected {origin.__name__}, got {type(value).__name__}")
            # Validate generic type arguments if possible
            if args and hasattr(value, "__iter__"):
                for item in value:
                    for arg in args:
                        if not isinstance(item, arg):
                            raise TypeError(f"Invalid item type {type(item).__name__}, expected {arg.__name__}")
            return value

        # Handle basic types
        if not isinstance(value, expected_type):
            try:
                return expected_type(value)
            except (TypeError, ValueError):
                raise TypeError(f"Expected {expected_type.__name__}, got {type(value).__name__}")

        return value

    async def validate(self) -> BlockValidationContext:
        """Validate node configuration"""
        context = BlockValidationContext(self)

        # Check channel types
        input_channel = self.get_input_channel("input")
        if input_channel and self._block.metadata.input_schema:
            if input_channel._type_hint != self._block.metadata.input_schema:
                context.add_warning(
                    f"Input channel type ({input_channel._type_hint}) doesn't match "
                    f"block input schema ({self._block.metadata.input_schema})"
                )

        output_channel = self.get_output_channel("output")
        if output_channel and self._block.metadata.output_schema:
            if output_channel._type_hint != self._block.metadata.output_schema:
                context.add_warning(
                    f"Output channel type ({output_channel._type_hint}) doesn't match "
                    f"block output schema ({self._block.metadata.output_schema})"
                )

        return context

    async def _execute(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Execute block with input from channels"""
        # Get input from channel
        input_channel = self.get_input_channel("input")
        if not input_channel:
            raise ValueError("Input channel not found")

        input_value = input_channel.get()
        if input_value is None:
            return None  # No input to process

        try:
            # Process with block
            output = await self._block(input_value)

            # Store response in output channel
            output_channel = self.get_output_channel("output")
            if output_channel:
                output_channel.set(output)

            return {
                "output": output
            }

        except (TypeError, BlockValidationError) as e:
            self._validation_context.add_error(str(e))
            raise

    def checkpoint(self) -> Dict[str, Any]:
        """Create checkpoint of node state"""
        checkpoint = super().checkpoint()
        # Add block-specific state
        checkpoint.update({
            "block_metadata": self._block.metadata.__dict__,
            "block_validate": self._block.validate,
            "validate_types": self._validate_types
        })
        return checkpoint

    def restore(self, checkpoint: Dict[str, Any]) -> None:
        """Restore node state from checkpoint"""
        super().restore(checkpoint)
        # Restore block metadata
        if "block_metadata" in checkpoint:
            self._block.metadata.__dict__.update(checkpoint["block_metadata"])
        if "block_validate" in checkpoint:
            self._block.validate = checkpoint["block_validate"]
        if "validate_types" in checkpoint:
            self._validate_types = checkpoint["validate_types"]
