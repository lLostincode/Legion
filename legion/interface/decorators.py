import asyncio
import inspect
import logging
from typing import Annotated, Any, Dict, List, Optional, Type, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.fields import FieldInfo

from legion.interface.schemas import SystemPrompt, SystemPromptSection
from legion.interface.tools import BaseTool

# Set up logging
logger = logging.getLogger(__name__)

def param(type_hint: Any, description: str):
    """Helper function to create annotated parameter types with descriptions

    Example:
    -------
    def my_tool(
        numbers: param(List[float], "List of numbers to analyze"),
        threshold: param(float, "Cutoff threshold for outlier detection") = 2.0
    ):
        ...

    """
    return Annotated[type_hint, Field(description=description)]

# Export FunctionTool for testing
__all__ = ["tool", "schema", "param", "FunctionTool"]

# Move FunctionTool class definition outside of tool decorator
class FunctionTool(BaseTool):
    _registry = {}  # Class-level registry of injected values per function

    @classmethod
    def clear_registry(cls):
        """Clear the registry - used for testing"""
        cls._registry.clear()

    def __init__(self, func, name=None, description=None, param_model=None, inject=None, defaults=None):
        logger.debug(f"Initializing FunctionTool for {func.__name__}")
        # Store function reference
        self.func = func
        self.instance = None
        self.is_instance_method = False  # Track if this is an instance method
        self._instance_values = {}  # Store instance-specific injected values
        self._defaults = defaults or {}  # Store default values

        # Initialize base class
        super().__init__(
            name=name or func.__name__,
            description=description,
            parameters=param_model,
            injected_params=set(inject or []),
            injected_values={}
        )

        # Initialize registry entry for this function if needed
        if func not in self._registry:
            self._registry[func] = {}

        logger.debug(f"FunctionTool initialized with injected_params: {self.injected_params}")
        logger.debug(f"FunctionTool defaults: {self._defaults}")

    def get_injected_values(self, message_injections=None):
        """Get injected values, combining defaults with message-specific injections"""
        # Start with defaults
        values = dict(self._defaults)

        # Update with message-specific injections if any
        if message_injections:
            for injection in message_injections:
                # Check if this injection is for this tool instance
                if (isinstance(injection["tool"], FunctionTool) and
                    injection["tool"].func == self.func):
                    # Update values with injected parameters
                    values.update(injection["parameters"])
                    break  # Only use first matching injection

        logger.debug(f"[TOOL VALUES] Final values for {self.name}:")
        logger.debug(f"  - Defaults: {self._defaults}")
        logger.debug(f"  - Message injections: {message_injections}")
        logger.debug(f"  - Combined values: {values}")  # This should now show merged values

        return values

    def run(self, **kwargs):
        """Execute the tool synchronously"""
        logger.debug(f"[TOOL RUN] Running {self.name} synchronously")
        logger.debug(f"[TOOL RUN] Input kwargs: {kwargs}")

        # Get injected values from context
        injected = self.get_injected_values(kwargs.pop("__injected_parameters__", None))
        logger.debug(f"[TOOL RUN] Injected values: {injected}")

        # Merge all parameters
        all_kwargs = dict(injected)
        all_kwargs.update(kwargs)
        logger.debug(f"[TOOL RUN] Merged kwargs: {all_kwargs}")

        if self.instance is not None and self.is_instance_method:
            logger.debug(f"[TOOL RUN] Calling as instance method with instance {self.instance}")
            return self.func(self.instance, **all_kwargs)
        else:
            logger.debug("[TOOL RUN] Calling as standalone function")
            return self.func(**all_kwargs)

    async def arun(self, **kwargs):
        """Execute the tool asynchronously"""
        logger.debug(f"[TOOL ARUN] Running {self.name} asynchronously")
        logger.debug(f"[TOOL ARUN] Input kwargs: {kwargs}")

        # Get injected values from context
        injected = self.get_injected_values(kwargs.pop("__injected_parameters__", None))
        logger.debug(f"[TOOL ARUN] Injected values: {injected}")

        # Merge all parameters
        all_kwargs = dict(injected)
        all_kwargs.update(kwargs)
        logger.debug(f"[TOOL ARUN] Merged kwargs: {all_kwargs}")

        if inspect.iscoroutinefunction(self.func):
            if self.instance is not None and self.is_instance_method:
                logger.debug("[TOOL ARUN] Calling async instance method")
                return await self.func(self.instance, **all_kwargs)
            logger.debug("[TOOL ARUN] Calling async standalone function")
            return await self.func(**all_kwargs)

        # Run sync function in thread pool
        logger.debug("[TOOL ARUN] Running sync function in thread pool")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.run(**all_kwargs)  # Pass the merged kwargs
        )

    def __get__(self, obj, objtype=None):
        """Support descriptor protocol for instance binding"""
        if obj is None:
            return self

        # Get or create bound instance
        bound_tool = self.bind_to(obj)

        # Only copy class-level injected values for non-test instances
        if not obj.__class__.__name__.endswith("Tester"):
            class_tool = getattr(self.func, "__tool__", None)
            if class_tool:
                bound_tool._injected_values.update(class_tool._injected_values)
                logger.debug(f"Updated bound tool with class-level values: {bound_tool._injected_values}")

        return bound_tool

    async def __call__(self, **kwargs) -> Any:
        """Override call to add logging"""
        logger.debug(f"Calling tool {self.name}")
        logger.debug(f"Current injected values: {self._injected_values}")
        logger.debug(f"Provided kwargs: {kwargs}")

        try:
            # Get injected values including defaults and message-specific injections
            injected = self.get_injected_values(kwargs.pop("__injected_parameters__", None))
            logger.debug(f"Using injected values: {injected}")

            # Create a new dict with injected values and update with provided kwargs
            all_kwargs = dict(injected)
            all_kwargs.update(kwargs)
            logger.debug(f"Final merged kwargs: {all_kwargs}")

            # Pass the merged kwargs to super().__call__
            return await super().__call__(**all_kwargs)
        except Exception as e:
            logger.error(f"Error calling tool {self.name}: {str(e)}")
            raise

    def bind_to(self, instance):
        """Bind the tool to an instance"""
        logger.debug(f"[TOOL BIND] Binding {self.name} to instance {instance}")

        # Create new instance
        bound_tool = FunctionTool.__new__(FunctionTool)

        # Copy all attributes
        for attr, value in self.__dict__.items():
            setattr(bound_tool, attr, value)

        # Set instance and determine if this is an instance method
        bound_tool.instance = instance
        bound_tool.is_instance_method = self.func.__name__ in instance.__class__.__dict__

        logger.debug(f"[TOOL BIND] Is instance method: {bound_tool.is_instance_method}")
        logger.debug(f"[TOOL BIND] Defaults: {bound_tool._defaults}")
        logger.debug(f"[TOOL BIND] Injectable params: {bound_tool.injected_params}")

        return bound_tool

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
                    "required": required
                }
            }
        }

def tool(
    func=None,  # Allow positional function argument for @tool syntax
    *,         # Force remaining arguments to be keyword-only
    inject: Optional[List[str]] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    defaults: Optional[Dict[str, Any]] = None
):
    """Decorator to create a tool from a function

    Can be used as @tool or @tool(inject=['param'], name='custom_name')

    Args:
    ----
        func: The function to decorate (automatically passed when using @tool)
        inject: List of parameter names to inject (not exposed to LLM)
        name: Override tool name
        description: Override tool description
        defaults: Default values for injectable parameters

    """
    # Define the actual decorator function
    def decorator(func):
        sig = inspect.signature(func)

        # Extract description from docstring or override
        doc_lines = []
        if func.__doc__:
            for line in func.__doc__.split("\n"):
                line = line.strip()
                if line:
                    doc_lines.append(line)
                    if len(doc_lines) == 1:
                        break
        tool_description = description or doc_lines[0] if doc_lines else f"Tool for {func.__name__}"

        # Create parameter model dynamically
        fields = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            # Get type annotation
            annotation = param.annotation if param.annotation != inspect._empty else Any

            # Get or create field info
            if isinstance(param.default, FieldInfo):
                field_info = param.default
            else:
                default = ... if param.default == inspect._empty else param.default
                field_info = Field(
                    default=default,
                    description=f"{param_name} parameter"
                )

            fields[param_name] = (annotation, field_info)

        # Create parameter model with modern Pydantic V2 config
        param_model = create_model(
            f"{func.__name__.title()}Parameters",
            __base__=BaseModel,
            model_config=ConfigDict(extra="allow"),
            **fields
        )

        tool_instance = FunctionTool(
            func=func,
            name=name,
            description=tool_description,
            param_model=param_model,
            inject=inject,
            defaults=defaults
        )
        return tool_instance

    # Handle both @tool and @tool() syntax
    if func is not None:  # @tool syntax
        return decorator(func)
    # @tool() syntax
    return decorator

def schema(name: Optional[str] = None, description: Optional[str] = None):
    """Decorator for creating input/parameter schemas

    Example:
    -------
    @schema
    class WeatherReport:
        temperature: float = Field(description="Current temperature in Celsius")
        conditions: str = Field(description="Weather conditions")
        forecast: List[str] = Field(description="3-day forecast")

    """
    def decorator(cls):
        # Get class annotations and create fields
        fields = {}
        for field_name, field_type in get_type_hints(cls).items():
            field_obj = getattr(cls, field_name, None)
            if isinstance(field_obj, FieldInfo):
                fields[field_name] = (field_type, field_obj)
            else:
                fields[field_name] = (
                    field_type,
                    Field(..., description=f"{field_name} field")
                )

        # Create model
        model = create_model(
            name or cls.__name__,
            __doc__=description or cls.__doc__ or f"Schema for {name or cls.__name__}",
            **fields
        )

        return model

    # Allow using as @schema without parentheses
    if isinstance(name, type):
        cls = name
        name = None
        return decorator(cls)

    return decorator

def output_schema(cls=None, *, name: Optional[str] = None, description: Optional[str] = None):
    """Decorator to create an output schema model

    Example:
    -------
    @output_schema
    class Analysis:
        summary: str = Field(description="Analysis summary")
        metrics: Dict[str, float] = Field(description="Key metrics")
        recommendations: List[str] = Field(description="Action items")

    """
    def decorator(cls):
        # Get class annotations
        fields = {}
        for field_name, field_type in get_type_hints(cls).items():
            field_obj = getattr(cls, field_name, None)
            if isinstance(field_obj, FieldInfo):
                fields[field_name] = (field_type, field_obj)
            else:
                fields[field_name] = (
                    field_type,
                    Field(..., description=f"{field_name} field")
                )

        # Add standard output fields
        output_fields = {
            "error": (Optional[str], Field(None, description="Error message if any")),
            "metadata": (Optional[Dict[str, Any]], Field(None, description="Additional metadata")),
            "confidence_score": (Optional[float], Field(None, ge=0, le=1, description="Confidence in the response"))
        }

        for field_name, field_info in output_fields.items():
            if field_name not in fields:
                fields[field_name] = field_info

        # Create model
        model = create_model(
            name or cls.__name__,
            __doc__=description or cls.__doc__ or f"Output schema for {name or cls.__name__}",
            **fields
        )

        return model

    # Handle both @output_schema and @output_schema() syntax
    if cls is None:
        return decorator
    return decorator(cls)

def system_prompt(
    template: Optional[str] = None,
    sections: Optional[List[Dict[str, Any]]] = None
):
    """Decorator for defining dynamic system prompts

    Can be used either with a template string or a list of sections:

    @system_prompt("I am a {role} that specializes in {specialty}")
    @agent(...)
    class MyAgent:
        ...

    @system_prompt(sections=[
        {"content": "Base capabilities", "is_dynamic": False},
        {"content": "{specialty}", "is_dynamic": True, "section_id": "specialty"}
    ])
    @agent(...)
    class MyAgent:
        ...
    """
    def decorator(cls: Type) -> Type:
        if template:
            # Create a single dynamic section from template
            cls._system_prompt = SystemPrompt(
                sections=[
                    SystemPromptSection(
                        content=template,
                        is_dynamic=True,
                        section_id="template"
                    )
                ]
            )
        elif sections:
            # Create sections from provided list
            cls._system_prompt = SystemPrompt(
                sections=[
                    SystemPromptSection(**section)
                    for section in sections
                ]
            )
        else:
            # Use docstring as static prompt
            cls._system_prompt = SystemPrompt(
                static_prompt=cls.__doc__ or ""
            )
        return cls
    return decorator
