import asyncio
import json
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Type, TypeVar, Union

from pydantic import BaseModel

from ..errors import ProviderError
from .schemas import Message, ModelResponse, ProviderConfig, Role, TokenUsage
from .tools import BaseTool

T = TypeVar("T")

def supports_async(sync_func: Callable[..., T]) -> Callable[..., Union[T, Awaitable[T]]]:
    """Decorator to support both sync and async implementations"""
    @wraps(sync_func)
    async def async_wrapper(self, *args, **kwargs):
        # If async implementation exists, use it
        async_method = f"a{sync_func.__name__}"
        if hasattr(self, async_method):
            return await getattr(self, async_method)(*args, **kwargs)
        # Otherwise, run sync version in thread pool
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: sync_func(self, *args, **kwargs)
        )
    return async_wrapper

class LLMInterface(ABC):
    """Abstract base class defining the LLM provider interface"""

    def __init__(
        self,
        config: ProviderConfig,
        debug: bool = False
    ):
        """Initialize LLM provider

        Args:
        ----
            config: Provider configuration
            debug: Enable debug logging

        """
        self.config = config
        self.debug = debug
        self._setup_client()

    @abstractmethod
    def _setup_client(self) -> None:
        """Initialize provider-specific client"""
        pass

    @abstractmethod
    def _format_messages(self, messages: List[Message]) -> Any:
        """Convert standard messages to provider format"""
        pass

    @abstractmethod
    def _extract_tool_calls(self, response: Any) -> Optional[List[Dict[str, Any]]]:
        """Extract tool calls from provider response"""
        pass

    @abstractmethod
    def _extract_content(self, response: Any) -> str:
        """Extract content from provider response"""
        pass

    @abstractmethod
    def _extract_usage(self, response: Any) -> TokenUsage:
        """Extract token usage from provider response"""
        pass

    def _create_json_conversation(
        self,
        tool_conversation: List[Message],
        schema: Type[BaseModel]
    ) -> List[Message]:
        """Format a tool-using conversation for JSON conversion"""
        # Create a narrative of the conversation
        narrative = []

        # Process each message
        for msg in tool_conversation:
            if msg.role == Role.USER:
                narrative.append(f"User: {msg.content}")
            elif msg.role == Role.ASSISTANT:
                if msg.tool_calls:
                    # Document tool usage
                    for tool_call in msg.tool_calls:
                        narrative.append(
                            f"Assistant used {tool_call['function']['name']} "
                            f"with arguments: {tool_call['function']['arguments']}"
                        )
                elif msg.content:
                    narrative.append(f"Assistant: {msg.content}")
            elif msg.role == Role.TOOL:
                narrative.append(f"Tool response: {msg.content}")

        # Get the schema as JSON for the prompt
        schema_json = schema.model_json_schema()

        # Create messages for JSON formatting
        return [
            Message(
                role=Role.SYSTEM,
                content=(
                    "Convert the assistant's final response into JSON matching the provided schema.\n"
                    "Use the conversation history for context, but only format the final response.\n\n"
                    f"Schema:\n{json.dumps(schema_json, indent=2)}\n\n"
                    "Respond ONLY with valid JSON matching this schema. No other text."
                )
            ),
            Message(
                role=Role.USER,
                content=(
                    "Here is the conversation history:\n\n" +
                    "\n".join(narrative) + "\n\n" +
                    "Format the assistant's final response according to the schema."
                )
            )
        ]

    @abstractmethod
    def _get_chat_completion(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get a basic chat completion"""
        pass

    @abstractmethod
    def _get_tool_completion(
        self,
        messages: List[Message],
        model: str,
        tools: Sequence[BaseTool],
        temperature: float,
        max_tokens: Optional[int] = None,
        format_json: bool = False,
        json_schema: Optional[Type[BaseModel]] = None
    ) -> ModelResponse:
        """Get a chat completion with tool use

        Args:
        ----
            messages: Conversation messages
            model: Model to use
            tools: Available tools
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            format_json: Whether to format final response as JSON
            json_schema: Schema to use for JSON formatting

        """
        pass

    @abstractmethod
    def _get_json_completion(
        self,
        messages: List[Message],
        model: str,
        schema: Type[BaseModel],
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get a chat completion formatted as JSON"""
        pass

    def _get_json_formatting_prompt(self, schema: Type[BaseModel], content: str) -> str:
        """Create generic JSON formatting prompt"""
        schema_json = schema.model_json_schema()

        # Extract field info
        fields = []
        for field_name, field_info in schema_json["properties"].items():
            field_type = field_info.get("type", "any")
            field_desc = field_info.get("description", "")
            required = field_name in schema_json.get("required", [])
            fields.append(f"- {field_name} ({field_type}): {field_desc}" + (" (required)" if required else ""))

        formatting_instructions = (
            "Create a JSON response with these fields:\n" +
            "\n".join(fields) + "\n\n" +
            "Example format:\n" +
            "{\n" +
            "  // Fill in actual values, not descriptions\n" +
            "  // String fields need quotes, numbers don't\n" +
            "  // Required fields must be included\n" +
            "  // Optional fields can be null\n" +
            "}\n\n" +
            "Rules:\n" +
            "1. Use exact field names shown above\n" +
            "2. Include all required fields\n" +
            "3. Fill in actual values, not field descriptions\n" +
            "4. Format values according to their types (strings in quotes, numbers without)\n" +
            "5. Do not add extra fields\n\n" +
            "Format response as valid JSON only. No other text."
        )

        return f"{formatting_instructions}\n\nContent to format:\n{content}"

    @supports_async
    def complete(
        self,
        messages: List[Message],
        model: str,
        response_schema: Optional[Type[BaseModel]] = None,
        tools: Optional[Sequence[BaseTool]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get completion from LLM (sync version)"""
        try:
            if tools and response_schema:
                return self._get_tool_completion(
                    messages=messages,
                    model=model,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    format_json=True,
                    json_schema=response_schema
                )
            elif response_schema:
                return self._get_json_completion(
                    messages=messages,
                    model=model,
                    schema=response_schema,
                    temperature=0.0,
                    max_tokens=max_tokens
                )
            elif tools:
                return self._get_tool_completion(
                    messages=messages,
                    model=model,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                return self._get_chat_completion(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
        except Exception as e:
            raise ProviderError(f"Error during completion: {str(e)}")

    @abstractmethod
    async def _asetup_client(self) -> None:
        """Initialize provider-specific async client"""
        pass

    @abstractmethod
    async def _aget_chat_completion(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get a basic chat completion asynchronously"""
        pass

    @abstractmethod
    async def _aget_tool_completion(
        self,
        messages: List[Message],
        model: str,
        tools: Sequence[BaseTool],
        temperature: float,
        max_tokens: Optional[int] = None,
        format_json: bool = False,
        json_schema: Optional[Type[BaseModel]] = None
    ) -> ModelResponse:
        """Get a chat completion with tool use asynchronously"""
        pass

    @abstractmethod
    async def _aget_json_completion(
        self,
        messages: List[Message],
        model: str,
        schema: Type[BaseModel],
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get a chat completion formatted as JSON asynchronously"""
        pass

    async def acomplete(
        self,
        messages: List[Message],
        model: str,
        response_schema: Optional[Type[BaseModel]] = None,
        tools: Optional[Sequence[BaseTool]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get completion from LLM asynchronously"""
        try:
            if tools and response_schema:
                return await self._aget_tool_completion(
                    messages=messages,
                    model=model,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    format_json=True,
                    json_schema=response_schema
                )
            elif response_schema:
                return await self._aget_json_completion(
                    messages=messages,
                    model=model,
                    schema=response_schema,
                    temperature=0.0,
                    max_tokens=max_tokens
                )
            elif tools:
                return await self._aget_tool_completion(
                    messages=messages,
                    model=model,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                return await self._aget_chat_completion(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
        except Exception as e:
            raise ProviderError(f"Error during async completion: {str(e)}")
