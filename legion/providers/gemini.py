"""Google's Gemini-specific implementation of the LLM interface"""

import json
from typing import Any, Dict, List, Optional, Sequence, Type

from openai import OpenAI
from pydantic import BaseModel

from legion.interface.base import LLMInterface

from ..errors import ProviderError
from ..interface.schemas import (
    ChatParameters,
    Message,
    ModelResponse,
    ProviderConfig,
    Role,
    TokenUsage,
)
from ..interface.tools import BaseTool
from . import ProviderFactory
from .openai import OpenAIProvider


class GeminiFactory(ProviderFactory):
    """Factory for creating Gemini providers"""

    def create_provider(self, config: Optional[ProviderConfig] = None, **kwargs) -> LLMInterface:
        """Create a new Gemini provider instance"""
        return GeminiProvider(config=config, **kwargs)

class GeminiProvider(OpenAIProvider):
    """Google's Gemini-specific provider implementation.
    Inherits from OpenAIProvider since Gemini's API is OpenAI-compatible.
    """

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    SUPPORTED_MODELS = {
        "gemini-1.5-flash-latest": "models/gemini-1.5-pro",
        "gemini-1.5-pro-latest": "models/gemini-1.5-pro-latest",
        "gemini-pro": "models/gemini-pro"
    }

    def _setup_client(self) -> None:
        """Initialize Gemini client using OpenAI client"""
        try:
            self.client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url or self.DEFAULT_BASE_URL,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries
            )
        except Exception as e:
            raise ProviderError(f"Failed to initialize Gemini client: {str(e)}")

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert messages to Gemini format"""
        formatted_messages = []

        for msg in messages:
            # Skip empty messages
            if not msg.content and not msg.tool_calls:
                continue

            if msg.role == Role.SYSTEM:
                gemini_instruction = (
                    "If a user requests multiple things and you can only do a subset of them, "
                    "use your available tools to complete what you can and explain what you cannot do. "
                )
                content = f"System instruction: {msg.content}\n\n{gemini_instruction}"
                formatted_messages.append({
                    "role": "user",
                    "content": content
                })
            elif msg.role == Role.TOOL:
                # Format tool results as assistant messages with function call context
                formatted_messages.append({
                    "role": "assistant",
                    "content": f"Function '{msg.name}' returned: {msg.content}"
                })
            else:
                formatted_messages.append({
                    "role": "user" if msg.role == Role.USER else "assistant",
                    "content": msg.content or ""
                })

        return formatted_messages

    def _get_chat_completion(self, messages: List[Message], model: str, params: ChatParameters) -> ModelResponse:
        """Get a basic chat completion with Gemini-specific handling"""
        try:
            # Map model name to Gemini format
            gemini_model = self.SUPPORTED_MODELS.get(model, model)

            # Prepare request
            request = {
                "model": gemini_model,
                "messages": self._format_messages(messages),
                "temperature": params.temperature,
                "max_tokens": params.max_tokens,
                "stream": params.stream
            }

            if self.debug:
                print("\nSending request to Gemini API:")
                print(f"Model: {gemini_model}")
                print(f"Messages: {json.dumps(request['messages'], indent=2)}")

            response = self.client.chat.completions.create(**request)

            return ModelResponse(
                content=self._extract_content(response),
                raw_response=response,
                usage=self._extract_usage(response),
                tool_calls=None
            )
        except Exception as e:
            raise ProviderError(f"Gemini completion failed: {str(e)}")

    def _extract_usage(self, response: Any) -> TokenUsage:
        """Extract token usage from Gemini response"""
        # Gemini might not provide detailed token counts
        try:
            return TokenUsage(
                prompt_tokens=getattr(response.usage, "prompt_tokens", 0),
                completion_tokens=getattr(response.usage, "completion_tokens", 0),
                total_tokens=getattr(response.usage, "total_tokens", 0)
            )
        except AttributeError:
            return TokenUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0
            )

    def _get_tool_completion(
        self,
        messages: List[Message],
        model: str,
        tools: Sequence[BaseTool],
        temperature: float,
        json_temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get completion with tool usage"""
        gemini_model = self.SUPPORTED_MODELS.get(model, model)
        formatted_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": tool.parameters.model_json_schema()["properties"],
                        "required": tool.parameters.model_json_schema()["required"]
                    }
                }
            }
            for tool in tools
        ]

        # Initialize conversation history
        current_messages = messages.copy()
        max_iterations = 5  # Prevent infinite loops
        iterations = 0
        last_response = None

        while iterations < max_iterations:
            iterations += 1

            # Format messages for this iteration
            formatted_messages = [
                msg for msg in self._format_messages(current_messages)
                if msg.get("content")
            ]

            if self.debug:
                print(f"\nSending tool request to Gemini API (iteration {iterations}):")
                print(f"Model: {gemini_model}")
                print(f"Messages: {json.dumps(formatted_messages, indent=2)}")
                print(f"Tools: {json.dumps(formatted_tools, indent=2)}")

            response = self.client.chat.completions.create(
                model=gemini_model,
                messages=formatted_messages,
                tools=formatted_tools,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = self._extract_content(response)
            tool_calls = self._extract_tool_calls(response)
            last_response = response

            # Process tool calls if present
            if tool_calls:
                for tool_call in tool_calls:
                    tool = next(
                        (t for t in tools if t.name == tool_call["function"]["name"]),
                        None
                    )
                    if tool:
                        try:
                            args = json.loads(tool_call["function"]["arguments"])
                            result = tool(**args)

                            # Add tool response to conversation
                            current_messages.append(Message(
                                role=Role.TOOL,
                                content=str(result),
                                tool_call_id=tool_call["id"],
                                name=tool_call["function"]["name"]
                            ))
                        except Exception as e:
                            raise ProviderError(f"Error executing {tool.name}: {str(e)}")
            else:
                # No more tool calls, return the final response
                return ModelResponse(
                    content=content,
                    raw_response=last_response,
                    usage=self._extract_usage(last_response),
                    tool_calls=None
                )

        raise ProviderError("Exceeded maximum number of tool call iterations")

    def _get_json_completion(
        self,
        messages: List[Message],
        model: str,
        schema: Optional[Type[BaseModel]],
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get a chat completion formatted as JSON"""
        try:
            # Get generic JSON formatting prompt
            formatting_prompt = self._get_json_formatting_prompt(schema, messages[-1].content)

            # Create messages for Gemini
            gemini_messages = [
                {"role": "system", "content": formatting_prompt}
            ]

            # Add remaining messages, skipping system
            gemini_messages.extend([
                msg.model_dump() for msg in messages
                if msg.role != Role.SYSTEM
            ])

            response = self.client.beta.chat.completions.parse(
                model=model,
                messages=gemini_messages,
                response_format=schema,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Convert parsed response to dict before JSON serialization
            parsed_dict = response.choices[0].message.parsed.model_dump()

            return ModelResponse(
                content=json.dumps(parsed_dict),
                raw_response=response,
                usage=self._extract_usage(response),
                tool_calls=None
            )
        except Exception as e:
            raise ProviderError(f"Gemini JSON completion failed: {str(e)}")
