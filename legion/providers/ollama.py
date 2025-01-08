"""Ollama-specific implementation of the LLM interface"""

import json
from typing import Any, Dict, List, Optional, Sequence, Type

from pydantic import BaseModel

from ..errors import ProviderError
from ..interface.base import LLMInterface
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


class OllamaFactory(ProviderFactory):
    """Factory for creating Ollama providers"""

    def create_provider(self, config: Optional[ProviderConfig] = None, **kwargs) -> LLMInterface:
        """Create a new Ollama provider instance"""
        return OllamaProvider(config=config, **kwargs)

class OllamaProvider(LLMInterface):
    """Ollama-specific provider implementation"""

    def _setup_client(self) -> None:
        """Initialize Ollama client"""
        try:
            from ollama import Client
            self.client = Client(host=self.config.base_url or "http://localhost:11434")
        except Exception as e:
            raise ProviderError(f"Failed to initialize Ollama client: {str(e)}")

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert messages to Ollama format"""
        ollama_messages = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                # Ollama handles system messages as special user messages
                ollama_messages.append({
                    "role": "system",
                    "content": msg.content
                })
                continue

            if msg.role == Role.TOOL:
                # Format tool results
                ollama_messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "name": msg.name
                })
            else:
                # Format regular messages
                ollama_messages.append({
                    "role": "user" if msg.role == Role.USER else "assistant",
                    "content": msg.content
                })

        return ollama_messages

    def _get_chat_completion(
        self,
        messages: List[Message],
        model: str,
        params: ChatParameters
    ) -> ModelResponse:
        """Get a basic chat completion"""
        try:
            # Build options dictionary
            options = {
                "temperature": params.temperature
            }

            response = self.client.chat(
                model=model,
                messages=self._format_messages(messages),
                options=options,
                stream=params.stream
            )

            return ModelResponse(
                content=self._extract_content(response),
                raw_response=response,
                usage=self._extract_usage(response),
                tool_calls=None
            )
        except Exception as e:
            raise ProviderError(f"Ollama completion failed: {str(e)}")

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
        try:
            # Build options dictionary
            options = {
                "temperature": temperature
            }

            response = self.client.chat(
                model=model,
                messages=self._format_messages(messages),
                tools=[tool.get_schema() for tool in tools],
                options=options,
                stream=False
            )

            # Process tool calls if any
            tool_calls = self._extract_tool_calls(response)

            return ModelResponse(
                content=self._extract_content(response),
                raw_response=response,
                usage=self._extract_usage(response),
                tool_calls=tool_calls
            )
        except Exception as e:
            raise ProviderError(f"Ollama tool completion failed: {str(e)}")

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
            # Format schema for system prompt
            schema_json = schema.model_json_schema()
            schema_prompt = (
                "You must respond with valid JSON that matches this schema:\n"
                f"{json.dumps(schema_json, indent=2)}\n\n"
                "Respond ONLY with valid JSON. No other text."
            )

            # Create new messages list with modified system message
            formatted_messages = []
            system_content = schema_prompt

            for msg in messages:
                if msg.role == Role.SYSTEM:
                    # Combine existing system message with schema prompt
                    system_content = f"{msg.content}\n\n{schema_prompt}"
                else:
                    formatted_messages.append(msg)

            # Add system message at the start
            formatted_messages.insert(0, Message(
                role=Role.SYSTEM,
                content=system_content
            ))

            # Build options dictionary
            options = {
                "temperature": temperature,
                "format": "json"  # Enable JSON mode
            }

            response = self.client.chat(
                model=model,
                messages=self._format_messages(formatted_messages),
                options=options,
                stream=False
            )

            # Validate response against schema
            try:
                content = self._extract_content(response)
                data = json.loads(content)
                schema.model_validate(data)
            except Exception as e:
                raise ProviderError(f"Invalid JSON response: {str(e)}")

            return ModelResponse(
                content=content,
                raw_response=response,
                usage=self._extract_usage(response),
                tool_calls=None
            )
        except Exception as e:
            raise ProviderError(f"Ollama JSON completion failed: {str(e)}")

    def _extract_usage(self, response: Any) -> TokenUsage:
        """Extract token usage from Ollama response"""
        # Ollama might not provide token counts
        return TokenUsage(
            prompt_tokens=getattr(response, "prompt_tokens", 0),
            completion_tokens=getattr(response, "completion_tokens", 0),
            total_tokens=getattr(response, "total_tokens", 0)
        )

    def _extract_content(self, response: Any) -> str:
        """Extract content from Ollama response"""
        if not hasattr(response, "message"):
            return ""

        if hasattr(response.message, "content"):
            return response.message.content.strip()

        return ""

    def _extract_tool_calls(self, response: Any) -> Optional[List[Dict[str, Any]]]:
        """Extract tool calls from Ollama response"""
        if not hasattr(response, "message") or not hasattr(response.message, "tool_calls"):
            return None

        tool_calls = []
        for tool_call in response.message.tool_calls:
            # Generate a unique ID if none provided
            call_id = getattr(tool_call, "id", f"call_{len(tool_calls)}")

            tool_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": json.dumps(tool_call.function.arguments)
                }
            })

            if self.debug:
                print(f"\nExtracted tool call: {json.dumps(tool_calls[-1], indent=2)}")

        return tool_calls if tool_calls else None
