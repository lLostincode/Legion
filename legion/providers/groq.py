"""Groq-specific implementation of the LLM interface"""

from typing import Optional, List, Sequence, Dict, Any, Type
import json

from openai import OpenAI
from pydantic import BaseModel

from legion.interface.base import LLMInterface

from ..interface.schemas import ProviderConfig, Message, Role, ChatParameters, ModelResponse
from ..errors import ProviderError, ToolError
from .openai import OpenAIProvider
from ..interface.tools import BaseTool
from . import ProviderFactory

class GroqFactory(ProviderFactory):
    """Factory for creating Groq providers"""
    
    def create_provider(self, config: Optional[ProviderConfig] = None, **kwargs) -> LLMInterface:
        """Create a new Groq provider instance"""
        return GroqProvider(config=config, **kwargs)

class GroqProvider(OpenAIProvider):
    """
    Groq-specific provider implementation.
    Inherits from OpenAIProvider since Groq's API is OpenAI-compatible.
    """
    
    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
    GROQ_SYSTEM_INSTRUCTION = (
        "DO NOT attempt to use tools that you do not have access to. "
        "If a user requests something that is outside the scope of your capabilities, "
        "do the best you can with the tools you have available."
    )
    
    def _setup_client(self) -> None:
        """Initialize Groq client using OpenAI's client"""
        try:
            self.client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url or self.DEFAULT_BASE_URL,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries
            )
        except Exception as e:
            raise ProviderError(f"Failed to initialize Groq client: {str(e)}")
    
    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for Groq API"""
        formatted_messages = []
        
        for msg in messages:
            if msg.role == Role.SYSTEM:
                # Add Groq-specific instruction to system message
                content = f"{msg.content}\n\n{self.GROQ_SYSTEM_INSTRUCTION}" if msg.content else self.GROQ_SYSTEM_INSTRUCTION
                formatted_messages.append({
                    "role": "system",
                    "content": content
                })
                continue
            
            message = {"role": msg.role.value}
            
            # Handle tool results
            if msg.role == Role.TOOL:
                if not msg.tool_call_id:
                    continue  # Skip tool messages without tool_call_id
                message.update({
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                    "name": msg.name
                })
            # Handle assistant messages with tool calls
            elif msg.role == Role.ASSISTANT and msg.tool_calls:
                message.update({
                    "content": msg.content,
                    "tool_calls": msg.tool_calls
                })
            # Handle regular messages
            else:
                message["content"] = msg.content
            
            formatted_messages.append(message)
        
        return formatted_messages
    
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
            # Initialize conversation
            current_messages = messages.copy()
            final_response = None
            
            while True:
                if self.debug:
                    print(f"\nSending request to Groq with {len(current_messages)} messages...")
                
                # Get response with tools
                response = self.client.chat.completions.create(
                    model=model,
                    messages=self._format_messages(current_messages),
                    tools=[tool.get_schema() for tool in tools],
                    tool_choice="auto",
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Get assistant's message and tool calls
                assistant_message = response.choices[0].message
                content = assistant_message.content
                tool_calls = self._extract_tool_calls(response)
                
                if self.debug:
                    if tool_calls:
                        print("\nGroq tool calls triggered:")
                        for call in tool_calls:
                            print(f"- {call['function']['name']}: {call['function']['arguments']}")
                    else:
                        print("\nNo tool calls in Groq response")
                
                # Add assistant's response to conversation
                assistant_msg = Message(
                    role=Role.ASSISTANT,
                    content=content,
                    tool_calls=tool_calls
                )
                current_messages.append(assistant_msg)
                
                # If no tool calls, this is our final response
                if not tool_calls:
                    if self.debug:
                        print("\nFinal response received from Groq")
                    return ModelResponse(
                        content=content or "",
                        raw_response=response,
                        usage=self._extract_usage(response),
                        tool_calls=None
                    )
                
                # Process tool calls
                for tool_call in tool_calls:
                    tool = next(
                        (t for t in tools if t.name == tool_call["function"]["name"]),
                        None
                    )
                    if tool:
                        try:
                            args = json.loads(tool_call["function"]["arguments"])
                            result = tool(**args)
                            
                            if self.debug:
                                print(f"\nTool {tool.name} returned: {result}")
                            
                            # Add tool response to conversation
                            tool_msg = Message(
                                role=Role.TOOL,
                                content=str(result),
                                tool_call_id=tool_call["id"],
                                name=tool_call["function"]["name"]
                            )
                            current_messages.append(tool_msg)
                        except Exception as e:
                            raise ToolError(f"Error executing {tool.name}: {str(e)}")
                
        except Exception as e:
            raise ProviderError(f"Groq tool completion failed: {str(e)}")
    
    def _validate_request(self, **kwargs) -> dict:
        """Validate and modify request parameters for Groq"""
        # Ensure N=1 as Groq doesn't support other values
        if kwargs.get('n', 1) != 1:
            raise ProviderError("Groq only supports n=1")
        
        # Handle temperature=0 case
        if kwargs.get('temperature', 1.0) == 0:
            kwargs['temperature'] = 1e-8
        
        # Remove unsupported parameters
        unsupported = ['logprobs', 'logit_bias', 'top_logprobs']
        for param in unsupported:
            kwargs.pop(param, None)
        
        return kwargs
    
    def _get_chat_completion(self, *args, **kwargs):
        """Override to validate params and customize debug output"""
        kwargs = self._validate_request(**kwargs)
        if self.debug:
            print(f"\nSending chat completion request to Groq...")
        return super()._get_chat_completion(*args, **kwargs)
    
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
            
            # Create messages for Groq
            groq_messages = [
                {"role": "system", "content": formatting_prompt}
            ]
            
            # Add remaining messages, skipping system
            groq_messages.extend([
                msg.model_dump() for msg in messages
                if msg.role != Role.SYSTEM
            ])
            
            # Handle temperature=0 case for Groq
            groq_temp = 1e-8 if temperature == 0 else temperature
            
            response = self.client.chat.completions.create(
                model=model,
                messages=groq_messages,
                response_format={"type": "json_object"},
                temperature=groq_temp,
                max_tokens=max_tokens
            )
            
            # Validate response against schema
            content = response.choices[0].message.content
            try:
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
            raise ProviderError(f"Groq JSON completion failed: {str(e)}")
    