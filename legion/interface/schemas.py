from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Message role"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"

class TokenUsage(BaseModel):
    """Token usage information"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ToolCall(BaseModel):
    """Tool call information"""

    id: str = Field(description="Unique identifier for the tool call")
    name: str = Field(description="Name of the tool to call")
    arguments: str = Field(description="Arguments to pass to the tool as a JSON string")

class Message(BaseModel):
    """Message schema"""

    content: str
    role: Role = Role.USER
    name: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Union[ToolCall, Dict[str, Any]]]] = None
    tool_call_id: Optional[str] = None

class ModelResponse(BaseModel):
    """Model response schema"""

    content: str
    role: Role = Role.ASSISTANT
    tool_calls: Optional[List[Union[ToolCall, Dict[str, Any]]]] = None
    raw_response: Optional[Dict[str, Any]] = None
    usage: Optional[TokenUsage] = None

class ProviderConfig(BaseModel):
    """Base provider configuration"""

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization_id: Optional[str] = None
    timeout: int = Field(default=60, ge=1)
    max_retries: int = Field(default=3, ge=0)
    model: Optional[str] = None

    class Config:
        extra = "allow"  # Allow provider-specific config options

class ChatParameters(BaseModel):
    """Standard parameters for chat completions"""

    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None

    class Config:
        extra = "allow"  # Allow provider-specific parameters

class SystemPromptSection(BaseModel):
    """Section of a system prompt that can be static or dynamic"""

    content: Union[str, Callable[[], str]]  # Allow callable for dynamic content
    is_dynamic: bool = False
    section_id: Optional[str] = None
    default_value: Optional[Union[str, Callable[[], str]]] = None  # Add default value support

    model_config = {
        "arbitrary_types_allowed": True  # Allow callable types
    }

    def render(self, dynamic_values: Optional[Dict[str, str]] = None) -> str:
        """Render the section content with dynamic values"""
        if not self.is_dynamic:
            return self.content() if callable(self.content) else self.content

        # Get the value to use
        value = None

        # If we have dynamic values and a section_id, use that value
        if dynamic_values and self.section_id and self.section_id in dynamic_values:
            value = dynamic_values[self.section_id]
        # If content is callable, call it
        elif callable(self.content):
            value = self.content()
        # If content is a string template and we have dynamic values, try to format it
        elif isinstance(self.content, str) and dynamic_values:
            try:
                value = self.content.format(**dynamic_values)
            except (KeyError, ValueError):
                pass  # Fall through to default if formatting fails

        # If no value yet, try default value
        if value is None and self.default_value is not None:
            if callable(self.default_value):
                value = self.default_value()
            elif isinstance(self.default_value, str):
                try:
                    value = self.default_value.format(**dynamic_values) if dynamic_values else self.default_value
                except (KeyError, ValueError):
                    value = self.default_value

        # If still no value, use raw content
        if value is None:
            value = self.content() if callable(self.content) else self.content

        # Format with key-value pair if we have a section_id
        if self.section_id:
            return f"{self.section_id}: {value}"
        return value

class SystemPrompt(BaseModel):
    """Template-based system prompt with static and dynamic sections"""

    sections: List[SystemPromptSection] = Field(
        default_factory=list,
        description="Ordered sections of the system prompt"
    )
    static_prompt: Optional[str] = Field(
        None,
        description="Static prompt text (used if no sections defined)"
    )

    def render(self, dynamic_values: Optional[Dict[str, str]] = None) -> str:
        """Render the complete system prompt with any dynamic values"""
        if not self.sections:
            return self.static_prompt or ""

        rendered_sections = []
        dynamic_values = dynamic_values or {}

        for section in self.sections:
            rendered_sections.append(section.render(dynamic_values))

        return "\n\n".join(rendered_sections)

    def __str__(self) -> str:
        """String representation should be the rendered content"""
        return self.render()
