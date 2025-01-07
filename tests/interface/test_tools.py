import sys
import pytest
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import asyncio

from legion.interface.tools import BaseTool
from legion.errors.exceptions import ToolError

# Test parameter models
class SimpleParams(BaseModel):
    message: str
    optional_param: Optional[str] = None

class ComplexParams(BaseModel):
    required_str: str
    required_int: int
    optional_float: float = 0.0
    nested_dict: Dict[str, Any] = Field(default_factory=dict)

class InjectableParams(BaseModel):
    api_key: str
    user_id: str
    message: str

# Test tool implementations
class SimpleTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="simple_tool",
            description="A simple test tool",
            parameters=SimpleParams
        )
        self._is_async = False  # Explicitly set sync
    
    def run(self, message: str, optional_param: Optional[str] = None) -> str:
        """Implement the sync run method"""
        if optional_param:
            return f"Tool response: {message} (optional: {optional_param})"
        return f"Tool response: {message}"

class AsyncTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="async_tool",
            description="An async test tool",
            parameters=SimpleParams
        )
        self._is_async = True  # Explicitly set async
    
    def run(self, message: str, optional_param: Optional[str] = None) -> str:
        """Implement the sync run method (required by BaseTool)"""
        raise NotImplementedError("Use arun instead")
    
    async def arun(self, message: str, optional_param: Optional[str] = None) -> str:
        """Implement the async run method"""
        await asyncio.sleep(0.1)  # Simulate async operation
        if optional_param:
            return f"Async response: {message} (optional: {optional_param})"
        return f"Async response: {message}"

class ComplexTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="complex_tool",
            description="A tool with complex parameters",
            parameters=ComplexParams
        )
    
    def run(
        self,
        required_str: str,
        required_int: int,
        optional_float: float = 0.0,
        nested_dict: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Implement the sync run method with complex parameters"""
        return {
            "str_value": required_str,
            "int_value": required_int,
            "float_value": optional_float,
            "nested": nested_dict or {}
        }

class InjectableTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="injectable_tool",
            description="A tool with injectable parameters",
            parameters=InjectableParams,
            injected_params={"api_key", "user_id"}
        )
    
    def run(self, api_key: str, user_id: str, message: str) -> str:
        """Implement the sync run method with injected parameters"""
        return f"Auth: {api_key[:4]}... User: {user_id} Message: {message}"

# Test fixtures
@pytest.fixture
def simple_tool():
    return SimpleTool()

@pytest.fixture
def async_tool():
    return AsyncTool()

@pytest.fixture
def complex_tool():
    return ComplexTool()

@pytest.fixture
def injectable_tool():
    return InjectableTool()

def test_tool_initialization():
    """Test basic tool initialization"""
    tool = SimpleTool()
    
    assert tool.name == "simple_tool"
    assert tool.description == "A simple test tool"
    assert tool.parameters == SimpleParams
    assert not tool.injected_params
    assert not tool._is_async  # Now correctly detects sync tool

def test_async_tool_initialization():
    """Test async tool initialization"""
    tool = AsyncTool()
    
    assert tool.name == "async_tool"
    assert tool.description == "An async test tool"
    assert tool.parameters == SimpleParams
    assert tool._is_async  # Now correctly detects async tool

def test_tool_schema():
    """Test tool schema generation"""
    tool = ComplexTool()
    schema = tool.get_schema()
    
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "complex_tool"
    assert schema["function"]["description"] == "A tool with complex parameters"
    
    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert "required_str" in params["properties"]
    assert "required_int" in params["properties"]
    assert "optional_float" in params["properties"]
    assert "nested_dict" in params["properties"]
    assert set(params["required"]) == {"required_str", "required_int"}

def test_parameter_validation(simple_tool):
    """Test parameter validation"""
    # Valid parameters
    result = asyncio.run(simple_tool(message="test"))
    assert result == "Tool response: test"
    
    # Invalid parameters
    with pytest.raises(ToolError):
        asyncio.run(simple_tool(wrong_param="test"))
    
    with pytest.raises(ToolError):
        asyncio.run(simple_tool())  # Missing required parameter

def test_optional_parameters(simple_tool):
    """Test optional parameter handling"""
    # Without optional parameter
    result = asyncio.run(simple_tool(message="test"))
    assert result == "Tool response: test"
    
    # With optional parameter
    result = asyncio.run(simple_tool(message="test", optional_param="extra"))
    assert result == "Tool response: test (optional: extra)"

@pytest.mark.asyncio
async def test_async_execution(async_tool):
    """Test async tool execution"""
    result = await async_tool(message="test")
    assert result == "Async response: test"
    
    result = await async_tool(message="test", optional_param="extra")
    assert result == "Async response: test (optional: extra)"

def test_complex_parameters(complex_tool):
    """Test complex parameter handling"""
    nested = {"key": "value", "number": 42}
    result = asyncio.run(complex_tool(
        required_str="test",
        required_int=123,
        optional_float=3.14,
        nested_dict=nested
    ))
    
    assert result["str_value"] == "test"
    assert result["int_value"] == 123
    assert result["float_value"] == 3.14
    assert result["nested"] == nested

def test_parameter_injection(injectable_tool):
    """Test parameter injection"""
    # Inject parameters
    tool = injectable_tool.inject(
        api_key="secret_key_123",
        user_id="user_456"
    )
    
    # Verify injected values are used
    result = asyncio.run(tool(message="test message"))
    assert "Auth: secr" in result
    assert "User: user_456" in result
    assert "Message: test message" in result
    
    # Verify can't inject non-injectable parameters
    with pytest.raises(ValueError):
        tool.inject(message="can't inject this")

def test_tool_serialization(complex_tool):
    """Test tool serialization for provider APIs"""
    serialized = complex_tool.model_dump()
    
    assert serialized["type"] == "function"
    assert "function" in serialized
    assert serialized["function"]["name"] == "complex_tool"
    assert "parameters" in serialized["function"]

def test_error_handling(simple_tool):
    """Test error handling in tool execution"""
    # Test with invalid parameter type
    with pytest.raises(ToolError) as exc_info:
        asyncio.run(simple_tool(message=123))  # message should be string
    assert "Invalid parameters" in str(exc_info.value)
    
    # Test with missing required parameter
    with pytest.raises(ToolError) as exc_info:
        asyncio.run(simple_tool())
    assert "Invalid parameters" in str(exc_info.value)

def test_injected_parameter_schema(injectable_tool):
    """Test that injected parameters are removed from schema"""
    schema = injectable_tool.get_schema()
    properties = schema["function"]["parameters"]["properties"]
    
    # Verify injected parameters are not in schema
    assert "api_key" not in properties
    assert "user_id" not in properties
    assert "message" in properties
    
    # Verify required parameters are updated
    assert "message" in schema["function"]["parameters"]["required"]
    assert "api_key" not in schema["function"]["parameters"]["required"]
    assert "user_id" not in schema["function"]["parameters"]["required"]

@pytest.mark.asyncio
async def test_async_error_handling(async_tool):
    """Test error handling in async tool execution"""
    # Test with invalid parameter type
    with pytest.raises(ToolError) as exc_info:
        await async_tool(message=123)  # message should be string
    assert "Invalid parameters" in str(exc_info.value)
    
    # Test with missing required parameter
    with pytest.raises(ToolError) as exc_info:
        await async_tool()
    assert "Invalid parameters" in str(exc_info.value)

def test_tool_reuse_with_injection(injectable_tool):
    """Test that tool can be reused with different injected values"""
    # First use with injected values
    tool1 = InjectableTool().inject(
        api_key="key1",
        user_id="user1"
    )
    result1 = asyncio.run(tool1(message="test1"))
    assert "Auth: key1" in result1
    assert "User: user1" in result1
    
    # Second use with different values
    tool2 = InjectableTool().inject(
        api_key="key2",
        user_id="user2"
    )
    result2 = asyncio.run(tool2(message="test2"))
    assert "Auth: key2" in result2
    assert "User: user2" in result2
    
    # Original tool should not be modified
    assert not hasattr(injectable_tool, '_injected_values') or not injectable_tool._injected_values

if __name__ == "__main__":
    # Configure pytest arguments
    args = [
        __file__,
        "-v",
            "-p", "no:warnings",
            "--tb=short",
            "--asyncio-mode=auto"
        ]
    
    # Run tests
    sys.exit(pytest.main(args))