# errors/exceptions.py
"""Custom exception classes for legion"""

class LegionError(Exception):
    """Base exception for all legion errors"""

    pass

class ProviderError(LegionError):
    """Errors related to LLM provider operations"""

    pass

class ToolError(LegionError):
    """Errors related to tool execution"""

    pass

class JSONFormatError(LegionError):
    """Errors related to JSON formatting"""

    pass

class InvalidSchemaError(LegionError):
    """Errors related to invalid schemas"""

    pass

class AgentError(LegionError):
    """Error in agent operations"""

    pass
