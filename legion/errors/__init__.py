# Errors __init__.py
"""Custom exceptions for legion"""

from .exceptions import (
    LegionError,
    AgentError,
    ProviderError,
    ToolError,
    JSONFormatError,
    InvalidSchemaError
)

class LegionError(Exception):
    """Base exception class for Legion"""
    pass

class AgentError(LegionError):
    """Exception raised for errors in Agent operations"""
    pass

class ProviderError(LegionError):
    """Exception raised for errors in Provider operations"""
    pass

class ConfigError(LegionError):
    """Exception raised for configuration errors"""
    pass

class ValidationError(LegionError):
    """Exception raised for validation errors"""
    pass