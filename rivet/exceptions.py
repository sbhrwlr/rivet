"""Custom exceptions for Rivet framework."""


class RivetError(Exception):
    """Base exception for Rivet framework."""
    pass


class ParsingError(RivetError):
    """Exception raised when output parsing fails."""
    
    def __init__(self, message: str, original_text: str = None, parser_type: str = None):
        super().__init__(message)
        self.original_text = original_text
        self.parser_type = parser_type


class StreamingError(RivetError):
    """Exception raised when streaming operations fail."""
    pass


class UsageLimitError(RivetError):
    """Exception raised when usage limits are exceeded."""
    
    def __init__(self, message: str, limit_type: str = None, current_usage: float = None, limit: float = None):
        super().__init__(message)
        self.limit_type = limit_type
        self.current_usage = current_usage
        self.limit = limit


class ToolError(RivetError):
    """Exception raised when tool execution fails."""
    
    def __init__(self, message: str, tool_name: str = None):
        super().__init__(message)
        self.tool_name = tool_name