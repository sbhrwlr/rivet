"""
Rivet - A lightweight, developer-first framework to build AI agents.
Philosophy: less abstraction, more action.
"""

from .agent import Agent
from .inspector import Inspector
from .tools import tool, ToolRegistry
from .streaming import StreamingHandler
from .parsers import (
    OutputParser, JSONParser, ListParser, XMLParser, 
    SmartParser, RetryParser
)
from .callbacks import CallbackManager, CallbackEvent
from .middleware import (
    Middleware, MiddlewareChain, LoggingMiddleware, 
    TimingMiddleware, ValidationMiddleware, RateLimitMiddleware
)

# Import PydanticParser only if pydantic is available
try:
    from .parsers import PydanticParser
    _pydantic_available = True
except ImportError:
    _pydantic_available = False

__version__ = "0.1.0"

_all_exports = [
    "Agent", "Inspector", "tool", "ToolRegistry", "StreamingHandler",
    "OutputParser", "JSONParser", "ListParser", "XMLParser",
    "SmartParser", "RetryParser", "CallbackManager", "CallbackEvent",
    "Middleware", "MiddlewareChain", "LoggingMiddleware", 
    "TimingMiddleware", "ValidationMiddleware", "RateLimitMiddleware"
]

if _pydantic_available:
    _all_exports.append("PydanticParser")

__all__ = _all_exports