"""Tool system with decorators and registry."""

import asyncio
from typing import Callable, Dict, Any, List
from functools import wraps


class ToolRegistry:
    """Registry for managing tools."""
    
    def __init__(self):
        self.registry: Dict[str, Callable] = {}
        
    def register(self, name: str, func: Callable) -> None:
        """Register a tool function."""
        self.registry[name] = func
        
    def call(self, name: str, *args, **kwargs) -> Any:
        """Call a registered tool."""
        if name not in self.registry:
            raise ValueError(f"Tool '{name}' not found")
        return self.registry[name](*args, **kwargs)
    
    async def acall(self, name: str, *args, **kwargs) -> Any:
        """Call a registered tool asynchronously."""
        if name not in self.registry:
            raise ValueError(f"Tool '{name}' not found")
        
        func = self.registry[name]
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            result = await asyncio.to_thread(func, *args, **kwargs)
        
        # Ensure we don't return a coroutine
        if asyncio.iscoroutine(result):
            result = await result
            
        return result
        
    def list(self) -> List[str]:
        """List all registered tool names."""
        return list(self.registry.keys())
        
    def remove(self, name: str) -> None:
        """Remove a tool from registry."""
        if name in self.registry:
            del self.registry[name]


# Global registry instance
_global_registry = ToolRegistry()


def tool(name: str = None):
    """Decorator to register a function as a tool."""
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        _global_registry.register(tool_name, func)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _global_registry