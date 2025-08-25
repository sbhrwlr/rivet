"""Tool system with decorators and registry."""

import asyncio
import inspect
from typing import Callable, Dict, Any, List, get_type_hints
from functools import wraps
import typing


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
    """Decorator to register a function as a tool with an auto-generated schema."""
    
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        _global_registry.register(tool_name, func)

        # === Schema generation ===
        sig = inspect.signature(func)
        hints = get_type_hints(func)

        schema = {
            "type": "function",
            "name": tool_name,
            "description": (inspect.getdoc(func) or "").strip(),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        }

        for pname, p in sig.parameters.items():
            param_schema = {}
            hint = hints.get(pname, str)

            # map types
            if hint == str:
                param_schema["type"] = "string"
            elif hint == int:
                param_schema["type"] = "integer"
            elif hint == float:
                param_schema["type"] = "number"
            elif hint == bool:
                param_schema["type"] = "boolean"
            elif typing.get_origin(hint) is typing.Literal:
                param_schema["type"] = "string"
                param_schema["enum"] = list(typing.get_args(hint))
            else:
                param_schema["type"] = "string"  # fallback

            param_schema["description"] = f"{pname} parameter."

            schema["parameters"]["properties"][pname] = param_schema

            if p.default is inspect._empty:
                schema["parameters"]["required"].append(pname)

        func.__schema__ = schema

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__schema__ = schema  # also attach to wrapper
        return wrapper

    return decorator


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _global_registry