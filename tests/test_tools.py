"""Tests for the tool system."""

import pytest
from rivet.tools import ToolRegistry, tool, get_registry


def test_tool_registry():
    """Test basic tool registry functionality."""
    registry = ToolRegistry()
    
    def sample_tool():
        return "Tool result"
    
    # Test registration
    registry.register("sample", sample_tool)
    assert "sample" in registry.list()
    
    # Test calling
    result = registry.call("sample")
    assert result == "Tool result"
    
    # Test removal
    registry.remove("sample")
    assert "sample" not in registry.list()


def test_tool_decorator():
    """Test tool decorator functionality."""
    @tool("decorated_tool")
    def my_tool():
        return "Decorated result"
    
    # Tool should be in global registry
    global_registry = get_registry()
    assert "decorated_tool" in global_registry.list()
    
    # Should be callable
    result = global_registry.call("decorated_tool")
    assert result == "Decorated result"


def test_tool_decorator_auto_name():
    """Test tool decorator with automatic naming."""
    @tool()
    def auto_named_tool():
        return "Auto named"
    
    global_registry = get_registry()
    assert "auto_named_tool" in global_registry.list()
    
    result = global_registry.call("auto_named_tool")
    assert result == "Auto named"


def test_tool_not_found():
    """Test calling non-existent tool."""
    registry = ToolRegistry()
    
    with pytest.raises(ValueError, match="Tool 'nonexistent' not found"):
        registry.call("nonexistent")