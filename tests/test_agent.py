"""Tests for the core Agent class."""

import pytest
from rivet.agent import Agent
from rivet.models.base import ModelAdapter
from rivet.memory.json_adapter import JSONAdapter
from rivet.tools import ToolRegistry, tool


class MockModelAdapter(ModelAdapter):
    """Mock model for testing."""
    
    def __init__(self, response: str = "Mock response"):
        self.response = response
        
    def generate(self, prompt: str, available_tools=None) -> str:
        return self.response
        
    def configure(self, **kwargs) -> None:
        pass


def test_agent_basic_run():
    """Test basic agent execution."""
    model = MockModelAdapter("Hello, world!")
    agent = Agent(model=model)
    
    response = agent.run("Test message")
    assert response == "Hello, world!"


def test_agent_with_memory():
    """Test agent with memory storage."""
    model = MockModelAdapter("Response with memory")
    memory = JSONAdapter("test_memory.json")
    agent = Agent(model=model, memory=memory)
    
    response = agent.run("Test with memory")
    assert response == "Response with memory"
    
    # Check memory was stored
    memories = memory.retrieve("Test with memory")
    assert len(memories) > 0
    
    # Cleanup
    memory.clear()


def test_agent_with_tools():
    """Test agent with tool execution."""
    @tool("test_tool")
    def test_function():
        return "Tool executed"
    
    model = MockModelAdapter("TOOL_CALL: test_tool")
    tools = ToolRegistry()
    tools.register("test_tool", test_function)
    agent = Agent(model=model, tools=tools)
    
    response = agent.run("Use the tool")
    assert "TOOL_RESULT: Tool executed" in response


def test_agent_tool_error():
    """Test agent with unknown tool call."""
    model = MockModelAdapter("TOOL_CALL: unknown_tool")
    agent = Agent(model=model)
    
    response = agent.run("Use unknown tool")
    assert "TOOL_ERROR: Unknown tool unknown_tool" in response