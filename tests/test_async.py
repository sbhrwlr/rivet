"""Tests for async functionality."""

import asyncio
import pytest
from rivet.agent import Agent
from rivet.models.base import ModelAdapter
from rivet.memory.json_adapter import JSONAdapter
from rivet.tools import ToolRegistry, tool


class MockAsyncModelAdapter(ModelAdapter):
    """Mock async model for testing."""
    
    def __init__(self, response: str = "Mock async response"):
        self.response = response
        
    def generate(self, prompt: str, available_tools=None) -> str:
        return self.response
        
    async def agenerate(self, prompt: str, available_tools=None) -> str:
        await asyncio.sleep(0.01)  # Simulate async operation
        return f"Async: {self.response}"
        
    async def stream(self, prompt: str, available_tools=None):
        await asyncio.sleep(0.01)
        for chunk in self.response.split():
            yield chunk + " "
        
    def configure(self, **kwargs) -> None:
        pass


@pytest.mark.asyncio
async def test_agent_async_run():
    """Test async agent execution."""
    model = MockAsyncModelAdapter("Hello async world!")
    agent = Agent(model=model)
    
    response = await agent.arun("Test async message")
    assert "Async: Hello async world!" in response


@pytest.mark.asyncio
async def test_agent_async_stream():
    """Test async agent streaming."""
    model = MockAsyncModelAdapter("Hello streaming world")
    agent = Agent(model=model)
    
    chunks = []
    async for chunk in agent.astream("Test streaming"):
        chunks.append(chunk)
    
    assert len(chunks) > 0
    full_response = "".join(chunks)
    assert "Hello" in full_response
    assert "streaming" in full_response


@pytest.mark.asyncio
async def test_async_tools():
    """Test async tool execution."""
    async def async_test_function():
        await asyncio.sleep(0.01)
        return "Async tool result"
    
    model = MockAsyncModelAdapter("TOOL_CALL: async_test_tool")
    tools = ToolRegistry()
    tools.register("async_test_tool", async_test_function)
    agent = Agent(model=model, tools=tools)
    
    response = await agent.arun("Use async tool")
    assert "TOOL_RESULT: Async tool result" in response


@pytest.mark.asyncio
async def test_mixed_sync_async_tools():
    """Test mixing sync and async tools."""
    def sync_function():
        return "Sync result"
    
    async def async_function():
        await asyncio.sleep(0.01)
        return "Async result"
    
    tools = ToolRegistry()
    tools.register("sync_tool", sync_function)
    tools.register("async_tool", async_function)
    
    # Test async call on sync tool
    result = await tools.acall("sync_tool")
    assert result == "Sync result"
    
    # Test async call on async tool
    result = await tools.acall("async_tool")
    assert result == "Async result"


@pytest.mark.asyncio
async def test_async_memory():
    """Test async memory operations."""
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
        memory = JSONAdapter(f.name)
        
        # Test async store and retrieve
        await memory.astore("Test input", "Test output")
        memories = await memory.aretrieve("Test input")
        
        assert len(memories) == 1
        assert "Test input" in memories[0]
        assert "Test output" in memories[0]
        
        # Test async clear
        await memory.aclear()
        memories = await memory.aretrieve("Test input")
        assert len(memories) == 0
        
        # Cleanup
        os.unlink(f.name)