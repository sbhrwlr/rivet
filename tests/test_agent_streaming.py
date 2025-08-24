"""Tests for Agent streaming functionality."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from rivet.agent import Agent
from rivet.models.base import ModelAdapter
from rivet.memory.base import MemoryAdapter
from rivet.tools import ToolRegistry
from rivet.callbacks import CallbackManager, CallbackEvent
from rivet.parsers.base import OutputParser


class MockModelAdapter(ModelAdapter):
    """Mock model adapter for testing."""
    
    def __init__(self, responses=None, stream_chunks=None):
        self.responses = responses or ["Mock response"]
        self.stream_chunks = stream_chunks or ["Mock ", "stream ", "response"]
        self.call_count = 0
        
    def generate(self, prompt: str, available_tools=None) -> str:
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response
        
    def configure(self, **kwargs) -> None:
        pass
    
    async def stream(self, prompt: str, available_tools=None):
        for chunk in self.stream_chunks:
            yield chunk


class MockMemoryAdapter(MemoryAdapter):
    """Mock memory adapter for testing."""
    
    def __init__(self):
        self.stored_data = []
        
    def store(self, key: str, value: str) -> None:
        self.stored_data.append((key, value))
        
    def retrieve(self, key: str) -> list:
        return []
    
    def clear(self) -> None:
        self.stored_data.clear()
    
    async def astore(self, key: str, value: str) -> None:
        self.stored_data.append((key, value))
        
    async def aretrieve(self, key: str) -> list:
        return []
    
    async def aclear(self) -> None:
        self.stored_data.clear()


class MockOutputParser(OutputParser):
    """Mock output parser for testing."""
    
    def parse(self, text: str):
        return f"Parsed: {text}"
    
    def parse_with_fallback(self, text: str):
        return self.parse(text)


@pytest.fixture
def tools():
    """Create a tool registry with test tools."""
    registry = ToolRegistry()
    
    def test_tool():
        return "Tool result"
    
    async def async_tool():
        return "Async tool result"
    
    registry.register("test_tool", test_tool)
    registry.register("async_tool", async_tool)
    
    return registry


@pytest.fixture
def memory():
    """Create a mock memory adapter."""
    return MockMemoryAdapter()


@pytest.fixture
def callbacks():
    """Create a callback manager."""
    return CallbackManager()


@pytest.mark.asyncio
async def test_agent_stream_basic(tools, memory, callbacks):
    """Test basic agent streaming without tool calls."""
    model = MockModelAdapter(stream_chunks=["Hello ", "world ", "from ", "agent"])
    agent = Agent(model=model, memory=memory, tools=tools, callbacks=callbacks)
    
    chunks = []
    async for chunk in agent.astream("test message"):
        chunks.append(chunk)
    
    assert len(chunks) == 4
    assert "".join(chunks) == "Hello world from agent"
    
    # Verify memory storage happened after complete response
    assert len(memory.stored_data) == 1
    assert memory.stored_data[0][0] == "test message"
    assert memory.stored_data[0][1] == "Hello world from agent"


@pytest.mark.asyncio
async def test_agent_stream_with_tool_calls(tools, memory, callbacks):
    """Test agent streaming with tool calls."""
    model = MockModelAdapter(stream_chunks=["Using ", "TOOL_CALL: test_tool", " done"])
    agent = Agent(model=model, memory=memory, tools=tools, callbacks=callbacks)
    
    chunks = []
    async for chunk in agent.astream("test with tools"):
        chunks.append(chunk)
    
    # Should get: "Using ", "TOOL_RESULT: Tool result", " done"
    assert len(chunks) == 3
    assert chunks[0] == "Using "
    assert "TOOL_RESULT: Tool result" in chunks[1]
    assert chunks[2] == " done"


@pytest.mark.asyncio
async def test_agent_stream_with_async_tool_calls(tools, memory, callbacks):
    """Test agent streaming with async tool calls."""
    model = MockModelAdapter(stream_chunks=["Async ", "TOOL_CALL: async_tool", " complete"])
    agent = Agent(model=model, memory=memory, tools=tools, callbacks=callbacks)
    
    chunks = []
    async for chunk in agent.astream("test async tools"):
        chunks.append(chunk)
    
    assert len(chunks) == 3
    assert chunks[0] == "Async "
    assert "TOOL_RESULT: Async tool result" in chunks[1]
    assert chunks[2] == " complete"


@pytest.mark.asyncio
async def test_agent_stream_with_output_parser(tools, memory, callbacks):
    """Test agent streaming with output parser."""
    model = MockModelAdapter(stream_chunks=["Raw ", "response"])
    parser = MockOutputParser()
    agent = Agent(model=model, memory=memory, tools=tools, callbacks=callbacks, output_parser=parser)
    
    chunks = []
    async for chunk in agent.astream("test parsing"):
        chunks.append(chunk)
    
    # Chunks should be raw during streaming
    assert "".join(chunks) == "Raw response"
    
    # But memory should store the raw response (not parsed)
    assert len(memory.stored_data) == 1
    assert memory.stored_data[0][1] == "Raw response"


@pytest.mark.asyncio
async def test_agent_stream_callbacks(tools, memory):
    """Test that callbacks are triggered during streaming."""
    model = MockModelAdapter(stream_chunks=["Test ", "TOOL_CALL: test_tool", " stream"])
    callbacks = CallbackManager()
    agent = Agent(model=model, memory=memory, tools=tools, callbacks=callbacks)
    
    callback_events = []
    
    async def callback_handler(data):
        callback_events.append(data)
    
    # Register callbacks for different events
    callbacks.register(CallbackEvent.AGENT_START, callback_handler)
    callbacks.register(CallbackEvent.MODEL_CALL, callback_handler)
    callbacks.register(CallbackEvent.TOOL_CALL, callback_handler)
    callbacks.register(CallbackEvent.STREAMING_CHUNK, callback_handler)
    callbacks.register(CallbackEvent.AGENT_END, callback_handler)
    
    chunks = []
    async for chunk in agent.astream("test callbacks"):
        chunks.append(chunk)
    
    # Verify callbacks were triggered
    assert len(callback_events) >= 5  # At least start, model_call, tool_call, chunks, end
    
    # Check specific callback types
    start_events = [e for e in callback_events if "streaming" in e and e["streaming"] is True]
    assert len(start_events) > 0


@pytest.mark.asyncio
async def test_agent_stream_error_handling(tools, memory, callbacks):
    """Test error handling during streaming."""
    # Create a model that raises an exception during streaming
    class FailingModel(MockModelAdapter):
        async def stream(self, prompt: str, available_tools=None):
            yield "Start "
            raise Exception("Stream failed")
    
    model = FailingModel()
    agent = Agent(model=model, memory=memory, tools=tools, callbacks=callbacks)
    
    error_callbacks = []
    
    async def error_callback(data):
        error_callbacks.append(data)
    
    callbacks.register(CallbackEvent.ERROR, error_callback)
    
    with pytest.raises(Exception) as exc_info:
        async for chunk in agent.astream("test error"):
            pass
    
    assert "Stream failed" in str(exc_info.value)
    
    # Verify error callback was triggered
    assert len(error_callbacks) == 1
    assert error_callbacks[0]["streaming"] is True


@pytest.mark.asyncio
async def test_agent_stream_with_context(tools, memory, callbacks):
    """Test agent streaming with context."""
    model = MockModelAdapter(stream_chunks=["Context ", "response"])
    agent = Agent(model=model, memory=memory, tools=tools, callbacks=callbacks)
    
    context = {"user_id": "123", "session": "abc"}
    
    chunks = []
    async for chunk in agent.astream("test message", context=context):
        chunks.append(chunk)
    
    assert "".join(chunks) == "Context response"


@pytest.mark.asyncio
async def test_agent_stream_memory_storage_after_completion(tools, callbacks):
    """Test that memory storage happens after complete response in streaming mode."""
    memory = MockMemoryAdapter()
    model = MockModelAdapter(stream_chunks=["Part ", "1 ", "Part ", "2"])
    agent = Agent(model=model, memory=memory, tools=tools, callbacks=callbacks)
    
    chunks = []
    async for chunk in agent.astream("test memory timing"):
        chunks.append(chunk)
        # Memory should not be stored yet during streaming
        if len(chunks) < 4:  # Before completion
            assert len(memory.stored_data) == 0
    
    # After completion, memory should be stored
    assert len(memory.stored_data) == 1
    assert memory.stored_data[0][1] == "Part 1 Part 2"


@pytest.mark.asyncio
async def test_agent_stream_multiple_tool_calls(tools, memory, callbacks):
    """Test agent streaming with multiple tool calls."""
    model = MockModelAdapter(stream_chunks=[
        "First ", 
        "TOOL_CALL: test_tool", 
        " then ", 
        "TOOL_CALL: async_tool", 
        " done"
    ])
    agent = Agent(model=model, memory=memory, tools=tools, callbacks=callbacks)
    
    chunks = []
    async for chunk in agent.astream("test multiple tools"):
        chunks.append(chunk)
    
    assert len(chunks) == 5
    assert chunks[0] == "First "
    assert "TOOL_RESULT: Tool result" in chunks[1]
    assert chunks[2] == " then "
    assert "TOOL_RESULT: Async tool result" in chunks[3]
    assert chunks[4] == " done"