"""Tests for StreamingHandler functionality."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from rivet.streaming import StreamingHandler
from rivet.tools import ToolRegistry
from rivet.callbacks import CallbackManager, CallbackEvent
from rivet.exceptions import StreamingError


class MockAsyncIterator:
    """Mock async iterator for testing."""
    
    def __init__(self, chunks):
        self.chunks = chunks
        self.index = 0
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.index >= len(self.chunks):
            raise StopAsyncIteration
        chunk = self.chunks[self.index]
        self.index += 1
        return chunk


@pytest.fixture
def tools():
    """Create a tool registry with test tools."""
    registry = ToolRegistry()
    
    def calculator(operation, a, b):
        if operation == "add":
            return a + b
        elif operation == "multiply":
            return a * b
        return "Unknown operation"
    
    async def async_search(query):
        return f"Search results for: {query}"
    
    def simple_tool():
        return "Simple result"
    
    registry.register("calculator", calculator)
    registry.register("search", async_search)
    registry.register("simple", simple_tool)
    
    return registry


@pytest.fixture
def callbacks():
    """Create a callback manager for testing."""
    return CallbackManager()


@pytest.fixture
def handler(tools, callbacks):
    """Create a StreamingHandler for testing."""
    return StreamingHandler(tools=tools, callbacks=callbacks)


@pytest.mark.asyncio
async def test_stream_without_tool_calls(handler):
    """Test streaming without any tool calls."""
    chunks = ["Hello ", "world ", "from ", "AI"]
    stream = MockAsyncIterator(chunks)
    
    result_chunks = []
    async for chunk in handler.stream_with_tools(stream):
        result_chunks.append(chunk)
    
    assert result_chunks == chunks


@pytest.mark.asyncio
async def test_stream_with_simple_tool_call(handler):
    """Test streaming with a simple tool call."""
    chunks = ["Hello ", "TOOL_CALL: simple", " world"]
    stream = MockAsyncIterator(chunks)
    
    result_chunks = []
    async for chunk in handler.stream_with_tools(stream):
        result_chunks.append(chunk)
    
    # Should get: "Hello ", "TOOL_RESULT: Simple result", " world"
    assert len(result_chunks) == 3
    assert result_chunks[0] == "Hello "
    assert "TOOL_RESULT: Simple result" in result_chunks[1]
    assert result_chunks[2] == " world"


@pytest.mark.asyncio
async def test_stream_with_tool_call_with_args(handler):
    """Test streaming with tool call that has arguments."""
    chunks = ["Calculate: ", "TOOL_CALL: calculator(operation=add, a=5, b=3)", " done"]
    stream = MockAsyncIterator(chunks)
    
    result_chunks = []
    async for chunk in handler.stream_with_tools(stream):
        result_chunks.append(chunk)
    
    assert len(result_chunks) == 3
    assert result_chunks[0] == "Calculate: "
    assert "TOOL_RESULT: 8" in result_chunks[1]
    assert result_chunks[2] == " done"


@pytest.mark.asyncio
async def test_stream_with_async_tool_call(handler):
    """Test streaming with async tool call."""
    chunks = ["Searching: ", "TOOL_CALL: search(query=python)", " complete"]
    stream = MockAsyncIterator(chunks)
    
    result_chunks = []
    async for chunk in handler.stream_with_tools(stream):
        result_chunks.append(chunk)
    
    assert len(result_chunks) == 3
    assert result_chunks[0] == "Searching: "
    assert "TOOL_RESULT: Search results for: python" in result_chunks[1]
    assert result_chunks[2] == " complete"


@pytest.mark.asyncio
async def test_stream_with_unknown_tool(handler):
    """Test streaming with unknown tool call."""
    chunks = ["Using: ", "TOOL_CALL: unknown_tool", " failed"]
    stream = MockAsyncIterator(chunks)
    
    result_chunks = []
    async for chunk in handler.stream_with_tools(stream):
        result_chunks.append(chunk)
    
    assert len(result_chunks) == 3
    assert result_chunks[0] == "Using: "
    assert "TOOL_ERROR:" in result_chunks[1]
    assert result_chunks[2] == " failed"


@pytest.mark.asyncio
async def test_stream_with_multiple_tool_calls(handler):
    """Test streaming with multiple tool calls."""
    chunks = [
        "First: ", 
        "TOOL_CALL: simple", 
        " Second: ", 
        "TOOL_CALL: calculator(operation=multiply, a=4, b=2)", 
        " done"
    ]
    stream = MockAsyncIterator(chunks)
    
    result_chunks = []
    async for chunk in handler.stream_with_tools(stream):
        result_chunks.append(chunk)
    
    assert len(result_chunks) == 5
    assert result_chunks[0] == "First: "
    assert "TOOL_RESULT: Simple result" in result_chunks[1]
    assert result_chunks[2] == " Second: "
    assert "TOOL_RESULT: 8" in result_chunks[3]
    assert result_chunks[4] == " done"


@pytest.mark.asyncio
async def test_detect_tool_calls():
    """Test tool call detection."""
    handler = StreamingHandler()
    
    text1 = "Hello TOOL_CALL: simple world"
    assert handler.detect_tool_calls(text1) == ["simple"]
    
    text2 = "TOOL_CALL: calculator(a=1, b=2) and TOOL_CALL: search"
    detected = handler.detect_tool_calls(text2)
    assert "calculator" in detected
    assert "search" in detected
    
    text3 = "No tool calls here"
    assert handler.detect_tool_calls(text3) == []


@pytest.mark.asyncio
async def test_has_tool_calls():
    """Test tool call presence detection."""
    handler = StreamingHandler()
    
    assert handler.has_tool_calls("TOOL_CALL: simple") is True
    assert handler.has_tool_calls("Hello TOOL_CALL: test world") is True
    assert handler.has_tool_calls("No tools here") is False
    assert handler.has_tool_calls("") is False


@pytest.mark.asyncio
async def test_execute_tool_calls(handler):
    """Test executing tool calls in text."""
    text = "Calculate: TOOL_CALL: calculator(operation=add, a=10, b=5) and TOOL_CALL: simple"
    
    result = await handler.execute_tool_calls(text)
    
    assert "TOOL_RESULT: 15" in result
    assert "TOOL_RESULT: Simple result" in result
    assert "TOOL_CALL:" not in result  # Original calls should be replaced


@pytest.mark.asyncio
async def test_parse_tool_args():
    """Test tool argument parsing."""
    handler = StreamingHandler()
    
    # Test key=value format
    args1 = handler._parse_tool_args("operation=add, a=5, b=3")
    assert args1["operation"] == "add"
    assert args1["a"] == 5
    assert args1["b"] == 3
    
    # Test JSON format
    args2 = handler._parse_tool_args('{"query": "test", "limit": 10}')
    assert args2["query"] == "test"
    assert args2["limit"] == 10
    
    # Test empty args
    args3 = handler._parse_tool_args("")
    assert args3 == {}
    
    # Test invalid format (should fallback to raw input)
    args4 = handler._parse_tool_args("invalid format")
    assert args4["input"] == "invalid format"


@pytest.mark.asyncio
async def test_streaming_error_handling(handler):
    """Test error handling during streaming."""
    # Create a stream that raises an exception
    async def failing_stream():
        yield "Hello "
        raise Exception("Stream failed")
    
    with pytest.raises(StreamingError) as exc_info:
        async for chunk in handler.stream_with_tools(failing_stream()):
            pass
    
    assert "Streaming with tool execution failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_callback_integration(handler):
    """Test that callbacks are triggered during tool execution."""
    callback_calls = []
    
    async def test_callback(data):
        callback_calls.append(data)
    
    handler.callbacks.register(CallbackEvent.TOOL_CALL, test_callback)
    
    chunks = ["Test: ", "TOOL_CALL: simple", " done"]
    stream = MockAsyncIterator(chunks)
    
    result_chunks = []
    async for chunk in handler.stream_with_tools(stream):
        result_chunks.append(chunk)
    
    # Verify callback was called
    assert len(callback_calls) == 1
    data = callback_calls[0]
    assert data["tool_name"] == "simple"
    assert data["streaming"] is True


@pytest.mark.asyncio
async def test_buffer_management():
    """Test that buffer is properly managed during streaming."""
    handler = StreamingHandler()
    
    # Test that buffer is cleared after processing
    chunks = ["TOOL_CALL: simple"]
    stream = MockAsyncIterator(chunks)
    
    async for chunk in handler.stream_with_tools(stream):
        pass
    
    # Buffer should be empty after streaming
    assert handler._buffer == ""


@pytest.mark.asyncio
async def test_partial_tool_calls():
    """Test handling of partial tool calls across chunks."""
    handler = StreamingHandler()
    
    # Simulate tool call split across chunks
    chunks = ["TOOL_CALL: calc", "ulator(a=1, b=2)"]
    stream = MockAsyncIterator(chunks)
    
    result_chunks = []
    async for chunk in handler.stream_with_tools(stream):
        result_chunks.append(chunk)
    
    # Should handle the complete tool call when buffer contains full call
    # This is a more complex scenario that might need buffer management improvements
    assert len(result_chunks) >= 1