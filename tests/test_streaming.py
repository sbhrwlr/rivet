"""Tests for streaming functionality."""

import pytest
import asyncio
import sys
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from rivet.models.base import ModelAdapter
from rivet.models.openai_adapter import OpenAIAdapter
from rivet.exceptions import StreamingError


class MockModelAdapter(ModelAdapter):
    """Mock model adapter for testing."""
    
    def __init__(self, should_fail_streaming=False, should_fail_fallback=False):
        self.should_fail_streaming = should_fail_streaming
        self.should_fail_fallback = should_fail_fallback
        
    def generate(self, prompt: str, available_tools=None) -> str:
        if self.should_fail_fallback:
            raise Exception("Fallback failed")
        return f"Response to: {prompt}"
        
    def configure(self, **kwargs) -> None:
        pass
    
    async def stream(self, prompt: str, available_tools=None):
        if self.should_fail_streaming:
            raise Exception("Streaming failed")
        
        # Simulate streaming chunks
        response = f"Response to: {prompt}"
        words = response.split()
        for word in words:
            yield word + " "


class ConcreteModelAdapter(ModelAdapter):
    """Concrete model adapter for testing base class functionality."""
    
    def generate(self, prompt: str, available_tools=None) -> str:
        return f"Response to: {prompt}"
        
    def configure(self, **kwargs) -> None:
        pass


@pytest.mark.asyncio
async def test_base_model_adapter_stream_success():
    """Test base model adapter streaming with successful generation."""
    adapter = MockModelAdapter()
    
    chunks = []
    async for chunk in adapter.stream("test prompt"):
        chunks.append(chunk)
    
    assert len(chunks) == 4  # "Response ", "to: ", "test ", "prompt "
    assert "".join(chunks).strip() == "Response to: test prompt"


@pytest.mark.asyncio
async def test_base_model_adapter_stream_fallback():
    """Test base model adapter falls back to agenerate when streaming fails."""
    adapter = ConcreteModelAdapter()
    
    # Mock agenerate to return a response
    adapter.agenerate = AsyncMock(return_value="Fallback response")
    
    chunks = []
    async for chunk in adapter.stream("test prompt"):
        chunks.append(chunk)
    
    assert len(chunks) == 1
    assert chunks[0] == "Fallback response"


@pytest.mark.asyncio
async def test_base_model_adapter_stream_error():
    """Test base model adapter raises StreamingError when agenerate fails."""
    adapter = ConcreteModelAdapter()
    
    # Mock agenerate to raise an exception
    adapter.agenerate = AsyncMock(side_effect=Exception("Generation failed"))
    
    with pytest.raises(StreamingError) as exc_info:
        async for chunk in adapter.stream("test prompt"):
            pass
    
    assert "Streaming failed: Generation failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_openai_adapter_stream_success():
    """Test OpenAI adapter streaming success."""
    adapter = OpenAIAdapter(api_key="test-key")
    
    # Mock the OpenAI client and streaming response
    mock_chunk = Mock()
    mock_chunk.choices = [Mock()]
    mock_chunk.choices[0].delta.content = "test chunk"
    
    mock_stream = AsyncMock()
    mock_stream.__aiter__.return_value = [mock_chunk]
    
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_stream
    
    # Mock the openai module
    mock_openai = MagicMock()
    mock_openai.AsyncOpenAI.return_value = mock_client
    
    with patch.dict('sys.modules', {'openai': mock_openai}):
        chunks = []
        async for chunk in adapter.stream("test prompt"):
            chunks.append(chunk)
        
        assert len(chunks) == 1
        assert chunks[0] == "test chunk"


@pytest.mark.asyncio
async def test_openai_adapter_stream_fallback():
    """Test OpenAI adapter falls back to non-streaming when streaming fails."""
    adapter = OpenAIAdapter(api_key="test-key")
    
    # Mock streaming to fail
    mock_client = AsyncMock()
    mock_client.chat.completions.create.side_effect = Exception("Streaming failed")
    
    # Mock agenerate to succeed
    adapter.agenerate = AsyncMock(return_value="Fallback response")
    
    # Mock the openai module
    mock_openai = MagicMock()
    mock_openai.AsyncOpenAI.return_value = mock_client
    
    with patch.dict('sys.modules', {'openai': mock_openai}):
        chunks = []
        async for chunk in adapter.stream("test prompt"):
            chunks.append(chunk)
        
        assert len(chunks) == 1
        assert chunks[0] == "Fallback response"


@pytest.mark.asyncio
async def test_openai_adapter_stream_complete_failure():
    """Test OpenAI adapter raises StreamingError when both streaming and fallback fail."""
    adapter = OpenAIAdapter(api_key="test-key")
    
    # Mock streaming to fail
    mock_client = AsyncMock()
    mock_client.chat.completions.create.side_effect = Exception("Streaming failed")
    
    # Mock agenerate to also fail
    adapter.agenerate = AsyncMock(side_effect=Exception("Fallback failed"))
    
    # Mock the openai module
    mock_openai = MagicMock()
    mock_openai.AsyncOpenAI.return_value = mock_client
    
    with patch.dict('sys.modules', {'openai': mock_openai}):
        with pytest.raises(StreamingError) as exc_info:
            async for chunk in adapter.stream("test prompt"):
                pass
        
        assert "Streaming failed" in str(exc_info.value)
        assert "Fallback also failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_openai_adapter_stream_missing_openai():
    """Test OpenAI adapter raises StreamingError when OpenAI package is missing."""
    adapter = OpenAIAdapter(api_key="test-key")
    
    # Mock import to raise ImportError for openai specifically
    original_import = __builtins__['__import__']
    
    def mock_import(name, *args, **kwargs):
        if name == 'openai':
            raise ImportError("No module named 'openai'")
        return original_import(name, *args, **kwargs)
    
    with patch('builtins.__import__', side_effect=mock_import):
        with pytest.raises(StreamingError) as exc_info:
            async for chunk in adapter.stream("test prompt"):
                pass
        
        assert "OpenAI package not installed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_openai_adapter_stream_with_tools():
    """Test OpenAI adapter streaming with available tools."""
    adapter = OpenAIAdapter(api_key="test-key")
    
    # Mock the OpenAI client and streaming response
    mock_chunk = Mock()
    mock_chunk.choices = [Mock()]
    mock_chunk.choices[0].delta.content = "Using tool"
    
    mock_stream = AsyncMock()
    mock_stream.__aiter__.return_value = [mock_chunk]
    
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_stream
    
    # Mock the openai module
    mock_openai = MagicMock()
    mock_openai.AsyncOpenAI.return_value = mock_client
    
    with patch.dict('sys.modules', {'openai': mock_openai}):
        chunks = []
        async for chunk in adapter.stream("test prompt", available_tools=["calculator", "search"]):
            chunks.append(chunk)
        
        # Verify tools were included in the prompt
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert "Available tools: calculator, search" in messages[0]["content"]
        assert "TOOL_CALL:" in messages[0]["content"]
        
        assert len(chunks) == 1
        assert chunks[0] == "Using tool"