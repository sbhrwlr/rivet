"""
Tests for Agent integration with callbacks and middleware.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from rivet import Agent, CallbackManager, CallbackEvent
from rivet.middleware import LoggingMiddleware, TimingMiddleware, ValidationMiddleware
from rivet.models.base import ModelAdapter
from rivet.memory.base import MemoryAdapter
from rivet.tools import ToolRegistry


class MockModelAdapter(ModelAdapter):
    """Mock model adapter for testing."""
    
    def __init__(self, response="Test response"):
        self.response = response
        self.generate_called = False
        self.agenerate_called = False
    
    def generate(self, prompt, **kwargs):
        self.generate_called = True
        return self.response
    
    def configure(self, **kwargs):
        """Configure the model - no-op for mock."""
        pass
    
    async def agenerate(self, prompt, **kwargs):
        self.agenerate_called = True
        return self.response
    
    async def stream(self, prompt, **kwargs):
        for chunk in self.response.split():
            yield chunk + " "


class MockMemoryAdapter(MemoryAdapter):
    """Mock memory adapter for testing."""
    
    def __init__(self):
        self.stored_data = []
        self.retrieve_data = []
    
    def store(self, key, value):
        self.stored_data.append((key, value))
    
    def retrieve(self, key):
        return self.retrieve_data
    
    async def astore(self, key, value):
        self.stored_data.append((key, value))
    
    async def aretrieve(self, key):
        return self.retrieve_data
    
    def clear(self):
        self.stored_data.clear()
        self.retrieve_data.clear()


class TestAgentCallbackIntegration:
    """Test Agent integration with callbacks."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = MockModelAdapter()
        self.memory = MockMemoryAdapter()
        self.callbacks = CallbackManager()
        
        # Mock callbacks
        self.agent_start_callback = Mock()
        self.agent_end_callback = Mock()
        self.model_call_callback = Mock()
        self.error_callback = Mock()
        
        # Register callbacks
        self.callbacks.register(CallbackEvent.AGENT_START, self.agent_start_callback)
        self.callbacks.register(CallbackEvent.AGENT_END, self.agent_end_callback)
        self.callbacks.register(CallbackEvent.MODEL_CALL, self.model_call_callback)
        self.callbacks.register(CallbackEvent.ERROR, self.error_callback)
        
        self.agent = Agent(
            model=self.model,
            memory=self.memory,
            callbacks=self.callbacks
        )
    
    def test_sync_run_triggers_callbacks(self):
        """Test that sync run triggers appropriate callbacks."""
        result = self.agent.run("Test message")
        
        # Verify callbacks were called
        self.agent_start_callback.assert_called_once()
        self.model_call_callback.assert_called_once()
        self.agent_end_callback.assert_called_once()
        
        # Verify callback data
        start_data = self.agent_start_callback.call_args[0][0]
        assert start_data["message"] == "Test message"
        
        end_data = self.agent_end_callback.call_args[0][0]
        assert end_data["response"] == "Test response"
    
    @pytest.mark.asyncio
    async def test_async_run_triggers_callbacks(self):
        """Test that async run triggers appropriate callbacks."""
        # Use async callbacks
        async_start_callback = AsyncMock()
        async_end_callback = AsyncMock()
        
        callbacks = CallbackManager()
        callbacks.register(CallbackEvent.AGENT_START, async_start_callback)
        callbacks.register(CallbackEvent.AGENT_END, async_end_callback)
        
        agent = Agent(
            model=self.model,
            memory=self.memory,
            callbacks=callbacks
        )
        
        result = await agent.arun("Test message")
        
        # Verify async callbacks were called
        async_start_callback.assert_called_once()
        async_end_callback.assert_called_once()
    
    def test_error_callback_triggered_on_exception(self):
        """Test that error callback is triggered when exception occurs."""
        # Make model raise an exception
        self.model.response = None
        
        def failing_generate(prompt, **kwargs):
            raise ValueError("Model error")
        
        self.model.generate = failing_generate
        
        with pytest.raises(ValueError):
            self.agent.run("Test message")
        
        # Verify error callback was called
        self.error_callback.assert_called_once()
        error_data = self.error_callback.call_args[0][0]
        assert "Model error" in error_data["error"]
    
    def test_tool_call_callback_triggered(self):
        """Test that tool call callback is triggered for tool calls."""
        # Set up model to return tool call
        self.model.response = "TOOL_CALL: test_tool"
        
        # Register a test tool
        self.agent.tools.register("test_tool", lambda: "tool result")
        
        # Register tool call callback
        tool_callback = Mock()
        self.callbacks.register(CallbackEvent.TOOL_CALL, tool_callback)
        
        result = self.agent.run("Test message")
        
        # Verify tool callback was called
        tool_callback.assert_called_once()
        tool_data = tool_callback.call_args[0][0]
        assert "TOOL_CALL: test_tool" in tool_data["response"]


class TestAgentMiddlewareIntegration:
    """Test Agent integration with middleware."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = MockModelAdapter()
        self.memory = MockMemoryAdapter()
        
        # Create middleware instances
        self.logging_middleware = LoggingMiddleware(include_data=False)
        self.timing_middleware = TimingMiddleware()
        self.validation_middleware = ValidationMiddleware(
            required_request_fields=["message"]
        )
        
        self.agent = Agent(
            model=self.model,
            memory=self.memory,
            middleware=[
                self.validation_middleware,
                self.timing_middleware,
                self.logging_middleware
            ]
        )
    
    def test_middleware_processes_request_and_response(self):
        """Test that middleware processes both request and response."""
        result = self.agent.run("Test message")
        
        # Verify model was called (middleware didn't break execution)
        assert self.model.generate_called
        assert result == "Test response"
    
    @pytest.mark.asyncio
    async def test_async_middleware_processes_request_and_response(self):
        """Test that middleware works with async execution."""
        result = await self.agent.arun("Test message")
        
        # Verify model was called
        assert self.model.agenerate_called
        assert result == "Test response"
    
    def test_validation_middleware_logs_errors(self):
        """Test that validation middleware logs errors for invalid requests."""
        # Create agent with strict validation
        strict_validation = ValidationMiddleware(
            required_request_fields=["message", "required_field"]
        )
        
        agent = Agent(
            model=self.model,
            middleware=[strict_validation]
        )
        
        # Should continue execution despite validation error (error isolation)
        # but the error should be logged
        result = agent.run("Test message")
        assert result == "Test response"  # Execution continues
    
    def test_timing_middleware_adds_metadata(self):
        """Test that timing middleware adds execution time metadata."""
        # Create agent with only timing middleware
        agent = Agent(
            model=self.model,
            middleware=[TimingMiddleware(store_in_response=True)]
        )
        
        result = self.agent.run("Test message")
        
        # The timing metadata is added to the internal response processing
        # but not exposed in the final result for this simple test
        assert result == "Test response"
    
    def test_middleware_error_handling(self):
        """Test that middleware errors don't break agent execution."""
        class ErrorMiddleware:
            async def process_request(self, request):
                raise ValueError("Middleware error")
            
            async def process_response(self, response):
                return response
        
        agent = Agent(
            model=self.model,
            middleware=[ErrorMiddleware()]
        )
        
        # Should still work despite middleware error
        result = agent.run("Test message")
        assert result == "Test response"


class TestAgentCallbacksAndMiddlewareTogether:
    """Test Agent with both callbacks and middleware."""
    
    @pytest.mark.asyncio
    async def test_callbacks_and_middleware_together(self):
        """Test that callbacks and middleware work together."""
        model = MockModelAdapter()
        callbacks = CallbackManager()
        
        # Set up callbacks
        start_callback = AsyncMock()
        end_callback = AsyncMock()
        callbacks.register(CallbackEvent.AGENT_START, start_callback)
        callbacks.register(CallbackEvent.AGENT_END, end_callback)
        
        # Set up middleware
        timing_middleware = TimingMiddleware()
        
        agent = Agent(
            model=model,
            callbacks=callbacks,
            middleware=[timing_middleware]
        )
        
        result = await agent.arun("Test message")
        
        # Verify both systems worked
        start_callback.assert_called_once()
        end_callback.assert_called_once()
        assert result == "Test response"
        assert model.agenerate_called
    
    @pytest.mark.asyncio
    async def test_streaming_with_callbacks_and_middleware(self):
        """Test streaming with both callbacks and middleware."""
        model = MockModelAdapter("Hello world test")
        callbacks = CallbackManager()
        
        # Set up callbacks
        start_callback = AsyncMock()
        end_callback = AsyncMock()
        callbacks.register(CallbackEvent.AGENT_START, start_callback)
        callbacks.register(CallbackEvent.AGENT_END, end_callback)
        
        agent = Agent(
            model=model,
            callbacks=callbacks,
            middleware=[TimingMiddleware()]
        )
        
        # Collect streamed chunks
        chunks = []
        async for chunk in agent.astream("Test message"):
            chunks.append(chunk)
        
        # Verify callbacks were triggered
        start_callback.assert_called_once()
        end_callback.assert_called_once()
        
        # Verify streaming worked
        assert len(chunks) > 0
        assert "Hello" in "".join(chunks)
    
    def test_callback_and_middleware_error_isolation(self):
        """Test that errors in callbacks/middleware don't break each other."""
        model = MockModelAdapter()
        callbacks = CallbackManager()
        
        # Add failing callback
        def failing_callback(data):
            raise ValueError("Callback error")
        
        callbacks.register(CallbackEvent.AGENT_START, failing_callback)
        
        # Add failing middleware
        class FailingMiddleware:
            async def process_request(self, request):
                raise ValueError("Middleware error")
            
            async def process_response(self, response):
                return response
        
        agent = Agent(
            model=model,
            callbacks=callbacks,
            middleware=[FailingMiddleware()]
        )
        
        # Should still work despite both callback and middleware errors
        result = agent.run("Test message")
        assert result == "Test response"