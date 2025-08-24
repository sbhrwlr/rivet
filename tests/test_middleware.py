"""
Tests for the middleware system.
"""

import asyncio
import time
import pytest
from unittest.mock import Mock, patch
from rivet.middleware import (
    Middleware, MiddlewareChain, LoggingMiddleware, TimingMiddleware,
    ValidationMiddleware, RateLimitMiddleware
)


class MockMiddleware(Middleware):
    """Mock middleware for testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.process_request_called = False
        self.process_response_called = False
    
    async def process_request(self, request):
        self.process_request_called = True
        request[f"{self.name}_request"] = True
        return request
    
    async def process_response(self, response):
        self.process_response_called = True
        response[f"{self.name}_response"] = True
        return response


class ErrorMiddleware(Middleware):
    """Middleware that raises errors for testing."""
    
    def __init__(self, error_on_request=False, error_on_response=False):
        self.error_on_request = error_on_request
        self.error_on_response = error_on_response
    
    async def process_request(self, request):
        if self.error_on_request:
            raise ValueError("Request error")
        return request
    
    async def process_response(self, response):
        if self.error_on_response:
            raise ValueError("Response error")
        return response


class TestMiddlewareChain:
    """Test MiddlewareChain functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.chain = MiddlewareChain()
        self.middleware1 = MockMiddleware("middleware1")
        self.middleware2 = MockMiddleware("middleware2")
    
    def test_add_middleware(self):
        """Test adding middleware to chain."""
        self.chain.add_middleware(self.middleware1)
        assert len(self.chain) == 1
        assert self.middleware1 in self.chain.middleware
    
    def test_remove_middleware(self):
        """Test removing middleware from chain."""
        self.chain.add_middleware(self.middleware1)
        
        result = self.chain.remove_middleware(self.middleware1)
        assert result is True
        assert len(self.chain) == 0
        assert self.middleware1 not in self.chain.middleware
    
    def test_remove_nonexistent_middleware(self):
        """Test removing middleware that doesn't exist."""
        result = self.chain.remove_middleware(self.middleware1)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_process_request_single_middleware(self):
        """Test processing request through single middleware."""
        self.chain.add_middleware(self.middleware1)
        
        request = {"message": "test"}
        result = await self.chain.process_request(request)
        
        assert self.middleware1.process_request_called
        assert result["middleware1_request"] is True
        assert result["message"] == "test"
    
    @pytest.mark.asyncio
    async def test_process_request_multiple_middleware(self):
        """Test processing request through multiple middleware."""
        self.chain.add_middleware(self.middleware1)
        self.chain.add_middleware(self.middleware2)
        
        request = {"message": "test"}
        result = await self.chain.process_request(request)
        
        assert self.middleware1.process_request_called
        assert self.middleware2.process_request_called
        assert result["middleware1_request"] is True
        assert result["middleware2_request"] is True
    
    @pytest.mark.asyncio
    async def test_process_response_single_middleware(self):
        """Test processing response through single middleware."""
        self.chain.add_middleware(self.middleware1)
        
        response = {"result": "test"}
        result = await self.chain.process_response(response)
        
        assert self.middleware1.process_response_called
        assert result["middleware1_response"] is True
        assert result["result"] == "test"
    
    @pytest.mark.asyncio
    async def test_process_response_multiple_middleware_reverse_order(self):
        """Test that responses are processed in reverse order."""
        execution_order = []
        
        class OrderTrackingMiddleware(Middleware):
            def __init__(self, name):
                self.name = name
            
            async def process_request(self, request):
                return request
            
            async def process_response(self, response):
                execution_order.append(self.name)
                return response
        
        middleware1 = OrderTrackingMiddleware("first")
        middleware2 = OrderTrackingMiddleware("second")
        
        self.chain.add_middleware(middleware1)
        self.chain.add_middleware(middleware2)
        
        await self.chain.process_response({"result": "test"})
        
        # Should process in reverse order for responses
        assert execution_order == ["second", "first"]
    
    @pytest.mark.asyncio
    async def test_middleware_error_handling_request(self):
        """Test that middleware errors don't break the chain."""
        error_middleware = ErrorMiddleware(error_on_request=True)
        
        self.chain.add_middleware(error_middleware)
        self.chain.add_middleware(self.middleware1)
        
        request = {"message": "test"}
        result = await self.chain.process_request(request)
        
        # Should continue processing despite error
        assert self.middleware1.process_request_called
        assert result["middleware1_request"] is True
    
    @pytest.mark.asyncio
    async def test_middleware_error_handling_response(self):
        """Test that middleware errors don't break the response chain."""
        error_middleware = ErrorMiddleware(error_on_response=True)
        
        self.chain.add_middleware(self.middleware1)
        self.chain.add_middleware(error_middleware)
        
        response = {"result": "test"}
        result = await self.chain.process_response(response)
        
        # Should continue processing despite error
        assert self.middleware1.process_response_called
        assert result["middleware1_response"] is True
    
    def test_clear_middleware(self):
        """Test clearing all middleware."""
        self.chain.add_middleware(self.middleware1)
        self.chain.add_middleware(self.middleware2)
        
        self.chain.clear()
        assert len(self.chain) == 0
    
    @pytest.mark.asyncio
    async def test_empty_chain(self):
        """Test processing with empty middleware chain."""
        request = {"message": "test"}
        response = {"result": "test"}
        
        request_result = await self.chain.process_request(request)
        response_result = await self.chain.process_response(response)
        
        assert request_result == request
        assert response_result == response


class TestLoggingMiddleware:
    """Test LoggingMiddleware functionality."""
    
    @pytest.mark.asyncio
    async def test_logging_middleware_with_data(self):
        """Test logging middleware that includes data."""
        with patch('rivet.middleware.logger') as mock_logger:
            middleware = LoggingMiddleware(include_data=True)
            
            request = {"message": "test"}
            response = {"result": "test"}
            
            await middleware.process_request(request)
            await middleware.process_response(response)
            
            # Check that logging was called
            assert mock_logger.getChild.called
    
    @pytest.mark.asyncio
    async def test_logging_middleware_without_data(self):
        """Test logging middleware that excludes data."""
        with patch('rivet.middleware.logger') as mock_logger:
            middleware = LoggingMiddleware(include_data=False)
            
            request = {"message": "test"}
            response = {"result": "test"}
            
            await middleware.process_request(request)
            await middleware.process_response(response)
            
            # Check that logging was called
            assert mock_logger.getChild.called


class TestTimingMiddleware:
    """Test TimingMiddleware functionality."""
    
    @pytest.mark.asyncio
    async def test_timing_middleware(self):
        """Test timing middleware functionality."""
        middleware = TimingMiddleware(store_in_response=True)
        
        request = {"message": "test"}
        processed_request = await middleware.process_request(request)
        
        # Simulate some processing time
        await asyncio.sleep(0.01)
        
        response = {"result": "test", "_request_id": id(processed_request)}
        processed_response = await middleware.process_response(response)
        
        # Check that timing data was added
        assert "_middleware_start_time" in processed_request
        assert "_middleware_end_time" in processed_response
        assert "_metadata" in processed_response
        assert "execution_time" in processed_response["_metadata"]
        assert processed_response["_metadata"]["execution_time"] > 0
    
    @pytest.mark.asyncio
    async def test_timing_middleware_without_storage(self):
        """Test timing middleware without storing in response."""
        middleware = TimingMiddleware(store_in_response=False)
        
        request = {"message": "test"}
        response = {"result": "test"}
        
        await middleware.process_request(request)
        processed_response = await middleware.process_response(response)
        
        # Should not store timing data in response
        assert "_metadata" not in processed_response or "execution_time" not in processed_response.get("_metadata", {})


class TestValidationMiddleware:
    """Test ValidationMiddleware functionality."""
    
    @pytest.mark.asyncio
    async def test_validation_middleware_valid_request(self):
        """Test validation middleware with valid request."""
        middleware = ValidationMiddleware(required_request_fields=["message"])
        
        request = {"message": "test", "context": {}}
        result = await middleware.process_request(request)
        
        assert result == request
    
    @pytest.mark.asyncio
    async def test_validation_middleware_invalid_request(self):
        """Test validation middleware with invalid request."""
        middleware = ValidationMiddleware(required_request_fields=["message", "context"])
        
        request = {"message": "test"}
        
        with pytest.raises(ValueError, match="Missing required request fields"):
            await middleware.process_request(request)
    
    @pytest.mark.asyncio
    async def test_validation_middleware_valid_response(self):
        """Test validation middleware with valid response."""
        middleware = ValidationMiddleware(required_response_fields=["result"])
        
        response = {"result": "test", "metadata": {}}
        result = await middleware.process_response(response)
        
        assert result == response
    
    @pytest.mark.asyncio
    async def test_validation_middleware_invalid_response(self):
        """Test validation middleware with invalid response."""
        middleware = ValidationMiddleware(required_response_fields=["result", "metadata"])
        
        response = {"result": "test"}
        
        with pytest.raises(ValueError, match="Missing required response fields"):
            await middleware.process_response(response)


class TestRateLimitMiddleware:
    """Test RateLimitMiddleware functionality."""
    
    @pytest.mark.asyncio
    async def test_rate_limit_middleware_under_limit(self):
        """Test rate limiting when under the limit."""
        middleware = RateLimitMiddleware(max_requests_per_minute=10)
        
        request = {"message": "test"}
        result = await middleware.process_request(request)
        
        assert result == request
        assert len(middleware.request_times) == 1
    
    @pytest.mark.asyncio
    async def test_rate_limit_middleware_at_limit(self):
        """Test rate limiting when at the limit."""
        middleware = RateLimitMiddleware(max_requests_per_minute=2)
        
        # Make requests up to the limit
        request = {"message": "test"}
        await middleware.process_request(request)
        await middleware.process_request(request)
        
        # Next request should fail
        with pytest.raises(ValueError, match="Rate limit exceeded"):
            await middleware.process_request(request)
    
    @pytest.mark.asyncio
    async def test_rate_limit_middleware_response_metadata(self):
        """Test that rate limit info is added to response."""
        middleware = RateLimitMiddleware(max_requests_per_minute=10)
        
        # Make a request first
        await middleware.process_request({"message": "test"})
        
        response = {"result": "test"}
        result = await middleware.process_response(response)
        
        assert "_metadata" in result
        assert "rate_limit" in result["_metadata"]
        assert result["_metadata"]["rate_limit"]["max_requests_per_minute"] == 10
        assert result["_metadata"]["rate_limit"]["current_requests"] == 1
    
    @pytest.mark.asyncio
    async def test_rate_limit_middleware_time_window(self):
        """Test that old requests are cleaned up."""
        middleware = RateLimitMiddleware(max_requests_per_minute=2)
        
        # Manually add old request times
        current_time = time.time()
        middleware.request_times = [current_time - 120, current_time - 90]  # 2 and 1.5 minutes ago
        
        # Should be able to make requests since old ones are cleaned up
        request = {"message": "test"}
        result = await middleware.process_request(request)
        
        assert result == request
        assert len(middleware.request_times) == 1  # Only the new request


class TestMiddlewareIntegration:
    """Test middleware integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_middleware_chain(self):
        """Test a complete middleware chain with multiple types."""
        chain = MiddlewareChain()
        
        # Add various middleware
        chain.add_middleware(ValidationMiddleware(required_request_fields=["message"]))
        chain.add_middleware(TimingMiddleware())
        chain.add_middleware(LoggingMiddleware(include_data=False))
        
        request = {"message": "test"}
        response = {"result": "test", "_request_id": id(request)}
        
        # Process request and response
        processed_request = await chain.process_request(request)
        processed_response = await chain.process_response(response)
        
        # Verify all middleware processed the data
        assert processed_request["message"] == "test"
        assert "_middleware_start_time" in processed_request
        assert processed_response["result"] == "test"
        assert "_middleware_end_time" in processed_response
    
    @pytest.mark.asyncio
    async def test_middleware_chain_with_errors(self):
        """Test middleware chain continues despite errors."""
        chain = MiddlewareChain()
        
        # Add middleware that will error and one that should still work
        chain.add_middleware(ErrorMiddleware(error_on_request=True))
        chain.add_middleware(MockMiddleware("working"))
        
        request = {"message": "test"}
        result = await chain.process_request(request)
        
        # Working middleware should still process despite error
        assert result["working_request"] is True