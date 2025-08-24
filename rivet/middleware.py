"""
Middleware system for Rivet agents.

Provides request/response processing pipeline for agents.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class Middleware(ABC):
    """
    Abstract base class for middleware components.
    
    Middleware can process requests before execution and responses after execution.
    """
    
    @abstractmethod
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process request before agent execution.
        
        Args:
            request: Request data containing message, context, etc.
            
        Returns:
            Modified request data
        """
        pass
    
    @abstractmethod
    async def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process response after agent execution.
        
        Args:
            response: Response data containing result, metadata, etc.
            
        Returns:
            Modified response data
        """
        pass


class MiddlewareChain:
    """
    Manages and executes a chain of middleware components.
    """
    
    def __init__(self, middleware: Optional[List[Middleware]] = None):
        """
        Initialize middleware chain.
        
        Args:
            middleware: List of middleware instances
        """
        self.middleware = middleware or []
        self._logger = logger.getChild("MiddlewareChain")
    
    def add_middleware(self, middleware: Middleware) -> None:
        """
        Add middleware to the chain.
        
        Args:
            middleware: Middleware instance to add
        """
        self.middleware.append(middleware)
        self._logger.debug(f"Added middleware: {middleware.__class__.__name__}")
    
    def remove_middleware(self, middleware: Middleware) -> bool:
        """
        Remove middleware from the chain.
        
        Args:
            middleware: Middleware instance to remove
            
        Returns:
            True if middleware was found and removed
        """
        try:
            self.middleware.remove(middleware)
            self._logger.debug(f"Removed middleware: {middleware.__class__.__name__}")
            return True
        except ValueError:
            return False
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process request through all middleware in order.
        
        Args:
            request: Initial request data
            
        Returns:
            Processed request data
        """
        current_request = request
        
        for middleware in self.middleware:
            try:
                current_request = await middleware.process_request(current_request)
            except Exception as e:
                self._logger.error(f"Error in middleware {middleware.__class__.__name__}.process_request: {e}")
                # Continue with unmodified request on error
        
        return current_request
    
    async def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process response through all middleware in reverse order.
        
        Args:
            response: Initial response data
            
        Returns:
            Processed response data
        """
        current_response = response
        
        # Process in reverse order for response
        for middleware in reversed(self.middleware):
            try:
                current_response = await middleware.process_response(current_response)
            except Exception as e:
                self._logger.error(f"Error in middleware {middleware.__class__.__name__}.process_response: {e}")
                # Continue with unmodified response on error
        
        return current_response
    
    def clear(self) -> None:
        """Clear all middleware from the chain."""
        self.middleware.clear()
        self._logger.debug("Cleared all middleware")
    
    def __len__(self) -> int:
        """Return number of middleware in the chain."""
        return len(self.middleware)


class LoggingMiddleware(Middleware):
    """
    Middleware that logs requests and responses.
    """
    
    def __init__(self, log_level: int = logging.INFO, include_data: bool = True):
        """
        Initialize logging middleware.
        
        Args:
            log_level: Logging level to use
            include_data: Whether to include request/response data in logs
        """
        self.log_level = log_level
        self.include_data = include_data
        self._logger = logger.getChild("LoggingMiddleware")
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Log incoming request."""
        if self.include_data:
            self._logger.log(self.log_level, f"Processing request: {request}")
        else:
            self._logger.log(self.log_level, "Processing request")
        
        return request
    
    async def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Log outgoing response."""
        if self.include_data:
            self._logger.log(self.log_level, f"Processing response: {response}")
        else:
            self._logger.log(self.log_level, "Processing response")
        
        return response


class TimingMiddleware(Middleware):
    """
    Middleware that tracks execution timing.
    """
    
    def __init__(self, store_in_response: bool = True):
        """
        Initialize timing middleware.
        
        Args:
            store_in_response: Whether to store timing data in response
        """
        self.store_in_response = store_in_response
        self._logger = logger.getChild("TimingMiddleware")
        self._start_times: Dict[str, float] = {}
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Record start time for request."""
        request_id = id(request)
        self._start_times[request_id] = time.time()
        
        # Add timestamp to request
        request["_middleware_start_time"] = datetime.now().isoformat()
        
        return request
    
    async def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate and log execution time."""
        # Try to find matching request ID from response metadata
        request_id = response.get("_request_id", id(response))
        
        if request_id in self._start_times:
            execution_time = time.time() - self._start_times[request_id]
            del self._start_times[request_id]
        else:
            execution_time = None
        
        if execution_time is not None:
            self._logger.info(f"Request execution time: {execution_time:.3f}s")
            
            if self.store_in_response:
                if "_metadata" not in response:
                    response["_metadata"] = {}
                response["_metadata"]["execution_time"] = execution_time
        
        # Add end timestamp
        response["_middleware_end_time"] = datetime.now().isoformat()
        
        return response


class ValidationMiddleware(Middleware):
    """
    Middleware that validates requests and responses.
    """
    
    def __init__(self, 
                 required_request_fields: Optional[List[str]] = None,
                 required_response_fields: Optional[List[str]] = None):
        """
        Initialize validation middleware.
        
        Args:
            required_request_fields: Fields that must be present in requests
            required_response_fields: Fields that must be present in responses
        """
        self.required_request_fields = required_request_fields or []
        self.required_response_fields = required_response_fields or []
        self._logger = logger.getChild("ValidationMiddleware")
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate request structure."""
        missing_fields = []
        for field in self.required_request_fields:
            if field not in request:
                missing_fields.append(field)
        
        if missing_fields:
            error_msg = f"Missing required request fields: {missing_fields}"
            self._logger.error(error_msg)
            raise ValueError(error_msg)
        
        return request
    
    async def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate response structure."""
        missing_fields = []
        for field in self.required_response_fields:
            if field not in response:
                missing_fields.append(field)
        
        if missing_fields:
            error_msg = f"Missing required response fields: {missing_fields}"
            self._logger.error(error_msg)
            raise ValueError(error_msg)
        
        return response


class RateLimitMiddleware(Middleware):
    """
    Middleware that implements rate limiting.
    """
    
    def __init__(self, max_requests_per_minute: int = 60):
        """
        Initialize rate limiting middleware.
        
        Args:
            max_requests_per_minute: Maximum requests allowed per minute
        """
        self.max_requests_per_minute = max_requests_per_minute
        self.request_times: List[float] = []
        self._logger = logger.getChild("RateLimitMiddleware")
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Check rate limits before processing request."""
        current_time = time.time()
        
        # Remove requests older than 1 minute
        cutoff_time = current_time - 60
        self.request_times = [t for t in self.request_times if t > cutoff_time]
        
        # Check if we're at the limit
        if len(self.request_times) >= self.max_requests_per_minute:
            error_msg = f"Rate limit exceeded: {self.max_requests_per_minute} requests per minute"
            self._logger.warning(error_msg)
            raise ValueError(error_msg)
        
        # Record this request
        self.request_times.append(current_time)
        
        return request
    
    async def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Add rate limit info to response."""
        if "_metadata" not in response:
            response["_metadata"] = {}
        
        response["_metadata"]["rate_limit"] = {
            "max_requests_per_minute": self.max_requests_per_minute,
            "current_requests": len(self.request_times)
        }
        
        return response