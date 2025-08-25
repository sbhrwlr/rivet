"""Base model adapter interface."""

import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, AsyncIterator

from rivet.tools import ToolRegistry


class ModelAdapter(ABC):
    """Base class for all model adapters."""
    
    @abstractmethod
    def generate(self, prompt: str, available_tools: Optional[List[str]] = None) -> str:
        """Generate a response from the model."""
        pass
        
    @abstractmethod
    def configure(self, **kwargs) -> None:
        """Configure the model with parameters."""
        pass
    
    async def agenerate(self, prompt: str, available_tools: Optional[List[str]] = None, tool_registry: Optional[ToolRegistry] = None) -> str:
        """Async generation - default implementation wraps sync method."""
        return await asyncio.to_thread(self.generate, prompt, available_tools)
    
    async def stream(self, prompt: str, available_tools: Optional[List[str]] = None) -> AsyncIterator[str]:
        """Stream generation - default implementation yields complete response."""
        try:
            response = await self.agenerate(prompt, available_tools)
            yield response
        except Exception as e:
            from ..exceptions import StreamingError
            raise StreamingError(f"Streaming failed: {str(e)}") from e
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text - default implementation estimates by word count."""
        return int(len(text.split()) * 1.3)  # Rough estimate: ~1.3 tokens per word
    
    def get_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for token usage - default implementation returns 0."""
        return 0.0
    
    def get_last_usage(self) -> Optional[Dict[str, int]]:
        """Get token usage from the last API call - default implementation returns None."""
        return None