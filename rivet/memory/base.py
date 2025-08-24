"""Base memory adapter interface."""

import asyncio
from abc import ABC, abstractmethod
from typing import List, Any, Optional


class MemoryAdapter(ABC):
    """Base class for all memory adapters."""
    
    @abstractmethod
    def store(self, input_text: str, output_text: str, metadata: Optional[dict] = None) -> None:
        """Store an interaction in memory."""
        pass
        
    @abstractmethod
    def retrieve(self, query: str, limit: int = 5) -> List[str]:
        """Retrieve relevant memories based on query."""
        pass
        
    @abstractmethod
    def clear(self) -> None:
        """Clear all stored memories."""
        pass
    
    async def astore(self, input_text: str, output_text: str, metadata: Optional[dict] = None) -> None:
        """Async store - default implementation wraps sync method."""
        await asyncio.to_thread(self.store, input_text, output_text, metadata)
    
    async def aretrieve(self, query: str, limit: int = 5) -> List[str]:
        """Async retrieve - default implementation wraps sync method."""
        return await asyncio.to_thread(self.retrieve, query, limit)
    
    async def aclear(self) -> None:
        """Async clear - default implementation wraps sync method."""
        await asyncio.to_thread(self.clear)