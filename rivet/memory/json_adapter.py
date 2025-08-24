"""JSON file memory adapter for simple storage."""

import json
import os
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from .base import MemoryAdapter

try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False


class JSONAdapter(MemoryAdapter):
    """JSON file-based memory storage."""
    
    def __init__(self, file_path: str = "rivet_memory.json"):
        self.file_path = file_path
        self.memories: List[Dict[str, Any]] = []
        self._load_memories()
        
    def _load_memories(self) -> None:
        """Load memories from JSON file."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    self.memories = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self.memories = []
                
    def _save_memories(self) -> None:
        """Save memories to JSON file."""
        with open(self.file_path, 'w') as f:
            json.dump(self.memories, f, indent=2)
            
    def store(self, input_text: str, output_text: str, metadata: Optional[dict] = None) -> None:
        """Store interaction in JSON file."""
        memory = {
            "input_text": input_text,
            "output_text": output_text,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.memories.append(memory)
        self._save_memories()
        
    def retrieve(self, query: str, limit: int = 5) -> List[str]:
        """Retrieve memories matching the query."""
        query_lower = query.lower()
        matching_memories = []
        
        for memory in reversed(self.memories):  # Most recent first
            if (query_lower in memory["input_text"].lower() or 
                query_lower in memory["output_text"].lower()):
                matching_memories.append(
                    f"Q: {memory['input_text']}\nA: {memory['output_text']}"
                )
                
            if len(matching_memories) >= limit:
                break
                
        return matching_memories
        
    def clear(self) -> None:
        """Clear all memories."""
        self.memories = []
        self._save_memories()
    
    async def _aload_memories(self) -> None:
        """Load memories from JSON file asynchronously."""
        if os.path.exists(self.file_path):
            try:
                if HAS_AIOFILES:
                    async with aiofiles.open(self.file_path, 'r') as f:
                        content = await f.read()
                        self.memories = json.loads(content)
                else:
                    await asyncio.to_thread(self._load_memories)
            except (json.JSONDecodeError, FileNotFoundError):
                self.memories = []
    
    async def _asave_memories(self) -> None:
        """Save memories to JSON file asynchronously."""
        if HAS_AIOFILES:
            try:
                async with aiofiles.open(self.file_path, 'w') as f:
                    await f.write(json.dumps(self.memories, indent=2))
            except Exception:
                # Fallback to sync if aiofiles fails
                await asyncio.to_thread(self._save_memories)
        else:
            # Fallback to sync if aiofiles not available
            await asyncio.to_thread(self._save_memories)
    
    async def astore(self, input_text: str, output_text: str, metadata: Optional[dict] = None) -> None:
        """Store interaction in JSON file asynchronously."""
        memory = {
            "input_text": input_text,
            "output_text": output_text,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.memories.append(memory)
        await self._asave_memories()
    
    async def aretrieve(self, query: str, limit: int = 5) -> List[str]:
        """Retrieve memories matching the query asynchronously."""
        # For JSON adapter, this is CPU-bound so we use thread pool
        return await asyncio.to_thread(self.retrieve, query, limit)
    
    async def aclear(self) -> None:
        """Clear all memories asynchronously."""
        self.memories = []
        await self._asave_memories()