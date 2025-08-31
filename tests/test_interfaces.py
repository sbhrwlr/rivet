"""Tests for abstract interfaces."""

import pytest
from typing import List, Dict, Any, AsyncIterator, Optional
from flowt.core.interfaces import LLMProvider, VectorStore
from flowt.core.models import LLMResponse, Vector, SearchResult, GenerationConfig


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    async def generate(self, prompt: str, config: GenerationConfig) -> LLMResponse:
        return LLMResponse(
            content=f"Generated response for: {prompt}",
            model=config.model,
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        )

    def stream_generate(self, prompt: str, config: GenerationConfig) -> AsyncIterator[str]:
        async def _generate() -> AsyncIterator[str]:
            tokens = ["Generated", " response", " for:", f" {prompt}"]
            for token in tokens:
                yield token
        return _generate()

    async def embed(self, texts: List[str]) -> List[List[float]]:
        # Return mock embeddings (3-dimensional for simplicity)
        return [[0.1, 0.2, 0.3] for _ in texts]


class MockVectorStore(VectorStore):
    """Mock vector store for testing."""

    def __init__(self) -> None:
        self.vectors: Dict[str, Vector] = {}

    async def upsert(self, vectors: List[Vector], collection_name: Optional[str] = None) -> None:
        for vector in vectors:
            self.vectors[vector.id] = vector

    async def search(
        self,
        query_vector: List[float],
        limit: int,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> List[SearchResult]:
        # Simple mock search - return first `limit` vectors
        results = []
        for i, (vector_id, vector) in enumerate(self.vectors.items()):
            if i >= limit:
                break
            results.append(SearchResult(
                content=f"Content for {vector_id}",
                score=0.9 - (i * 0.1),  # Decreasing scores
                vector_id=vector_id,
                metadata=vector.metadata
            ))
        return results

    async def delete(self, vector_ids: List[str], collection_name: Optional[str] = None) -> None:
        for vector_id in vector_ids:
            self.vectors.pop(vector_id, None)

    async def create_collection(
        self, 
        collection_name: str, 
        dimension: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        # Mock implementation - just pass
        pass

    async def delete_collection(self, collection_name: str) -> None:
        # Mock implementation - clear all vectors
        self.vectors.clear()


class TestLLMProvider:
    """Tests for LLM provider interface."""

    @pytest.mark.asyncio
    async def test_generate(self) -> None:
        """Test text generation."""
        provider = MockLLMProvider()
        config = GenerationConfig(model="test-model", temperature=0.5)
        
        response = await provider.generate("Hello", config)
        
        assert response.content == "Generated response for: Hello"
        assert response.model == "test-model"
        assert response.finish_reason == "stop"
        assert response.usage["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_stream_generate(self) -> None:
        """Test streaming text generation."""
        provider = MockLLMProvider()
        config = GenerationConfig()
        
        tokens = []
        async for token in provider.stream_generate("Hello", config):
            tokens.append(token)
        
        assert tokens == ["Generated", " response", " for:", " Hello"]

    @pytest.mark.asyncio
    async def test_embed(self) -> None:
        """Test text embedding."""
        provider = MockLLMProvider()
        texts = ["Hello", "World"]
        
        embeddings = await provider.embed(texts)
        
        assert len(embeddings) == 2
        assert all(len(emb) == 3 for emb in embeddings)
        assert embeddings[0] == [0.1, 0.2, 0.3]


class TestVectorStore:
    """Tests for vector store interface."""

    @pytest.mark.asyncio
    async def test_upsert_and_search(self) -> None:
        """Test vector upsert and search."""
        store = MockVectorStore()
        
        # Upsert some vectors
        vectors = [
            Vector(id="vec1", embedding=[0.1, 0.2, 0.3], metadata={"type": "test"}),
            Vector(id="vec2", embedding=[0.4, 0.5, 0.6], metadata={"type": "test"}),
        ]
        await store.upsert(vectors)
        
        # Search for vectors
        query_vector = [0.1, 0.2, 0.3]
        results = await store.search(query_vector, limit=2)
        
        assert len(results) == 2
        assert results[0].vector_id == "vec1"
        assert results[0].score == 0.9
        assert results[1].vector_id == "vec2"
        assert results[1].score == 0.8

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        """Test vector deletion."""
        store = MockVectorStore()
        
        # Upsert a vector
        vector = Vector(id="vec1", embedding=[0.1, 0.2, 0.3])
        await store.upsert([vector])
        
        # Verify it exists
        results = await store.search([0.1, 0.2, 0.3], limit=1)
        assert len(results) == 1
        
        # Delete it
        await store.delete(["vec1"])
        
        # Verify it's gone
        results = await store.search([0.1, 0.2, 0.3], limit=1)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_collection_management(self) -> None:
        """Test collection creation and deletion."""
        store = MockVectorStore()
        
        # These are just mock implementations, so we test they don't raise errors
        await store.create_collection("test_collection", dimension=3)
        await store.delete_collection("test_collection")