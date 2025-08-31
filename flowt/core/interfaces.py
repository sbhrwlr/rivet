"""Abstract base classes for LLM providers and vector stores."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Optional

from .models import GenerationConfig, LLMResponse, SearchResult, Vector


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(self, prompt: str, config: GenerationConfig) -> LLMResponse:
        """Generate text completion for the given prompt.

        Args:
            prompt: The input prompt text
            config: Generation configuration parameters

        Returns:
            LLMResponse containing the generated text and metadata
        """
        pass

    @abstractmethod
    def stream_generate(
        self, prompt: str, config: GenerationConfig
    ) -> AsyncIterator[str]:
        """Generate streaming text completion for the given prompt.

        Args:
            prompt: The input prompt text
            config: Generation configuration parameters

        Yields:
            Individual tokens or text chunks as they are generated
        """
        pass

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for the given texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (one per input text)
        """
        pass


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    async def upsert(
        self,
        vectors: list[Vector],
        collection_name: Optional[str] = None
    ) -> None:
        """Insert or update vectors in the store.

        Args:
            vectors: List of vectors to upsert
            collection_name: Optional collection name for multi-tenant stores
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        limit: int,
        filters: Optional[dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> list[SearchResult]:
        """Search for similar vectors.

        Args:
            query_vector: The query embedding vector
            limit: Maximum number of results to return
            filters: Optional metadata filters
            collection_name: Optional collection name for multi-tenant stores

        Returns:
            List of search results ordered by similarity score
        """
        pass

    @abstractmethod
    async def delete(
        self,
        vector_ids: list[str],
        collection_name: Optional[str] = None
    ) -> None:
        """Delete vectors by their IDs.

        Args:
            vector_ids: List of vector IDs to delete
            collection_name: Optional collection name for multi-tenant stores
        """
        pass

    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """Create a new collection.

        Args:
            collection_name: Name of the collection to create
            dimension: Dimension of vectors in this collection
            metadata: Optional collection metadata
        """
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection and all its vectors.

        Args:
            collection_name: Name of the collection to delete
        """
        pass
