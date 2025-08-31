"""Core data models for the LLM Pipeline Framework."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Enumeration of message roles in conversations."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Message(BaseModel):
    """Represents a message in a conversation.

    Attributes:
        role: The role of the message sender
        content: The message content
        metadata: Additional metadata for the message
        timestamp: When the message was created
    """

    role: MessageRole
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"use_enum_values": True}


class LLMResponse(BaseModel):
    """Response from an LLM provider.

    Attributes:
        content: The generated text content
        usage: Token usage statistics
        model: The model used for generation
        finish_reason: Why the generation stopped
        metadata: Additional response metadata
    """

    content: str
    usage: dict[str, int] = Field(default_factory=dict)
    model: str
    finish_reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """Represents a document with optional embedding.

    Attributes:
        content: The document text content
        metadata: Document metadata (title, source, etc.)
        embedding: Optional embedding vector
        score: Optional similarity score (used in search results)
    """

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[list[float]] = None
    score: Optional[float] = None

    @classmethod
    def from_search_result(cls, search_result: "SearchResult") -> "Document":
        """Create a Document from a SearchResult.

        Args:
            search_result: The search result to convert

        Returns:
            Document instance with content and score from search result
        """
        return cls(
            content=search_result.content,
            metadata=search_result.metadata,
            score=search_result.score,
        )


class Vector(BaseModel):
    """Represents a vector with ID and metadata.

    Attributes:
        id: Unique identifier for the vector
        embedding: The embedding vector
        metadata: Additional metadata for the vector
    """

    id: str
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)

    def validate_dimension(self, expected_dim: int) -> bool:
        """Validate that the embedding has the expected dimension.

        Args:
            expected_dim: Expected embedding dimension

        Returns:
            True if dimension matches, False otherwise
        """
        return len(self.embedding) == expected_dim


class SearchResult(BaseModel):
    """Result from a vector similarity search.

    Attributes:
        content: The content of the matching document
        score: Similarity score (higher = more similar)
        metadata: Metadata from the matching vector
        vector_id: ID of the matching vector
    """

    content: str
    score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    vector_id: str


# Configuration Models

class GenerationConfig(BaseModel):
    """Configuration for LLM text generation.

    Attributes:
        model: Model name to use for generation
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        frequency_penalty: Frequency penalty (-2.0 to 2.0)
        presence_penalty: Presence penalty (-2.0 to 2.0)
        stop: List of stop sequences
    """

    model: str = "gpt-3.5-turbo"
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, gt=0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    stop: Optional[list[str]] = None


class RetrievalConfig(BaseModel):
    """Configuration for document retrieval.

    Attributes:
        limit: Maximum number of documents to retrieve
        score_threshold: Minimum similarity score threshold
        filters: Metadata filters to apply during search
    """

    limit: int = Field(5, gt=0, le=100)
    score_threshold: float = Field(0.7, ge=0.0, le=1.0)
    filters: dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """Configuration for pipeline execution.

    Attributes:
        retry_attempts: Number of retry attempts for failed operations
        timeout_seconds: Timeout for individual operations
        enable_tracing: Whether to enable distributed tracing
        enable_logging: Whether to enable structured logging
    """

    retry_attempts: int = Field(default=3, ge=0, le=10)
    timeout_seconds: int = Field(default=30, gt=0)
    enable_tracing: bool = True
    enable_logging: bool = True
