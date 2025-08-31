"""Tests for core data models."""

import pytest
from datetime import datetime
from flowt.core.models import (
    Message,
    MessageRole,
    LLMResponse,
    Document,
    Vector,
    SearchResult,
    GenerationConfig,
    RetrievalConfig,
    PipelineConfig,
)


class TestMessage:
    """Tests for Message model."""

    def test_message_creation(self) -> None:
        """Test basic message creation."""
        message = Message(role=MessageRole.USER, content="Hello, world!")
        
        assert message.role == MessageRole.USER
        assert message.content == "Hello, world!"
        assert isinstance(message.timestamp, datetime)
        assert message.metadata == {}

    def test_message_with_metadata(self) -> None:
        """Test message creation with metadata."""
        metadata = {"source": "test", "priority": "high"}
        message = Message(
            role=MessageRole.ASSISTANT,
            content="Response",
            metadata=metadata
        )
        
        assert message.metadata == metadata


class TestLLMResponse:
    """Tests for LLMResponse model."""

    def test_llm_response_creation(self) -> None:
        """Test basic LLM response creation."""
        response = LLMResponse(
            content="Generated text",
            model="gpt-3.5-turbo",
            finish_reason="stop"
        )
        
        assert response.content == "Generated text"
        assert response.model == "gpt-3.5-turbo"
        assert response.finish_reason == "stop"
        assert response.usage == {}
        assert response.metadata == {}


class TestDocument:
    """Tests for Document model."""

    def test_document_creation(self) -> None:
        """Test basic document creation."""
        doc = Document(content="Document content")
        
        assert doc.content == "Document content"
        assert doc.metadata == {}
        assert doc.embedding is None
        assert doc.score is None

    def test_document_from_search_result(self) -> None:
        """Test creating document from search result."""
        search_result = SearchResult(
            content="Search result content",
            score=0.85,
            vector_id="vec_123",
            metadata={"source": "test"}
        )
        
        doc = Document.from_search_result(search_result)
        
        assert doc.content == "Search result content"
        assert doc.score == 0.85
        assert doc.metadata == {"source": "test"}


class TestVector:
    """Tests for Vector model."""

    def test_vector_creation(self) -> None:
        """Test basic vector creation."""
        embedding = [0.1, 0.2, 0.3]
        vector = Vector(id="vec_123", embedding=embedding)
        
        assert vector.id == "vec_123"
        assert vector.embedding == embedding
        assert vector.metadata == {}

    def test_validate_dimension(self) -> None:
        """Test dimension validation."""
        vector = Vector(id="vec_123", embedding=[0.1, 0.2, 0.3])
        
        assert vector.validate_dimension(3) is True
        assert vector.validate_dimension(4) is False


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_search_result_creation(self) -> None:
        """Test basic search result creation."""
        result = SearchResult(
            content="Result content",
            score=0.95,
            vector_id="vec_456"
        )
        
        assert result.content == "Result content"
        assert result.score == 0.95
        assert result.vector_id == "vec_456"
        assert result.metadata == {}


class TestGenerationConfig:
    """Tests for GenerationConfig model."""

    def test_default_config(self) -> None:
        """Test default generation config."""
        config = GenerationConfig()
        
        assert config.model == "gpt-3.5-turbo"
        assert config.temperature == 0.7
        assert config.max_tokens is None
        assert config.top_p == 1.0

    def test_config_validation(self) -> None:
        """Test config parameter validation."""
        # Valid config
        config = GenerationConfig(temperature=0.5, max_tokens=100)
        assert config.temperature == 0.5
        assert config.max_tokens == 100

        # Invalid temperature (too high)
        with pytest.raises(ValueError):
            GenerationConfig(temperature=3.0)

        # Invalid max_tokens (negative)
        with pytest.raises(ValueError):
            GenerationConfig(max_tokens=-1)


class TestRetrievalConfig:
    """Tests for RetrievalConfig model."""

    def test_default_config(self) -> None:
        """Test default retrieval config."""
        config = RetrievalConfig()
        
        assert config.limit == 5
        assert config.score_threshold == 0.7
        assert config.filters == {}

    def test_config_validation(self) -> None:
        """Test config parameter validation."""
        # Valid config
        config = RetrievalConfig(limit=10, score_threshold=0.8)
        assert config.limit == 10
        assert config.score_threshold == 0.8

        # Invalid limit (too high)
        with pytest.raises(ValueError):
            RetrievalConfig(limit=200)

        # Invalid score_threshold (negative)
        with pytest.raises(ValueError):
            RetrievalConfig(score_threshold=-0.1)


class TestPipelineConfig:
    """Tests for PipelineConfig model."""

    def test_default_config(self) -> None:
        """Test default pipeline config."""
        config = PipelineConfig()
        
        assert config.retry_attempts == 3
        assert config.timeout_seconds == 30
        assert config.enable_tracing is True
        assert config.enable_logging is True

    def test_config_validation(self) -> None:
        """Test config parameter validation."""
        # Valid config
        config = PipelineConfig(retry_attempts=5, timeout_seconds=60)
        assert config.retry_attempts == 5
        assert config.timeout_seconds == 60

        # Invalid retry_attempts (too high)
        with pytest.raises(ValueError):
            PipelineConfig(retry_attempts=15)

        # Invalid timeout_seconds (negative)
        with pytest.raises(ValueError):
            PipelineConfig(timeout_seconds=-1)

    def test_config_edge_cases(self) -> None:
        """Test config edge cases."""
        # Minimum valid values
        config = PipelineConfig(retry_attempts=0, timeout_seconds=1)
        assert config.retry_attempts == 0
        assert config.timeout_seconds == 1

        # Maximum valid values
        config = PipelineConfig(retry_attempts=10, timeout_seconds=3600)
        assert config.retry_attempts == 10
        assert config.timeout_seconds == 3600

        # Boolean flags
        config = PipelineConfig(enable_tracing=False, enable_logging=False)
        assert config.enable_tracing is False
        assert config.enable_logging is False