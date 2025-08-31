"""Tests for OpenAI provider implementation using the official SDK."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import openai
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.completion_usage import CompletionUsage
from openai.types.create_embedding_response import Usage

from flowt.core.models import GenerationConfig, LLMResponse
from flowt.providers.openai_provider import (
    OpenAIAPIError,
    OpenAIProvider,
    OpenAIProviderError,
    OpenAIRateLimitError,
    RateLimiter,
)


class TestRateLimiter:
    """Test cases for RateLimiter class."""

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_immediate_request(self):
        """Test that rate limiter allows the first request immediately."""
        limiter = RateLimiter(requests_per_minute=60)

        # First request should not wait
        start_time = asyncio.get_event_loop().time()
        await limiter.acquire()
        end_time = asyncio.get_event_loop().time()

        # Should complete almost immediately (allow small margin for execution time)
        assert end_time - start_time < 0.1

    @pytest.mark.asyncio
    async def test_rate_limiter_enforces_delay(self):
        """Test that rate limiter enforces delay between requests."""
        limiter = RateLimiter(requests_per_minute=60)  # 1 request per second

        # First request
        await limiter.acquire()

        # Second request should be delayed
        start_time = asyncio.get_event_loop().time()
        await limiter.acquire()
        end_time = asyncio.get_event_loop().time()

        # Should wait approximately 1 second (allow some margin)
        assert 0.9 < end_time - start_time < 1.2

    @pytest.mark.asyncio
    async def test_rate_limiter_disabled(self):
        """Test that rate limiter can be disabled."""
        limiter = RateLimiter(requests_per_minute=0)

        # Multiple requests should not wait
        for _ in range(3):
            start_time = asyncio.get_event_loop().time()
            await limiter.acquire()
            end_time = asyncio.get_event_loop().time()
            assert end_time - start_time < 0.1


class TestOpenAIProvider:
    """Test cases for OpenAIProvider class."""

    @pytest.fixture
    def provider(self):
        """Create OpenAI provider for testing."""
        return OpenAIProvider(
            api_key="test-api-key",
            timeout=10.0,
            max_retries=2,
            requests_per_minute=120,
        )

    @pytest.fixture
    def generation_config(self):
        """Create generation config for testing."""
        return GenerationConfig(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100,
            top_p=1.0,
        )

    @pytest.fixture
    def mock_chat_completion(self):
        """Mock ChatCompletion response."""
        return ChatCompletion(
            id="chatcmpl-123",
            object="chat.completion",
            created=1677652288,
            model="gpt-3.5-turbo",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="Hello! How can I help you today?",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=9,
                completion_tokens=12,
                total_tokens=21,
            ),
        )

    @pytest.fixture
    def mock_embedding_response(self):
        """Mock CreateEmbeddingResponse."""
        return CreateEmbeddingResponse(
            object="list",
            data=[
                Embedding(
                    object="embedding",
                    embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
                    index=0,
                ),
                Embedding(
                    object="embedding",
                    embedding=[0.6, 0.7, 0.8, 0.9, 1.0],
                    index=1,
                ),
            ],
            model="text-embedding-3-small",
            usage=Usage(prompt_tokens=8, total_tokens=8),
        )

    @pytest.mark.asyncio
    async def test_generate_success(self, provider, generation_config, mock_chat_completion):
        """Test successful text generation."""
        with patch.object(
            provider._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_chat_completion

            response = await provider.generate("Hello", generation_config)

            assert isinstance(response, LLMResponse)
            assert response.content == "Hello! How can I help you today?"
            assert response.model == "gpt-3.5-turbo"
            assert response.finish_reason == "stop"
            assert response.usage == {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21,
            }
            assert response.metadata["response_id"] == "chatcmpl-123"

            # Verify the call was made with correct parameters
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["model"] == "gpt-3.5-turbo"
            assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_generate_with_optional_params(self, provider, mock_chat_completion):
        """Test generation with optional parameters."""
        config = GenerationConfig(
            model="gpt-4",
            temperature=0.5,
            stop=["END", "STOP"],
            frequency_penalty=0.1,
            presence_penalty=0.2,
        )

        with patch.object(
            provider._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_chat_completion

            await provider.generate("Test prompt", config)

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["stop"] == ["END", "STOP"]
            assert call_kwargs["frequency_penalty"] == 0.1
            assert call_kwargs["presence_penalty"] == 0.2
            assert "max_tokens" not in call_kwargs  # Should not be included when None

    @pytest.mark.asyncio
    async def test_embed_success(self, provider, mock_embedding_response):
        """Test successful text embedding."""
        with patch.object(
            provider._client.embeddings, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_embedding_response

            texts = ["Hello world", "How are you?"]
            embeddings = await provider.embed(texts)

            assert len(embeddings) == 2
            assert embeddings[0] == [0.1, 0.2, 0.3, 0.4, 0.5]
            assert embeddings[1] == [0.6, 0.7, 0.8, 0.9, 1.0]

            # Verify the call was made with correct parameters
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["model"] == "text-embedding-3-small"
            assert call_kwargs["input"] == texts

    @pytest.mark.asyncio
    async def test_embed_empty_list(self, provider):
        """Test embedding empty list returns empty list."""
        embeddings = await provider.embed([])
        assert embeddings == []

    @pytest.mark.asyncio
    async def test_stream_generate_success(self, provider, generation_config):
        """Test successful streaming generation."""
        # Create mock streaming chunks
        chunks = [
            ChatCompletionChunk(
                id="chatcmpl-123",
                object="chat.completion.chunk",
                created=1677652288,
                model="gpt-3.5-turbo",
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChoiceDelta(content="Hello"),
                        finish_reason=None,
                    )
                ],
            ),
            ChatCompletionChunk(
                id="chatcmpl-123",
                object="chat.completion.chunk",
                created=1677652288,
                model="gpt-3.5-turbo",
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChoiceDelta(content=" there"),
                        finish_reason=None,
                    )
                ],
            ),
            ChatCompletionChunk(
                id="chatcmpl-123",
                object="chat.completion.chunk",
                created=1677652288,
                model="gpt-3.5-turbo",
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChoiceDelta(content="!"),
                        finish_reason="stop",
                    )
                ],
            ),
        ]

        async def mock_stream():
            for chunk in chunks:
                yield chunk

        with patch.object(
            provider._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_stream()

            result_chunks = []
            async for chunk in provider.stream_generate("Hello", generation_config):
                result_chunks.append(chunk)

            assert result_chunks == ["Hello", " there", "!"]
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_stream_generate_handles_empty_chunks(self, provider, generation_config):
        """Test that streaming handles chunks without content gracefully."""
        chunks = [
            ChatCompletionChunk(
                id="chatcmpl-123",
                object="chat.completion.chunk",
                created=1677652288,
                model="gpt-3.5-turbo",
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChoiceDelta(content="Hello"),
                        finish_reason=None,
                    )
                ],
            ),
            ChatCompletionChunk(
                id="chatcmpl-123",
                object="chat.completion.chunk",
                created=1677652288,
                model="gpt-3.5-turbo",
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChoiceDelta(content=None),  # Empty content
                        finish_reason=None,
                    )
                ],
            ),
            ChatCompletionChunk(
                id="chatcmpl-123",
                object="chat.completion.chunk",
                created=1677652288,
                model="gpt-3.5-turbo",
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChoiceDelta(content=" world"),
                        finish_reason="stop",
                    )
                ],
            ),
        ]

        async def mock_stream():
            for chunk in chunks:
                yield chunk

        with patch.object(
            provider._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_stream()

            result_chunks = []
            async for chunk in provider.stream_generate("Hello", generation_config):
                result_chunks.append(chunk)

            # Should skip the empty chunk
            assert result_chunks == ["Hello", " world"]

    def test_handle_rate_limit_error(self, provider):
        """Test that rate limit errors are properly handled."""
        # Create a real RateLimitError-like exception
        class MockRateLimitError(openai.RateLimitError, Exception):
            def __init__(self, message):
                Exception.__init__(self, message)
                self.message = message
            
            def __str__(self):
                return self.message

        error = MockRateLimitError("Rate limit exceeded")
        
        with pytest.raises(OpenAIRateLimitError) as exc_info:
            provider._handle_openai_error(error)

        assert "Rate limit exceeded" in str(exc_info.value)
        assert exc_info.value.original_error is error

    def test_handle_api_error(self, provider):
        """Test that API errors are properly handled."""
        # Create a real APIError-like exception
        class MockAPIError(openai.APIError, Exception):
            def __init__(self, message):
                Exception.__init__(self, message)
                self.message = message
            
            def __str__(self):
                return self.message

        error = MockAPIError("Invalid request")
        
        with pytest.raises(OpenAIAPIError) as exc_info:
            provider._handle_openai_error(error)

        assert "OpenAI API error" in str(exc_info.value)
        assert exc_info.value.original_error is error

    @pytest.mark.asyncio
    async def test_handle_unexpected_error(self, provider, generation_config):
        """Test that unexpected errors are properly handled."""
        with patch.object(
            provider._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = ValueError("Unexpected error")

            with pytest.raises(OpenAIProviderError) as exc_info:
                await provider.generate("Hello", generation_config)

            assert "Unexpected error" in str(exc_info.value)
            assert isinstance(exc_info.value.original_error, ValueError)

    def test_handle_connection_error(self, provider):
        """Test that connection errors are properly handled."""
        # Create a real APIConnectionError-like exception
        class MockConnectionError(openai.APIConnectionError, Exception):
            def __init__(self, message):
                Exception.__init__(self, message)
                self.message = message
            
            def __str__(self):
                return self.message

        error = MockConnectionError("Connection failed")
        
        with pytest.raises(OpenAIAPIError) as exc_info:
            provider._handle_openai_error(error)

        assert "OpenAI API error" in str(exc_info.value)
        assert exc_info.value.original_error is error

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test that provider works as async context manager."""
        async with OpenAIProvider(api_key="test-key") as provider:
            assert isinstance(provider, OpenAIProvider)
            # Client should be available
            assert provider._client is not None

    @pytest.mark.asyncio
    async def test_close_method(self, provider):
        """Test that close method properly closes the OpenAI client."""
        with patch.object(provider._client, "close", new_callable=AsyncMock) as mock_close:
            await provider.close()
            mock_close.assert_called_once()

    def test_initialization_with_custom_params(self):
        """Test provider initialization with custom parameters."""
        provider = OpenAIProvider(
            api_key="custom-key",
            base_url="https://custom.openai.com/v1",
            organization="org-123",
            project="proj-456",
            timeout=60.0,
            max_retries=5,
            requests_per_minute=30,
            default_model="gpt-4",
            default_embedding_model="text-embedding-3-large",
        )

        assert provider.timeout == 60.0
        assert provider.max_retries == 5
        assert provider.default_model == "gpt-4"
        assert provider.default_embedding_model == "text-embedding-3-large"
        assert provider.rate_limiter.requests_per_minute == 30

    def test_initialization_with_env_vars(self):
        """Test provider initialization using environment variables."""
        # Mock environment variable
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-env-key'}):
            provider = OpenAIProvider()
            assert provider.default_model == "gpt-3.5-turbo"
            assert provider.default_embedding_model == "text-embedding-3-small"

    def test_framework_methods(self, provider):
        """Test framework-oriented helper methods."""
        # Test with_model
        new_provider = provider.with_model("gpt-4")
        assert new_provider.default_model == "gpt-4"
        assert new_provider is not provider  # Should be a new instance

        # Test with_embedding_model
        new_provider = provider.with_embedding_model("text-embedding-3-large")
        assert new_provider.default_embedding_model == "text-embedding-3-large"

        # Test with_rate_limit
        new_provider = provider.with_rate_limit(30)
        assert new_provider.rate_limiter.requests_per_minute == 30

    def test_client_info_property(self, provider):
        """Test client_info property returns correct information."""
        info = provider.client_info
        
        assert "base_url" in info
        assert "timeout" in info
        assert "max_retries" in info
        assert "default_model" in info
        assert "default_embedding_model" in info
        assert "rate_limit_rpm" in info
        
        assert info["timeout"] == provider.timeout
        assert info["max_retries"] == provider.max_retries
        assert info["default_model"] == provider.default_model

    def test_config_to_openai_params(self, provider):
        """Test conversion of GenerationConfig to OpenAI parameters."""
        config = GenerationConfig(
            model="gpt-4",
            temperature=0.8,
            max_tokens=150,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            stop=["END"],
        )

        params = provider._config_to_openai_params(config)

        assert params["model"] == "gpt-4"
        assert params["temperature"] == 0.8
        assert params["max_tokens"] == 150
        assert params["top_p"] == 0.9
        assert params["frequency_penalty"] == 0.1
        assert params["presence_penalty"] == 0.2
        assert params["stop"] == ["END"]

    def test_config_to_openai_params_with_none_values(self, provider):
        """Test conversion handles None values correctly."""
        config = GenerationConfig(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=None,  # Should not be included
            stop=None,  # Should not be included
        )

        params = provider._config_to_openai_params(config)

        assert "max_tokens" not in params
        assert "stop" not in params
        assert params["model"] == "gpt-3.5-turbo"
        assert params["temperature"] == 0.7

    def test_build_chat_messages(self, provider):
        """Test building chat messages from prompt."""
        messages = provider._build_chat_messages("Hello, world!")
        
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, world!"