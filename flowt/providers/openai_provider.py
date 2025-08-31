"""OpenAI LLM provider implementation using the official OpenAI SDK."""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any, Optional, List, Dict, Union

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types import CreateEmbeddingResponse
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..core.interfaces import LLMProvider
from ..core.models import GenerationConfig, LLMResponse


class OpenAIProviderError(Exception):
    """Base exception for OpenAI provider errors."""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        self.message = message
        self.original_error = original_error
        super().__init__(message)


class OpenAIRateLimitError(OpenAIProviderError):
    """Exception raised when OpenAI rate limits are exceeded."""
    pass


class OpenAIAPIError(OpenAIProviderError):
    """Exception raised for OpenAI API errors."""
    pass


class RateLimiter:
    """Simple rate limiter for API calls with configurable limits."""

    def __init__(self, requests_per_minute: int = 60):
        """Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests allowed per minute
        """
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire rate limit token, waiting if necessary."""
        if self.min_interval == 0:
            return  # No rate limiting
            
        async with self._lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time

            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                await asyncio.sleep(wait_time)

            self.last_request_time = time.time()


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider using the official OpenAI SDK with retry logic and rate limiting."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        requests_per_minute: int = 60,
        default_model: str = "gpt-3.5-turbo",
        default_embedding_model: str = "text-embedding-3-small",
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            base_url: Base URL for OpenAI API (optional, for custom endpoints)
            organization: OpenAI organization ID (optional)
            project: OpenAI project ID (optional)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts for SDK retries
            requests_per_minute: Rate limit for API requests (0 to disable)
            default_model: Default model for text generation
            default_embedding_model: Default model for embeddings
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_model = default_model
        self.default_embedding_model = default_embedding_model
        self.rate_limiter = RateLimiter(requests_per_minute)

        # Initialize the OpenAI client with configuration
        client_kwargs: dict[str, Any] = {
            "timeout": timeout,
            "max_retries": max_retries,
        }
        
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        if organization is not None:
            client_kwargs["organization"] = organization
        if project is not None:
            client_kwargs["project"] = project

        self._client = AsyncOpenAI(**client_kwargs)

    async def __aenter__(self) -> "OpenAIProvider":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self._client.close()

    def _handle_openai_error(self, error: Exception) -> None:
        """Convert OpenAI SDK errors to provider-specific errors."""
        if isinstance(error, openai.RateLimitError):
            raise OpenAIRateLimitError(
                f"Rate limit exceeded: {str(error)}", original_error=error
            ) from error
        elif isinstance(error, (openai.APIError, openai.OpenAIError)):
            raise OpenAIAPIError(
                f"OpenAI API error: {str(error)}", original_error=error
            ) from error
        else:
            raise OpenAIProviderError(
                f"Unexpected error: {str(error)}", original_error=error
            ) from error

    @retry(
        retry=retry_if_exception_type((OpenAIRateLimitError, openai.APIConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def _execute_with_retry(self, operation_name: str, operation: Any) -> Any:
        """Execute an operation with retry logic and rate limiting."""
        await self.rate_limiter.acquire()
        
        try:
            return await operation()
        except Exception as e:
            self._handle_openai_error(e)

    def _build_chat_messages(self, prompt: str) -> List[ChatCompletionMessageParam]:
        """Build chat messages from a prompt."""
        return [{"role": "user", "content": prompt}]

    def _config_to_openai_params(self, config: GenerationConfig) -> dict[str, Any]:
        """Convert GenerationConfig to OpenAI API parameters."""
        params = {
            "model": config.model or self.default_model,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
        }

        if config.max_tokens is not None:
            params["max_tokens"] = config.max_tokens

        if config.stop is not None:
            params["stop"] = config.stop

        return params

    def _chat_completion_to_llm_response(
        self, completion: ChatCompletion
    ) -> LLMResponse:
        """Convert OpenAI ChatCompletion to LLMResponse."""
        choice = completion.choices[0]
        content = choice.message.content if choice.message.content is not None else ""

        usage_dict = {}
        if completion.usage:
            usage_dict = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }

        return LLMResponse(
            content=content,
            usage=usage_dict,
            model=completion.model,
            finish_reason=choice.finish_reason if choice.finish_reason is not None else "unknown",
            metadata={
                "response_id": completion.id,
                "created": completion.created,
                "system_fingerprint": completion.system_fingerprint,
            },
        )

    async def generate(self, prompt: str, config: GenerationConfig) -> LLMResponse:
        """Generate text completion for the given prompt.

        Args:
            prompt: The input prompt text
            config: Generation configuration parameters

        Returns:
            LLMResponse containing the generated text and metadata

        Raises:
            OpenAIProviderError: For various API and configuration errors
        """
        messages = self._build_chat_messages(prompt)
        params = self._config_to_openai_params(config)

        async def _generate() -> Any:
            return await self._client.chat.completions.create(
                messages=messages, **params
            )

        completion = await self._execute_with_retry("generate", _generate)
        return self._chat_completion_to_llm_response(completion)

    async def stream_generate(
        self, prompt: str, config: GenerationConfig
    ) -> AsyncIterator[str]:
        """Generate streaming text completion for the given prompt.

        Args:
            prompt: The input prompt text
            config: Generation configuration parameters

        Yields:
            Individual tokens or text chunks as they are generated

        Raises:
            OpenAIProviderError: For various API and configuration errors
        """
        messages = self._build_chat_messages(prompt)
        params = self._config_to_openai_params(config)
        params["stream"] = True

        async def _stream_generate() -> Any:
            return await self._client.chat.completions.create(
                messages=messages, **params
            )

        stream = await self._execute_with_retry("stream_generate", _stream_generate)

        try:
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    if choice.delta and choice.delta.content:
                        yield choice.delta.content
        except Exception as e:
            self._handle_openai_error(e)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for the given texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (one per input text)

        Raises:
            OpenAIProviderError: For various API and configuration errors
        """
        if not texts:
            return []

        async def _embed() -> Any:
            return await self._client.embeddings.create(
                model=self.default_embedding_model, input=texts
            )

        response = await self._execute_with_retry("embed", _embed)

        # Extract embeddings in the same order as input
        embeddings = []
        for item in sorted(response.data, key=lambda x: x.index):
            embeddings.append(item.embedding)

        return embeddings

    async def close(self) -> None:
        """Close the OpenAI client."""
        await self._client.close()

    # Framework-oriented helper methods

    def with_model(self, model: str) -> "OpenAIProvider":
        """Create a new provider instance with a different default model.
        
        Args:
            model: The model name to use as default
            
        Returns:
            New OpenAIProvider instance with the specified model
        """
        new_provider = OpenAIProvider(
            api_key=self._client.api_key,
            base_url=str(self._client.base_url) if self._client.base_url else None,
            organization=self._client.organization,
            project=self._client.project,
            timeout=self.timeout,
            max_retries=self.max_retries,
            requests_per_minute=self.rate_limiter.requests_per_minute,
            default_model=model,
            default_embedding_model=self.default_embedding_model,
        )
        return new_provider

    def with_embedding_model(self, model: str) -> "OpenAIProvider":
        """Create a new provider instance with a different default embedding model.
        
        Args:
            model: The embedding model name to use as default
            
        Returns:
            New OpenAIProvider instance with the specified embedding model
        """
        new_provider = OpenAIProvider(
            api_key=self._client.api_key,
            base_url=str(self._client.base_url) if self._client.base_url else None,
            organization=self._client.organization,
            project=self._client.project,
            timeout=self.timeout,
            max_retries=self.max_retries,
            requests_per_minute=self.rate_limiter.requests_per_minute,
            default_model=self.default_model,
            default_embedding_model=model,
        )
        return new_provider

    def with_rate_limit(self, requests_per_minute: int) -> "OpenAIProvider":
        """Create a new provider instance with different rate limiting.
        
        Args:
            requests_per_minute: New rate limit (0 to disable)
            
        Returns:
            New OpenAIProvider instance with the specified rate limit
        """
        new_provider = OpenAIProvider(
            api_key=self._client.api_key,
            base_url=str(self._client.base_url) if self._client.base_url else None,
            organization=self._client.organization,
            project=self._client.project,
            timeout=self.timeout,
            max_retries=self.max_retries,
            requests_per_minute=requests_per_minute,
            default_model=self.default_model,
            default_embedding_model=self.default_embedding_model,
        )
        return new_provider

    @property
    def client_info(self) -> dict[str, Any]:
        """Get information about the underlying OpenAI client."""
        return {
            "base_url": str(self._client.base_url) if self._client.base_url else None,
            "organization": self._client.organization,
            "project": self._client.project,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "default_model": self.default_model,
            "default_embedding_model": self.default_embedding_model,
            "rate_limit_rpm": self.rate_limiter.requests_per_minute,
        }