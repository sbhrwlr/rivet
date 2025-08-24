"""OpenAI model adapter."""

import os
import asyncio
from typing import List, Optional, AsyncIterator
from .base import ModelAdapter
import openai

class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI GPT models."""
    
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        self.async_client = None
        self._last_usage = None  # Store last API call usage info
        
    def configure(self, **kwargs) -> None:
        """Configure OpenAI parameters."""
        if "model" in kwargs:
            self.model = kwargs["model"]
        if "api_key" in kwargs:
            self.api_key = kwargs["api_key"]
            
    def generate(self, prompt: str, available_tools: Optional[List[str]] = None) -> str:
        """Generate response using OpenAI API."""
        if not self.client:
            try:
                self.client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                return "Error: OpenAI package not installed. Run: pip install openai"
                
        try:
            messages = [{"role": "user", "content": prompt}]
            
            if available_tools:
                tool_info = f"\nAvailable tools: {', '.join(available_tools)}"
                tool_info += "\nTo use a tool, respond with: TOOL_CALL: <tool_name>"
                messages[0]["content"] += tool_info
                
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500
            )
            
            # Store token usage information for potential tracking
            self._last_usage = {
                'input_tokens': response.usage.prompt_tokens if response.usage else 0,
                'output_tokens': response.usage.completion_tokens if response.usage else 0,
                'total_tokens': response.usage.total_tokens if response.usage else 0
            }
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    async def agenerate(self, prompt: str, available_tools: Optional[List[str]] = None) -> str:
        """Generate response using OpenAI API asynchronously."""
        if not self.async_client:
            try:
                import openai
                self.async_client = openai.AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                return "Error: OpenAI package not installed. Run: pip install openai"
                
        try:
            messages = [{"role": "user", "content": prompt}]
            
            if available_tools:
                tool_info = f"\nAvailable tools: {', '.join(available_tools)}"
                tool_info += "\nTo use a tool, respond with: TOOL_CALL: <tool_name>"
                messages[0]["content"] += tool_info
                
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500
            )
            
            # Store token usage information for potential tracking
            self._last_usage = {
                'input_tokens': response.usage.prompt_tokens if response.usage else 0,
                'output_tokens': response.usage.completion_tokens if response.usage else 0,
                'total_tokens': response.usage.total_tokens if response.usage else 0
            }
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    async def stream(self, prompt: str, available_tools: Optional[List[str]] = None) -> AsyncIterator[str]:
        """Stream response using OpenAI API with fallback to non-streaming."""
        if not self.async_client:
            try:
                import openai
                self.async_client = openai.AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                from ..exceptions import StreamingError
                raise StreamingError("OpenAI package not installed. Run: pip install openai")
                
        try:
            messages = [{"role": "user", "content": prompt}]
            
            if available_tools:
                tool_info = f"\nAvailable tools: {', '.join(available_tools)}"
                tool_info += "\nTo use a tool, respond with: TOOL_CALL: <tool_name>"
                messages[0]["content"] += tool_info
                
            stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            from ..exceptions import StreamingError
            # Try to fall back to non-streaming mode
            try:
                response = await self.agenerate(prompt, available_tools)
                yield response
            except Exception as fallback_error:
                raise StreamingError(f"Streaming failed: {str(e)}. Fallback also failed: {str(fallback_error)}") from e
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken if available, otherwise estimate."""
        try:
            import tiktoken
            # Handle different model names and their encodings
            model_name = self.model
            if model_name.startswith("gpt-4"):
                encoding = tiktoken.encoding_for_model("gpt-4")
            elif model_name.startswith("gpt-3.5"):
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                # Try the exact model name first
                try:
                    encoding = tiktoken.encoding_for_model(model_name)
                except KeyError:
                    # Fall back to cl100k_base encoding for most modern models
                    encoding = tiktoken.get_encoding("cl100k_base")
            
            return len(encoding.encode(text))
        except ImportError:
            # Fallback to word-based estimation (roughly 1.3 tokens per word)
            return int(len(text.split()) * 1.3)
        except Exception:
            # If tiktoken fails for any other reason, use word estimation
            return int(len(text.split()) * 1.3)
    
    def get_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on OpenAI pricing."""
        # Import CostCalculator to use centralized pricing
        from ..usage import CostCalculator
        calculator = CostCalculator()
        return calculator.calculate_cost(self.model, input_tokens, output_tokens)
    
    def get_last_usage(self) -> Optional[dict]:
        """Get token usage from the last API call."""
        return self._last_usage