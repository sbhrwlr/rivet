"""
Tests for token counting functionality in model adapters.
"""

import pytest
from unittest.mock import Mock, patch

from rivet.models.openai_adapter import OpenAIAdapter
from rivet.models.base import ModelAdapter


class TestTokenCounting:
    """Test token counting functionality."""
    
    def test_base_model_adapter_count_tokens_fallback(self):
        """Test base ModelAdapter token counting fallback."""
        # Create a concrete implementation for testing
        class TestAdapter(ModelAdapter):
            def generate(self, prompt, available_tools=None):
                return "test response"
            
            def configure(self, **kwargs):
                pass
        
        adapter = TestAdapter()
        
        # Test word-based estimation
        text = "Hello world this is a test"
        token_count = adapter.count_tokens(text)
        
        # Should be roughly 1.3 tokens per word
        expected = int(len(text.split()) * 1.3)  # 5 words * 1.3 = 6.5 -> 6
        assert token_count == expected
    
    def test_base_model_adapter_get_cost_default(self):
        """Test base ModelAdapter get_cost default implementation."""
        class TestAdapter(ModelAdapter):
            def generate(self, prompt, available_tools=None):
                return "test response"
            
            def configure(self, **kwargs):
                pass
        
        adapter = TestAdapter()
        cost = adapter.get_cost(1000, 500)
        assert cost == 0.0
    
    def test_base_model_adapter_get_last_usage_default(self):
        """Test base ModelAdapter get_last_usage default implementation."""
        class TestAdapter(ModelAdapter):
            def generate(self, prompt, available_tools=None):
                return "test response"
            
            def configure(self, **kwargs):
                pass
        
        adapter = TestAdapter()
        usage = adapter.get_last_usage()
        assert usage is None


class TestOpenAIAdapterTokenCounting:
    """Test OpenAI adapter token counting functionality."""
    
    def test_count_tokens_with_tiktoken_gpt35(self):
        """Test token counting with tiktoken for GPT-3.5."""
        adapter = OpenAIAdapter(model="gpt-3.5-turbo")
        
        # Test with a known string
        text = "Hello, how are you today?"
        
        try:
            import tiktoken
            # If tiktoken is available, should use it
            token_count = adapter.count_tokens(text)
            
            # Verify it's using tiktoken by checking it's not the word estimation
            word_estimate = int(len(text.split()) * 1.3)
            assert token_count != word_estimate  # Should be different from word estimate
            assert isinstance(token_count, int)
            assert token_count > 0
            
        except ImportError:
            # If tiktoken not available, should fall back to word estimation
            token_count = adapter.count_tokens(text)
            word_estimate = int(len(text.split()) * 1.3)
            assert token_count == word_estimate
    
    def test_count_tokens_with_tiktoken_gpt4(self):
        """Test token counting with tiktoken for GPT-4."""
        adapter = OpenAIAdapter(model="gpt-4")
        
        text = "This is a test message for GPT-4 token counting."
        
        try:
            import tiktoken
            token_count = adapter.count_tokens(text)
            
            # Should return a reasonable token count
            assert isinstance(token_count, int)
            assert token_count > 0
            assert token_count < len(text)  # Should be less than character count
            
        except ImportError:
            # Fallback to word estimation
            token_count = adapter.count_tokens(text)
            word_estimate = int(len(text.split()) * 1.3)
            assert token_count == word_estimate
    
    def test_count_tokens_custom_model(self):
        """Test token counting with custom model name."""
        adapter = OpenAIAdapter(model="custom-model-name")
        
        text = "Test message for custom model"
        token_count = adapter.count_tokens(text)
        
        # Should handle unknown models gracefully
        assert isinstance(token_count, int)
        assert token_count > 0
    
    def test_count_tokens_empty_string(self):
        """Test token counting with empty string."""
        adapter = OpenAIAdapter()
        
        token_count = adapter.count_tokens("")
        assert token_count == 0
    
    def test_count_tokens_fallback_on_tiktoken_error(self):
        """Test fallback to word estimation when tiktoken fails."""
        adapter = OpenAIAdapter()
        
        # Mock tiktoken import to raise an exception
        with patch('builtins.__import__', side_effect=lambda name, *args: 
                   Exception("Tiktoken error") if name == 'tiktoken' else __import__(name, *args)):
            
            text = "This should fall back to word estimation"
            token_count = adapter.count_tokens(text)
            
            # Should fall back to word estimation
            word_estimate = int(len(text.split()) * 1.3)
            assert token_count == word_estimate
    
    def test_get_cost_calculation(self):
        """Test cost calculation using CostCalculator."""
        adapter = OpenAIAdapter(model="gpt-3.5-turbo")
        
        # Test cost calculation
        cost = adapter.get_cost(1000, 500)
        
        # Should use CostCalculator for accurate pricing
        expected_cost = (1000/1000 * 0.0015) + (500/1000 * 0.002)  # GPT-3.5 pricing
        assert cost == expected_cost
    
    def test_get_cost_different_models(self):
        """Test cost calculation for different models."""
        # Test GPT-4
        adapter_gpt4 = OpenAIAdapter(model="gpt-4")
        cost_gpt4 = adapter_gpt4.get_cost(1000, 500)
        expected_gpt4 = (1000/1000 * 0.03) + (500/1000 * 0.06)
        assert cost_gpt4 == expected_gpt4
        
        # Test GPT-4o
        adapter_gpt4o = OpenAIAdapter(model="gpt-4o")
        cost_gpt4o = adapter_gpt4o.get_cost(1000, 500)
        expected_gpt4o = (1000/1000 * 0.005) + (500/1000 * 0.015)
        assert cost_gpt4o == expected_gpt4o
    
    def test_get_last_usage_after_generate(self):
        """Test getting usage information after API call."""
        adapter = OpenAIAdapter()
        
        # Mock OpenAI client and response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('openai.OpenAI', return_value=mock_client):
            response = adapter.generate("Test prompt")
            
            # Check that usage was stored
            usage = adapter.get_last_usage()
            assert usage is not None
            assert usage['input_tokens'] == 100
            assert usage['output_tokens'] == 50
            assert usage['total_tokens'] == 150
    
    def test_get_last_usage_after_agenerate(self):
        """Test getting usage information after async API call."""
        adapter = OpenAIAdapter()
        
        # Mock async OpenAI client and response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.prompt_tokens = 200
        mock_response.usage.completion_tokens = 100
        mock_response.usage.total_tokens = 300
        
        mock_client = Mock()
        # Make the async method return a coroutine
        async def mock_create(*args, **kwargs):
            return mock_response
        mock_client.chat.completions.create = mock_create
        
        async def test_async():
            with patch('openai.AsyncOpenAI', return_value=mock_client):
                response = await adapter.agenerate("Test prompt")
                
                # Check that usage was stored
                usage = adapter.get_last_usage()
                assert usage is not None
                assert usage['input_tokens'] == 200
                assert usage['output_tokens'] == 100
                assert usage['total_tokens'] == 300
        
        import asyncio
        asyncio.run(test_async())
    
    def test_get_last_usage_no_usage_info(self):
        """Test handling when API response has no usage information."""
        adapter = OpenAIAdapter()
        
        # Mock response without usage info
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = None
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('openai.OpenAI', return_value=mock_client):
            response = adapter.generate("Test prompt")
            
            # Should store zeros when no usage info available
            usage = adapter.get_last_usage()
            assert usage is not None
            assert usage['input_tokens'] == 0
            assert usage['output_tokens'] == 0
            assert usage['total_tokens'] == 0
    
    def test_get_last_usage_initially_none(self):
        """Test that get_last_usage returns None initially."""
        adapter = OpenAIAdapter()
        
        usage = adapter.get_last_usage()
        assert usage is None


class TestTokenCountingIntegration:
    """Test token counting integration with other components."""
    
    def test_token_counting_with_tools(self):
        """Test token counting when tools are available."""
        adapter = OpenAIAdapter()
        
        # Test that tool information is included in token count
        text_without_tools = "Hello, how are you?"
        text_with_tools = text_without_tools + "\nAvailable tools: calculator, weather\nTo use a tool, respond with: TOOL_CALL: <tool_name>"
        
        count_without_tools = adapter.count_tokens(text_without_tools)
        count_with_tools = adapter.count_tokens(text_with_tools)
        
        # Token count should be higher when tools are included
        assert count_with_tools > count_without_tools
    
    def test_cost_calculation_accuracy(self):
        """Test that cost calculations are accurate for various token amounts."""
        adapter = OpenAIAdapter(model="gpt-3.5-turbo")
        
        test_cases = [
            (0, 0, 0.0),
            (1000, 0, 0.0015),  # Only input tokens
            (0, 1000, 0.002),   # Only output tokens
            (1000, 1000, 0.0035),  # Both input and output
            (500, 250, 0.00125),   # Smaller amounts
        ]
        
        for input_tokens, output_tokens, expected_cost in test_cases:
            cost = adapter.get_cost(input_tokens, output_tokens)
            assert abs(cost - expected_cost) < 0.000001  # Allow for floating point precision