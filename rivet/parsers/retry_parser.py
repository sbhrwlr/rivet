"""Retry parser with fallback mechanisms."""

import time
from typing import Any, Optional, List, Callable
from .base import OutputParser
from ..exceptions import ParsingError


class RetryParser(OutputParser):
    """Parser that retries parsing with fallback strategies."""
    
    def __init__(
        self,
        primary_parser: OutputParser,
        fallback_parsers: Optional[List[OutputParser]] = None,
        max_retries: int = 3,
        retry_delay: float = 0.1,
        fallback_value: Any = None
    ):
        """
        Initialize retry parser.
        
        Args:
            primary_parser: Main parser to try first.
            fallback_parsers: List of fallback parsers to try if primary fails.
            max_retries: Maximum number of retry attempts.
            retry_delay: Delay between retries in seconds.
            fallback_value: Value to return if all parsing attempts fail.
        """
        self.primary_parser = primary_parser
        self.fallback_parsers = fallback_parsers or []
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.fallback_value = fallback_value
    
    def parse(self, text: str) -> Any:
        """Parse text with retry and fallback logic."""
        last_error = None
        
        # Try primary parser with retries
        for attempt in range(self.max_retries):
            try:
                result = self.primary_parser.parse(text)
                if self.primary_parser.validate_output(result):
                    return result
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        # Try fallback parsers
        for fallback_parser in self.fallback_parsers:
            try:
                result = fallback_parser.parse(text)
                if fallback_parser.validate_output(result):
                    return result
            except Exception:
                continue
        
        # If all parsers fail, return fallback value or raise error
        if self.fallback_value is not None:
            return self.fallback_value
        
        # Return original text as last resort
        if last_error:
            raise ParsingError(
                f"All parsing attempts failed. Last error: {str(last_error)}",
                original_text=text,
                parser_type=type(self.primary_parser).__name__
            )
        
        return text
    
    def validate_output(self, parsed_output: Any) -> bool:
        """Validate output using primary parser's validation."""
        return self.primary_parser.validate_output(parsed_output)