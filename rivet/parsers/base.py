"""Base output parser interface."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class OutputParser(ABC):
    """Base class for all output parsers."""
    
    @abstractmethod
    def parse(self, text: str) -> Any:
        """Parse text into structured format."""
        pass
    
    def parse_with_fallback(self, text: str, fallback: Optional[Any] = None) -> Any:
        """Parse text with fallback value if parsing fails."""
        try:
            return self.parse(text)
        except Exception:
            return fallback if fallback is not None else text
    
    def validate_output(self, parsed_output: Any) -> bool:
        """Validate the parsed output. Override in subclasses for custom validation."""
        return parsed_output is not None