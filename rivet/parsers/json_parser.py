"""JSON output parser."""

import json
import re
from typing import Dict, Any, Union
from .base import OutputParser


class JSONParser(OutputParser):
    """Parser for JSON responses."""
    
    def __init__(self, strict: bool = False):
        """
        Initialize JSON parser.
        
        Args:
            strict: If True, requires valid JSON. If False, attempts to extract JSON from text.
        """
        self.strict = strict
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse text as JSON."""
        if self.strict:
            return json.loads(text)
        
        # Try to parse as-is first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from text
        json_match = self._extract_json(text)
        if json_match:
            try:
                return json.loads(json_match)
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Could not parse JSON from text: {text[:100]}...")
    
    def _extract_json(self, text: str) -> Union[str, None]:
        """Extract JSON from text using regex patterns."""
        # Look for JSON objects
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Simple nested objects
            r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]',  # Arrays
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    json.loads(match)  # Validate it's valid JSON
                    return match
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def validate_output(self, parsed_output: Any) -> bool:
        """Validate that output is a valid dict or list."""
        return isinstance(parsed_output, (dict, list))