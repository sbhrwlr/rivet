"""Smart parser that automatically detects format and chooses appropriate parser."""

import json
import re
from typing import Any, Dict, List, Optional
from .base import OutputParser
from .json_parser import JSONParser
from .list_parser import ListParser
from .xml_parser import XMLParser
from ..exceptions import ParsingError


class SmartParser(OutputParser):
    """Parser that automatically detects content format and uses appropriate parser."""
    
    def __init__(self, preferred_format: Optional[str] = None):
        """
        Initialize smart parser.
        
        Args:
            preferred_format: Preferred format to try first ('json', 'xml', 'list').
        """
        self.preferred_format = preferred_format
        self.parsers = {
            'json': JSONParser(strict=False),
            'xml': XMLParser(),
            'list': ListParser()
        }
    
    def parse(self, text: str) -> Any:
        """Parse text by auto-detecting format."""
        text = text.strip()
        
        if not text:
            return ""
        
        # Try preferred format first
        if self.preferred_format and self.preferred_format in self.parsers:
            try:
                return self.parsers[self.preferred_format].parse(text)
            except Exception:
                pass
        
        # Auto-detect format and try appropriate parsers
        detected_formats = self._detect_formats(text)
        
        for format_name in detected_formats:
            if format_name in self.parsers:
                try:
                    result = self.parsers[format_name].parse(text)
                    return result
                except Exception:
                    continue
        
        # If no parser works, return original text
        return text
    
    def _detect_formats(self, text: str) -> List[str]:
        """Detect possible formats in order of likelihood."""
        formats = []
        
        # JSON detection
        if self._looks_like_json(text):
            formats.append('json')
        
        # XML detection
        if self._looks_like_xml(text):
            formats.append('xml')
        
        # List detection
        if self._looks_like_list(text):
            formats.append('list')
        
        # If no specific format detected, try all in order
        if not formats:
            formats = ['json', 'list', 'xml']
        
        return formats
    
    def _looks_like_json(self, text: str) -> bool:
        """Check if text looks like JSON."""
        text = text.strip()
        
        # Quick checks for JSON-like structure
        if (text.startswith('{') and text.endswith('}')) or \
           (text.startswith('[') and text.endswith(']')):
            return True
        
        # Look for JSON patterns in text
        json_patterns = [
            r'\{[^{}]*"[^"]*"[^{}]*:[^{}]*\}',  # Simple object
            r'\[[^\[\]]*"[^"]*"[^\[\]]*\]',     # Simple array
        ]
        
        for pattern in json_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _looks_like_xml(self, text: str) -> bool:
        """Check if text looks like XML."""
        text = text.strip()
        
        # Basic XML structure check
        if text.startswith('<') and text.endswith('>'):
            # Look for closing tags
            if re.search(r'<[^>]+>.*</[^>]+>', text, re.DOTALL):
                return True
        
        return False
    
    def _looks_like_list(self, text: str) -> bool:
        """Check if text looks like a list."""
        lines = text.strip().split('\n')
        
        # Multiple lines suggest a list
        if len(lines) > 1:
            return True
        
        # Single line with common separators
        separators = [',', ';', '|']
        for sep in separators:
            if sep in text and len(text.split(sep)) > 1:
                return True
        
        return False
    
    def validate_output(self, parsed_output: Any) -> bool:
        """Validate that we got some output."""
        return parsed_output is not None