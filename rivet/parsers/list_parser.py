"""List output parser."""

import re
from typing import List, Union, Callable, Optional
from .base import OutputParser


class ListParser(OutputParser):
    """Parser for list responses."""
    
    def __init__(
        self, 
        separator: Optional[str] = None,
        item_parser: Optional[Callable[[str], any]] = None,
        strip_items: bool = True
    ):
        """
        Initialize list parser.
        
        Args:
            separator: String to split on. If None, tries to detect common separators.
            item_parser: Function to parse individual items.
            strip_items: Whether to strip whitespace from items.
        """
        self.separator = separator
        self.item_parser = item_parser or (lambda x: x)
        self.strip_items = strip_items
    
    def parse(self, text: str) -> List[any]:
        """Parse text into a list."""
        if self.separator:
            items = text.split(self.separator)
        else:
            items = self._auto_split(text)
        
        if self.strip_items:
            items = [item.strip() for item in items]
        
        # Remove empty items
        items = [item for item in items if item]
        
        # Apply item parser
        try:
            return [self.item_parser(item) for item in items]
        except Exception as e:
            raise ValueError(f"Failed to parse list items: {str(e)}")
    
    def _auto_split(self, text: str) -> List[str]:
        """Automatically detect separator and split text."""
        # Try numbered/bulleted lists first
        if self._is_numbered_list(text):
            return self._parse_numbered_list(text)
        
        # Try common separators in order of preference
        separators = ['\n', ',', ';', '|', '\t']
        
        for sep in separators:
            if sep in text:
                parts = text.split(sep)
                if len(parts) > 1:
                    return parts
        
        # Fallback to single item
        return [text]
    
    def _is_numbered_list(self, text: str) -> bool:
        """Check if text appears to be a numbered or bulleted list."""
        patterns = [
            r'^\d+\.',  # 1. 2. 3.
            r'^[-*•]',  # - * •
            r'^\([a-zA-Z0-9]+\)',  # (a) (1) (i)
        ]
        
        lines = text.strip().split('\n')
        if len(lines) < 2:
            return False
        
        for pattern in patterns:
            matching_lines = [line for line in lines if line.strip() and re.match(pattern, line.strip())]
            if len(matching_lines) >= 2:  # At least 2 lines match the pattern
                return True
        
        return False
    
    def _parse_numbered_list(self, text: str) -> List[str]:
        """Parse numbered or bulleted list."""
        lines = text.strip().split('\n')
        items = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove common list prefixes
            line = re.sub(r'^\d+\.\s*', '', line)  # 1. 2. 3.
            line = re.sub(r'^[-*•]\s*', '', line)  # - * •
            line = re.sub(r'^\([a-zA-Z0-9]+\)\s*', '', line)  # (a) (1)
            
            if line:
                items.append(line)
        
        return items
    
    def validate_output(self, parsed_output: any) -> bool:
        """Validate that output is a list."""
        return isinstance(parsed_output, list)