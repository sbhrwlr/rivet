"""XML output parser."""

import xml.etree.ElementTree as ET
from typing import Dict, Any, Union
from .base import OutputParser


class XMLParser(OutputParser):
    """Parser for XML responses."""
    
    def __init__(self, root_tag: str = None):
        """
        Initialize XML parser.
        
        Args:
            root_tag: Expected root tag name. If None, accepts any root tag.
        """
        self.root_tag = root_tag
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse text as XML and convert to dictionary."""
        try:
            # Clean up the text
            text = text.strip()
            
            # Parse XML
            root = ET.fromstring(text)
            
            # Validate root tag if specified
            if self.root_tag and root.tag != self.root_tag:
                raise ValueError(f"Expected root tag '{self.root_tag}', got '{root.tag}'")
            
            # Convert to dictionary
            return self._element_to_dict(root)
            
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML: {str(e)}")
    
    def _element_to_dict(self, element: ET.Element) -> Dict[str, Any]:
        """Convert XML element to dictionary."""
        result = {}
        
        # Add attributes
        if element.attrib:
            result['@attributes'] = element.attrib
        
        # Add text content
        if element.text and element.text.strip():
            if len(element) == 0:  # Leaf node
                return element.text.strip()
            else:
                result['@text'] = element.text.strip()
        
        # Add child elements
        children = {}
        for child in element:
            child_data = self._element_to_dict(child)
            
            if child.tag in children:
                # Multiple children with same tag - convert to list
                if not isinstance(children[child.tag], list):
                    children[child.tag] = [children[child.tag]]
                children[child.tag].append(child_data)
            else:
                children[child.tag] = child_data
        
        result.update(children)
        
        # If only text content, return just the text
        if len(result) == 1 and '@text' in result:
            return result['@text']
        
        return result
    
    def validate_output(self, parsed_output: Any) -> bool:
        """Validate that output is a dictionary."""
        return isinstance(parsed_output, (dict, str))