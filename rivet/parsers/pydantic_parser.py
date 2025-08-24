"""Pydantic output parser."""

import json
from typing import Type, TypeVar, Any
from .base import OutputParser

try:
    from pydantic import BaseModel, ValidationError
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = None
    ValidationError = Exception

T = TypeVar('T', bound='BaseModel')


class PydanticParser(OutputParser):
    """Parser for Pydantic models."""
    
    def __init__(self, model_class: Type[T]):
        """
        Initialize Pydantic parser.
        
        Args:
            model_class: Pydantic model class to parse into.
        """
        if not HAS_PYDANTIC:
            raise ImportError("Pydantic is required for PydanticParser. Install with: pip install pydantic")
        
        if not issubclass(model_class, BaseModel):
            raise ValueError("model_class must be a Pydantic BaseModel")
        
        self.model_class = model_class
    
    def parse(self, text: str) -> T:
        """Parse text into Pydantic model."""
        # Try to parse as JSON first
        try:
            data = json.loads(text)
            return self.model_class(**data)
        except (json.JSONDecodeError, ValidationError):
            pass
        
        # Try to extract JSON from text
        from .json_parser import JSONParser
        json_parser = JSONParser(strict=False)
        
        try:
            data = json_parser.parse(text)
            return self.model_class(**data)
        except (ValueError, ValidationError) as e:
            raise ValueError(f"Could not parse text into {self.model_class.__name__}: {str(e)}")
    
    def validate_output(self, parsed_output: Any) -> bool:
        """Validate that output is an instance of the model class."""
        return isinstance(parsed_output, self.model_class)
    
    def get_schema(self) -> dict:
        """Get JSON schema for the model."""
        if hasattr(self.model_class, 'model_json_schema'):
            return self.model_class.model_json_schema()
        elif hasattr(self.model_class, 'schema'):
            return self.model_class.schema()
        else:
            return {}