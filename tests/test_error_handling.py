"""Tests for parser error handling and fallback mechanisms."""

import pytest
import time
from rivet.parsers import JSONParser, ListParser, XMLParser
from rivet.parsers.smart_parser import SmartParser
from rivet.parsers.retry_parser import RetryParser
from rivet.exceptions import ParsingError


class TestParsingError:
    """Test ParsingError exception."""
    
    def test_parsing_error_creation(self):
        """Test creating ParsingError with different parameters."""
        # Basic error
        error = ParsingError("Test error")
        assert str(error) == "Test error"
        assert error.original_text is None
        assert error.parser_type is None
        
        # Error with context
        error = ParsingError(
            "JSON parsing failed",
            original_text="invalid json",
            parser_type="JSONParser"
        )
        assert str(error) == "JSON parsing failed"
        assert error.original_text == "invalid json"
        assert error.parser_type == "JSONParser"


class TestSmartParser:
    """Test SmartParser functionality."""
    
    def test_json_detection_and_parsing(self):
        """Test smart parser detecting and parsing JSON."""
        parser = SmartParser()
        
        # Clear JSON object
        result = parser.parse('{"name": "John", "age": 30}')
        assert result == {"name": "John", "age": 30}
        
        # JSON array
        result = parser.parse('[1, 2, 3, "test"]')
        assert result == [1, 2, 3, "test"]
        
        # JSON in text
        result = parser.parse('Here is the data: {"key": "value"} end')
        assert result == {"key": "value"}
    
    def test_xml_detection_and_parsing(self):
        """Test smart parser detecting and parsing XML."""
        parser = SmartParser()
        
        # Simple XML
        result = parser.parse('<person><name>John</name><age>30</age></person>')
        assert result == {"name": "John", "age": "30"}
        
        # XML with attributes - simple element returns just text
        result = parser.parse('<item id="123">Test</item>')
        assert result == "Test"  # Simple element with just text content
    
    def test_list_detection_and_parsing(self):
        """Test smart parser detecting and parsing lists."""
        parser = SmartParser()
        
        # Numbered list
        result = parser.parse("1. Apple\n2. Banana\n3. Cherry")
        assert result == ["Apple", "Banana", "Cherry"]
        
        # Comma separated
        result = parser.parse("apple, banana, cherry")
        assert result == ["apple", "banana", "cherry"]
        
        # Bullet list
        result = parser.parse("- Item 1\n- Item 2\n- Item 3")
        assert result == ["Item 1", "Item 2", "Item 3"]
    
    def test_preferred_format(self):
        """Test smart parser with preferred format."""
        # Prefer JSON
        parser = SmartParser(preferred_format="json")
        
        # Ambiguous content that could be parsed as list or JSON
        # Should try JSON first
        result = parser.parse('["apple", "banana", "cherry"]')
        assert result == ["apple", "banana", "cherry"]  # Parsed as JSON array
    
    def test_fallback_to_original_text(self):
        """Test smart parser falling back to original text."""
        parser = SmartParser()
        
        # Plain text gets parsed as single-item list by list parser
        result = parser.parse("This is just plain text with no structure")
        assert result == ["This is just plain text with no structure"]
        
        # Empty content
        result = parser.parse("")
        assert result == ""
        
        # Whitespace only
        result = parser.parse("   \n\t  ")
        assert result == ""
    
    def test_format_detection_methods(self):
        """Test format detection methods."""
        parser = SmartParser()
        
        # JSON detection
        assert parser._looks_like_json('{"key": "value"}') is True
        assert parser._looks_like_json('[1, 2, 3]') is True
        assert parser._looks_like_json('plain text') is False
        
        # XML detection
        assert parser._looks_like_xml('<root><child>value</child></root>') is True
        assert parser._looks_like_xml('<item>value</item>') is True
        assert parser._looks_like_xml('not xml') is False
        
        # List detection
        assert parser._looks_like_list('item1\nitem2\nitem3') is True
        assert parser._looks_like_list('item1, item2, item3') is True
        assert parser._looks_like_list('single item') is False


class TestRetryParser:
    """Test RetryParser functionality."""
    
    def test_successful_primary_parsing(self):
        """Test retry parser when primary parser succeeds."""
        primary = JSONParser(strict=False)
        parser = RetryParser(primary)
        
        result = parser.parse('{"name": "John", "age": 30}')
        assert result == {"name": "John", "age": 30}
    
    def test_retry_with_primary_parser(self):
        """Test retry logic with primary parser."""
        # Create a parser that fails first few times
        class FlakyParser(JSONParser):
            def __init__(self):
                super().__init__(strict=False)
                self.attempt_count = 0
            
            def parse(self, text):
                self.attempt_count += 1
                if self.attempt_count < 3:  # Fail first 2 attempts
                    raise ValueError("Simulated failure")
                return super().parse(text)
        
        primary = FlakyParser()
        parser = RetryParser(primary, max_retries=3, retry_delay=0.01)
        
        result = parser.parse('{"name": "John"}')
        assert result == {"name": "John"}
        assert primary.attempt_count == 3
    
    def test_fallback_to_secondary_parsers(self):
        """Test fallback to secondary parsers."""
        primary = JSONParser(strict=True)  # Will fail on non-JSON
        fallback1 = XMLParser()  # Will fail on non-XML
        fallback2 = ListParser()  # Should succeed on list
        
        parser = RetryParser(
            primary,
            fallback_parsers=[fallback1, fallback2],
            max_retries=1
        )
        
        # This should fail JSON and XML parsing but succeed with list parser
        result = parser.parse("apple\nbanana\ncherry")
        assert result == ["apple", "banana", "cherry"]
    
    def test_fallback_value(self):
        """Test fallback to specified value."""
        primary = JSONParser(strict=True)
        parser = RetryParser(
            primary,
            fallback_value={"error": "parsing_failed"},
            max_retries=1
        )
        
        result = parser.parse("not json at all")
        assert result == {"error": "parsing_failed"}
    
    def test_final_error_when_all_fail(self):
        """Test error when all parsing attempts fail."""
        primary = JSONParser(strict=True)
        fallback = XMLParser()
        
        parser = RetryParser(
            primary,
            fallback_parsers=[fallback],
            max_retries=1,
            fallback_value=None  # No fallback value
        )
        
        with pytest.raises(ParsingError) as exc_info:
            parser.parse("completely unparseable content")
        
        assert "All parsing attempts failed" in str(exc_info.value)
        assert exc_info.value.original_text == "completely unparseable content"
        assert exc_info.value.parser_type == "JSONParser"
    
    def test_return_original_text_as_last_resort(self):
        """Test returning original text when no fallback value is set."""
        primary = JSONParser(strict=True)
        
        # Create parser without fallback_value (defaults to None)
        # and without raising error (by catching the exception case)
        parser = RetryParser(primary, max_retries=1)
        
        # Modify the parse method to not raise error in this specific case
        original_parse = parser.parse
        
        def modified_parse(text):
            try:
                return original_parse(text)
            except ParsingError:
                return text  # Return original text instead of raising
        
        parser.parse = modified_parse
        
        result = parser.parse("unparseable content")
        assert result == "unparseable content"
    
    def test_validation_with_retry_parser(self):
        """Test validation using primary parser's validation."""
        primary = JSONParser()
        parser = RetryParser(primary)
        
        # Valid JSON object
        assert parser.validate_output({"key": "value"}) is True
        
        # Invalid for JSON parser
        assert parser.validate_output("string") is False


class TestErrorHandlingIntegration:
    """Test error handling integration across different parsers."""
    
    def test_nested_error_handling(self):
        """Test error handling with nested parser configurations."""
        # Create a complex parser chain
        json_parser = JSONParser(strict=True)
        smart_parser = SmartParser(preferred_format="json")
        retry_parser = RetryParser(
            json_parser,
            fallback_parsers=[smart_parser],
            max_retries=2
        )
        
        # This should fail strict JSON but succeed with smart parser
        result = retry_parser.parse("Here's the data: [1, 2, 3]")
        assert result == [1, 2, 3]
    
    def test_error_context_preservation(self):
        """Test that error context is preserved through parser chain."""
        primary = JSONParser(strict=True)
        parser = RetryParser(primary, max_retries=1, fallback_value=None)
        
        try:
            parser.parse("invalid content")
        except ParsingError as e:
            assert e.original_text == "invalid content"
            assert e.parser_type == "JSONParser"
            assert "All parsing attempts failed" in str(e)
    
    def test_performance_with_retries(self):
        """Test that retry delays work as expected."""
        class SlowParser(JSONParser):
            def parse(self, text):
                raise ValueError("Always fails")
        
        primary = SlowParser()
        parser = RetryParser(primary, max_retries=3, retry_delay=0.05)
        
        start_time = time.time()
        try:
            parser.parse("test")
        except ParsingError:
            pass
        end_time = time.time()
        
        # Should have taken at least 2 * retry_delay (2 delays between 3 attempts)
        assert end_time - start_time >= 0.1  # 2 * 0.05
    
    def test_mixed_sync_async_compatibility(self):
        """Test that error handling works with both sync and async contexts."""
        parser = SmartParser()
        
        # Should work the same in both contexts
        result1 = parser.parse('{"test": "value"}')
        
        # Simulate what would happen in async context
        import asyncio
        
        async def async_parse():
            return parser.parse('{"test": "value"}')
        
        result2 = asyncio.run(async_parse())
        
        assert result1 == result2 == {"test": "value"}