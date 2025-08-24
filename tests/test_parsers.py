"""Tests for output parsers."""

import pytest
from rivet.parsers import OutputParser, JSONParser, ListParser, XMLParser

try:
    from pydantic import BaseModel
    from rivet.parsers import PydanticParser
    HAS_PYDANTIC = True
    
    class TestModel(BaseModel):
        name: str
        age: int
        active: bool = True
        
except ImportError:
    HAS_PYDANTIC = False


def test_json_parser_strict():
    """Test JSON parser in strict mode."""
    parser = JSONParser(strict=True)
    
    # Valid JSON
    result = parser.parse('{"name": "John", "age": 30}')
    assert result == {"name": "John", "age": 30}
    
    # Invalid JSON should raise error
    with pytest.raises(ValueError):
        parser.parse("This is not JSON")


def test_json_parser_non_strict():
    """Test JSON parser in non-strict mode."""
    parser = JSONParser(strict=False)
    
    # Valid JSON
    result = parser.parse('{"name": "John", "age": 30}')
    assert result == {"name": "John", "age": 30}
    
    # JSON embedded in text
    result = parser.parse('Here is the data: {"name": "John", "age": 30} and that\'s it.')
    assert result == {"name": "John", "age": 30}
    
    # Array
    result = parser.parse('The list is: [1, 2, 3]')
    assert result == [1, 2, 3]


@pytest.mark.skipif(not HAS_PYDANTIC, reason="Pydantic not available")
def test_pydantic_parser():
    """Test Pydantic parser."""
    parser = PydanticParser(TestModel)
    
    # Valid JSON
    result = parser.parse('{"name": "John", "age": 30}')
    assert isinstance(result, TestModel)
    assert result.name == "John"
    assert result.age == 30
    assert result.active is True
    
    # JSON embedded in text
    result = parser.parse('Here is the data: {"name": "Jane", "age": 25, "active": false}')
    assert isinstance(result, TestModel)
    assert result.name == "Jane"
    assert result.age == 25
    assert result.active is False
    
    # Invalid data should raise error
    with pytest.raises(ValueError):
        parser.parse('{"name": "John"}')  # Missing required age field


def test_list_parser_with_separator():
    """Test list parser with explicit separator."""
    parser = ListParser(separator=',')
    
    result = parser.parse('apple,banana,cherry')
    assert result == ['apple', 'banana', 'cherry']
    
    # With whitespace
    result = parser.parse('apple, banana , cherry ')
    assert result == ['apple', 'banana', 'cherry']


def test_list_parser_auto_detect():
    """Test list parser with auto-detection."""
    parser = ListParser()
    
    # Newline separated
    result = parser.parse('apple\nbanana\ncherry')
    assert result == ['apple', 'banana', 'cherry']
    
    # Comma separated
    result = parser.parse('apple, banana, cherry')
    assert result == ['apple', 'banana', 'cherry']
    
    # Numbered list
    result = parser.parse('1. apple\n2. banana\n3. cherry')
    assert result == ['apple', 'banana', 'cherry']
    
    # Bulleted list
    result = parser.parse('- apple\n- banana\n- cherry')
    assert result == ['apple', 'banana', 'cherry']


def test_list_parser_with_item_parser():
    """Test list parser with item parser."""
    parser = ListParser(separator=',', item_parser=int)
    
    result = parser.parse('1, 2, 3, 4')
    assert result == [1, 2, 3, 4]


def test_xml_parser():
    """Test XML parser."""
    parser = XMLParser()
    
    # Simple XML
    xml = '<person><name>John</name><age>30</age></person>'
    result = parser.parse(xml)
    assert result == {'name': 'John', 'age': '30'}
    
    # XML with attributes
    xml = '<person id="123"><name>John</name><age>30</age></person>'
    result = parser.parse(xml)
    assert result['@attributes'] == {'id': '123'}
    assert result['name'] == 'John'
    
    # XML with repeated elements
    xml = '<people><person>John</person><person>Jane</person></people>'
    result = parser.parse(xml)
    assert result == {'person': ['John', 'Jane']}


def test_xml_parser_with_root_tag():
    """Test XML parser with expected root tag."""
    parser = XMLParser(root_tag='person')
    
    # Correct root tag
    xml = '<person><name>John</name></person>'
    result = parser.parse(xml)
    assert result == {'name': 'John'}
    
    # Wrong root tag
    with pytest.raises(ValueError):
        parser.parse('<user><name>John</name></user>')


def test_parser_fallback():
    """Test parser fallback functionality."""
    parser = JSONParser(strict=True)
    
    # With fallback value
    result = parser.parse_with_fallback("invalid json", fallback={"error": "parse_failed"})
    assert result == {"error": "parse_failed"}
    
    # Without fallback value (returns original text)
    result = parser.parse_with_fallback("invalid json")
    assert result == "invalid json"
    
    # Valid parsing (no fallback needed)
    result = parser.parse_with_fallback('{"valid": "json"}')
    assert result == {"valid": "json"}


def test_parser_validation():
    """Test parser validation."""
    json_parser = JSONParser()
    list_parser = ListParser()
    
    # Valid outputs
    assert json_parser.validate_output({"key": "value"}) is True
    assert json_parser.validate_output([1, 2, 3]) is True
    assert list_parser.validate_output([1, 2, 3]) is True
    
    # Invalid outputs
    assert json_parser.validate_output("string") is False
    assert list_parser.validate_output("string") is False