"""Tests for Agent integration with output parsing."""

import pytest
from rivet import Agent, JSONParser, ListParser, XMLParser, SmartParser, RetryParser
from rivet.models.base import ModelAdapter
from rivet.memory.json_adapter import JSONAdapter
from rivet.tools import ToolRegistry, tool

try:
    from pydantic import BaseModel
    from rivet.parsers import PydanticParser
    HAS_PYDANTIC = True
    
    class TaskModel(BaseModel):
        task: str
        priority: int
        completed: bool = False
        
except ImportError:
    HAS_PYDANTIC = False
    PydanticParser = None
    TaskModel = None


class MockModelAdapter(ModelAdapter):
    """Mock model adapter for testing."""
    
    def __init__(self, response: str):
        self.response = response
        
    def generate(self, prompt: str, available_tools=None) -> str:
        return self.response
        
    async def agenerate(self, prompt: str, available_tools=None) -> str:
        return self.response
        
    async def stream(self, prompt: str, available_tools=None):
        # Simple streaming - yield response in chunks
        chunk_size = 10
        for i in range(0, len(self.response), chunk_size):
            yield self.response[i:i+chunk_size]
    
    def configure(self, **kwargs):
        pass


class TestAgentJSONParsing:
    """Test Agent with JSON output parsing."""
    
    def test_agent_json_parsing_success(self):
        """Test successful JSON parsing."""
        model = MockModelAdapter('{"task": "Complete project", "priority": 1}')
        parser = JSONParser(strict=False)
        agent = Agent(model=model, output_parser=parser)
        
        result = agent.run("Create a task")
        
        assert isinstance(result, dict)
        assert result["task"] == "Complete project"
        assert result["priority"] == 1
    
    def test_agent_json_parsing_with_text(self):
        """Test JSON parsing from text with embedded JSON."""
        model = MockModelAdapter('Here is your task: {"task": "Review code", "priority": 2}')
        parser = JSONParser(strict=False)
        agent = Agent(model=model, output_parser=parser)
        
        result = agent.run("Create a task")
        
        assert isinstance(result, dict)
        assert result["task"] == "Review code"
        assert result["priority"] == 2
    
    def test_agent_json_parsing_fallback(self):
        """Test JSON parsing fallback to original text."""
        model = MockModelAdapter('This is not JSON at all')
        parser = JSONParser(strict=True)  # Will fail on non-JSON
        agent = Agent(model=model, output_parser=parser)
        
        result = agent.run("Create a task")
        
        # Should fallback to original text
        assert result == 'This is not JSON at all'
    
    @pytest.mark.asyncio
    async def test_agent_async_json_parsing(self):
        """Test async agent with JSON parsing."""
        model = MockModelAdapter('{"task": "Async task", "priority": 3}')
        parser = JSONParser(strict=False)
        agent = Agent(model=model, output_parser=parser)
        
        result = await agent.arun("Create an async task")
        
        assert isinstance(result, dict)
        assert result["task"] == "Async task"
        assert result["priority"] == 3


@pytest.mark.skipif(not HAS_PYDANTIC, reason="Pydantic not available")
class TestAgentPydanticParsing:
    """Test Agent with Pydantic model parsing."""
    
    def test_agent_pydantic_parsing_success(self):
        """Test successful Pydantic model parsing."""
        model = MockModelAdapter('{"task": "Write tests", "priority": 1, "completed": false}')
        parser = PydanticParser(TaskModel)
        agent = Agent(model=model, output_parser=parser)
        
        result = agent.run("Create a task")
        
        assert isinstance(result, TaskModel)
        assert result.task == "Write tests"
        assert result.priority == 1
        assert result.completed is False
    
    def test_agent_pydantic_parsing_with_defaults(self):
        """Test Pydantic parsing with default values."""
        model = MockModelAdapter('{"task": "Default task", "priority": 2}')
        parser = PydanticParser(TaskModel)
        agent = Agent(model=model, output_parser=parser)
        
        result = agent.run("Create a task")
        
        assert isinstance(result, TaskModel)
        assert result.task == "Default task"
        assert result.priority == 2
        assert result.completed is False  # Default value
    
    def test_agent_pydantic_parsing_validation_error(self):
        """Test Pydantic parsing with validation error."""
        model = MockModelAdapter('{"task": "Invalid task"}')  # Missing required priority
        parser = PydanticParser(TaskModel)
        agent = Agent(model=model, output_parser=parser)
        
        result = agent.run("Create a task")
        
        # Should fallback to original text on validation error
        assert result == '{"task": "Invalid task"}'


class TestAgentListParsing:
    """Test Agent with list output parsing."""
    
    def test_agent_list_parsing_numbered(self):
        """Test list parsing with numbered list."""
        model = MockModelAdapter("1. Buy groceries\n2. Walk the dog\n3. Finish report")
        parser = ListParser()
        agent = Agent(model=model, output_parser=parser)
        
        result = agent.run("List my tasks")
        
        assert isinstance(result, list)
        assert result == ["Buy groceries", "Walk the dog", "Finish report"]
    
    def test_agent_list_parsing_comma_separated(self):
        """Test list parsing with comma-separated values."""
        model = MockModelAdapter("apple, banana, cherry, date")
        parser = ListParser()
        agent = Agent(model=model, output_parser=parser)
        
        result = agent.run("List fruits")
        
        assert isinstance(result, list)
        assert result == ["apple", "banana", "cherry", "date"]
    
    def test_agent_list_parsing_with_separator(self):
        """Test list parsing with custom separator."""
        model = MockModelAdapter("red|green|blue|yellow")
        parser = ListParser(separator="|")
        agent = Agent(model=model, output_parser=parser)
        
        result = agent.run("List colors")
        
        assert isinstance(result, list)
        assert result == ["red", "green", "blue", "yellow"]


class TestAgentXMLParsing:
    """Test Agent with XML output parsing."""
    
    def test_agent_xml_parsing_simple(self):
        """Test simple XML parsing."""
        model = MockModelAdapter('<person><name>John</name><age>30</age></person>')
        parser = XMLParser()
        agent = Agent(model=model, output_parser=parser)
        
        result = agent.run("Get person data")
        
        assert isinstance(result, dict)
        assert result["name"] == "John"
        assert result["age"] == "30"
    
    def test_agent_xml_parsing_with_attributes(self):
        """Test XML parsing with attributes."""
        model = MockModelAdapter('<task id="123" status="pending">Complete project</task>')
        parser = XMLParser()
        agent = Agent(model=model, output_parser=parser)
        
        result = agent.run("Get task")
        
        # Simple element with attributes returns just the text content
        assert result == "Complete project"


class TestAgentSmartParsing:
    """Test Agent with smart parser."""
    
    def test_agent_smart_parsing_json(self):
        """Test smart parser detecting JSON."""
        model = MockModelAdapter('{"type": "json", "data": [1, 2, 3]}')
        parser = SmartParser()
        agent = Agent(model=model, output_parser=parser)
        
        result = agent.run("Get data")
        
        assert isinstance(result, dict)
        assert result["type"] == "json"
        assert result["data"] == [1, 2, 3]
    
    def test_agent_smart_parsing_list(self):
        """Test smart parser detecting list."""
        model = MockModelAdapter("- Task 1\n- Task 2\n- Task 3")
        parser = SmartParser()
        agent = Agent(model=model, output_parser=parser)
        
        result = agent.run("Get tasks")
        
        assert isinstance(result, list)
        assert result == ["Task 1", "Task 2", "Task 3"]
    
    def test_agent_smart_parsing_xml(self):
        """Test smart parser detecting XML."""
        model = MockModelAdapter('<config><debug>true</debug><port>8080</port></config>')
        parser = SmartParser()
        agent = Agent(model=model, output_parser=parser)
        
        result = agent.run("Get config")
        
        assert isinstance(result, dict)
        assert result["debug"] == "true"
        assert result["port"] == "8080"
    
    def test_agent_smart_parsing_preferred_format(self):
        """Test smart parser with preferred format."""
        model = MockModelAdapter('[1, 2, 3, 4]')  # Could be JSON or list
        parser = SmartParser(preferred_format="json")
        agent = Agent(model=model, output_parser=parser)
        
        result = agent.run("Get numbers")
        
        # Should be parsed as JSON array
        assert isinstance(result, list)
        assert result == [1, 2, 3, 4]


class TestAgentRetryParsing:
    """Test Agent with retry parser."""
    
    def test_agent_retry_parsing_success(self):
        """Test retry parser succeeding on first try."""
        model = MockModelAdapter('{"success": true, "message": "Task completed"}')
        primary_parser = JSONParser(strict=False)
        parser = RetryParser(primary_parser)
        agent = Agent(model=model, output_parser=parser)
        
        result = agent.run("Complete task")
        
        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["message"] == "Task completed"
    
    def test_agent_retry_parsing_fallback(self):
        """Test retry parser falling back to secondary parser."""
        model = MockModelAdapter("Task 1\nTask 2\nTask 3")  # Not JSON, but valid list
        primary_parser = JSONParser(strict=True)  # Will fail
        fallback_parser = ListParser()  # Will succeed
        parser = RetryParser(primary_parser, fallback_parsers=[fallback_parser])
        agent = Agent(model=model, output_parser=parser)
        
        result = agent.run("Get tasks")
        
        assert isinstance(result, list)
        assert result == ["Task 1", "Task 2", "Task 3"]
    
    def test_agent_retry_parsing_with_fallback_value(self):
        """Test retry parser with fallback value."""
        model = MockModelAdapter("Completely unparseable content")
        primary_parser = JSONParser(strict=True)
        parser = RetryParser(
            primary_parser, 
            fallback_value={"error": "parsing_failed", "raw": "Completely unparseable content"}
        )
        agent = Agent(model=model, output_parser=parser)
        
        result = agent.run("Parse this")
        
        assert isinstance(result, dict)
        assert result["error"] == "parsing_failed"
        assert result["raw"] == "Completely unparseable content"


class TestAgentParsingWithMemory:
    """Test Agent parsing integration with memory."""
    
    def test_agent_parsing_with_memory_storage(self):
        """Test that raw response is stored in memory, not parsed result."""
        model = MockModelAdapter('{"task": "Test task", "priority": 1}')
        parser = JSONParser(strict=False)
        memory = JSONAdapter()
        agent = Agent(model=model, output_parser=parser, memory=memory)
        
        result = agent.run("Create a task")
        
        # Result should be parsed
        assert isinstance(result, dict)
        assert result["task"] == "Test task"
        
        # Memory should contain raw response
        memories = memory.retrieve("Create a task")
        assert len(memories) > 0
        # The stored response should be the raw JSON string
        assert '{"task": "Test task", "priority": 1}' in memories[0]


class TestAgentParsingWithTools:
    """Test Agent parsing integration with tools."""
    
    def test_agent_parsing_with_tool_calls(self):
        """Test parsing works with tool execution."""
        def get_time():
            """Get current time."""
            return "2023-01-01 12:00:00"
        
        tools = ToolRegistry()
        tools.register("get_time", get_time)
        
        model = MockModelAdapter('TOOL_CALL: get_time\n{"result": "time_retrieved"}')
        parser = JSONParser(strict=False)
        agent = Agent(model=model, tools=tools, output_parser=parser)
        
        result = agent.run("What time is it?")
        
        # Should parse the JSON part after tool execution
        assert isinstance(result, dict)
        assert result["result"] == "time_retrieved"
    
    def test_agent_parsing_tool_result_only(self):
        """Test parsing when response is only tool results."""
        def get_data():
            """Get structured data."""
            return '{"data": "from_tool", "source": "tool"}'
        
        tools = ToolRegistry()
        tools.register("get_data", get_data)
        
        model = MockModelAdapter('TOOL_CALL: get_data')
        parser = JSONParser(strict=False)
        agent = Agent(model=model, tools=tools, output_parser=parser)
        
        result = agent.run("Get data")
        
        # Should parse the tool result
        assert isinstance(result, dict)
        assert result["data"] == "from_tool"
        assert result["source"] == "tool"


class TestAgentParsingBackwardCompatibility:
    """Test backward compatibility when no parser is specified."""
    
    def test_agent_without_parser_returns_string(self):
        """Test that agent without parser returns raw string."""
        model = MockModelAdapter('{"this": "is", "raw": "json"}')
        agent = Agent(model=model)  # No parser specified
        
        result = agent.run("Get data")
        
        # Should return raw string
        assert isinstance(result, str)
        assert result == '{"this": "is", "raw": "json"}'
    
    def test_agent_none_parser_returns_string(self):
        """Test that agent with None parser returns raw string."""
        model = MockModelAdapter('Raw response text')
        agent = Agent(model=model, output_parser=None)
        
        result = agent.run("Get data")
        
        # Should return raw string
        assert isinstance(result, str)
        assert result == 'Raw response text'