# Rivet 🔗

A lightweight, developer-first framework to build AI agents.

**Philosophy: Less abstraction, more action.**

## Why Rivet?

Rivet fastens together Models, Memory, and Tools with minimal boilerplate. No "chains of chains" - just simple, hackable components that work together.

## Quick Start

```python
from rivet import Agent, tool
from rivet.models.openai_adapter import OpenAIAdapter
from rivet.memory.json_adapter import JSONAdapter

# Define a tool
@tool("get_time")
def get_current_time():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Create agent
agent = Agent(
    model=OpenAIAdapter(),
    memory=JSONAdapter()
)

# Run it
response = agent.run("What time is it?")
print(response)
```

That's it! Your agent has memory, can call tools, and is ready to go.

## Core Components

### Agent

The central orchestrator that coordinates everything:

- Manages conversation flow
- Retrieves relevant memories
- Executes tool calls
- Handles model interactions

### Model Adapters

Plug-and-play wrappers for different LLM providers:

- `OpenAIAdapter` - GPT models
- Easy to extend for Anthropic, local models, etc.

### Memory Adapters

Storage backends for conversation history:

- `JSONAdapter` - Simple file-based storage
- `SQLiteAdapter` - Persistent database storage
- Extensible for vector databases, Redis, etc.

### Tools

Simple callable functions with decorator registration:

```python
@tool("weather")
def get_weather(city: str):
    return f"It's sunny in {city}"
```

### Inspector

Transparent debugging layer:

```python
from rivet import Inspector

inspector = Inspector(log_file="debug.log")
agent = Agent(model=model, inspector=inspector)

# View logs
print(inspector.summary())
```

## Architecture Principles

1. **Simple API** - MVP agent in <10 lines
2. **Adapter Pattern** - Easy to extend and swap components
3. **Batteries Included** - Works out of the box with sensible defaults
4. **Transparent Debugging** - See exactly what your agent is doing
5. **No Heavy Abstractions** - Readable, hackable code
6. **Developer First** - Built for indie developers who want control

## Installation

```bash
pip install rivet-ai
```

Optional dependencies:

```bash
pip install openai  # For OpenAI models
```

## Examples

### Basic Chat Agent

```python
from rivet import Agent
from rivet.models.openai_adapter import OpenAIAdapter

agent = Agent(model=OpenAIAdapter())
response = agent.run("Hello, how are you?")
```

### Agent with Memory

```python
from rivet.memory.sqlite_adapter import SQLiteAdapter

agent = Agent(
    model=OpenAIAdapter(),
    memory=SQLiteAdapter("my_agent.db")
)

# Conversations are remembered
agent.run("My name is Alice")
agent.run("What's my name?")  # Will remember Alice
```

### Agent with Tools

```python
@tool("calculate")
def calculator(expression: str):
    return eval(expression)  # Don't do this in production!

agent = Agent(
    model=OpenAIAdapter(),
    tools=get_registry()  # Uses global tool registry
)

agent.run("What's 15 * 23?")
```

## Extending Rivet

### Custom Model Adapter

```python
from rivet.models.base import ModelAdapter

class MyModelAdapter(ModelAdapter):
    def generate(self, prompt: str, available_tools=None) -> str:
        # Your model logic here
        return "Response from my model"

    def configure(self, **kwargs) -> None:
        # Configuration logic
        pass
```

### Custom Memory Adapter

```python
from rivet.memory.base import MemoryAdapter

class RedisAdapter(MemoryAdapter):
    def store(self, input_text: str, output_text: str, metadata=None) -> None:
        # Store in Redis
        pass

    def retrieve(self, query: str, limit: int = 5) -> List[str]:
        # Retrieve from Redis
        return []

    def clear(self) -> None:
        # Clear Redis
        pass
```

## Contributing

Rivet is designed to be hackable. The codebase is intentionally simple and well-documented.

1. Fork the repo
2. Make your changes
3. Add tests
4. Submit a PR

## License

MIT License - see LICENSE file for details.

---

**Built with ❤️ for developers who want to build AI agents, not fight frameworks.**
