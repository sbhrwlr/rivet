# LLM Pipeline Framework

A lightweight, type-safe Python framework for building LLM applications with explicit pipeline definitions and minimal dependencies.

## Features

- **Type Safety**: Complete type annotations with runtime validation via Pydantic
- **Minimal Dependencies**: Fewer than 10 direct dependencies
- **Explicit Pipelines**: Clear, explicit arrays of steps with no hidden logic
- **Pluggable Adapters**: Support for multiple LLM providers and vector stores
- **Production Ready**: Built-in retry logic, logging, tracing, and error handling
- **Developer Experience**: Excellent IDE support with full autocomplete

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv add llm-pipeline-framework

# Or using pip
pip install llm-pipeline-framework
```

### Basic Usage

```python
from flowt import Pipeline, PipelineStep, PipelineContext

# Define a simple pipeline step
class EchoStep(PipelineStep[str, str]):
    async def execute(self, input_data: str, context: PipelineContext) -> str:
        return f"Echo: {input_data}"

# Create and execute a pipeline
pipeline = Pipeline([EchoStep()])
result = await pipeline.execute("Hello, world!")
print(result)  # Output: Echo: Hello, world!
```

### OpenAI Provider Example

```python
import asyncio
import os
from flowt.providers import OpenAIProvider
from flowt.core.models import GenerationConfig

async def main():
    # Set your API key: export OPENAI_API_KEY="your-key-here"
    provider = OpenAIProvider()

    config = GenerationConfig(
        temperature=0.7,
        max_tokens=100,
    )

    async with provider as p:
        response = await p.generate(
            "Explain quantum computing in simple terms.",
            config
        )
        print(response.content)

asyncio.run(main())
```

For more comprehensive examples, see:

- [`examples/basic_pipeline.py`](examples/basic_pipeline.py) - Introduction to pipeline framework
- [`examples/quick_start.py`](examples/quick_start.py) - Simple getting started examples
- [`examples/simple_pipeline.py`](examples/simple_pipeline.py) - Common pipeline patterns with OpenAI
- [`examples/pipeline_with_openai.py`](examples/pipeline_with_openai.py) - Advanced pipeline workflows
- [`examples/openai_provider_example.py`](examples/openai_provider_example.py) - Direct provider usage

## Core Concepts

### Pipeline Steps

Pipeline steps are the building blocks of the framework. Each step takes an input of type `T` and produces an output of type `U`:

```python
class MyStep(PipelineStep[InputType, OutputType]):
    async def execute(self, input_data: InputType, context: PipelineContext) -> OutputType:
        # Your processing logic here
        return processed_output
```

### Providers

The framework uses abstract providers for LLM and vector store integrations:

- `LLMProvider`: Abstract base for LLM integrations (OpenAI, Anthropic, etc.)
- `VectorStore`: Abstract base for vector databases (Qdrant, Pinecone, etc.)

### Data Models

Core data models with full type safety:

- `Message`: Conversation messages with roles and metadata
- `Document`: Text documents with optional embeddings
- `Vector`: Embedding vectors with IDs and metadata
- `LLMResponse`: Structured LLM responses with usage statistics

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and Python tooling.

### Installing uv

If you don't have uv installed, you can install it using:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv

# Or with homebrew (macOS)
brew install uv
```

### Setup

```bash
# Clone the repository
git clone https://github.com/llm-pipeline-framework/llm-pipeline-framework.git
cd llm-pipeline-framework

# Install with uv (recommended)
uv sync --dev

# Or install with pip
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
uv run pre-commit install
```

### Common Commands

We provide a Makefile for common development tasks:

```bash
# Show all available commands
make help

# Install dependencies
make dev-install

# Run all checks (lint, type-check, test)
make check

# Run tests
make test

# Run tests with coverage
make test-cov

# Run linting
make lint

# Format code
make format

# Run type checking
make type-check

# Run the example
make example

# Clean build artifacts
make clean
```

### Manual Commands

If you prefer to run commands directly:

```bash
# Running Tests
uv run pytest tests/ -v

# Type Checking
uv run mypy flowt

# Linting and Formatting
uv run ruff check flowt
uv run ruff format flowt

# Run example
uv run python examples/basic_pipeline.py
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.
