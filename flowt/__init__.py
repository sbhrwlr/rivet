"""LLM Pipeline Framework - A lightweight, type-safe framework for building LLM applications."""

__version__ = "0.1.0"

from .core.interfaces import LLMProvider, VectorStore
from .core.models import Document, LLMResponse, Message, SearchResult, Vector
from .core.pipeline import Pipeline, PipelineContext, PipelineStep

__all__ = [
    "Message",
    "LLMResponse",
    "Document",
    "Vector",
    "SearchResult",
    "LLMProvider",
    "VectorStore",
    "Pipeline",
    "PipelineStep",
    "PipelineContext",
]
