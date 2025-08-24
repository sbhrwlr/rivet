"""Tests for memory adapters."""

import os
import tempfile
import pytest
from rivet.memory.json_adapter import JSONAdapter
from rivet.memory.sqlite_adapter import SQLiteAdapter


def test_json_adapter():
    """Test JSON memory adapter."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
        adapter = JSONAdapter(f.name)
        
        # Test store and retrieve
        adapter.store("Hello", "Hi there")
        adapter.store("Weather", "It's sunny")
        
        memories = adapter.retrieve("Hello")
        assert len(memories) == 1
        assert "Hello" in memories[0]
        assert "Hi there" in memories[0]
        
        # Test clear
        adapter.clear()
        memories = adapter.retrieve("Hello")
        assert len(memories) == 0
        
        # Cleanup
        os.unlink(f.name)


def test_sqlite_adapter():
    """Test SQLite memory adapter."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
        adapter = SQLiteAdapter(f.name)
        
        # Test store and retrieve
        adapter.store("Question", "Answer")
        adapter.store("Another", "Response")
        
        memories = adapter.retrieve("Question")
        assert len(memories) == 1
        assert "Question" in memories[0]
        assert "Answer" in memories[0]
        
        # Test clear
        adapter.clear()
        memories = adapter.retrieve("Question")
        assert len(memories) == 0
        
        # Cleanup
        os.unlink(f.name)