"""SQLite memory adapter for persistent storage."""

import sqlite3
import json
from typing import List, Optional
from .base import MemoryAdapter


class SQLiteAdapter(MemoryAdapter):
    """SQLite-based memory storage."""
    
    def __init__(self, db_path: str = "rivet_memory.db"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input_text TEXT NOT NULL,
                    output_text TEXT NOT NULL,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            
    def store(self, input_text: str, output_text: str, metadata: Optional[dict] = None) -> None:
        """Store interaction in SQLite database."""
        metadata_json = json.dumps(metadata) if metadata else None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO memories (input_text, output_text, metadata) VALUES (?, ?, ?)",
                (input_text, output_text, metadata_json)
            )
            conn.commit()
            
    def retrieve(self, query: str, limit: int = 5) -> List[str]:
        """Retrieve memories matching the query."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT input_text, output_text FROM memories 
                WHERE input_text LIKE ? OR output_text LIKE ?
                ORDER BY timestamp DESC LIMIT ?
                """,
                (f"%{query}%", f"%{query}%", limit)
            )
            
            results = []
            for row in cursor.fetchall():
                results.append(f"Q: {row[0]}\nA: {row[1]}")
                
            return results
            
    def clear(self) -> None:
        """Clear all memories."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM memories")
            conn.commit()