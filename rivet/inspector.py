"""Debugging and tracing layer for transparent agent operations."""

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


class Inspector:
    """Debugging and tracing layer for agent operations."""
    
    def __init__(self, enabled: bool = True, log_file: Optional[str] = None):
        self.enabled = enabled
        self.log_file = log_file
        self.logs: List[Dict[str, Any]] = []
        
    def log(self, event: str, data: Any = None) -> None:
        """Log an event with optional data."""
        if not self.enabled:
            return
            
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "data": data
        }
        
        self.logs.append(log_entry)
        
        if self.log_file:
            self._write_to_file(log_entry)
        else:
            self._print_log(log_entry)
            
    def _print_log(self, entry: Dict[str, Any]) -> None:
        """Print log entry to console."""
        timestamp = entry["timestamp"]
        event = entry["event"]
        data = entry.get("data", "")
        
        print(f"[{timestamp}] {event}: {data}")
        
    def _write_to_file(self, entry: Dict[str, Any]) -> None:
        """Write log entry to file."""
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")
            
    def get_logs(self, event_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all logs, optionally filtered by event type."""
        if event_filter:
            return [log for log in self.logs if log["event"] == event_filter]
        return self.logs.copy()
        
    def clear_logs(self) -> None:
        """Clear all stored logs."""
        self.logs.clear()
        
    def summary(self) -> Dict[str, Any]:
        """Get a summary of logged events."""
        event_counts = {}
        for log in self.logs:
            event = log["event"]
            event_counts[event] = event_counts.get(event, 0) + 1
            
        return {
            "total_events": len(self.logs),
            "event_counts": event_counts,
            "first_event": self.logs[0]["timestamp"] if self.logs else None,
            "last_event": self.logs[-1]["timestamp"] if self.logs else None
        }