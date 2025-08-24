"""
Usage tracking and cost calculation for Rivet agents.

This module provides classes for tracking token usage and calculating costs
for AI model operations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path


@dataclass
class UsageRecord:
    """Record of token usage for a single model operation."""
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    operation: str  # 'generate', 'stream', 'tool_call'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'model': self.model,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'cost': self.cost,
            'operation': self.operation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UsageRecord':
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            model=data['model'],
            input_tokens=data['input_tokens'],
            output_tokens=data['output_tokens'],
            cost=data['cost'],
            operation=data['operation']
        )


class CostCalculator:
    """Calculate costs for AI model usage based on current pricing."""
    
    # Model pricing data (per 1K tokens)
    PRICING = {
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-32k": {"input": 0.06, "output": 0.12},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "text-embedding-ada-002": {"input": 0.0001, "output": 0.0},
        "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
        "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
    }
    
    def __init__(self, custom_pricing: Optional[Dict[str, Dict[str, float]]] = None):
        """Initialize with optional custom pricing."""
        self.pricing = self.PRICING.copy()
        if custom_pricing:
            self.pricing.update(custom_pricing)
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for token usage."""
        if model not in self.pricing:
            # Return 0 for unknown models rather than raising an error
            return 0.0
        
        pricing = self.pricing[model]
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return round(input_cost + output_cost, 6)  # Round to 6 decimal places
    
    def get_model_pricing(self, model: str) -> Optional[Dict[str, float]]:
        """Get pricing information for a specific model."""
        return self.pricing.get(model)
    
    def add_model_pricing(self, model: str, input_price: float, output_price: float):
        """Add or update pricing for a model."""
        self.pricing[model] = {"input": input_price, "output": output_price}


class UsageTracker:
    """Track and manage usage records for AI model operations."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize usage tracker with optional persistent storage."""
        self.usage_data: List[UsageRecord] = []
        self.cost_calculator = CostCalculator()
        self.storage_path = Path(storage_path) if storage_path else None
        
        # Load existing data if storage path is provided
        if self.storage_path and self.storage_path.exists():
            self._load_data()
    
    def track_usage(self, model: str, input_tokens: int, output_tokens: int, 
                   operation: str = "generate") -> UsageRecord:
        """Track token usage for a model call."""
        cost = self.cost_calculator.calculate_cost(model, input_tokens, output_tokens)
        
        record = UsageRecord(
            timestamp=datetime.now(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            operation=operation
        )
        
        self.usage_data.append(record)
        
        # Save to storage if configured
        if self.storage_path:
            self._save_data()
        
        return record
    
    def get_total_cost(self, time_range: Optional[Tuple[datetime, datetime]] = None,
                      model: Optional[str] = None) -> float:
        """Calculate total cost for time range and/or model."""
        filtered_records = self._filter_records(time_range, model)
        return sum(record.cost for record in filtered_records)
    
    def get_total_tokens(self, time_range: Optional[Tuple[datetime, datetime]] = None,
                        model: Optional[str] = None) -> Dict[str, int]:
        """Get total token counts for time range and/or model."""
        filtered_records = self._filter_records(time_range, model)
        
        return {
            "input_tokens": sum(record.input_tokens for record in filtered_records),
            "output_tokens": sum(record.output_tokens for record in filtered_records),
            "total_tokens": sum(record.input_tokens + record.output_tokens 
                              for record in filtered_records)
        }
    
    def get_usage_summary(self, time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """Get detailed usage statistics."""
        filtered_records = self._filter_records(time_range)
        
        if not filtered_records:
            return {
                "total_cost": 0.0,
                "total_tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "by_model": {},
                "by_operation": {},
                "record_count": 0,
                "time_range": time_range
            }
        
        # Group by model
        by_model = {}
        for record in filtered_records:
            if record.model not in by_model:
                by_model[record.model] = {
                    "cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "count": 0
                }
            
            by_model[record.model]["cost"] += record.cost
            by_model[record.model]["input_tokens"] += record.input_tokens
            by_model[record.model]["output_tokens"] += record.output_tokens
            by_model[record.model]["count"] += 1
        
        # Group by operation
        by_operation = {}
        for record in filtered_records:
            if record.operation not in by_operation:
                by_operation[record.operation] = {
                    "cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "count": 0
                }
            
            by_operation[record.operation]["cost"] += record.cost
            by_operation[record.operation]["input_tokens"] += record.input_tokens
            by_operation[record.operation]["output_tokens"] += record.output_tokens
            by_operation[record.operation]["count"] += 1
        
        return {
            "total_cost": self.get_total_cost(time_range),
            "total_tokens": self.get_total_tokens(time_range),
            "by_model": by_model,
            "by_operation": by_operation,
            "record_count": len(filtered_records),
            "time_range": time_range
        }
    
    def get_daily_cost(self, days_back: int = 30) -> float:
        """Get total cost for the last N days."""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        return self.get_total_cost((start_time, end_time))
    
    def clear_usage_data(self, before_date: Optional[datetime] = None):
        """Clear usage data, optionally only before a specific date."""
        if before_date:
            self.usage_data = [record for record in self.usage_data 
                             if record.timestamp >= before_date]
        else:
            self.usage_data.clear()
        
        # Save changes if storage is configured
        if self.storage_path:
            self._save_data()
    
    def export_data(self, format: str = "json") -> str:
        """Export usage data in specified format."""
        if format.lower() == "json":
            return json.dumps([record.to_dict() for record in self.usage_data], 
                            indent=2, default=str)
        elif format.lower() == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(["timestamp", "model", "input_tokens", "output_tokens", 
                           "cost", "operation"])
            
            # Write data
            for record in self.usage_data:
                writer.writerow([
                    record.timestamp.isoformat(),
                    record.model,
                    record.input_tokens,
                    record.output_tokens,
                    record.cost,
                    record.operation
                ])
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _filter_records(self, time_range: Optional[Tuple[datetime, datetime]] = None,
                       model: Optional[str] = None) -> List[UsageRecord]:
        """Filter records by time range and/or model."""
        filtered = self.usage_data
        
        if time_range:
            start_time, end_time = time_range
            filtered = [record for record in filtered 
                       if start_time <= record.timestamp <= end_time]
        
        if model:
            filtered = [record for record in filtered if record.model == model]
        
        return filtered
    
    def _save_data(self):
        """Save usage data to storage file."""
        if not self.storage_path:
            return
        
        # Ensure directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [record.to_dict() for record in self.usage_data]
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_data(self):
        """Load usage data from storage file."""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            self.usage_data = [UsageRecord.from_dict(record) for record in data]
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # If data is corrupted, start fresh
            self.usage_data = []