"""
Tests for usage tracking and cost calculation functionality.
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json

from rivet.usage import UsageTracker, CostCalculator, UsageRecord


class TestUsageRecord:
    """Test UsageRecord dataclass."""
    
    def test_create_usage_record(self):
        """Test creating a usage record."""
        timestamp = datetime.now()
        record = UsageRecord(
            timestamp=timestamp,
            model="gpt-3.5-turbo",
            input_tokens=100,
            output_tokens=50,
            cost=0.225,
            operation="generate"
        )
        
        assert record.timestamp == timestamp
        assert record.model == "gpt-3.5-turbo"
        assert record.input_tokens == 100
        assert record.output_tokens == 50
        assert record.cost == 0.225
        assert record.operation == "generate"
    
    def test_to_dict(self):
        """Test converting usage record to dictionary."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        record = UsageRecord(
            timestamp=timestamp,
            model="gpt-4",
            input_tokens=200,
            output_tokens=100,
            cost=9.0,
            operation="stream"
        )
        
        result = record.to_dict()
        expected = {
            'timestamp': '2024-01-01T12:00:00',
            'model': 'gpt-4',
            'input_tokens': 200,
            'output_tokens': 100,
            'cost': 9.0,
            'operation': 'stream'
        }
        
        assert result == expected
    
    def test_from_dict(self):
        """Test creating usage record from dictionary."""
        data = {
            'timestamp': '2024-01-01T12:00:00',
            'model': 'gpt-4',
            'input_tokens': 200,
            'output_tokens': 100,
            'cost': 9.0,
            'operation': 'stream'
        }
        
        record = UsageRecord.from_dict(data)
        
        assert record.timestamp == datetime(2024, 1, 1, 12, 0, 0)
        assert record.model == "gpt-4"
        assert record.input_tokens == 200
        assert record.output_tokens == 100
        assert record.cost == 9.0
        assert record.operation == "stream"


class TestCostCalculator:
    """Test CostCalculator class."""
    
    def test_calculate_cost_gpt35_turbo(self):
        """Test cost calculation for GPT-3.5 Turbo."""
        calculator = CostCalculator()
        
        # 1000 input tokens, 500 output tokens
        cost = calculator.calculate_cost("gpt-3.5-turbo", 1000, 500)
        expected = (1000/1000 * 0.0015) + (500/1000 * 0.002)
        expected = 0.0015 + 0.001  # 0.0025
        
        assert cost == expected
    
    def test_calculate_cost_gpt4(self):
        """Test cost calculation for GPT-4."""
        calculator = CostCalculator()
        
        # 500 input tokens, 250 output tokens
        cost = calculator.calculate_cost("gpt-4", 500, 250)
        expected = (500/1000 * 0.03) + (250/1000 * 0.06)
        expected = 0.015 + 0.015  # 0.03
        
        assert cost == expected
    
    def test_calculate_cost_unknown_model(self):
        """Test cost calculation for unknown model returns 0."""
        calculator = CostCalculator()
        
        cost = calculator.calculate_cost("unknown-model", 1000, 500)
        assert cost == 0.0
    
    def test_custom_pricing(self):
        """Test calculator with custom pricing."""
        custom_pricing = {
            "custom-model": {"input": 0.01, "output": 0.02}
        }
        calculator = CostCalculator(custom_pricing)
        
        cost = calculator.calculate_cost("custom-model", 1000, 500)
        expected = (1000/1000 * 0.01) + (500/1000 * 0.02)
        expected = 0.01 + 0.01  # 0.02
        
        assert cost == expected
    
    def test_get_model_pricing(self):
        """Test getting pricing for a model."""
        calculator = CostCalculator()
        
        pricing = calculator.get_model_pricing("gpt-3.5-turbo")
        expected = {"input": 0.0015, "output": 0.002}
        
        assert pricing == expected
    
    def test_get_model_pricing_unknown(self):
        """Test getting pricing for unknown model."""
        calculator = CostCalculator()
        
        pricing = calculator.get_model_pricing("unknown-model")
        assert pricing is None
    
    def test_add_model_pricing(self):
        """Test adding new model pricing."""
        calculator = CostCalculator()
        
        calculator.add_model_pricing("new-model", 0.005, 0.01)
        pricing = calculator.get_model_pricing("new-model")
        
        assert pricing == {"input": 0.005, "output": 0.01}


class TestUsageTracker:
    """Test UsageTracker class."""
    
    def test_track_usage(self):
        """Test tracking usage."""
        tracker = UsageTracker()
        
        record = tracker.track_usage("gpt-3.5-turbo", 1000, 500, "generate")
        
        assert len(tracker.usage_data) == 1
        assert record.model == "gpt-3.5-turbo"
        assert record.input_tokens == 1000
        assert record.output_tokens == 500
        assert record.operation == "generate"
        assert record.cost == 0.0025  # Calculated cost
        assert isinstance(record.timestamp, datetime)
    
    def test_get_total_cost(self):
        """Test getting total cost."""
        tracker = UsageTracker()
        
        tracker.track_usage("gpt-3.5-turbo", 1000, 500)
        tracker.track_usage("gpt-4", 500, 250)
        
        total_cost = tracker.get_total_cost()
        expected = 0.0025 + 0.03  # GPT-3.5 + GPT-4 costs
        
        assert total_cost == expected
    
    def test_get_total_cost_with_model_filter(self):
        """Test getting total cost filtered by model."""
        tracker = UsageTracker()
        
        tracker.track_usage("gpt-3.5-turbo", 1000, 500)
        tracker.track_usage("gpt-4", 500, 250)
        
        gpt35_cost = tracker.get_total_cost(model="gpt-3.5-turbo")
        assert gpt35_cost == 0.0025
        
        gpt4_cost = tracker.get_total_cost(model="gpt-4")
        assert gpt4_cost == 0.03
    
    def test_get_total_cost_with_time_range(self):
        """Test getting total cost with time range filter."""
        tracker = UsageTracker()
        
        # Add some usage records
        tracker.track_usage("gpt-3.5-turbo", 1000, 500)
        
        # Create time range that includes the record
        now = datetime.now()
        start_time = now - timedelta(hours=1)
        end_time = now + timedelta(hours=1)
        
        cost = tracker.get_total_cost((start_time, end_time))
        assert cost == 0.0025
        
        # Create time range that excludes the record
        old_start = now - timedelta(days=2)
        old_end = now - timedelta(days=1)
        
        cost = tracker.get_total_cost((old_start, old_end))
        assert cost == 0.0
    
    def test_get_total_tokens(self):
        """Test getting total token counts."""
        tracker = UsageTracker()
        
        tracker.track_usage("gpt-3.5-turbo", 1000, 500)
        tracker.track_usage("gpt-4", 200, 100)
        
        tokens = tracker.get_total_tokens()
        
        assert tokens["input_tokens"] == 1200
        assert tokens["output_tokens"] == 600
        assert tokens["total_tokens"] == 1800
    
    def test_get_usage_summary(self):
        """Test getting usage summary."""
        tracker = UsageTracker()
        
        tracker.track_usage("gpt-3.5-turbo", 1000, 500, "generate")
        tracker.track_usage("gpt-4", 200, 100, "stream")
        tracker.track_usage("gpt-3.5-turbo", 500, 250, "tool_call")
        
        summary = tracker.get_usage_summary()
        
        assert summary["total_cost"] == 0.0025 + 0.012 + 0.00125  # Sum of all costs
        assert summary["total_tokens"]["input_tokens"] == 1700
        assert summary["total_tokens"]["output_tokens"] == 850
        assert summary["total_tokens"]["total_tokens"] == 2550
        assert summary["record_count"] == 3
        
        # Check by_model breakdown
        assert "gpt-3.5-turbo" in summary["by_model"]
        assert "gpt-4" in summary["by_model"]
        assert summary["by_model"]["gpt-3.5-turbo"]["count"] == 2
        assert summary["by_model"]["gpt-4"]["count"] == 1
        
        # Check by_operation breakdown
        assert "generate" in summary["by_operation"]
        assert "stream" in summary["by_operation"]
        assert "tool_call" in summary["by_operation"]
    
    def test_get_usage_summary_empty(self):
        """Test getting usage summary with no data."""
        tracker = UsageTracker()
        
        summary = tracker.get_usage_summary()
        
        assert summary["total_cost"] == 0.0
        assert summary["total_tokens"]["input_tokens"] == 0
        assert summary["total_tokens"]["output_tokens"] == 0
        assert summary["total_tokens"]["total_tokens"] == 0
        assert summary["record_count"] == 0
        assert summary["by_model"] == {}
        assert summary["by_operation"] == {}
    
    def test_get_daily_cost(self):
        """Test getting daily cost."""
        tracker = UsageTracker()
        
        tracker.track_usage("gpt-3.5-turbo", 1000, 500)
        
        daily_cost = tracker.get_daily_cost(1)  # Last 1 day
        assert daily_cost == 0.0025
    
    def test_clear_usage_data(self):
        """Test clearing usage data."""
        tracker = UsageTracker()
        
        tracker.track_usage("gpt-3.5-turbo", 1000, 500)
        tracker.track_usage("gpt-4", 200, 100)
        
        assert len(tracker.usage_data) == 2
        
        tracker.clear_usage_data()
        assert len(tracker.usage_data) == 0
    
    def test_clear_usage_data_before_date(self):
        """Test clearing usage data before a specific date."""
        tracker = UsageTracker()
        
        # Add records with different timestamps
        old_record = UsageRecord(
            timestamp=datetime.now() - timedelta(days=2),
            model="gpt-3.5-turbo",
            input_tokens=1000,
            output_tokens=500,
            cost=0.0025,
            operation="generate"
        )
        tracker.usage_data.append(old_record)
        
        tracker.track_usage("gpt-4", 200, 100)  # Recent record
        
        assert len(tracker.usage_data) == 2
        
        # Clear records older than 1 day
        cutoff_date = datetime.now() - timedelta(days=1)
        tracker.clear_usage_data(cutoff_date)
        
        assert len(tracker.usage_data) == 1
        assert tracker.usage_data[0].model == "gpt-4"
    
    def test_export_data_json(self):
        """Test exporting data as JSON."""
        tracker = UsageTracker()
        
        tracker.track_usage("gpt-3.5-turbo", 1000, 500, "generate")
        
        json_data = tracker.export_data("json")
        parsed_data = json.loads(json_data)
        
        assert len(parsed_data) == 1
        assert parsed_data[0]["model"] == "gpt-3.5-turbo"
        assert parsed_data[0]["input_tokens"] == 1000
        assert parsed_data[0]["output_tokens"] == 500
        assert parsed_data[0]["operation"] == "generate"
    
    def test_export_data_csv(self):
        """Test exporting data as CSV."""
        tracker = UsageTracker()
        
        tracker.track_usage("gpt-3.5-turbo", 1000, 500, "generate")
        
        csv_data = tracker.export_data("csv")
        lines = csv_data.strip().split('\n')
        
        # Check header
        assert lines[0] == "timestamp,model,input_tokens,output_tokens,cost,operation"
        
        # Check data row
        data_row = lines[1].split(',')
        assert data_row[1] == "gpt-3.5-turbo"
        assert data_row[2] == "1000"
        assert data_row[3] == "500"
        assert data_row[5] == "generate"
    
    def test_export_data_invalid_format(self):
        """Test exporting data with invalid format."""
        tracker = UsageTracker()
        
        with pytest.raises(ValueError, match="Unsupported export format"):
            tracker.export_data("xml")
    
    def test_persistent_storage(self):
        """Test persistent storage functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "usage.json"
            
            # Create tracker with storage
            tracker = UsageTracker(str(storage_path))
            tracker.track_usage("gpt-3.5-turbo", 1000, 500)
            
            # Verify file was created
            assert storage_path.exists()
            
            # Create new tracker with same storage path
            tracker2 = UsageTracker(str(storage_path))
            
            # Should load existing data
            assert len(tracker2.usage_data) == 1
            assert tracker2.usage_data[0].model == "gpt-3.5-turbo"
            assert tracker2.usage_data[0].input_tokens == 1000
    
    def test_persistent_storage_corrupted_data(self):
        """Test handling of corrupted storage data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "usage.json"
            
            # Write corrupted JSON
            with open(storage_path, 'w') as f:
                f.write("invalid json")
            
            # Should handle gracefully and start fresh
            tracker = UsageTracker(str(storage_path))
            assert len(tracker.usage_data) == 0