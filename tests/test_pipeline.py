"""Tests for pipeline functionality."""

import pytest
from typing import Any, List
from flowt.core.pipeline import Pipeline, PipelineStep, PipelineContext
from flowt.core.models import PipelineConfig


class EchoStep(PipelineStep[str, str]):
    """Simple step that echoes input with a prefix."""

    def __init__(self, prefix: str = "Echo"):
        self.prefix = prefix

    async def execute(self, input_data: str, context: PipelineContext) -> str:
        return f"{self.prefix}: {input_data}"


class UppercaseStep(PipelineStep[str, str]):
    """Step that converts input to uppercase."""

    async def execute(self, input_data: str, context: PipelineContext) -> str:
        return input_data.upper()


class TestPipelineContext:
    """Tests for PipelineContext."""

    def test_context_creation(self) -> None:
        """Test basic context creation."""
        context = PipelineContext.create()
        
        assert context.execution_id is not None
        assert len(context.execution_id) > 0
        assert context.metadata == {}
        assert isinstance(context.config, PipelineConfig)

    def test_context_with_config(self) -> None:
        """Test context creation with custom config."""
        config = PipelineConfig(retry_attempts=5, timeout_seconds=60)
        context = PipelineContext.create(config)
        
        assert context.config.retry_attempts == 5
        assert context.config.timeout_seconds == 60

    def test_context_unique_execution_ids(self) -> None:
        """Test that each context gets a unique execution ID."""
        context1 = PipelineContext.create()
        context2 = PipelineContext.create()
        
        assert context1.execution_id != context2.execution_id

    def test_context_metadata_modification(self) -> None:
        """Test that context metadata can be modified."""
        context = PipelineContext.create()
        
        assert context.metadata == {}
        
        context.metadata["key1"] = "value1"
        context.metadata["key2"] = 42
        
        assert context.metadata == {"key1": "value1", "key2": 42}


class TestPipelineStep:
    """Tests for PipelineStep."""

    @pytest.mark.asyncio
    async def test_echo_step(self) -> None:
        """Test basic step execution."""
        step = EchoStep("Test")
        context = PipelineContext.create()
        
        result = await step.execute("Hello", context)
        assert result == "Test: Hello"

    def test_step_name(self) -> None:
        """Test step name generation."""
        step = EchoStep()
        assert step.get_step_name() == "EchoStep"

    def test_input_validation(self) -> None:
        """Test input validation."""
        step = EchoStep()
        
        assert step.validate_input("valid input") is True
        # Note: We can't test None validation with strict typing, so test empty string instead
        assert step.validate_input("") is True  # Empty string is still valid


class NumberStep(PipelineStep[int, int]):
    """Step that multiplies input by a factor."""

    def __init__(self, factor: int = 2):
        self.factor = factor

    async def execute(self, input_data: int, context: PipelineContext) -> int:
        return input_data * self.factor


class StringToIntStep(PipelineStep[str, int]):
    """Step that converts string to integer."""

    async def execute(self, input_data: str, context: PipelineContext) -> int:
        return len(input_data)


class TestPipeline:
    """Tests for Pipeline."""

    def test_pipeline_creation(self) -> None:
        """Test basic pipeline creation."""
        steps: List[PipelineStep[Any, Any]] = [EchoStep(), UppercaseStep()]
        pipeline: Pipeline[str, str] = Pipeline(steps)
        
        assert pipeline.get_step_count() == 2
        assert pipeline.get_step_names() == ["EchoStep", "UppercaseStep"]

    def test_empty_pipeline_raises_error(self) -> None:
        """Test that empty pipeline raises error."""
        with pytest.raises(ValueError, match="Pipeline must have at least one step"):
            Pipeline([])

    @pytest.mark.asyncio
    async def test_single_step_pipeline(self) -> None:
        """Test pipeline with single step."""
        pipeline: Pipeline[str, str] = Pipeline([EchoStep("Test")])
        
        result = await pipeline.execute("Hello")
        assert result == "Test: Hello"

    @pytest.mark.asyncio
    async def test_multi_step_pipeline(self) -> None:
        """Test pipeline with multiple steps."""
        steps: List[PipelineStep[Any, Any]] = [EchoStep("Test"), UppercaseStep()]
        pipeline: Pipeline[str, str] = Pipeline(steps)
        
        result = await pipeline.execute("hello")
        assert result == "TEST: HELLO"

    @pytest.mark.asyncio
    async def test_pipeline_with_config(self) -> None:
        """Test pipeline with custom configuration."""
        config = PipelineConfig(retry_attempts=1, timeout_seconds=10)
        pipeline: Pipeline[str, str] = Pipeline([EchoStep()], config)
        
        result = await pipeline.execute("test")
        assert result == "Echo: test"

    @pytest.mark.asyncio
    async def test_pipeline_validation_error(self) -> None:
        """Test pipeline with validation error."""
        class FailingValidationStep(PipelineStep[str, str]):
            def validate_input(self, input_data: str) -> bool:
                return False
            
            async def execute(self, input_data: str, context: PipelineContext) -> str:
                return input_data

        pipeline: Pipeline[str, str] = Pipeline([FailingValidationStep()])
        
        with pytest.raises(ValueError, match="Input validation failed"):
            await pipeline.execute("test")

    @pytest.mark.asyncio
    async def test_pipeline_execution_error(self) -> None:
        """Test pipeline with execution error."""
        class FailingStep(PipelineStep[str, str]):
            async def execute(self, input_data: str, context: PipelineContext) -> str:
                raise RuntimeError("Step failed")

        pipeline: Pipeline[str, str] = Pipeline([FailingStep()])
        
        with pytest.raises(Exception, match="Step 0 \\(FailingStep\\) failed"):
            await pipeline.execute("test")

    @pytest.mark.asyncio
    async def test_pipeline_type_transformation(self) -> None:
        """Test pipeline with type transformations."""
        # Pipeline that transforms string -> int -> int
        steps: List[PipelineStep[Any, Any]] = [StringToIntStep(), NumberStep(3)]
        pipeline: Pipeline[str, int] = Pipeline(steps)
        
        result = await pipeline.execute("hello")  # len("hello") = 5, then 5 * 3 = 15
        assert result == 15

    @pytest.mark.asyncio
    async def test_pipeline_context_sharing(self) -> None:
        """Test that context is shared between steps."""
        class ContextWriterStep(PipelineStep[str, str]):
            async def execute(self, input_data: str, context: PipelineContext) -> str:
                context.metadata["test_key"] = "test_value"
                return input_data

        class ContextReaderStep(PipelineStep[str, str]):
            async def execute(self, input_data: str, context: PipelineContext) -> str:
                value = context.metadata.get("test_key", "not_found")
                return f"{input_data}_{value}"

        steps: List[PipelineStep[Any, Any]] = [ContextWriterStep(), ContextReaderStep()]
        pipeline: Pipeline[str, str] = Pipeline(steps)
        result = await pipeline.execute("test")
        
        assert result == "test_test_value"

    @pytest.mark.asyncio
    async def test_pipeline_execution_id_consistency(self) -> None:
        """Test that execution ID is consistent across steps."""
        execution_ids: List[str] = []

        class ExecutionIdCapturerStep(PipelineStep[str, str]):
            async def execute(self, input_data: str, context: PipelineContext) -> str:
                execution_ids.append(context.execution_id)
                return input_data

        steps: List[PipelineStep[Any, Any]] = [ExecutionIdCapturerStep(), ExecutionIdCapturerStep()]
        pipeline: Pipeline[str, str] = Pipeline(steps)
        
        await pipeline.execute("test")
        
        assert len(execution_ids) == 2
        assert execution_ids[0] == execution_ids[1]
        assert len(execution_ids[0]) > 0