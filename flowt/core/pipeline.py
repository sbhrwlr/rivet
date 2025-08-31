"""Core pipeline orchestration classes."""

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from .models import PipelineConfig

# Type variables for generic pipeline steps
T = TypeVar("T")  # Input type
U = TypeVar("U")  # Output type


class PipelineContext(BaseModel):
    """Context object passed between pipeline steps.

    Attributes:
        execution_id: Unique identifier for this pipeline execution
        metadata: Arbitrary metadata that can be shared between steps
        config: Pipeline configuration
    """

    execution_id: str
    metadata: dict[str, Any] = {}
    config: PipelineConfig

    @classmethod
    def create(cls, config: Optional[PipelineConfig] = None) -> "PipelineContext":
        """Create a new pipeline context with a unique execution ID.

        Args:
            config: Optional pipeline configuration

        Returns:
            New PipelineContext instance
        """
        return cls(
            execution_id=str(uuid4()),
            metadata={},
            config=config or PipelineConfig(),
        )


class PipelineStep(ABC, Generic[T, U]):
    """Abstract base class for all pipeline steps.

    Pipeline steps are the building blocks of pipelines. Each step takes
    an input of type T and produces an output of type U.
    """

    @abstractmethod
    async def execute(self, input_data: T, context: PipelineContext) -> U:
        """Execute the pipeline step.

        Args:
            input_data: Input data for this step
            context: Pipeline execution context

        Returns:
            Output data from this step
        """
        pass

    def validate_input(self, input_data: T) -> bool:
        """Validate input data for this step.

        Args:
            input_data: Input data to validate

        Returns:
            True if input is valid, False otherwise
        """
        # Default implementation - override in subclasses for custom validation
        return input_data is not None

    def get_step_name(self) -> str:
        """Get a human-readable name for this step.

        Returns:
            Step name (defaults to class name)
        """
        return self.__class__.__name__


class Pipeline(Generic[T, U]):
    """Main pipeline orchestrator.

    A pipeline is composed of an explicit array of steps that are executed
    in sequence. Each step's output becomes the input to the next step.
    """

    def __init__(self, steps: list[PipelineStep[Any, Any]], config: Optional[PipelineConfig] = None):
        """Initialize the pipeline.

        Args:
            steps: List of pipeline steps to execute in order
            config: Optional pipeline configuration
        """
        if not steps:
            raise ValueError("Pipeline must have at least one step")

        self.steps = steps
        self.config = config or PipelineConfig()

    async def execute(self, input_data: T) -> U:
        """Execute the pipeline with the given input.

        Args:
            input_data: Initial input data for the pipeline

        Returns:
            Final output from the last pipeline step

        Raises:
            ValueError: If input validation fails
            Exception: If any step execution fails
        """
        context = PipelineContext.create(self.config)
        current_data: Any = input_data

        for i, step in enumerate(self.steps):
            # Validate input for this step
            if not step.validate_input(current_data):
                raise ValueError(
                    f"Input validation failed for step {i} ({step.get_step_name()})"
                )

            # Execute the step
            try:
                current_data = await step.execute(current_data, context)
            except Exception as e:
                # Add context to the error
                raise Exception(
                    f"Step {i} ({step.get_step_name()}) failed: {str(e)}"
                ) from e

        return current_data  # type: ignore

    def get_step_count(self) -> int:
        """Get the number of steps in this pipeline.

        Returns:
            Number of pipeline steps
        """
        return len(self.steps)

    def get_step_names(self) -> list[str]:
        """Get the names of all steps in this pipeline.

        Returns:
            List of step names in execution order
        """
        return [step.get_step_name() for step in self.steps]
