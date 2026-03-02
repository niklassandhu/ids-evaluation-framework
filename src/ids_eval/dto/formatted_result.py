from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ids_eval.dto.metric_config import MetricCategory


class FormattedMetric(BaseModel):
    key: str = Field(..., description="Original metric key")
    display_name: str = Field(..., description="Human-readable name")
    value: Any = Field(..., description="The metric value")
    category: MetricCategory = Field(..., description="Metric category")
    unit: str | None = Field(default=None, description="Unit of measurement")
    higher_is_better: bool | None = Field(default=None, description="Whether higher values are better")


class FormattedPerformanceDrop(BaseModel):
    """Performance drop between clean and adversarial results."""

    metric_name: str = Field(..., description="Name of the metric")
    clean_value: float = Field(..., description="Value on clean data")
    adversarial_value: float = Field(..., description="Value on adversarial data")
    absolute_drop: float = Field(..., description="Absolute performance drop")
    relative_drop_percent: float = Field(..., description="Relative drop as percentage")


class FormattedAdversarialResult(BaseModel):
    """Results from testing with adversarial samples."""

    attack_name: str = Field(..., description="Name of the adversarial attack")
    metrics: dict[str, list[FormattedMetric]] = Field(default_factory=dict, description="Metrics grouped by category")
    performance_drops: list[FormattedPerformanceDrop] = Field(
        default_factory=list, description="Performance drop compared to clean results"
    )


class FormattedTestResult(BaseModel):
    dataset: str = Field(..., description="Test dataset name")
    is_cross_dataset: bool = Field(..., description="True if tested on different dataset than trained")
    is_adversarial: bool = Field(default=False, description="True if this is an adversarial test result")
    attack_name: str | None = Field(default=None, description="Name of the adversarial attack (if adversarial)")
    metrics: dict[str, list[FormattedMetric]] = Field(
        default_factory=dict, description="Metrics grouped by category: detection, runtime, resource, other"
    )
    adversarial_results: list[FormattedAdversarialResult] = Field(
        default_factory=list, description="Adversarial test results with performance drops"
    )


class FormattedTraining(BaseModel):
    metrics: dict[str, list[FormattedMetric]] = Field(
        default_factory=dict, description="Training metrics grouped by category"
    )
    sample_count: int | None = Field(default=None, description="Number of training samples")


class FormattedEvaluation(BaseModel):
    trained_on: str = Field(..., description="Training dataset name")
    model: str = Field(..., description="Model/plugin name")
    model_size_mb: float | None = Field(default=None, description="Model storage size")
    model_cached: bool | None = Field(default=None, description="Whether model was loaded from cache")
    training: FormattedTraining | None = Field(default=None, description="Training metrics and info")
    test_results: list[FormattedTestResult] = Field(
        default_factory=list, description="List of test results on different datasets"
    )


class FormattedResults(BaseModel):
    evaluations: list[FormattedEvaluation] = Field(default_factory=list, description="All evaluation entries")
    metadata_index: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Index of all metric metadata by key"
    )
