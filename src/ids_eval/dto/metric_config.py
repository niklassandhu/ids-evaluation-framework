from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# Valid metric categories
MetricCategory = Literal["detection", "runtime", "resource", "other"]


class MetricPluginConfig(BaseModel):
    plugin: str = Field(..., description="Plugin name (e.g., 'accuracy_metric')")
    params: dict[str, Any] | None = Field(
        default=None, description="Optional parameters to pass to the plugin's prepare() method"
    )


ComparisonChartType = Literal["grouped_bar", "horizontal_bar"]


class MetricMetadata(BaseModel):
    key: str = Field(..., description="The metric key in results (e.g., 'test_accuracy')")
    display_name: str = Field(..., description="Human-readable name (e.g., 'Accuracy')")
    category: MetricCategory = Field(
        default="other", description="Category for grouping: 'detection', 'runtime', 'resource', 'other'"
    )
    unit: str | None = Field(default=None, description="Unit of measurement (e.g., 'ratio', 'seconds', 'MB')")
    higher_is_better: bool | None = Field(
        default=None, description="True if higher values are better, False if lower is better"
    )
    description: str | None = Field(default=None, description="Brief description of what this metric measures")
    comparison_group: str | None = Field(
        default=None,
        description="Group identifier for comparison charts. " "Metrics with the same group are plotted together.",
    )
    comparison_chart_type: ComparisonChartType | None = Field(
        default=None,
        description="Chart type for comparison visualization: "
        "'grouped_bar' (vertical, models on x-axis) or "
        "'horizontal_bar' (horizontal, models on y-axis).",
    )
