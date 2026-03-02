from typing import Any

from ids_eval.dto.metric_config import MetricMetadata
from ids_eval.interface.abstract_static_metric import AbstractStaticMetric


class AccuracyMetric(AbstractStaticMetric):
    def __init__(self):
        super().__init__()

    def _static_metric_metadata(self, is_multiclass: bool) -> list[MetricMetadata]:
        return [
            MetricMetadata(
                key="test_accuracy",
                display_name="Accuracy",
                category="detection",
                unit="ratio",
                higher_is_better=True,
                description="Proportion of correctly classified samples",
                comparison_group="detection_performance",
                comparison_chart_type="grouped_bar",
            )
        ]

    def _static_metric_prepare(self) -> None:
        from sklearn.metrics import accuracy_score

        self.accuracy_score = accuracy_score

    def _static_metric_calculate(self, metrics: dict[str, Any], is_multiclass: bool) -> dict[str, Any]:
        return {"test_accuracy": round(self.accuracy_score(metrics["test_y_true"], metrics["test_y_pred"]), 5)}

    def _static_metric_visualize(self, metrics: dict[str, Any], visual_name_prefix: str, is_multiclass: bool) -> None:
        pass
