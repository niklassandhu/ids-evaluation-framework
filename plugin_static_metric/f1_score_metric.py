from typing import Any

from ids_eval.dto.metric_config import MetricMetadata
from ids_eval.interface.abstract_static_metric import AbstractStaticMetric


class F1ScoreMetric(AbstractStaticMetric):
    def __init__(self):
        super().__init__()

    def _static_metric_metadata(self, is_multiclass: bool) -> list[MetricMetadata]:
        if is_multiclass:
            return [
                MetricMetadata(
                    key="test_f1_score_macro",
                    display_name="F1 Score (Macro)",
                    category="detection",
                    unit="ratio",
                    higher_is_better=True,
                    description="Macro-averaged F1 score across all classes",
                    comparison_group="detection_performance",
                    comparison_chart_type="grouped_bar",
                )
            ]

        return [
            MetricMetadata(
                key="test_f1_score",
                display_name="F1 Score",
                category="detection",
                unit="ratio",
                higher_is_better=True,
                description="Harmonic mean of precision and recall",
                comparison_group="detection_performance",
                comparison_chart_type="grouped_bar",
            )
        ]

    def _static_metric_prepare(self) -> None:
        from sklearn.metrics import f1_score

        self.f1_score = f1_score

    def _static_metric_calculate(self, metrics: dict[str, Any], is_multiclass: bool) -> dict[str, Any]:
        if is_multiclass:
            return {
                "test_f1_score_macro": round(
                    self.f1_score(metrics["test_y_true"], metrics["test_y_pred"], average="macro", zero_division=0), 5
                )
            }

        return {"test_f1_score": round(self.f1_score(metrics["test_y_true"], metrics["test_y_pred"]), 5)}

    def _static_metric_visualize(self, metrics: dict[str, Any], visual_name_prefix: str, is_multiclass: bool) -> None:
        pass
