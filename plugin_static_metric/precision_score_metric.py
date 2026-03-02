from typing import Any

from ids_eval.dto.metric_config import MetricMetadata
from ids_eval.interface.abstract_static_metric import AbstractStaticMetric


class PrecisionScoreMetric(AbstractStaticMetric):
    def __init__(self):
        super().__init__()

    def _static_metric_metadata(self, is_multiclass: bool) -> list[MetricMetadata]:
        if is_multiclass:
            return [
                MetricMetadata(
                    key="test_precision_score_macro",
                    display_name="Precision (Macro)",
                    category="detection",
                    unit="ratio",
                    higher_is_better=True,
                    description="Macro-averaged precision across all classes",
                )
            ]

        return [
            MetricMetadata(
                key="test_precision_score",
                display_name="Precision",
                category="detection",
                unit="ratio",
                higher_is_better=True,
                description="Ratio of true positives to all positive predictions",
            )
        ]

    def _static_metric_prepare(self) -> None:
        from sklearn.metrics import precision_score

        self.precision_score = precision_score

    def _static_metric_calculate(self, metrics: dict[str, Any], is_multiclass: bool) -> dict[str, Any]:
        if is_multiclass:
            return {
                "test_precision_score_macro": round(
                    self.precision_score(
                        metrics["test_y_true"], metrics["test_y_pred"], average="macro", zero_division=0
                    ),
                    5,
                )
            }

        return {
            "test_precision_score": round(
                self.precision_score(metrics["test_y_true"], metrics["test_y_pred"], zero_division=0), 5
            )
        }

    def _static_metric_visualize(self, metrics: dict[str, Any], visual_name_prefix: str, is_multiclass: bool) -> None:
        pass
