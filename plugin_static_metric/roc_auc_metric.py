from typing import Any

import numpy as np

from ids_eval.dto.metric_config import MetricMetadata
from ids_eval.interface.abstract_static_metric import AbstractStaticMetric


class RocAucMetric(AbstractStaticMetric):
    def __init__(self):
        super().__init__()

    def _static_metric_metadata(self, is_multiclass: bool) -> list[MetricMetadata]:
        if is_multiclass:
            return [
                MetricMetadata(
                    key="test_roc_auc_ovr",
                    display_name="ROC AUC (OvR)",
                    category="detection",
                    unit="ratio",
                    higher_is_better=True,
                    description="One-vs-Rest ROC AUC for multiclass classification",
                    comparison_group="detection_performance",
                    comparison_chart_type="grouped_bar",
                )
            ]

        return [
            MetricMetadata(
                key="test_roc_auc",
                display_name="ROC AUC",
                category="detection",
                unit="ratio",
                higher_is_better=True,
                description="Area under the Receiver Operating Characteristic curve",
                comparison_group="detection_performance",
                comparison_chart_type="grouped_bar",
            )
        ]

    def _static_metric_prepare(self) -> None:
        from sklearn.metrics import auc, roc_auc_score, roc_curve

        self.roc_curve_fc = roc_curve
        self.auc_fc = auc
        self.roc_auc_score_fc = roc_auc_score

    def _static_metric_calculate(self, metrics: dict[str, Any], is_multiclass: bool) -> dict[str, Any]:
        y_true = metrics["test_y_true"]
        y_score = metrics["test_y_proba"]

        if is_multiclass:
            y_true = np.where(y_true == "benign", 0, 1)
            # benign index is last column of y_score
            benign_index = y_score.shape[1] - 1
            y_score = 1.0 - y_score[:, benign_index]
        else:
            if y_score.ndim > 1 and y_score.shape[1] > 1:
                y_score = y_score[:, 1]

        fpr, tpr, _ = self.roc_curve_fc(y_true, y_score)
        return {"test_roc_auc": round(self.auc_fc(fpr, tpr), 5)}

    def _static_metric_visualize(self, metrics: dict[str, Any], visual_name_prefix: str, is_multiclass: bool) -> None:
        pass
