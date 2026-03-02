from typing import Any

import numpy as np

from ids_eval.dto.metric_config import MetricMetadata
from ids_eval.interface.abstract_static_metric import AbstractStaticMetric

"""
Calculation of Intrusion Detection Capability (C_ID) metric as per the paper:
G. Gu, P. Fogla, D. Dagon, W. Lee, and B. Skoric, Measuring intrusion detection
capability: an information-theoretic approach. Mar. 2006, Pages: 101. doi:
10.1145/1128817.1128834
"""


class IntrusionDetectionCapability(AbstractStaticMetric):
    def __init__(self):
        super().__init__()

    def _static_metric_metadata(self, is_multiclass: bool) -> list[MetricMetadata]:
        if is_multiclass:
            # IDC requires binary classification, return a binarized version for multiclass
            return [
                MetricMetadata(
                    key="test_intrusion_detection_capability_binary",
                    display_name="IDC (Binary)",
                    category="detection",
                    unit="ratio",
                    higher_is_better=True,
                    description="IDC using binarized labels (attack vs benign)",
                    comparison_group="detection_performance",
                    comparison_chart_type="grouped_bar",
                )
            ]

        return [
            MetricMetadata(
                key="test_intrusion_detection_capability",
                display_name="IDC",
                category="detection",
                unit="ratio",
                higher_is_better=True,
                description="Intrusion Detection Capability (information-theoretic metric)",
                comparison_group="detection_performance",
                comparison_chart_type="grouped_bar",
            )
        ]

    def _static_metric_prepare(self) -> None:
        from numpy import log2
        from sklearn.metrics import confusion_matrix

        self.confusion_matrix_fc = confusion_matrix
        self.log2_fc = log2

    def _static_metric_calculate(self, metrics: dict[str, Any], is_multiclass: bool) -> dict[str, Any]:
        y_true = metrics["test_y_true"]
        y_pred = metrics["test_y_pred"]

        if is_multiclass:
            y_true = np.where(y_true == "benign", 0, 1)
            y_pred = np.where(y_pred == "benign", 0, 1)
            key = "test_intrusion_detection_capability_binary"
        else:
            key = "test_intrusion_detection_capability"

        tn, fp, fn, tp = self.confusion_matrix_fc(y_true, y_pred).ravel().tolist()

        total_events = tn + fp + fn + tp
        base_rate = (tp + fn) / total_events if total_events != 0 else 0.0
        prob_alarm = (tp + fp) / total_events if total_events != 0 else 0.0
        prob_no_alarm = (tn + fn) / total_events if total_events != 0 else 0.0

        ppv = tp / (tp + fp) if (tp + fp) != 0 else 0.0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) != 0 else 0.0  # Negative Predictive Value

        def entropy_fc(p: float) -> float:
            return -1 * (p * self.log2_fc(p) + (1 - p) * self.log2_fc(1 - p)) if p not in [0, 1] else 0.0

        entropy_base = entropy_fc(base_rate)
        conditional_entropy = prob_alarm * entropy_fc(ppv) + prob_no_alarm * entropy_fc(npv)

        idc = (entropy_base - conditional_entropy) / entropy_base if entropy_base != 0 else 0.0

        return {key: round(float(idc), 5)}

    def _static_metric_visualize(self, metrics: dict[str, Any], visual_name_prefix: str, is_multiclass: bool) -> None:
        pass
