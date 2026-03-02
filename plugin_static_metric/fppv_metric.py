from typing import Any

import numpy as np

from ids_eval.dto.metric_config import MetricMetadata
from ids_eval.interface.abstract_static_metric import AbstractStaticMetric


class FppvMetric(AbstractStaticMetric):
    """
    False Positives Per Volume (FPPV) and False Positives Per Million Samples metrics.

    FPPV measures false positives relative to the in-memory size of the test
    DataFrame (in GB). This gives a rough estimate of false-alarm density per
    data volume. Note that the "GB" value depends on the DataFrame's
    in-memory representation and may differ from raw network traffic volume.

    Formula: FPPV = FP / V_data (in GB)
    where V_data = in-memory size of the test DataFrame.

    FP per million samples is an alternative, volume-independent metric that
    normalises false positives by the number of samples (packets / CSV rows):

    Formula: FP_per_million = (FP / n_samples) * 1,000,000
    """

    def __init__(self):
        super().__init__()

    def _static_metric_metadata(self, is_multiclass: bool) -> list[MetricMetadata]:
        return [
            MetricMetadata(
                key="test_fppv",
                display_name="FPPV",
                category="detection",
                unit="FP/GB",
                higher_is_better=False,
                description="False Positives per GB of in-memory test data",
            ),
            MetricMetadata(
                key="test_fp_per_million",
                display_name="FP per Million Samples",
                category="detection",
                unit="FP/1M samples",
                higher_is_better=False,
                description="False Positives per 1,000,000 samples (packets/rows)",
            ),
        ]

    def _static_metric_prepare(self) -> None:
        from sklearn.metrics import confusion_matrix

        self.confusion_matrix_fc = confusion_matrix

    def _static_metric_calculate(self, metrics: dict[str, Any], is_multiclass: bool) -> dict[str, Any]:
        y_true = metrics["test_y_true"]
        y_pred = metrics["test_y_pred"]

        if is_multiclass:
            y_true = np.where(y_true == "benign", 0, 1)
            y_pred = np.where(y_pred == "benign", 0, 1)

        cm = self.confusion_matrix_fc(y_true, y_pred)
        false_positives = int(cm[0, 1])  # FP is at position [0, 1]

        data_size_gb = metrics.get("test_data_size_gb", 0.0)
        if data_size_gb and data_size_gb > 0:
            fppv = false_positives / data_size_gb
        else:
            fppv = None

        n_samples = metrics.get("test_n_samples", len(y_true))
        if n_samples > 0:
            fp_per_million = (false_positives / n_samples) * 1_000_000
        else:
            fp_per_million = None

        return {
            "test_fppv": round(float(fppv), 5) if fppv is not None else None,
            "test_fp_per_million": round(float(fp_per_million), 5) if fp_per_million is not None else None,
        }

    def _static_metric_visualize(self, metrics: dict[str, Any], visual_name_prefix: str, is_multiclass: bool) -> None:
        pass
