from pathlib import Path
from typing import Any

import numpy as np

from ids_eval.dto.metric_config import MetricMetadata
from ids_eval.interface.abstract_static_metric import AbstractStaticMetric


class RocCurveMetric(AbstractStaticMetric):
    def __init__(self):
        super().__init__()

    def _static_metric_metadata(self, is_multiclass: bool) -> list[MetricMetadata]:
        return []

    def _static_metric_prepare(self) -> None:
        import matplotlib.pyplot as plt
        from sklearn.metrics import RocCurveDisplay

        self.roc_curve_display_fc = RocCurveDisplay
        self.plt = plt

    def _static_metric_calculate(self, metrics: dict[str, Any], is_multiclass: bool) -> dict[str, Any]:
        return {}

    def _static_metric_visualize(self, metrics: dict[str, Any], visual_name_prefix: str, is_multiclass: bool) -> None:
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

        self.plt.figure(figsize=(8, 6))
        axis = self.plt.gca()

        self.roc_curve_display_fc.from_predictions(
            y_true,
            y_score,
            ax=axis,
            plot_chance_level=True,
            name="ROC Curve (Binarized: Attack vs Benign)" if is_multiclass else "ROC Curve",
        )
        visual_name = visual_name_prefix + "roc_curve.png"
        self.plt.tight_layout()
        self.plt.savefig(Path(self.visual_path) / visual_name)
        self.plt.close()
