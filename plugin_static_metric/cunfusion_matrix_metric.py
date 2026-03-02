from pathlib import Path
from typing import Any

from ids_eval.dto.metric_config import MetricMetadata
from ids_eval.interface.abstract_static_metric import AbstractStaticMetric


class ConfusionMatrixMetric(AbstractStaticMetric):
    def __init__(self):
        super().__init__()

    def _static_metric_metadata(self, is_multiclass: bool) -> list[MetricMetadata]:
        return [
            MetricMetadata(
                key="test_confusion_matrix",
                display_name="Confusion Matrix",
                category="detection",
                description="Confusion matrix",
            ),
            MetricMetadata(
                key="test_confusion_matrix_normalized",
                display_name="Confusion Matrix (Normalized)",
                category="detection",
                description="Normalized confusion matrix",
            ),
        ]

    def _static_metric_prepare(self) -> None:
        from sklearn.metrics import confusion_matrix

        self.confusion_matrix_fc = confusion_matrix

    def _static_metric_calculate(self, metrics: dict[str, Any], is_multiclass: bool) -> dict[str, Any]:
        self.test_confusion_matrix = self.confusion_matrix_fc(metrics["test_y_true"], metrics["test_y_pred"])
        self.test_confusion_matrix_normalized = self.confusion_matrix_fc(
            metrics["test_y_true"], metrics["test_y_pred"], normalize="true"
        )

        cm = self.test_confusion_matrix.tolist()

        if is_multiclass:
            # For multiclass, return the full matrix as nested list
            cm_formatted = [[int(cell) for cell in row] for row in cm]
            cm_normalized = self.test_confusion_matrix_normalized.tolist()
            cm_normalized_formatted = [[round(cell, 5) for cell in row] for row in cm_normalized]
        else:
            # For binary, return named values
            cm_formatted = {
                "true_negative": int(cm[0][0]),
                "false_positive": int(cm[0][1]),
                "false_negative": int(cm[1][0]),
                "true_positive": int(cm[1][1]),
            }
            cm_normalized = self.test_confusion_matrix_normalized.tolist()
            cm_normalized_formatted = {
                "true_negative": round(cm_normalized[0][0], 5),
                "false_positive": round(cm_normalized[0][1], 5),
                "false_negative": round(cm_normalized[1][0], 5),
                "true_positive": round(cm_normalized[1][1], 5),
            }

        return {"test_confusion_matrix": cm_formatted, "test_confusion_matrix_normalized": cm_normalized_formatted}

    def _static_metric_visualize(self, metrics: dict[str, Any], visual_name_prefix: str, is_multiclass: bool) -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # For multiclass, use smaller font size and larger figure
        if is_multiclass:
            n_classes = self.test_confusion_matrix.shape[0]
            figsize = max(8, n_classes * 0.8)
            plt.figure(figsize=(figsize, figsize))
            annot_kws = {"size": max(6, 12 - n_classes // 2)}
        else:
            plt.figure(figsize=(8, 6))
            annot_kws = {"size": 12}

        sns.heatmap(self.test_confusion_matrix, annot=True, fmt="d", annot_kws=annot_kws)
        visual_name = visual_name_prefix + "confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(Path(self.visual_path) / visual_name)
        plt.close()

        if is_multiclass:
            plt.figure(figsize=(figsize, figsize))
        else:
            plt.figure(figsize=(8, 6))

        visual_name = visual_name_prefix + "confusion_matrix_normalized.png"
        sns.heatmap(self.test_confusion_matrix_normalized, annot=True, fmt=".2f", annot_kws=annot_kws)
        plt.tight_layout()
        plt.savefig(Path(self.visual_path) / visual_name)
        plt.close()
