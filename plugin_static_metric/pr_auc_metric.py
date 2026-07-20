from typing import Any
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score

from ids_eval.dto.metric_config import MetricMetadata
from ids_eval.interface.abstract_static_metric import AbstractStaticMetric


class PrAucMetric(AbstractStaticMetric):
    """Average Precision (Area under the precision-recall-curve)."""

    def __init__(self):
        super().__init__()

    def _static_metric_metadata(self, is_multiclass: bool) -> list[MetricMetadata]:
        return [MetricMetadata(
            key="test_pr_auc",
            display_name="PR-AUC (Average Precision)",
            category="detection",
            unit="ratio",
            higher_is_better=True,
            description="Area under the precision-recall curve",
            comparison_group="detection_performance",
            comparison_chart_type="grouped_bar",
        )]

    def _static_metric_prepare(self) -> None:
        self.average_precision_score = average_precision_score

    def _static_metric_calculate(self, metrics: dict[str, Any], is_multiclass: bool) -> dict[str, Any]:
        """Average Precision; multiclass is reduced to attack-vs-benign (1 - P(benign))."""
        y_true = metrics["test_y_true"]
        y_score = metrics["test_y_proba"]
        if is_multiclass:
            #set up 2 classes: benign = 0, else =1
            y_true = np.where(y_true == "benign", 0, 1)
            benign_index = y_score.shape[1] - 1
            y_score = 1.0 - y_score[:, benign_index]
            # print(y_score[:, benign_index] ,y_score)
        else:
            if getattr(y_score, "ndim", 1) > 1 and y_score.shape[1] > 1:
                y_score = y_score[:, 1]
        return {"test_pr_auc": round(float(self.average_precision_score(y_true, y_score)), 5)}

    def _static_metric_visualize(self, metrics, visual_name_prefix, is_multiclass) -> None:
        """Plot the PR curve with the prevalence (no-skill) baseline."""
        y_true = metrics["test_y_true"]
        y_score = metrics["test_y_proba"]
        if is_multiclass:
            y_true = np.where(y_true == "benign", 0, 1)
            benign_index = y_score.shape[1] - 1
            y_score = 1.0 - y_score[:, benign_index]
        else:
            if getattr(y_score, "ndim", 1) > 1 and y_score.shape[1] > 1:
                y_score = y_score[:, 1]

        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        PrecisionRecallDisplay.from_predictions(
            y_true, y_score, ax=ax,
            name="PR Curve (Attack vs Benign)" if is_multiclass else "PR Curve",
        )

        prevalence = float(np.mean(y_true))
        ax.axhline(prevalence, ls="--", lw=1, color="grey", label=f"Baseline (prevalence={prevalence:.3f})")
        ax.set_ylim(0, 1)
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(Path(self.visual_path) / (visual_name_prefix + "pr_curve.png"), dpi=150)
        plt.close()