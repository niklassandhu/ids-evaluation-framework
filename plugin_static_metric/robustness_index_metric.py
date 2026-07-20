from __future__ import annotations

from typing import Any
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from ids_eval.dto.metric_config import MetricMetadata
from ids_eval.interface.abstract_static_metric import AbstractStaticMetric


def _ri_from_curve(curve: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, float]:
    """Sort by epsilon; return (eps, acc, RI = trapz(acc, eps) / eps-span)"""
    eps = np.array([p["epsilon"] for p in curve], dtype=float)
    acc = np.array([p["accuracy"] for p in curve], dtype=float)

    order = np.argsort(eps)
    eps, acc = eps[order], acc[order]

    span = float(eps[-1] - eps[0])
    if span <= 0:
        ri = float(acc[0])
    else:
        ri = float(np.trapezoid(acc, eps) / span)

    return eps, acc, ri


class RobustnessIndexMetric(AbstractStaticMetric):
    """
    Computes the Robustness Index (RI) from a robustness curve, as in  M. Rajhans and V. Khawarey,
    "Empirical analysis of adversarial robustness and explainability drift in cybersecurity classifiers", 2026.
    arXiv:2602.06395

       RI = (1/eps_max) * integral of raw accuracy from eps=0 to eps_max)

    Magnitudes of epsilon pertubations are set in run_confing.py via config.yml or default.
    """

    def __init__(self):
        super().__init__()

    def _static_metric_metadata(self, is_multiclass: bool) -> list[MetricMetadata]:
        return [
            MetricMetadata(
                key="test_robustness_index",
                display_name="Robustness Index",
                category="other",
                unit="ratio",
                higher_is_better=True,
                description="Area under the accuracy-vs-perturbation curve, normalized by the epsilon range)",
                comparison_group="robustness",
                comparison_chart_type="grouped_bar",
            )
        ]

    def _static_metric_prepare(self) -> None:
        pass

    def _static_metric_calculate(
            self,
            metrics: dict[str, Any],
            is_multiclass: bool,
        ) -> dict[str, Any]:
            """Compute the RI from the robustness curve; empty dict if none present."""

            curve = metrics.get("robustness_curve", [])

            if not curve:
                return {}

            _, _, ri = _ri_from_curve(curve)

            return {
                "test_robustness_index": round(ri, 5)
        }

    def _static_metric_visualize(self, metrics: dict[str, Any], visual_name_prefix: str, is_multiclass: bool) -> None:
        """Plot accuracy vs perturbation and shade the RI area."""
        curve = metrics.get("robustness_curve", [])
        if not curve:
            return

        eps, acc, ri = _ri_from_curve(curve)

        base_acc = float(acc[0])
        if eps[0] == 0.0:
            base_label = f"Clean accuracy = {base_acc:.3f}"
        else:
            self.logger.warning("eps must be 0.0 in first sweep for baseline")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(eps, acc, marker="o", label="Accuracy under attack")
        ax.fill_between(eps, acc, alpha=0.15)
        ax.axhline(base_acc, ls="--", lw=1, color="grey", label=base_label)
        ax.set_xlabel("Perturbation magnitude ε")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.set_title(f"Robustness curve (RI = {ri:.3f})")
        ax.spines[["right", "top"]].set_visible(False)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(Path(self.visual_path) / (visual_name_prefix + "robustness_curve.png"), dpi=150)
        plt.close(fig)
