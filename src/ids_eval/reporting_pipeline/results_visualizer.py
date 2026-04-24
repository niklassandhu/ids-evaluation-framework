from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ids_eval.dto.formatted_result import (
    FormattedEvaluation,
    FormattedResults,
    FormattedTestResult,
)
from ids_eval.dto.run_config import RunConfig
from ids_eval.run_config_pipeline.config_manager import ConfigManager


@dataclass
class _MetricInfo:
    key: str
    display_name: str
    unit: str | None
    higher_is_better: bool | None
    comparison_chart_type: str  # "grouped_bar" or "horizontal_bar"


@dataclass
class _ComparisonGroup:
    group_id: str
    chart_type: str  # "grouped_bar" or "horizontal_bar"
    metrics: list[_MetricInfo] = field(default_factory=list)


class ResultsVisualizer:
    """Generates comparative visualizations based on metrics plugin metadata."""

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.reports_root = ConfigManager.get_report_directory(config)
        self.visual_dir = self.reports_root / "visuals_comparison"
        self.visual_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, formatted_results: FormattedResults, summary: dict[str, Any]) -> None:
        evaluations = formatted_results.evaluations
        if not evaluations:
            self.logger.warning("No evaluations available -- skipping comparison visualizations.")
            return

        groups = self._build_comparison_groups(formatted_results.metadata_index)
        if not groups:
            self.logger.info("No metrics with comparison_group defined -- skipping comparison visualizations.")
            return

        is_cross = summary.get("overview", {}).get("evaluation_type") == "cross_dataset"

        for group in groups.values():
            if group.chart_type == "grouped_bar":
                self._plot_grouped_bar_for_group(evaluations, group)
                if is_cross:
                    self._plot_heatmaps_for_group(evaluations, group)
            elif group.chart_type == "horizontal_bar":
                self._plot_horizontal_bar_for_group(evaluations, group)

        model_sizes = self._collect_model_sizes(evaluations)
        if model_sizes:
            self._render_simple_horizontal_bar(
                labels=list(model_sizes.keys()),
                values=list(model_sizes.values()),
                title="Resource Comparison -- Model Size",
                xlabel="Model Size (MB)",
                filename="resource_model_size.png",
            )

        self.logger.info(f"Comparison visualizations saved to {self.visual_dir}")

    @staticmethod
    def _build_comparison_groups(metadata_index: dict[str, dict[str, Any]]) -> dict[str, _ComparisonGroup]:
        groups: dict[str, _ComparisonGroup] = {}

        for key, meta in metadata_index.items():
            group_id = meta.get("comparison_group")
            chart_type = meta.get("comparison_chart_type")
            if group_id is None or chart_type is None:
                continue

            info = _MetricInfo(
                key=key,
                display_name=meta.get("display_name", key),
                unit=meta.get("unit"),
                higher_is_better=meta.get("higher_is_better"),
                comparison_chart_type=chart_type,
            )

            if group_id not in groups:
                groups[group_id] = _ComparisonGroup(group_id=group_id, chart_type=chart_type, metrics=[info])
            else:
                groups[group_id].metrics.append(info)

        return groups

    def _plot_grouped_bar_for_group(
        self,
        evaluations: list[FormattedEvaluation],
        group: _ComparisonGroup,
    ) -> None:
        all_data: dict[str, dict[str, dict[str, float]]] = {}
        for mi in group.metrics:
            collected = self._collect_test_metric(evaluations, mi.key)
            if collected:
                all_data[mi.display_name] = collected

        if not all_data:
            self.logger.info(f"No data for group '{group.group_id}' -- skipping grouped bar chart.")
            return

        models: set[str] = set()
        for model_dict in all_data.values():
            models.update(model_dict.keys())
        models_sorted = sorted(models)

        all_dataset_labels: set[str] = set()
        for model_dict in all_data.values():
            for ds_dict in model_dict.values():
                all_dataset_labels.update(ds_dict.keys())
        datasets_sorted = sorted(all_dataset_labels)

        single_dataset = len(datasets_sorted) == 1

        series: list[tuple[str, dict[str, float]]] = []
        if single_dataset:
            ds = datasets_sorted[0]
            for metric_name, model_dict in all_data.items():
                values: dict[str, float] = {}
                for model in models_sorted:
                    val = model_dict.get(model, {}).get(ds)
                    if val is not None:
                        values[model] = val
                if values:
                    series.append((metric_name, values))
        else:
            for metric_name, model_dict in all_data.items():
                for ds in datasets_sorted:
                    values = {}
                    for model in models_sorted:
                        val = model_dict.get(model, {}).get(ds)
                        if val is not None:
                            values[model] = val
                    if values:
                        series.append((f"{metric_name} ({ds})", values))

        if not series:
            return

        safe_group = group.group_id.replace(" ", "_")
        self._render_grouped_bar_chart(
            models=models_sorted,
            series=series,
            title=f"Comparison -- {group.group_id.replace('_', ' ').title()}",
            ylabel="Value",
            filename=f"comparison_{safe_group}.png",
        )

    def _plot_heatmaps_for_group(
        self,
        evaluations: list[FormattedEvaluation],
        group: _ComparisonGroup,
    ) -> None:
        model_evals: dict[str, list[FormattedEvaluation]] = defaultdict(list)
        for ev in evaluations:
            model_evals[ev.model].append(ev)

        for mi in group.metrics:
            for model, evals in model_evals.items():
                matrix, train_labels, test_labels = self._build_heatmap_matrix(evals, mi.key)
                if matrix is None:
                    continue

                safe_model = model.replace(" ", "_").replace("/", "_")
                safe_key = mi.key.replace(" ", "_")
                self._render_heatmap(
                    matrix=matrix,
                    row_labels=train_labels,
                    col_labels=test_labels,
                    title=f"{mi.display_name} -- {model}",
                    filename=f"heatmap_{safe_model}_{safe_key}.png",
                )

    def _plot_horizontal_bar_for_group(
        self,
        evaluations: list[FormattedEvaluation],
        group: _ComparisonGroup,
    ) -> None:
        series: list[tuple[str, dict[str, float]]] = []

        for mi in group.metrics:
            test_data = self._collect_test_metric(evaluations, mi.key)
            if not test_data:
                continue

            model_avgs: dict[str, float] = {}
            for model, ds_dict in test_data.items():
                vals = [v for v in ds_dict.values() if v is not None]
                if vals:
                    model_avgs[model] = sum(vals) / len(vals)
            if model_avgs:
                unit_suffix = f" ({mi.unit})" if mi.unit else ""
                series.append((f"{mi.display_name}{unit_suffix}", model_avgs))

        if not series:
            self.logger.info(f"No test data for group '{group.group_id}' -- skipping horizontal bar chart.")
            return

        all_models: set[str] = set()
        for _, model_dict in series:
            all_models.update(model_dict.keys())
        models_sorted = sorted(all_models)

        safe_group = group.group_id.replace(" ", "_")

        if len(series) == 1:
            # Single metric in group -- simple horizontal bar
            label, model_dict = series[0]
            labels = [m for m in models_sorted if m in model_dict]
            values = [model_dict[m] for m in labels]
            self._render_simple_horizontal_bar(
                labels=labels,
                values=values,
                title=f"Resource Comparison -- {group.group_id.replace('_', ' ').title()}",
                xlabel=label,
                filename=f"resource_{safe_group}.png",
            )
        else:
            # Multiple metrics -- grouped horizontal bar
            self._render_grouped_horizontal_bar(
                models=models_sorted,
                series=series,
                title=f"Resource Comparison -- {group.group_id.replace('_', ' ').title()}",
                filename=f"resource_{safe_group}.png",
            )

    @staticmethod
    def _collect_test_metric(
        evaluations: list[FormattedEvaluation],
        key: str,
    ) -> dict[str, dict[str, float]]:
        candidates = [key]
        if not key.startswith("test_"):
            candidates.append(f"test_{key}")

        data: dict[str, dict[str, float]] = defaultdict(dict)

        for ev in evaluations:
            for tr in ev.test_results:
                if tr.is_adversarial:
                    continue
                value = ResultsVisualizer._find_metric_in_test_result(tr, candidates)
                if value is not None:
                    label = f"{ev.trained_on} \u2192 {tr.dataset}" if tr.is_cross_dataset else tr.dataset
                    data[ev.model][label] = value

        return dict(data)

    @staticmethod
    def _find_metric_in_test_result(
        tr: FormattedTestResult,
        candidates: list[str],
    ) -> Optional[float]:
        for metrics_list in tr.metrics.values():
            for m in metrics_list:
                if m.key in candidates and isinstance(m.value, (int, float)):
                    return float(m.value)
        return None

    @staticmethod
    def _collect_model_sizes(evaluations: list[FormattedEvaluation]) -> dict[str, float]:
        sizes: dict[str, float] = {}
        for ev in evaluations:
            if ev.model_size_mb is not None and ev.model not in sizes:
                sizes[ev.model] = ev.model_size_mb
        return sizes

    def _build_heatmap_matrix(
        self,
        evaluations: list[FormattedEvaluation],
        key: str,
    ) -> tuple[Optional[np.ndarray], list[str], list[str]]:
        candidates = [key]
        if not key.startswith("test_"):
            candidates.append(f"test_{key}")

        raw: dict[str, dict[str, float]] = defaultdict(dict)
        for ev in evaluations:
            for tr in ev.test_results:
                if tr.is_adversarial:
                    continue
                value = self._find_metric_in_test_result(tr, candidates)
                if value is not None:
                    raw[ev.trained_on][tr.dataset] = value

        if not raw:
            return None, [], []

        train_labels = sorted(raw.keys())
        test_labels = sorted({ds for inner in raw.values() for ds in inner})

        if len(train_labels) < 1 or len(test_labels) < 2:
            return None, [], []

        matrix = np.full((len(train_labels), len(test_labels)), np.nan)
        for i, train_ds in enumerate(train_labels):
            for j, test_ds in enumerate(test_labels):
                matrix[i, j] = raw[train_ds].get(test_ds, np.nan)

        return matrix, train_labels, test_labels

    def _render_grouped_bar_chart(
        self,
        models: list[str],
        series: list[tuple[str, dict[str, float]]],
        title: str,
        ylabel: str,
        filename: str,
    ) -> None:
        n_models = len(models)
        n_series = len(series)
        if n_models == 0 or n_series == 0:
            return

        x = np.arange(n_models)
        width = 0.8 / n_series

        fig, ax = plt.subplots(figsize=(max(8, n_models * 2.5), 6))

        for idx, (label, model_dict) in enumerate(series):
            values = [model_dict.get(model, 0.0) for model in models]
            offset = (idx - (n_series - 1) / 2) * width
            bars = ax.bar(x + offset, values, width, label=label)

            for bar, val in zip(bars, values):
                if val != 0.0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xlabel("Model")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.legend(title="Metric", fontsize=7, loc="best")
        #ax.grid(axis="y", alpha=0.3)
        ax.grid(False)

        plt.tight_layout()
        plt.savefig(self.visual_dir / filename, dpi=150)
        plt.close(fig)

    def _render_heatmap(
        self,
        matrix: np.ndarray,
        row_labels: list[str],
        col_labels: list[str],
        title: str,
        filename: str,
    ) -> None:
        n_rows, n_cols = matrix.shape
        fig_w = max(8, n_cols * 1.5)
        fig_h = max(6, n_rows * 1.2)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            xticklabels=col_labels,
            yticklabels=row_labels,
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            ax=ax,
        )

        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xlabel("Test Dataset")
        ax.set_ylabel("Train Dataset")
        ax.set_title(title)
        ax.grid(False)

        plt.tight_layout()
        plt.savefig(self.visual_dir / filename, dpi=150)
        plt.close(fig)

    def _render_grouped_horizontal_bar(
        self,
        models: list[str],
        series: list[tuple[str, dict[str, float]]],
        title: str,
        filename: str,
    ) -> None:
        """Render a grouped horizontal bar chart (models on y-axis, bars per series)."""
        n_models = len(models)
        n_series = len(series)
        if n_models == 0 or n_series == 0:
            return

        y = np.arange(n_models)
        height = 0.8 / n_series

        fig, ax = plt.subplots(figsize=(10, max(4, n_models * 1.2)))

        all_vals = [v for _, md in series for v in md.values() if v != 0.0]
        max_val = max(abs(v) for v in all_vals) if all_vals else 1.0

        for idx, (label, model_dict) in enumerate(series):
            values = [model_dict.get(model, 0.0) for model in models]
            offset = (idx - (n_series - 1) / 2) * height
            bars = ax.barh(y + offset, values, height, label=label)

            for bar, val in zip(bars, values):
                if val != 0.0:
                    ax.text(
                        bar.get_width() + 0.01 * max_val,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.2f}",
                        ha="left",
                        va="center",
                        fontsize=8,
                    )
        
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_ylabel("Model")
        ax.set_xlabel("Value")
        ax.set_title(title)
        ax.set_yticks(y)
        ax.set_yticklabels(models)
        ax.legend(title="Metric", fontsize=8, loc="best")
        # ax.grid(axis="x", alpha=0.3)
        ax.grid(False)

        plt.tight_layout()
        plt.savefig(self.visual_dir / filename, dpi=150)
        plt.close(fig)

    def _render_simple_horizontal_bar(
        self,
        labels: list[str],
        values: list[float],
        title: str,
        xlabel: str,
        filename: str,
    ) -> None:
        if not labels:
            return

        y = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.8)))

        bars = ax.barh(y, values, 0.5)

        max_val = max(abs(v) for v in values) if values else 1.0
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 0.01 * max_val,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                ha="left",
                va="center",
                fontsize=9,
            )

        ax.spines[['right', 'top']].set_visible(False)
        ax.set_ylabel("Model")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        #ax.grid(axis="x", alpha=0.3)
        ax.grid(False)

        plt.tight_layout()
        plt.savefig(self.visual_dir / filename, dpi=150)
        plt.close(fig)
