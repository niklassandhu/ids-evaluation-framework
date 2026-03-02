from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from matplotlib import pyplot as plt

from ids_eval.dto.metric_config import MetricMetadata
from ids_eval.dto.run_config import RunConfig
from ids_eval.registry.static_metric_registry import StaticMetricRegistry


class MetricsCalculator:
    def __init__(self, config: RunConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._static_metric_registry: StaticMetricRegistry | None = None

    def calculate_metrics(
        self, train_metrics: list[dict[str, Any]], test_metrics: list[dict[str, Any]]
    ) -> Tuple[list[dict[str, Any]], Dict[str, MetricMetadata]]:
        metrics_per_model = self._merge_metrics(train_metrics, test_metrics)

        self._static_metric_registry = StaticMetricRegistry(self.config)
        self.logger.info("Importing static metrics plugins...")
        loaded_plugins = self._static_metric_registry.load_plugins()
        self.logger.info(f"Loaded {len(loaded_plugins)} static metric plugins.")

        self.logger.info("Computing static metrics...")

        all_metadata = {}
        for plugin, plugin_config in loaded_plugins:
            plugin.prepare(self.config, params=plugin_config.params)
            metadata_list = plugin.metadata()
            for meta in metadata_list:
                all_metadata[meta.key] = meta

        for metric in metrics_per_model:
            static_metrics_per_model = {}

            for plugin, plugin_config in loaded_plugins:
                calculated = plugin.calculate(metric)
                static_metrics_per_model.update(calculated)

                plugin.visualize(metric, metric["run_id"] + "_")
                plt.close("all")

            metric.update(static_metrics_per_model)
            for key in ("test_y_true", "test_y_pred", "test_y_proba"):
                metric.pop(key, None)

        return metrics_per_model, all_metadata

    def _merge_metrics(
        self, train_metrics: list[dict[str, Any]], test_metrics: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        metrics: list[dict[str, Any]] = []
        train_metric_lookup: dict[str, dict[str, Any]] = {}

        for metric in train_metrics:
            run_id = metric.get("run_id")
            if run_id:
                train_metric_lookup[run_id] = metric

        for test_metric in test_metrics:
            train_ref = test_metric.get("train_run_id")
            combined_metric: dict[str, Any] = {}
            if train_ref and train_ref in train_metric_lookup:
                combined_metric.update(train_metric_lookup[train_ref])
            else:
                self.logger.warning("No matching training metrics found for test run '%s'.", test_metric.get("run_id"))
            combined_metric.update(test_metric)
            metrics.append(combined_metric)

        return metrics
