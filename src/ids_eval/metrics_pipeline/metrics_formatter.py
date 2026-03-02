from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List

from ids_eval.dto.formatted_result import (
    FormattedAdversarialResult,
    FormattedEvaluation,
    FormattedMetric,
    FormattedPerformanceDrop,
    FormattedResults,
    FormattedTestResult,
    FormattedTraining,
)
from ids_eval.dto.metric_config import MetricMetadata
from ids_eval.dto.run_config import RunConfig


class MetricsFormatter:
    # Internal keys that should not appear in the formatted output
    INTERNAL_KEYS = {
        "run_id",
        "train_run_id",
        "train_dataset_index",
        "test_dataset_index",
        "train_fold",
        "test_fold",
        "train_ids_plugin",
        "test_ids_plugin",
        "train_dataset_indices",
        "test_dataset_indices",
        "evaluation_scope",
        "test_y_true",
        "test_y_pred",
        "test_y_proba",
        "is_adversarial",
        "test_attack_name",
        "test_n_samples",
        "test_data_size_gb",
    }

    # Metrics to track for performance drop calculation
    ROBUSTNESS_METRICS = {"test_accuracy", "test_f1_score", "test_precision", "test_recall", "test_roc_auc"}

    def __init__(self, config: RunConfig, metadata: dict[str, MetricMetadata]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._metadata = metadata

        # Build dataset index to name mapping
        self._dataset_names: Dict[int, str] = {}
        if config.data_manager and config.data_manager.dataset:
            for i, ds in enumerate(config.data_manager.dataset):
                self._dataset_names[i] = ds.name

    def format_results(self, raw_metrics: list[dict[str, Any]]) -> FormattedResults:
        if not raw_metrics:
            return FormattedResults(evaluations=[], metadata_index={})

        # Separate clean and adversarial metrics
        clean_metrics = [m for m in raw_metrics if not m.get("is_adversarial", False)]
        adv_metrics = [m for m in raw_metrics if m.get("is_adversarial", False)]

        # Group clean metrics by (train_dataset_index, model)
        groups: Dict[tuple, List[dict]] = defaultdict(list)
        for m in clean_metrics:
            train_idx = m.get("train_dataset_index", 0)
            model = m.get("train_ids_plugin", m.get("test_ids_plugin", "unknown"))
            key = (train_idx, model)
            groups[key].append(m)

        # Group adversarial metrics by (train_dataset_index, model, test_dataset_index)
        adv_groups: Dict[tuple, List[dict]] = defaultdict(list)
        for m in adv_metrics:
            train_idx = m.get("train_dataset_index", 0)
            test_idx = m.get("test_dataset_index", 0)
            model = m.get("train_ids_plugin", m.get("test_ids_plugin", "unknown"))
            key = (train_idx, model, test_idx)
            adv_groups[key].append(m)

        evaluations = []
        for (train_idx, model), test_results in groups.items():
            # Find corresponding adversarial results
            adv_results_for_eval: Dict[int, List[dict]] = {}
            for (adv_train, adv_model, adv_test), adv_list in adv_groups.items():
                if adv_train == train_idx and adv_model == model:
                    adv_results_for_eval[adv_test] = adv_list

            evaluation = self._format_evaluation(train_idx, model, test_results, adv_results_for_eval)
            evaluations.append(evaluation)

        metadata_index = {key: meta.model_dump() for key, meta in self._metadata.items()}

        return FormattedResults(evaluations=evaluations, metadata_index=metadata_index)

    def _format_evaluation(
        self, train_idx: int, model: str, test_results: List[dict], adv_results: Dict[int, List[dict]] | None = None
    ) -> FormattedEvaluation:
        first_result = test_results[0] if test_results else {}
        training = self._extract_training(first_result)

        formatted_tests = []
        for result in test_results:
            test_idx = result.get("test_dataset_index", 0)
            # Get adversarial results for this test dataset
            adv_for_test = adv_results.get(test_idx, []) if adv_results else []
            formatted = self._format_test_result(result, train_idx, adv_for_test)
            formatted_tests.append(formatted)

        return FormattedEvaluation(
            trained_on=self._get_dataset_name(train_idx),
            model=model,
            model_size_mb=first_result.get("model_storage_size_mb"),
            model_cached=first_result.get("model_loaded_from_cache"),
            training=training,
            test_results=formatted_tests,
        )

    def _extract_training(self, result: dict) -> FormattedTraining | None:
        training_metrics: Dict[str, List[FormattedMetric]] = defaultdict(list)
        sample_count = None

        for key, value in result.items():
            if not key.startswith("train_") or value is None:
                continue

            if key in self.INTERNAL_KEYS:
                continue

            if "sample_count" in key:
                sample_count = value
                continue

            formatted = self._format_metric(key, value, prefix="train_")
            if formatted:
                training_metrics[formatted.category].append(formatted)

        if not training_metrics and sample_count is None:
            return None

        return FormattedTraining(metrics=dict(training_metrics), sample_count=sample_count)

    def _format_test_result(
        self, result: dict, train_idx: int, adv_results: List[dict] | None = None
    ) -> FormattedTestResult:
        test_idx = result.get("test_dataset_index", 0)

        # Collect and categorize test metrics
        metrics: Dict[str, List[FormattedMetric]] = defaultdict(list)
        clean_metric_values: Dict[str, float] = {}

        for key, value in result.items():
            # Skip internal keys, None values, and training metrics
            if key in self.INTERNAL_KEYS or value is None:
                continue
            if key.startswith("train_"):
                continue

            formatted = self._format_metric(key, value, prefix="test_")
            if formatted:
                metrics[formatted.category].append(formatted)

            # Track metrics for performance drop calculation
            if key in self.ROBUSTNESS_METRICS and isinstance(value, (int, float)):
                clean_metric_values[key] = float(value)

        # Process adversarial results
        adversarial_results: List[FormattedAdversarialResult] = []
        if adv_results:
            adversarial_results = self._format_adversarial_results(adv_results, clean_metric_values)

        return FormattedTestResult(
            dataset=self._get_dataset_name(test_idx),
            is_cross_dataset=train_idx != test_idx,
            is_adversarial=False,
            metrics=dict(metrics),
            adversarial_results=adversarial_results,
        )

    def _format_adversarial_results(
        self, adv_results: List[dict], clean_metric_values: Dict[str, float]
    ) -> List[FormattedAdversarialResult]:
        formatted_adv_results: List[FormattedAdversarialResult] = []

        for adv_result in adv_results:
            attack_name = adv_result.get("test_attack_name", "unknown")

            # Collect metrics for this adversarial result
            metrics: Dict[str, List[FormattedMetric]] = defaultdict(list)
            adv_metric_values: Dict[str, float] = {}

            for key, value in adv_result.items():
                if key in self.INTERNAL_KEYS or value is None:
                    continue
                if key.startswith("train_"):
                    continue

                formatted = self._format_metric(key, value, prefix="test_")
                if formatted:
                    metrics[formatted.category].append(formatted)

                # Track metrics for performance drop
                if key in self.ROBUSTNESS_METRICS and isinstance(value, (int, float)):
                    adv_metric_values[key] = float(value)

            # Calculate performance drops
            performance_drops = self._calculate_performance_drops(clean_metric_values, adv_metric_values)

            formatted_adv_results.append(
                FormattedAdversarialResult(
                    attack_name=attack_name, metrics=dict(metrics), performance_drops=performance_drops
                )
            )

        return formatted_adv_results

    def _calculate_performance_drops(
        self, clean_values: Dict[str, float], adv_values: Dict[str, float]
    ) -> List[FormattedPerformanceDrop]:
        drops: List[FormattedPerformanceDrop] = []

        for key in self.ROBUSTNESS_METRICS:
            if key in clean_values and key in adv_values:
                clean_val = clean_values[key]
                adv_val = adv_values[key]
                absolute_drop = clean_val - adv_val
                relative_drop = (absolute_drop / clean_val * 100) if clean_val > 0 else 0.0

                # Get display name
                display_name = self._key_to_display_name(key.replace("test_", ""))

                drops.append(
                    FormattedPerformanceDrop(
                        metric_name=display_name,
                        clean_value=round(clean_val, 5),
                        adversarial_value=round(adv_val, 5),
                        absolute_drop=round(absolute_drop, 5),
                        relative_drop_percent=round(relative_drop, 5),
                    )
                )

        return drops

    def _format_metric(self, key: str, value: Any, prefix: str = "") -> FormattedMetric | None:
        meta = self._metadata.get(key)

        if meta is None and prefix and key.startswith(prefix):
            meta = self._metadata.get(key[len(prefix) :])

        if meta:
            display_name = meta.display_name
            category = meta.category
            unit = meta.unit
            higher_is_better = meta.higher_is_better
        else:
            # Fallback: infer from key name
            base_key = key[len(prefix) :] if key.startswith(prefix) else key
            display_name = self._key_to_display_name(base_key)
            category = "other"
            unit = "unknown"
            higher_is_better = None

        formatted_value = self._format_value(value)

        return FormattedMetric(
            key=key,
            display_name=display_name,
            value=formatted_value,
            category=category,
            unit=unit,
            higher_is_better=higher_is_better,
        )

    @staticmethod
    def _key_to_display_name(key: str) -> str:
        for prefix in ("test_", "train_"):
            if key.startswith(prefix):
                key = key[len(prefix) :]

        return key.replace("_", " ").title()

    @staticmethod
    def _format_value(value: Any) -> Any:
        if isinstance(value, float):
            if abs(value) < 0.001 and value != 0:
                return value
            return round(value, 5)
        return value

    def _get_dataset_name(self, index: int) -> str:
        return self._dataset_names.get(index, f"Dataset_{index}")
