from __future__ import annotations

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

from ids_eval.dto.formatted_result import (
    FormattedAdversarialResult,
    FormattedEvaluation,
    FormattedMetric,
    FormattedPerformanceDrop,
    FormattedResults,
    FormattedTestResult,
    FormattedTraining,
)
from ids_eval.dto.run_config import RunConfig
from ids_eval.run_config_pipeline.config_manager import ConfigManager


class ReportWriter:
    def __init__(self, config: RunConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.reports_root = ConfigManager.get_report_directory(config)

    def write_dataset_report(self, report_dict: dict[str, Any], filename: str = "dataset_report.yaml") -> Path:
        return self._write_yaml(report_dict, filename)

    def write_ids_report(self, formatted_results: FormattedResults, filename: str = "ids_report.yaml") -> Path:
        report = {
            "report_metadata": self._build_report_metadata(),
            "summary": self._build_summary(formatted_results),
            "evaluations": [self._evaluation_to_dict(ev) for ev in formatted_results.evaluations],
        }

        return self._write_yaml(report, filename)

    def write_summary(self, summary: dict[str, Any], filename: str = "evaluation_summary.yaml") -> Path:
        report = {"report_metadata": self._build_report_metadata()}
        report.update(summary)

        return self._write_yaml(report, filename)

    def _build_report_metadata(self) -> Dict[str, Any]:
        return {
            "generated_at": datetime.now().isoformat(),
            "name": self.config.general.name,
            "description": self.config.general.description,
            "config_hash": self.config.get_config_file_hash(),
        }

    @staticmethod
    def _build_summary(formatted_results: FormattedResults) -> Dict[str, Any]:
        evaluations = formatted_results.evaluations

        if not evaluations:
            return {"info": "No evaluation results available"}

        models = set()
        datasets = set()
        attacks = set()
        total_tests = 0
        total_adv_tests = 0

        for ev in evaluations:
            models.add(ev.model)
            datasets.add(ev.trained_on)
            for tr in ev.test_results:
                datasets.add(tr.dataset)
                total_tests += 1
                for adv in tr.adversarial_results:
                    attacks.add(adv.attack_name)
                    total_adv_tests += 1

        # Determine evaluation type
        has_cross = any(tr.is_cross_dataset for ev in evaluations for tr in ev.test_results)
        has_adversarial = total_adv_tests > 0

        summary: Dict[str, Any] = {
            "models_evaluated": sorted(list(models)),
            "datasets": sorted(list(datasets)),
            "evaluation_type": "cross_dataset" if has_cross else "single_dataset",
            "total_test_runs": total_tests,
        }

        if has_adversarial:
            summary["adversarial_evaluation"] = {
                "enabled": True,
                "attacks_used": sorted(list(attacks)),
                "total_adversarial_tests": total_adv_tests,
            }

        return summary

    def _evaluation_to_dict(self, ev: FormattedEvaluation) -> Dict[str, Any]:
        result: Dict[str, Any] = {"trained_on": ev.trained_on, "model": ev.model}

        if ev.model_size_mb is not None:
            result["model_size_mb"] = round(ev.model_size_mb, 5)

        if ev.model_cached is not None:
            result["model_cached"] = ev.model_cached

        if ev.training:
            result["training"] = self._training_to_dict(ev.training)

        result["tested_on"] = [self._test_result_to_dict(tr) for tr in ev.test_results]

        return result

    def _training_to_dict(self, training: FormattedTraining) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        if training.sample_count is not None:
            result["sample_count"] = training.sample_count

        for category, metrics_list in training.metrics.items():
            if metrics_list:
                result[category] = self._metrics_list_to_dict(metrics_list)

        return result

    def _test_result_to_dict(self, tr: FormattedTestResult) -> Dict[str, Any]:
        result: Dict[str, Any] = {"dataset": tr.dataset, "is_cross_dataset": tr.is_cross_dataset}

        # Add clean performance metrics
        clean_performance: Dict[str, Any] = {}
        for category, metrics_list in tr.metrics.items():
            if metrics_list:
                clean_performance[category] = self._metrics_list_to_dict(metrics_list)

        if clean_performance:
            result["clean_performance"] = clean_performance

        # Add adversarial results if present
        if tr.adversarial_results:
            result["adversarial_performance"] = [
                self._adversarial_result_to_dict(adv) for adv in tr.adversarial_results
            ]

            # Add robustness summary
            robustness_summary = self._build_robustness_summary(tr.adversarial_results)
            if robustness_summary:
                result["robustness_summary"] = robustness_summary

        return result

    def _adversarial_result_to_dict(self, adv: FormattedAdversarialResult) -> Dict[str, Any]:
        result: Dict[str, Any] = {"attack": adv.attack_name}

        # Add metrics by category
        for category, metrics_list in adv.metrics.items():
            if metrics_list:
                result[category] = self._metrics_list_to_dict(metrics_list)

        # Add performance drops
        if adv.performance_drops:
            result["performance_drop"] = self._performance_drops_to_dict(adv.performance_drops)

        return result

    @staticmethod
    def _performance_drops_to_dict(drops: list[FormattedPerformanceDrop]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for drop in drops:
            key = drop.metric_name.lower().replace(" ", "_")
            result[key] = {
                "clean": drop.clean_value,
                "adversarial": drop.adversarial_value,
                "absolute_drop": drop.absolute_drop,
                "relative_drop_percent": drop.relative_drop_percent,
            }
        return result

    @staticmethod
    def _build_robustness_summary(adversarial_results: list[FormattedAdversarialResult]) -> Dict[str, Any]:
        if not adversarial_results:
            return {}

        summary: Dict[str, Any] = {}

        # Collect drops across all attacks
        all_drops: Dict[str, list[float]] = {}
        for adv in adversarial_results:
            for drop in adv.performance_drops:
                key = drop.metric_name.lower().replace(" ", "_")
                if key not in all_drops:
                    all_drops[key] = []
                all_drops[key].append(drop.relative_drop_percent)

        # Calculate average and max drops
        for metric, drops in all_drops.items():
            if drops:
                summary[metric] = {
                    "avg_drop_percent": round(sum(drops) / len(drops), 5),
                    "max_drop_percent": round(max(drops), 5),
                    "min_drop_percent": round(min(drops), 5),
                }

        return summary

    @staticmethod
    def _metrics_list_to_dict(metrics_list: list[FormattedMetric]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for m in metrics_list:
            key = m.display_name.lower().replace(" ", "_")
            result[key] = m.value
        return result

    def _write_yaml(self, data: dict[str, Any], filename: str) -> Path:
        path = self.reports_root / filename
        try:
            with path.open("w", encoding="utf-8") as f:
                yaml.dump(data, f, sort_keys=False, default_flow_style=False, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to write YAML report {path}: {e}")
        return path
