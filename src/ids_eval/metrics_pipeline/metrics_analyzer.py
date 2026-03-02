from __future__ import annotations

import logging
from typing import Any, Dict

from ids_eval.dto.formatted_result import (
    FormattedEvaluation,
    FormattedResults,
)
from ids_eval.dto.run_config import RunConfig


class MetricsAnalyzer:
    def __init__(self, config: RunConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def analyze(self, formatted_results: FormattedResults) -> dict[str, Any]:
        evaluations = formatted_results.evaluations

        if not evaluations:
            return {"overview": {"info": "No evaluation results to analyze"}}

        summary: Dict[str, Any] = {"overview": self._build_overview(evaluations)}

        return summary

    @staticmethod
    def _build_overview(evaluations: list[FormattedEvaluation]) -> Dict[str, Any]:
        models = set()
        datasets = set()
        total_tests = 0

        for ev in evaluations:
            models.add(ev.model)
            datasets.add(ev.trained_on)
            for tr in ev.test_results:
                datasets.add(tr.dataset)
                total_tests += 1

        eval_type = "single_dataset"
        has_cross = any(tr.is_cross_dataset for ev in evaluations for tr in ev.test_results)
        if has_cross:
            eval_type = "cross_dataset"

        return {
            "evaluation_type": eval_type,
            "models": sorted(list(models)),
            "datasets": sorted(list(datasets)),
            "total_evaluations": total_tests,
        }
