import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ids_eval.dto.metric_config import MetricMetadata
from ids_eval.dto.run_config import RunConfig
from ids_eval.run_config_pipeline.config_manager import ConfigManager


class AbstractStaticMetric(ABC):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config: RunConfig | None = None
        self.visual_path: Path | None = None
        self._params: dict[str, Any] = {}

    def metadata(self) -> list[MetricMetadata]:
        is_multiclass = self.config.evaluation.general.is_multiclass
        return self._static_metric_metadata(is_multiclass)

    def prepare(self, config: RunConfig, params: dict[str, Any] | None = None) -> None:
        self.config = config
        self._params = params or {}
        report_root = ConfigManager.get_report_directory(config)
        self.visual_path = report_root / "visuals_evaluation"
        self.visual_path.mkdir(exist_ok=True)
        try:
            self._static_metric_prepare()
        except Exception as e:
            self.logger.error(f"Error in static metric preparation: {e}")
            self.logger.exception(e)
            raise RuntimeError("Error in static metric preparation")

    def calculate(self, metrics: dict[str, Any]) -> dict[str, Any]:
        try:
            is_multiclass = self.config.evaluation.general.is_multiclass
            return self._static_metric_calculate(metrics, is_multiclass)
        except Exception as e:
            self.logger.error(f"Error in static metric calculation: {e}")
            self.logger.exception(e)
            raise RuntimeError("Error in static metric calculation")

    def visualize(self, metrics: dict[str, Any], visual_name_prefix: str) -> None:
        try:
            is_multiclass = self.config.evaluation.general.is_multiclass
            return self._static_metric_visualize(metrics, visual_name_prefix, is_multiclass)
        except Exception as e:
            self.logger.error(f"Error in static metric visualization: {e}")
            self.logger.exception(e)
            raise RuntimeError("Error in static metric visualization")

    @abstractmethod
    def _static_metric_metadata(self, is_multiclass: bool) -> list[MetricMetadata]:
        return []

    @abstractmethod
    def _static_metric_prepare(self) -> None:
        pass

    @abstractmethod
    def _static_metric_calculate(self, metrics: dict[str, Any], is_multiclass: bool) -> dict[str, Any]:
        pass

    def _static_metric_visualize(self, metrics: dict[str, Any], visual_name_prefix: str, is_multiclass: bool) -> None:
        pass
