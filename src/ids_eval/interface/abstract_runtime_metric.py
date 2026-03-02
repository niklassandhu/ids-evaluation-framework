import logging
from abc import ABC, abstractmethod
from typing import Any

from ids_eval.dto.metric_config import MetricMetadata
from ids_eval.dto.run_config import RunConfig


class AbstractRuntimeMetric(ABC):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config: RunConfig | None = None
        self._params: dict[str, Any] = {}

    def metadata(self) -> list[MetricMetadata]:
        return self._runtime_metric_metadata()

    def prepare(self, params: dict[str, Any] | None = None) -> None:
        self._params = params or {}
        self._runtime_metric_prepare()

    def start(self) -> None:
        self._runtime_metric_start()

    def stop(self) -> None:
        self._runtime_metric_stop()

    def calculate(self) -> dict[str, Any]:
        return self._runtime_metric_calculate()

    @abstractmethod
    def _runtime_metric_metadata(self) -> list[MetricMetadata]:
        return []

    @abstractmethod
    def _runtime_metric_prepare(self) -> None:
        pass

    @abstractmethod
    def _runtime_metric_start(self) -> None:
        pass

    @abstractmethod
    def _runtime_metric_stop(self) -> None:
        pass

    @abstractmethod
    def _runtime_metric_calculate(self) -> dict[str, Any]:
        return {}
