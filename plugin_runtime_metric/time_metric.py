import time

from ids_eval.dto.metric_config import MetricMetadata
from ids_eval.interface.abstract_runtime_metric import AbstractRuntimeMetric


class TimeMetric(AbstractRuntimeMetric):
    def __init__(self):
        super().__init__()
        self.__t0 = 0.0
        self.__t1 = 0.0

    def _runtime_metric_metadata(self) -> list[MetricMetadata]:
        return [
            MetricMetadata(
                key="runtime_s",
                display_name="Runtime",
                category="runtime",
                unit="seconds",
                higher_is_better=False,
                description="Wall-clock time for the operation",
                comparison_group="test_runtime",
                comparison_chart_type="horizontal_bar",
            )
        ]

    def _runtime_metric_prepare(self):
        pass

    def _runtime_metric_start(self):
        self.__t0 = time.perf_counter()

    def _runtime_metric_stop(self):
        self.__t1 = time.perf_counter()

    def _runtime_metric_calculate(self) -> dict[str, float]:
        return {"runtime_s": round(self.__t1 - self.__t0, 5)}
