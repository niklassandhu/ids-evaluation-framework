import math
import threading
from typing import Any

import psutil

from ids_eval.dto.metric_config import MetricMetadata
from ids_eval.interface.abstract_runtime_metric import AbstractRuntimeMetric


class CpuLoadMetric(AbstractRuntimeMetric):
    """
    Runtime metric that measures the average CPU load of the IDS process
    during training and testing using thread-based sampling.
    """

    SAMPLING_INTERVAL_SECONDS = 0.01  # 10ms default

    def __init__(self):
        super().__init__()
        self._process: psutil.Process | None = None
        self._count: int = 0
        self._sum: float = 0.0
        self._min: float = math.inf
        self._max: float = -math.inf
        self._sampling_thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None

    def _runtime_metric_metadata(self) -> list[MetricMetadata]:
        return [
            MetricMetadata(
                key="avg_cpu_percent",
                display_name="Avg CPU",
                category="resource",
                unit="percent",
                higher_is_better=False,
                description="Average CPU load during execution",
                comparison_group="cpu_load",
                comparison_chart_type="horizontal_bar",
            ),
            MetricMetadata(
                key="max_cpu_percent",
                display_name="Max CPU",
                category="resource",
                unit="percent",
                higher_is_better=False,
                description="Maximum CPU load during execution",
                comparison_group="cpu_load",
                comparison_chart_type="horizontal_bar",
            ),
            MetricMetadata(
                key="min_cpu_percent",
                display_name="Min CPU",
                category="resource",
                unit="percent",
                description="Minimum CPU load during execution",
            ),
            MetricMetadata(
                key="cpu_sample_count",
                display_name="CPU Samples",
                category="resource",
                description="Number of CPU samples collected",
            ),
        ]

    def _runtime_metric_prepare(self) -> None:
        if self._params.get("interval"):
            self.SAMPLING_INTERVAL_SECONDS = float(self._params["interval"])

        self._process = psutil.Process()
        # Initial call to cpu_percent() to establish baseline
        # (first call always returns 0.0)
        self._process.cpu_percent()

    def _runtime_metric_start(self) -> None:
        self._count = 0
        self._sum = 0.0
        self._min = math.inf
        self._max = -math.inf
        self._stop_event = threading.Event()
        self._sampling_thread = threading.Thread(target=self._sample_cpu_percent, daemon=True)
        self._sampling_thread.start()

    def _runtime_metric_stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._sampling_thread is not None:
            self._sampling_thread.join(timeout=1.0)
            self._sampling_thread = None

    def _runtime_metric_calculate(self) -> dict[str, Any]:
        if self._count == 0:
            return {"avg_cpu_percent": None, "max_cpu_percent": None, "min_cpu_percent": None, "cpu_sample_count": None}

        return {
            "avg_cpu_percent": round(self._sum / self._count, 3),
            "max_cpu_percent": round(self._max, 5),
            "min_cpu_percent": round(self._min, 5),
            "cpu_sample_count": self._count,
        }

    def _sample_cpu_percent(self) -> None:
        """
        Background thread function that samples CPU percent at regular intervals.
        Runs until _stop_event is set.
        """
        while not self._stop_event.is_set():
            if self._process is not None:
                try:
                    cpu_percent = self._process.cpu_percent()
                    self._count += 1
                    self._sum += cpu_percent
                    if cpu_percent > self._max:
                        self._max = cpu_percent
                    if cpu_percent < self._min:
                        self._min = cpu_percent
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Process may have terminated or we lost access
                    break
            self._stop_event.wait(self.SAMPLING_INTERVAL_SECONDS)
