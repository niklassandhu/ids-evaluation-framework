import math
import threading
from typing import Any

import psutil

from ids_eval.dto.metric_config import MetricMetadata
from ids_eval.interface.abstract_runtime_metric import AbstractRuntimeMetric


class RamUsageMetric(AbstractRuntimeMetric):
    """
    Runtime metric that measures the RAM usage of the IDS process
    during training and testing using thread-based sampling.
    """

    SAMPLING_INTERVAL_SECONDS = 0.01

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
                key="avg_ram_mb",
                display_name="Avg RAM",
                category="resource",
                unit="MB",
                higher_is_better=False,
                description="Average RAM usage during execution",
                comparison_group="ram_usage",
                comparison_chart_type="horizontal_bar",
            ),
            MetricMetadata(
                key="max_ram_mb",
                display_name="Max RAM",
                category="resource",
                unit="MB",
                higher_is_better=False,
                description="Maximum RAM usage during execution",
                comparison_group="ram_usage",
                comparison_chart_type="horizontal_bar",
            ),
            MetricMetadata(
                key="min_ram_mb",
                display_name="Min RAM",
                category="resource",
                unit="MB",
                description="Minimum RAM usage during execution",
            ),
            MetricMetadata(
                key="ram_sample_count",
                display_name="RAM Samples",
                category="resource",
                description="Number of RAM samples collected",
            ),
        ]

    def _runtime_metric_prepare(self) -> None:
        if self._params.get("interval"):
            self.SAMPLING_INTERVAL_SECONDS = float(self._params["interval"])

        self._process = psutil.Process()

    def _runtime_metric_start(self) -> None:
        self._count = 0
        self._sum = 0.0
        self._min = math.inf
        self._max = -math.inf
        self._stop_event = threading.Event()
        self._sampling_thread = threading.Thread(target=self._sample_ram_usage, daemon=True)
        self._sampling_thread.start()

    def _runtime_metric_stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._sampling_thread is not None:
            self._sampling_thread.join(timeout=1.0)
            self._sampling_thread = None

    def _runtime_metric_calculate(self) -> dict[str, Any]:
        if self._count == 0:
            return {"avg_ram_mb": None, "max_ram_mb": None, "min_ram_mb": None, "ram_sample_count": None}

        return {
            "avg_ram_mb": round(self._sum / self._count, 5),
            "max_ram_mb": round(self._max, 5),
            "min_ram_mb": round(self._min, 5),
            "ram_sample_count": self._count,
        }

    def _sample_ram_usage(self) -> None:
        """
        Background thread function that samples RAM usage at regular intervals.
        Runs until _stop_event is set.
        """
        while not self._stop_event.is_set():
            if self._process is not None:
                try:
                    # RSS (Resident Set Size) = actual physical memory used
                    ram_bytes = self._process.memory_info().rss
                    ram_mb = ram_bytes / (1024 * 1024)
                    self._count += 1
                    self._sum += ram_mb
                    if ram_mb > self._max:
                        self._max = ram_mb
                    if ram_mb < self._min:
                        self._min = ram_mb
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Process may have terminated or we lost access
                    break
            self._stop_event.wait(self.SAMPLING_INTERVAL_SECONDS)
