from typing import Any

from ids_eval.dto.metric_config import MetricMetadata
from ids_eval.interface.abstract_static_metric import AbstractStaticMetric


class TimePerSampleMetric(AbstractStaticMetric):
    """
    Time Per Sample metric.

    Derives the average detection time per individual sample (packet / CSV row)
    from the total test runtime and the number of test samples.

    Formulas:
        time_per_sample_ms = (test_runtime_s / n_samples) * 1000
        throughput_samples_per_s = n_samples / test_runtime_s
    """

    def __init__(self):
        super().__init__()

    def _static_metric_metadata(self, is_multiclass: bool) -> list[MetricMetadata]:
        return [
            MetricMetadata(
                key="test_time_per_sample_ms",
                display_name="Time per Data Sample",
                category="runtime",
                unit="ms/sample",
                higher_is_better=False,
                description="Average detection time per sample (packet/row) in milliseconds",
                comparison_group="time_per_sample",
                comparison_chart_type="horizontal_bar",
            ),
            MetricMetadata(
                key="test_throughput_samples_per_s",
                display_name="Throughput",
                category="runtime",
                unit="samples/s",
                higher_is_better=True,
                description="Detection throughput in samples (packets/rows) per second",
                comparison_group="throughput",
                comparison_chart_type="horizontal_bar",
            ),
        ]

    def _static_metric_prepare(self) -> None:
        pass

    def _static_metric_calculate(self, metrics: dict[str, Any], is_multiclass: bool) -> dict[str, Any]:
        runtime_s = self.__resolve_test_runtime(metrics)
        n_samples = self.__resolve_n_samples(metrics)

        if runtime_s is not None and n_samples is not None and n_samples > 0 and runtime_s > 0:
            time_per_sample_ms = (runtime_s / n_samples) * 1000.0
            throughput = n_samples / runtime_s
        else:
            time_per_sample_ms = None
            throughput = None

        return {
            "test_time_per_sample_ms": round(time_per_sample_ms, 6) if time_per_sample_ms is not None else None,
            "test_throughput_samples_per_s": round(throughput, 5) if throughput is not None else None,
        }

    @staticmethod
    def __resolve_test_runtime(metrics: dict[str, Any]) -> float | None:
        plain_runtime = "test_runtime_s"
        if plain_runtime in metrics and metrics[plain_runtime] is not None:
            return float(metrics[plain_runtime])

        # Fallback: k-fold
        for k, v in metrics.items():
            if k.startswith("test_fold_") and k.endswith("_runtime_s") and v is not None:
                return float(v)

        return None

    @staticmethod
    def __resolve_n_samples(metrics: dict[str, Any]) -> int | None:
        if "test_n_samples" in metrics and metrics["test_n_samples"] is not None:
            return int(metrics["test_n_samples"])

        # Fallback: derive from y_true length
        y_true = metrics.get("test_y_true")
        if y_true is not None:
            return len(y_true)

        return None

    def _static_metric_visualize(self, metrics: dict[str, Any], visual_name_prefix: str, is_multiclass: bool) -> None:
        pass
