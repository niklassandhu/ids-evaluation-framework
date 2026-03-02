from typing import Any

from ids_eval.dto.metric_config import MetricMetadata
from ids_eval.interface.abstract_static_metric import AbstractStaticMetric


class RetrainCostMetric(AbstractStaticMetric):
    """
    Model Retrain Cost metric.

    Measures the time and resources required to train a model and calculates
    a cost score that can be used to estimate whether retraining is worthwhile.

    Formula: C_retrain = ω₁ · T_train + ω₂ · (CPU_rel · MEM_peak)

    Where:
    - ω₁, ω₂ = weighting factors
    - T_train = training time in seconds
    - CPU_rel = average CPU utilization during training (%)
    - MEM_peak = peak RAM usage during training (MB)
    """

    OMEGA_1 = 1.0  # Weight for time component
    OMEGA_2 = 1e-6  # Weight for resource component

    def __init__(self):
        super().__init__()

    def _static_metric_metadata(self, is_multiclass: bool) -> list[MetricMetadata]:
        return [
            MetricMetadata(
                key="train_model_retrain_cost",
                display_name="Retrain Cost",
                category="resource",
                higher_is_better=False,
                description="Combined time + resource cost for model retraining",
                comparison_group="retrain_cost",
                comparison_chart_type="horizontal_bar",
            ),
            MetricMetadata(
                key="train_model_retrain_time_cost",
                display_name="Retrain Time Cost",
                category="resource",
                unit="weighted seconds",
                higher_is_better=False,
                description="Time component of retrain cost",
            ),
            MetricMetadata(
                key="train_model_retrain_resource_cost",
                display_name="Retrain Resource Cost",
                category="resource",
                higher_is_better=False,
                description="Resource component of retrain cost",
            ),
        ]

    def _static_metric_prepare(self) -> None:
        pass

    def _static_metric_calculate(self, metrics: dict[str, Any], is_multiclass: bool) -> dict[str, Any]:
        t_train = metrics.get("train_runtime_s", 0.0)
        cpu_rel = metrics.get("train_avg_cpu_percent", 0.0)
        mem_peak = metrics.get("train_max_ram_mb", 0.0)

        t_train = t_train if t_train is not None else 0.0
        cpu_rel = cpu_rel if cpu_rel is not None else 0.0
        mem_peak = mem_peak if mem_peak is not None else 0.0

        time_cost = self.OMEGA_1 * t_train
        resource_cost = self.OMEGA_2 * (cpu_rel * mem_peak)
        c_retrain = time_cost + resource_cost

        return {
            "train_model_retrain_cost": float(c_retrain),
            "train_model_retrain_time_cost": float(time_cost),
            "train_model_retrain_resource_cost": float(resource_cost),
        }

    def _static_metric_visualize(self, metrics: dict[str, Any], visual_name_prefix: str, is_multiclass: bool) -> None:
        pass
