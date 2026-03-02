import tempfile

from codecarbon import EmissionsTracker

from ids_eval.dto.metric_config import MetricMetadata
from ids_eval.interface.abstract_runtime_metric import AbstractRuntimeMetric


class EmissionMetric(AbstractRuntimeMetric):
    def __init__(self):
        super().__init__()
        self.__tracker = None

    def _runtime_metric_metadata(self) -> list[MetricMetadata]:
        return [
            MetricMetadata(
                key="emissions",
                display_name="CO2 Emissions",
                category="resource",
                unit="kg CO2",
                higher_is_better=False,
                description="Estimated carbon emissions during execution",
                comparison_group="emissions",
                comparison_chart_type="horizontal_bar",
            ),
            MetricMetadata(
                key="emissions_rate",
                display_name="Emissions Rate",
                category="resource",
                unit="kg CO2/s",
                higher_is_better=False,
                description="Rate of carbon emissions",
            ),
            MetricMetadata(
                key="energy_consumed",
                display_name="Energy Consumed",
                category="resource",
                unit="kWh",
                higher_is_better=False,
                description="Total energy consumed during execution",
                comparison_group="energy",
                comparison_chart_type="horizontal_bar",
            ),
            MetricMetadata(
                key="cpu_power",
                display_name="CPU Power",
                category="resource",
                unit="W",
                description="CPU power consumption",
            ),
            MetricMetadata(
                key="gpu_power",
                display_name="GPU Power",
                category="resource",
                unit="W",
                description="GPU power consumption",
            ),
            MetricMetadata(
                key="ram_power",
                display_name="RAM Power",
                category="resource",
                unit="W",
                description="RAM power consumption",
            ),
        ]

    def _runtime_metric_prepare(self):
        try:
            self.__tracker = EmissionsTracker(
                save_to_file=False,
                project_name="ids_eval",
                log_level="error",
                allow_multiple_runs=False,
                output_dir=tempfile.mkdtemp(),
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize EmissionsTracker: {e}")
            self.__tracker = None

    def _runtime_metric_start(self):
        self.__tracker.start()

    def _runtime_metric_stop(self):
        self.__tracker.stop()

    def _runtime_metric_calculate(self) -> dict[str, float]:
        if self.__tracker is None or self.__tracker.final_emissions_data is None:
            # Tracker failed to collect data (e.g., another instance was running)
            return {
                "emissions": 0.0,
                "emissions_rate": 0.0,
                "cpu_power": 0.0,
                "gpu_power": 0.0,
                "ram_power": 0.0,
                "cpu_energy": 0.0,
                "gpu_energy": 0.0,
                "ram_energy": 0.0,
                "energy_consumed": 0.0,
                "os": "unknown",
                "cpu_model": "unknown",
                "gpu_model": "unknown",
                "ram_total_size": 0.0,
            }

        emissions_data = self.__tracker.final_emissions_data.values

        return {
            "emissions": emissions_data["emissions"],
            "emissions_rate": emissions_data["emissions_rate"],
            "cpu_power": round(float(emissions_data["cpu_power"]), 5),
            "gpu_power": round(float(emissions_data["gpu_power"]), 5),
            "ram_power": emissions_data["ram_power"],
            "cpu_energy": emissions_data["cpu_energy"],
            "gpu_energy": emissions_data["gpu_energy"],
            "ram_energy": emissions_data["ram_energy"],
            "energy_consumed": emissions_data["energy_consumed"],
            "os": emissions_data["os"],
            "cpu_model": emissions_data["cpu_model"],
            "gpu_model": emissions_data["gpu_model"],
            "ram_total_size": emissions_data["ram_total_size"],
        }
