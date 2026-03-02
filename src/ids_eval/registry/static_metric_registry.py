import importlib
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Any

from ids_eval.dto.metric_config import MetricMetadata, MetricPluginConfig
from ids_eval.dto.run_config import RunConfig
from ids_eval.interface.abstract_static_metric import AbstractStaticMetric


class StaticMetricRegistry:

    def __init__(self, config: RunConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.static_metric_path = self.config.general.plugin_static_metric_path
        self._loaded_plugins: list[tuple[AbstractStaticMetric, MetricPluginConfig]] = []
        self._all_metadata: dict[str, MetricMetadata] = {}

    def load_plugins(self) -> list[tuple[AbstractStaticMetric, MetricPluginConfig]]:
        self._loaded_plugins = []
        self._all_metadata = {}

        # Get configured metrics or load all if not specified
        configured_metrics: list[MetricPluginConfig] | None = None
        if self.config.evaluation and self.config.evaluation.static_metrics:
            configured_metrics = self.config.evaluation.static_metrics

        # Import all available plugins
        available_plugins = self._import_static_metrics()

        if configured_metrics:
            # Load only configured plugins
            for metric_config in configured_metrics:
                plugin_name = metric_config.plugin.lower()
                # Normalize: remove underscores for matching (e.g., "accuracy_metric" -> "accuracymetric")
                plugin_name_normalized = plugin_name.replace("_", "")
                # Try to find a matching plugin class
                plugin_class = None
                for name, cls in available_plugins.items():
                    # Match against: exact name, normalized name, or with "metric" suffix
                    if (
                        name == plugin_name
                        or name == plugin_name_normalized
                        or name == f"{plugin_name_normalized}metric"
                    ):
                        plugin_class = cls
                        break

                if plugin_class is None:
                    self.logger.warning(f"Static metric plugin '{metric_config.plugin}' not found, skipping")
                    continue

                try:
                    plugin = plugin_class()
                    self._loaded_plugins.append((plugin, metric_config))
                    self.logger.info(f"  > Loaded static metric plugin: {plugin_class.__name__}")
                except Exception as e:
                    self.logger.error(f"Could not load static metric plugin: {e}")
                    continue
        else:
            # Load all available plugins with default config
            for plugin_name, plugin_class in available_plugins.items():
                try:
                    plugin = plugin_class()
                    # Create default config for this plugin
                    default_config = MetricPluginConfig(plugin=plugin_name)
                    self._loaded_plugins.append((plugin, default_config))
                    self.logger.info(f"  > Loaded static metric plugin: {plugin_class.__name__}")
                except Exception as e:
                    self.logger.error(f"Could not load static metric plugin: {e}")
                    continue

        return self._loaded_plugins

    def get_plugins_with_config(self) -> list[tuple[AbstractStaticMetric, MetricPluginConfig]]:
        return self._loaded_plugins

    def _import_static_metrics(self) -> dict[str, Any]:
        if not Path(self.static_metric_path).is_dir():
            raise FileNotFoundError(f"Static metric directory not found: {self.static_metric_path}")

        static_metrics: dict[str, AbstractStaticMetric] = {}

        for file in Path(self.static_metric_path).glob("*.py"):
            module_name = file.stem

            try:
                spec = importlib.util.spec_from_file_location(module_name, file)
                if spec is None or spec.loader is None:
                    self.logger.warning(f"Could not load static metric {module_name}")
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, AbstractStaticMetric) and obj is not AbstractStaticMetric:
                        plugin_name = f"{name.lower()}"
                        static_metrics[plugin_name] = obj
            except Exception as e:
                self.logger.warning(f"Could not load static metric {module_name}: {e}")

        return static_metrics
