import importlib
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Any

from ids_eval.dto.metric_config import MetricMetadata, MetricPluginConfig
from ids_eval.dto.run_config import RunConfig
from ids_eval.interface.abstract_runtime_metric import AbstractRuntimeMetric


class RuntimeMetricRegistry:
    def __init__(self, config: RunConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.runtime_metric_path = self.config.general.plugin_runtime_metric_path
        self._loaded_plugins: list[tuple[AbstractRuntimeMetric, MetricPluginConfig]] = []
        self._all_metadata: dict[str, MetricMetadata] = {}

    def load_plugins(self) -> list[AbstractRuntimeMetric]:
        self._loaded_plugins = []
        self._all_metadata = {}

        # Get configured metrics or load all if not specified
        configured_metrics: list[MetricPluginConfig] | None = None
        if self.config.evaluation and self.config.evaluation.runtime_metrics:
            configured_metrics = self.config.evaluation.runtime_metrics

        # Import all available plugins
        available_plugins = self._import_runtime_metrics()

        if configured_metrics:
            # Load only configured plugins
            for metric_config in configured_metrics:
                plugin_name = metric_config.plugin.lower()
                # Normalize: remove underscores for matching (e.g., "time_metric" -> "timemetric")
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
                    self.logger.warning(f"Runtime metric plugin '{metric_config.plugin}' not found, skipping")
                    continue

                try:
                    plugin = plugin_class()
                    self._loaded_plugins.append((plugin, metric_config))
                    self.logger.info(f"  > Loaded runtime metric plugin: {plugin_class.__name__}")
                except Exception as e:
                    self.logger.error(f"Could not load runtime metric plugin: {e}")
                    continue
        else:
            # Load all available plugins with default config
            for plugin_name, plugin_class in available_plugins.items():
                try:
                    plugin = plugin_class()
                    # Create default config for this plugin
                    default_config = MetricPluginConfig(plugin=plugin_name)
                    self._loaded_plugins.append((plugin, default_config))
                    self.logger.info(f"  > Loaded runtime metric plugin: {plugin_class.__name__}")
                except Exception as e:
                    self.logger.error(f"Could not load runtime metric plugin: {e}")
                    continue

        # Collect metadata from all loaded plugins
        self._collect_metadata()

        # Return just the plugins for backward compatibility
        return [plugin for plugin, _ in self._loaded_plugins]

    def get_plugins_with_config(self) -> list[tuple[AbstractRuntimeMetric, MetricPluginConfig]]:
        return self._loaded_plugins

    def get_all_metadata(self) -> dict[str, MetricMetadata]:
        return self._all_metadata

    def _collect_metadata(self) -> None:
        self._all_metadata = {}
        for plugin, _ in self._loaded_plugins:
            try:
                metadata_list = plugin.metadata()
                for meta in metadata_list:
                    self._all_metadata[meta.key] = meta
            except Exception as e:
                self.logger.warning(f"Could not get metadata from plugin {plugin.__class__.__name__}: {e}")

    def _import_runtime_metrics(self) -> dict[str, Any]:
        if not Path(self.runtime_metric_path).is_dir():
            raise FileNotFoundError(f"Runtime metric directory not found: {self.runtime_metric_path}")

        runtime_metrics = {}

        for file in Path(self.runtime_metric_path).glob("*.py"):
            module_name = file.stem

            try:
                spec = importlib.util.spec_from_file_location(module_name, file)
                if spec is None or spec.loader is None:
                    self.logger.warning(f"Could not load runtime metric {module_name}")
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, AbstractRuntimeMetric) and obj is not AbstractRuntimeMetric:
                        plugin_name = f"{name.lower()}"
                        runtime_metrics[plugin_name] = obj
            except Exception as e:
                self.logger.warning(f"Could not load runtime metric {module_name}: {e}")

        return runtime_metrics
