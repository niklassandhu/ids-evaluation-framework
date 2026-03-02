from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Any

from ids_eval.dto.evaluation_config import MlModelConfig, SignatureModelConfig
from ids_eval.dto.run_config import RunConfig
from ids_eval.interface.abstract_ids_connector import AbstractIDSConnector

Loaded_ML_IDS_Plugins = list[tuple[AbstractIDSConnector, MlModelConfig]]
Loaded_Sig_IDS_Plugins = list[tuple[AbstractIDSConnector, SignatureModelConfig]]


class IdsConnectorRegistry:
    def __init__(self, config: RunConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.ids_plugin_path = self.config.general.plugin_ids_path

    def load_ids_plugins(self) -> tuple[Loaded_ML_IDS_Plugins, Loaded_Sig_IDS_Plugins]:
        ids_plugins = self._import_ids_plugins()
        loaded_ml_ids_plugins: list[tuple[AbstractIDSConnector, MlModelConfig]] = []
        loaded_sig_ids_plugins: list[tuple[AbstractIDSConnector, SignatureModelConfig]] = []

        ids_plugin_configs = [self.config.evaluation.anomaly_models, self.config.evaluation.signature_models]

        for ids_plugin_config in ids_plugin_configs:
            if ids_plugin_config is None:
                continue
            for model_cfg in ids_plugin_config:
                if model_cfg.plugin is None:
                    continue
                try:
                    IDSModel = ids_plugins[model_cfg.plugin.lower().replace("_", "")]
                except KeyError:
                    self.logger.error(f"IDS plugin '{model_cfg.plugin}' not found. Skipping.")
                    continue
                ids_model: AbstractIDSConnector = IDSModel()
                if isinstance(model_cfg, SignatureModelConfig):
                    loaded_sig_ids_plugins.append((ids_model, model_cfg))
                    self.logger.info(f"  > Loaded Signature IDS plugin: {model_cfg.plugin}")
                else:
                    loaded_ml_ids_plugins.append((ids_model, model_cfg))
                    self.logger.info(f"  > Loaded ML IDS plugin: {model_cfg.plugin}")

        return loaded_ml_ids_plugins, loaded_sig_ids_plugins

    def _import_ids_plugins(self) -> dict[str, Any]:
        if not Path(self.ids_plugin_path).is_dir():
            raise FileNotFoundError(f"IDS plugin directory not found: {self.ids_plugin_path}")

        plugins = {}

        for file in Path(self.ids_plugin_path).glob("*.py"):
            module_name = file.stem

            try:
                spec = importlib.util.spec_from_file_location(module_name, file)
                if spec is None or spec.loader is None:
                    self.logger.warning(f"Could not load model plugin {module_name}")
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, AbstractIDSConnector) and obj is not AbstractIDSConnector:
                        plugin_name = f"{name.lower()}"
                        plugins[plugin_name] = obj
            except Exception as e:
                self.logger.warning(f"Could not load IDS plugin {module_name}: {e}")

        return plugins
