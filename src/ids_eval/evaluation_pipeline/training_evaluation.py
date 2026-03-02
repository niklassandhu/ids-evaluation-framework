import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from pandas import DataFrame, Series

from ids_eval.dto.evaluation_config import MlModelConfig
from ids_eval.dto.run_config import RunConfig
from ids_eval.interface.abstract_ids_connector import AbstractIDSConnector
from ids_eval.interface.abstract_runtime_metric import AbstractRuntimeMetric

# A single split of data into training and testing sets
Split = Tuple[DataFrame, DataFrame, Series, Series]

# A list of splits to represent folds in cross-validation
Folds = List[Split]


class TrainingEvaluation:
    """Handles the training and saving of machine learning models."""

    def __init__(self, config: RunConfig, force_train: bool = False, force_model: bool = False):
        self.config = config
        self.force_train = force_train
        self.force_model = force_model
        self.logger = logging.getLogger(__name__)

    def train_model(
        self,
        dataset_index: int,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        loaded_plugin: tuple[AbstractIDSConnector, MlModelConfig],
        loaded_runtime_metrics: list[AbstractRuntimeMetric],
        fold: int | None = None,
    ) -> List[Dict[str, Any]]:
        if not self.config.evaluation:
            raise ValueError("Evaluation config is not provided in RunConfig.")

        split_train_metrics = []
        ids_model, model_cfg = loaded_plugin
        model_loaded = False
        if model_cfg.save_model and not self.force_train:
            model_loaded = self._try_load_model(ids_model, model_cfg, dataset_index, fold)

        model_size_mb: float | None = None

        if model_loaded:
            self.logger.info(
                f"Loaded pre-trained model: {model_cfg.plugin} for dataset index {dataset_index}, fold {fold}"
            )
            runtime_metric_dict = {}
            model_path = self._get_model_path(model_cfg, dataset_index, fold)
            model_size_mb = self._get_model_size_mb(model_path)
        else:
            self.logger.info(f"Training model: {model_cfg.plugin} for dataset index {dataset_index}, fold {fold}")

            for runtime_metric in loaded_runtime_metrics:
                runtime_metric.start()

            ids_model.prepare(x_train, y_train)

            for runtime_metric in loaded_runtime_metrics:
                runtime_metric.stop()

            runtime_metric_dict = {}

            for runtime_metric in loaded_runtime_metrics:
                calculated_runtime_metric = runtime_metric.calculate()
                for metric_name, metric_value in calculated_runtime_metric.items():
                    if fold is not None:
                        metric_name = f"train_fold_{fold}_{metric_name}"
                    else:
                        metric_name = f"train_{metric_name}"
                    runtime_metric_dict[metric_name] = metric_value

            if model_cfg.save_model:
                self._save_model(ids_model, model_cfg, dataset_index, fold)
                model_path = self._get_model_path(model_cfg, dataset_index, fold)
                model_size_mb = self._get_model_size_mb(model_path)

        internal_metric_dict = {
            "train_dataset_index": dataset_index,
            "train_fold": fold,
            "train_ids_plugin": model_cfg.plugin,
            "model_loaded_from_cache": model_loaded,
            "model_storage_size_mb": model_size_mb,
        }

        split_train_metrics.append({**runtime_metric_dict, **internal_metric_dict})

        return split_train_metrics

    def _get_model_path(self, model_cfg: MlModelConfig, dataset_index: int, fold: int | None) -> Path:
        base_path = Path(self.config.general.model_storage_path)
        config_hash = self._get_config_file_hash()
        plugin_dir = model_cfg.model_path if model_cfg.model_path else model_cfg.plugin

        if fold is not None:
            model_dir = base_path / config_hash / plugin_dir / f"ds{dataset_index}_fold{fold}"
        else:
            model_dir = base_path / config_hash / plugin_dir / f"ds{dataset_index}"

        return model_dir

    def _get_config_file_hash(self) -> str:
        config_hash = self.config.get_config_file_hash()
        if config_hash:
            return config_hash
        raise ValueError("Config file hash is not set in RunConfig.")

    def _try_load_model(
        self, ids_model: AbstractIDSConnector, model_cfg: MlModelConfig, dataset_index: int, fold: int | None
    ) -> bool:
        model_path = self._get_model_path(model_cfg, dataset_index, fold)
        hash_file = model_path / "config.hash"

        if not model_path.exists():
            self.logger.debug(f"No saved model found at {model_path}")
            return False

        # Skip hash validation if force_model is enabled
        if self.force_model:
            self.logger.info(f"Force-model mode: Skipping config hash validation for {model_cfg.plugin}")
        else:
            if not hash_file.exists():
                self.logger.debug(f"No config hash file found at {hash_file}")
                return False

            current_hash = self._get_config_file_hash()
            try:
                saved_hash = hash_file.read_text().strip()
            except Exception as e:
                self.logger.warning(f"Failed to read config hash from {hash_file}: {e}")
                return False

            if current_hash != saved_hash:
                self.logger.info(f"Config hash mismatch for {model_cfg.plugin}. Retraining required.")
                return False

        success = ids_model.load(model_path)
        return success

    def _save_model(
        self, ids_model: AbstractIDSConnector, model_cfg: MlModelConfig, dataset_index: int, fold: int | None
    ) -> None:
        model_path = self._get_model_path(model_cfg, dataset_index, fold)
        model_path.mkdir(parents=True, exist_ok=True)

        # Save the model
        ids_model.save(model_path)

        # Save the config file hash for validation on load
        current_hash = self._get_config_file_hash()
        hash_file = model_path / "config.hash"
        hash_file.write_text(current_hash)

        self.logger.info(f"Model saved to {model_path}")

    def _get_model_size_mb(self, model_path: Path) -> float:
        if not model_path.exists():
            return 0.0
        total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file() and f.name != "config.hash")
        return total_size / (1024 * 1024)
