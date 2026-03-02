import logging
from typing import Any, List, Tuple

import numpy as np
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


class TestingEvaluation:
    """Handles the evaluation of trained models on test data."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def test_model(
        self,
        dataset_index: int,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        loaded_plugin: tuple[AbstractIDSConnector, MlModelConfig],
        loaded_runtime_metrics: list[AbstractRuntimeMetric],
        fold: int | None = None,
        adversarial_samples: dict[str, pd.DataFrame] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not self.config.evaluation:
            raise ValueError("Evaluation config is not provided in RunConfig.")

        ids_model, model_cfg = loaded_plugin

        self.logger.info(f"Testing model: {model_cfg.plugin} for dataset index {dataset_index}, fold {fold}")

        clean_metrics = self._test_on_data(
            x_test=x_test,
            y_test=y_test,
            ids_model=ids_model,
            model_cfg=model_cfg,
            loaded_runtime_metrics=loaded_runtime_metrics,
            dataset_index=dataset_index,
            fold=fold,
            attack_name=None,  # Clean data
        )

        adv_metrics_list: list[dict[str, Any]] = []
        if adversarial_samples:
            for attack_name, x_adv in adversarial_samples.items():
                self.logger.info(f"Testing model on adversarial samples from {attack_name}")
                adv_metrics = self._test_on_data(
                    x_test=x_adv,
                    y_test=y_test,  # Same labels as clean data
                    ids_model=ids_model,
                    model_cfg=model_cfg,
                    loaded_runtime_metrics=[],  # Disable runtime metrics for adversarial tests
                    dataset_index=dataset_index,
                    fold=fold,
                    attack_name=attack_name,
                )
                adv_metrics_list.extend(adv_metrics)

        return clean_metrics, adv_metrics_list

    @staticmethod
    def _test_on_data(
        x_test: pd.DataFrame,
        y_test: pd.Series,
        ids_model: AbstractIDSConnector,
        model_cfg: MlModelConfig,
        loaded_runtime_metrics: list[AbstractRuntimeMetric],
        dataset_index: int,
        fold: int | None,
        attack_name: str | None,
    ) -> list[dict[str, Any]]:
        split_test_metrics = []
        test_data_size_bytes = x_test.memory_usage(deep=True).sum()
        test_data_size_gb = test_data_size_bytes / (1024**3)

        # Start runtime metrics
        for runtime_metric in loaded_runtime_metrics:
            runtime_metric.start()

        # Run detection
        y_pred, y_proba = ids_model.detect(x_test)

        # Stop runtime metrics
        for runtime_metric in loaded_runtime_metrics:
            runtime_metric.stop()

        runtime_metric_dict = {}
        for runtime_metric in loaded_runtime_metrics:
            calculated_runtime_metric = runtime_metric.calculate()
            for metric_name, metric_value in calculated_runtime_metric.items():
                if fold is not None:
                    metric_name = f"test_fold_{fold}_{metric_name}"
                else:
                    metric_name = f"test_{metric_name}"
                runtime_metric_dict[metric_name] = metric_value

        internal_metric_dict = {
            "test_dataset_index": dataset_index,
            "test_fold": fold,
            "test_ids_plugin": model_cfg.plugin,
            "test_y_true": y_test.to_numpy(),
            "test_y_pred": np.asarray(y_pred),
            "test_y_proba": y_proba,
            "test_data_size_gb": float(test_data_size_gb),
            "test_n_samples": len(x_test),
        }

        if attack_name is not None:
            internal_metric_dict["test_attack_name"] = attack_name
            internal_metric_dict["is_adversarial"] = True
        else:
            internal_metric_dict["is_adversarial"] = False

        split_test_metrics.append({**runtime_metric_dict, **internal_metric_dict})

        return split_test_metrics
