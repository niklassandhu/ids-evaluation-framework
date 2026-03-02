import gc
import logging
from typing import Any, List, Tuple

import pandas as pd
from pandas import DataFrame, Series

from ids_eval.adversarial_pipeline.adversarial_generator import AdversarialGenerator
from ids_eval.dto.evaluation_config import MlModelConfig
from ids_eval.dto.run_config import RunConfig
from ids_eval.enumeration.split_method import SplitMethod
from ids_eval.evaluation_pipeline.evaluation_checkpoint import EvaluationCheckpointStore
from ids_eval.evaluation_pipeline.testing_evaluation import TestingEvaluation
from ids_eval.evaluation_pipeline.training_evaluation import TrainingEvaluation
from ids_eval.interface.abstract_ids_connector import AbstractIDSConnector
from ids_eval.interface.abstract_runtime_metric import AbstractRuntimeMetric

# A single split of data into training and testing sets
Split = Tuple[DataFrame, DataFrame, Series, Series]

# A list of splits to represent folds in cross-validation
Folds = List[Split]


class MLEvaluationOrchestrator:
    def __init__(
        self, config: RunConfig, train_only: bool = False, force_train: bool = False, force_model: bool = False
    ):
        self.config = config
        self.train_only = train_only
        self.force_train = force_train
        self.force_model = force_model
        self.logger = logging.getLogger(__name__)
        self.adversarial_generator = AdversarialGenerator(config)

        if self.adversarial_generator.is_enabled():
            self.logger.info("Adversarial attack evaluation is enabled")

    def run_ml_ids_evaluation(
        self,
        dataset_splits: List[Split] | List[Folds],
        loaded_plugins: list[tuple[AbstractIDSConnector, MlModelConfig]],
        loaded_runtime_metrics: list[AbstractRuntimeMetric],
        checkpoint: EvaluationCheckpointStore,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Runs the full training and testing pipeline for ML models."""
        for ids_model, model_cfg in loaded_plugins:
            ids_model.deploy(model_cfg.params)
            self.logger.info(f"Deployed model {model_cfg.plugin} for evaluation.")

        for runtime_metric in loaded_runtime_metrics:
            runtime_metric.prepare()

        trainer = TrainingEvaluation(self.config, force_train=self.force_train, force_model=self.force_model)
        tester = TestingEvaluation(self.config)

        if self.force_train:
            self.logger.info("Force-train mode enabled. Ignoring saved models and retraining.")
        if self.force_model:
            self.logger.info("Force-model mode enabled. Loading saved models without config validation.")
        if self.train_only:
            self.logger.info("Train-only mode enabled. Skipping testing phase.")

        is_kfold: bool = len(dataset_splits) > 0 and isinstance(dataset_splits[0], list)
        is_cross_dataset: bool = (
            self.config.data_manager.split.method == SplitMethod.CROSS_DATASET
            or self.config.data_manager.split.method == SplitMethod.CROSS_DATASET_BENIGN
        )

        if is_cross_dataset:
            return self._run_cross_dataset_evaluation(
                dataset_splits, loaded_plugins, loaded_runtime_metrics, trainer, tester, checkpoint
            )

        if is_kfold:
            return self._run_kfold_evaluation(
                dataset_splits, loaded_plugins, loaded_runtime_metrics, trainer, tester, checkpoint
            )

        return self._run_intra_dataset_evaluation(
            dataset_splits, loaded_plugins, loaded_runtime_metrics, trainer, tester, checkpoint
        )

    def _run_intra_dataset_evaluation(
        self,
        dataset_splits: List[Split],
        loaded_plugins: list[tuple[AbstractIDSConnector, MlModelConfig]],
        loaded_runtime_metrics: list[AbstractRuntimeMetric],
        trainer: TrainingEvaluation,
        tester: TestingEvaluation,
        checkpoint: EvaluationCheckpointStore,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        train_metrics: list[dict[str, Any]] = []
        test_metrics: list[dict[str, Any]] = []

        evaluation_scope = "intra_dataset"

        self.logger.info("Starting intra-dataset evaluation...")

        for ds_idx, (X_train, X_test, y_train, y_test) in enumerate(dataset_splits):
            for loaded_plugin in loaded_plugins:
                plugin_name = loaded_plugin[1].plugin

                train_run_id = self._build_run_id(plugin_name, evaluation_scope, (ds_idx,), None)
                test_run_id = self._build_run_id(plugin_name, evaluation_scope, (ds_idx,), None, (ds_idx,))

                train_checkpointed = checkpoint.is_train_completed(train_run_id)
                test_checkpointed = checkpoint.is_test_completed(test_run_id)

                # --- TRAINING ---
                if train_checkpointed:
                    self.logger.info(f"Resuming: training '{train_run_id}' already completed, restoring metrics.")
                    train_metrics.extend(checkpoint.get_train_metrics(train_run_id))
                else:
                    new_train = self._capture_training_metrics(
                        trainer,
                        ds_idx,
                        X_train,
                        y_train,
                        loaded_plugin,
                        loaded_runtime_metrics,
                        evaluation_scope,
                        None,
                        [ds_idx],
                    )
                    train_metrics.extend(new_train)
                    checkpoint.save_train_step(train_run_id, new_train)

                # --- TESTING ---
                if self.train_only:
                    pass  # skip testing
                elif test_checkpointed:
                    self.logger.info(f"Resuming: testing '{test_run_id}' already completed, restoring metrics.")
                    test_metrics.extend(checkpoint.get_test_metrics(test_run_id))
                else:
                    if train_checkpointed:
                        trainer.train_model(ds_idx, X_train, y_train, loaded_plugin, [], None)

                    new_test = self._capture_testing_metrics(
                        tester,
                        ds_idx,
                        X_test,
                        y_test,
                        loaded_plugin,
                        loaded_runtime_metrics,
                        evaluation_scope,
                        None,
                        train_run_id,
                        [ds_idx],
                        [ds_idx],
                        x_train=X_train,
                        y_train=y_train,
                    )
                    test_metrics.extend(new_test)
                    checkpoint.save_test_step(test_run_id, new_test)

                gc.collect()

        return test_metrics, train_metrics

    def _run_kfold_evaluation(
        self,
        dataset_splits: List[Split] | List[Folds],
        loaded_plugins: list[tuple[AbstractIDSConnector, MlModelConfig]],
        loaded_runtime_metrics: list[AbstractRuntimeMetric],
        trainer: TrainingEvaluation,
        tester: TestingEvaluation,
        checkpoint: EvaluationCheckpointStore,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        train_metrics: list[dict[str, Any]] = []
        test_metrics: list[dict[str, Any]] = []

        evaluation_scope = "kfold"

        self.logger.info("Starting k-fold evaluation...")

        for ds_idx, folds in enumerate(dataset_splits):
            if not isinstance(folds, list):
                continue
            for fold_idx, (X_train, X_test, y_train, y_test) in enumerate(folds):
                for loaded_plugin in loaded_plugins:
                    plugin_name = loaded_plugin[1].plugin

                    train_run_id = self._build_run_id(plugin_name, evaluation_scope, (ds_idx,), fold_idx)
                    test_run_id = self._build_run_id(plugin_name, evaluation_scope, (ds_idx,), fold_idx, (ds_idx,))

                    train_checkpointed = checkpoint.is_train_completed(train_run_id)
                    test_checkpointed = checkpoint.is_test_completed(test_run_id)

                    # --- TRAINING ---
                    if train_checkpointed:
                        self.logger.info(f"Resuming: training '{train_run_id}' already completed, restoring metrics.")
                        train_metrics.extend(checkpoint.get_train_metrics(train_run_id))
                    else:
                        new_train = self._capture_training_metrics(
                            trainer,
                            ds_idx,
                            X_train,
                            y_train,
                            loaded_plugin,
                            loaded_runtime_metrics,
                            evaluation_scope,
                            fold_idx,
                            [ds_idx],
                        )
                        train_metrics.extend(new_train)
                        checkpoint.save_train_step(train_run_id, new_train)

                    # --- TESTING ---
                    if self.train_only:
                        pass
                    elif test_checkpointed:
                        self.logger.info(f"Resuming: testing '{test_run_id}' already completed, restoring metrics.")
                        test_metrics.extend(checkpoint.get_test_metrics(test_run_id))
                    else:
                        if train_checkpointed:
                            trainer.train_model(ds_idx, X_train, y_train, loaded_plugin, [], fold_idx)

                        new_test = self._capture_testing_metrics(
                            tester,
                            ds_idx,
                            X_test,
                            y_test,
                            loaded_plugin,
                            loaded_runtime_metrics,
                            evaluation_scope,
                            fold_idx,
                            train_run_id,
                            [ds_idx],
                            [ds_idx],
                            x_train=X_train,
                            y_train=y_train,
                        )
                        test_metrics.extend(new_test)
                        checkpoint.save_test_step(test_run_id, new_test)

                    gc.collect()

        return test_metrics, train_metrics

    def _run_cross_dataset_evaluation(
        self,
        dataset_splits: List[Split] | List[Folds],
        loaded_plugins: list[tuple[AbstractIDSConnector, MlModelConfig]],
        loaded_runtime_metrics: list[AbstractRuntimeMetric],
        trainer: TrainingEvaluation,
        tester: TestingEvaluation,
        checkpoint: EvaluationCheckpointStore,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        base_splits: list[Split] = []
        for splits in dataset_splits:
            if isinstance(splits, list):
                raise ValueError("Cross dataset evaluation does not support folded splits.")
            base_splits.append(splits)

        scope = "cross_dataset"

        self.logger.info("Starting cross-dataset evaluation...")

        train_metrics: list[dict[str, Any]] = []
        test_metrics: list[dict[str, Any]] = []

        for train_idx, (X_train, _, y_train, _) in enumerate(base_splits):
            for loaded_plugin in loaded_plugins:
                plugin_name = loaded_plugin[1].plugin

                train_run_id = self._build_run_id(plugin_name, scope, (train_idx,), None)

                train_checkpointed = checkpoint.is_train_completed(train_run_id)

                # --- TRAINING ---
                if train_checkpointed:
                    self.logger.info(f"Resuming: training '{train_run_id}' already completed, restoring metrics.")
                    train_metrics.extend(checkpoint.get_train_metrics(train_run_id))
                else:
                    new_train = self._capture_training_metrics(
                        trainer,
                        train_idx,
                        X_train,
                        y_train,
                        loaded_plugin,
                        loaded_runtime_metrics,
                        scope,
                        None,
                        [train_idx],
                    )
                    train_metrics.extend(new_train)
                    checkpoint.save_train_step(train_run_id, new_train)

                # --- TESTING (for each test dataset) ---
                if not self.train_only:
                    model_loaded_for_testing = False

                    for test_idx, (_, X_test, _, y_test) in enumerate(base_splits):
                        test_run_id = self._build_run_id(plugin_name, scope, (train_idx,), None, (test_idx,))
                        test_checkpointed = checkpoint.is_test_completed(test_run_id)

                        if test_checkpointed:
                            self.logger.info(f"Resuming: testing '{test_run_id}' already completed, restoring metrics.")
                            test_metrics.extend(checkpoint.get_test_metrics(test_run_id))
                        else:
                            if train_checkpointed and not model_loaded_for_testing:
                                trainer.train_model(train_idx, X_train, y_train, loaded_plugin, [], None)
                                model_loaded_for_testing = True

                            new_test = self._capture_testing_metrics(
                                tester,
                                test_idx,
                                X_test,
                                y_test,
                                loaded_plugin,
                                loaded_runtime_metrics,
                                scope,
                                None,
                                train_run_id,
                                [train_idx],
                                [test_idx],
                                x_train=X_train,
                                y_train=y_train,
                            )
                            test_metrics.extend(new_test)
                            checkpoint.save_test_step(test_run_id, new_test)

                        gc.collect()

        return test_metrics, train_metrics

    def _capture_training_metrics(
        self,
        trainer: TrainingEvaluation,
        dataset_index: int,
        x_train,
        y_train,
        loaded_plugin: tuple[AbstractIDSConnector, MlModelConfig],
        loaded_runtime_metrics: list[AbstractRuntimeMetric],
        scope: str,
        fold: int | None,
        train_dataset_indices: list[int | str],
    ) -> list[dict[str, Any]]:
        metrics = trainer.train_model(dataset_index, x_train, y_train, loaded_plugin, loaded_runtime_metrics, fold)
        plugin_name = loaded_plugin[1].plugin
        run_id = self._build_run_id(plugin_name, scope, tuple(train_dataset_indices), fold)

        for metric in metrics:
            metric.update(
                {
                    "evaluation_scope": scope,
                    "train_dataset_indices": list(train_dataset_indices),
                    "train_run_id": run_id,
                    "run_id": run_id,
                }
            )

        return metrics

    def _capture_testing_metrics(
        self,
        tester: TestingEvaluation,
        dataset_index: int,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        loaded_plugin: tuple[AbstractIDSConnector, MlModelConfig],
        loaded_runtime_metrics: list[AbstractRuntimeMetric],
        scope: str,
        fold: int | None,
        train_run_id: str,
        train_dataset_indices: list[int | str],
        test_dataset_indices: list[int | str],
        x_train: pd.DataFrame | None = None,
        y_train: pd.Series | None = None,
    ) -> list[dict[str, Any]]:
        ids_model, model_cfg = loaded_plugin

        adversarial_samples: dict[str, pd.DataFrame] | None = None
        if self.adversarial_generator.is_enabled():
            self.logger.info(f"Generating adversarial samples for {model_cfg.plugin}...")
            adversarial_samples = self.adversarial_generator.generate_adversarial_samples(
                x_test=x_test, y_test=y_test, ids_model=ids_model, x_train=x_train, y_train=y_train
            )

        # Test model on clean and adversarial data
        clean_metrics, adv_metrics = tester.test_model(
            dataset_index,
            x_test,
            y_test,
            loaded_plugin,
            loaded_runtime_metrics,
            fold,
            adversarial_samples=adversarial_samples,
        )

        del adversarial_samples

        plugin_name = model_cfg.plugin
        run_id = self._build_run_id(plugin_name, scope, tuple(train_dataset_indices), fold, tuple(test_dataset_indices))

        all_test_metrics: list[dict[str, Any]] = []

        for metric in clean_metrics:
            metric.update(
                {
                    "evaluation_scope": scope,
                    "train_dataset_indices": list(train_dataset_indices),
                    "test_dataset_indices": list(test_dataset_indices),
                    "train_run_id": train_run_id,
                    "run_id": run_id,
                }
            )
        all_test_metrics.extend(clean_metrics)

        for i, adv_metric in enumerate(adv_metrics):
            attack_name = adv_metric.get("test_attack_name", "unknown")
            adv_run_id = f"{run_id}|attack{i}_{attack_name.replace(' ', '_').lower()}"
            adv_metric.update(
                {
                    "evaluation_scope": scope,
                    "train_dataset_indices": list(train_dataset_indices),
                    "test_dataset_indices": list(test_dataset_indices),
                    "train_run_id": train_run_id,
                    "run_id": adv_run_id,
                }
            )
            all_test_metrics.append(adv_metric)

        return all_test_metrics

    @staticmethod
    def _build_run_id(
        plugin_name: str,
        scope: str,
        train_indices: tuple[int | str, ...],
        fold: int | None,
        test_indices: tuple[int | str, ...] | None = None,
    ) -> str:
        train_fragment = "-".join(str(idx) for idx in train_indices) or "none"
        fold_fragment = "" if fold is None else f"_fold {str(fold)})"
        base = f"{scope}_{plugin_name}_train-{train_fragment}{fold_fragment}"
        if test_indices is not None:
            test_fragment = "-".join(str(idx) for idx in test_indices) or "none"
            return f"{base}_test-{test_fragment}"
        return base
