import gc
import logging
import pickle
from pathlib import Path
from typing import Any, List, Tuple, Union

from pandas import DataFrame, Series

from ids_eval.dto.evaluation_config import SignatureModelConfig
from ids_eval.dto.run_config import RunConfig
from ids_eval.evaluation_pipeline.evaluation_checkpoint import EvaluationCheckpointStore
from ids_eval.evaluation_pipeline.ml_evaluation_orchestrator import (
    MLEvaluationOrchestrator,
)
from ids_eval.interface.abstract_ids_connector import AbstractIDSConnector
from ids_eval.metrics_pipeline.metrics_analyzer import MetricsAnalyzer
from ids_eval.metrics_pipeline.metrics_calculator import MetricsCalculator
from ids_eval.metrics_pipeline.metrics_formatter import MetricsFormatter
from ids_eval.registry.ids_connector_registry import IdsConnectorRegistry
from ids_eval.registry.runtime_metric_registry import RuntimeMetricRegistry
from ids_eval.reporting_pipeline.report_writer import ReportWriter
from ids_eval.reporting_pipeline.results_visualizer import ResultsVisualizer
from ids_eval.run_config_pipeline.config_manager import ConfigManager

# A single split of data into training and testing sets
Split = Tuple[DataFrame, DataFrame, Series, Series]

# A list of splits to represent folds in cross-validation
Folds = List[Split]


class EvaluationManager:
    """Orchestrates the entire model evaluation pipeline."""

    def __init__(
        self,
        config: RunConfig,
        train_only: bool = False,
        force_train: bool = False,
        force_model: bool = False,
        clear_checkpoints: bool = False,
    ):
        self.config = config
        self.train_only = train_only
        self.force_train = force_train
        self.force_model = force_model
        self.clear_checkpoints = clear_checkpoints
        self.logger = logging.getLogger(__name__)

    def run(self) -> None:
        """Executes the full model evaluation pipeline with checkpoint-based resume."""
        if not self.config.evaluation:
            self.logger.error("Evaluation config is not provided in RunConfig.")
            raise ValueError("Evaluation config is not provided in RunConfig.")

        checkpoint = EvaluationCheckpointStore.load_or_create(self._get_checkpoint_path())
        if self.force_train or self.clear_checkpoints:
            checkpoint.clear()

        stage = checkpoint.stage

        if stage == EvaluationCheckpointStore.STAGE_COMPLETE:
            self.logger.info(
                "Previous evaluation run already completed for this config. "
                "Use --force-train or --clear-checkpoints to re-run from scratch."
            )
            return

        # --- Load Processed Data, IDS Plugins, Metrics ---
        dataset_splits = self._load_processed_data()
        self.logger.info(f"Loaded {len(dataset_splits)} dataset splits.")

        ids_connector_registry = IdsConnectorRegistry(self.config)
        self.logger.info("Loading IDS plugins...")
        loaded_ml_ids_plugins, loaded_sig_ids_plugins = ids_connector_registry.load_ids_plugins()
        self.logger.info(f"Loaded {len(loaded_ml_ids_plugins)} ML IDS plugins.")

        runtime_metric_registry = RuntimeMetricRegistry(self.config)
        self.logger.info("Loading runtime metrics...")
        loaded_runtime_metrics = runtime_metric_registry.load_plugins()
        self.logger.info(f"Loaded {len(loaded_runtime_metrics)} runtime metrics.")

        # --- Run Evaluations ---
        train_metrics: list[dict[str, Any]] = []
        test_metrics: list[dict[str, Any]] = []

        if stage in (EvaluationCheckpointStore.STAGE_EVALUATION_IN_PROGRESS,):
            if self.config.evaluation.anomaly_models:
                self.logger.info("Starting ML IDS evaluation pipeline...")
                ml_evaluation_orchestrator = MLEvaluationOrchestrator(
                    self.config,
                    train_only=self.train_only,
                    force_train=self.force_train,
                    force_model=self.force_model,
                )
                test_metrics, train_metrics = ml_evaluation_orchestrator.run_ml_ids_evaluation(
                    dataset_splits, loaded_ml_ids_plugins, loaded_runtime_metrics, checkpoint
                )
                self.logger.info("ML IDS evaluation pipeline finished.")
                gc.collect()

            if self.config.evaluation.signature_models:
                self.logger.info("Starting Signature IDS evaluation pipeline...")
                self._run_sig_ids_evaluation(dataset_splits, loaded_sig_ids_plugins)
                self.logger.info("Signature IDS evaluation pipeline finished.")

            checkpoint.set_stage(EvaluationCheckpointStore.STAGE_EVALUATION_COMPLETE)
        else:
            # Restore metrics from checkpoint
            self.logger.info("Resuming: evaluation steps already completed, restoring metrics from checkpoint.")
            train_metrics = checkpoint.get_all_train_metrics()
            test_metrics = checkpoint.get_all_test_metrics()

        # --- Calculate Static Metrics ---
        if stage != EvaluationCheckpointStore.STAGE_METRICS_COMPLETE:
            self.logger.info("Calculating performance metrics...")
            metrics_calculator = MetricsCalculator(self.config)
            raw_metrics, static_metadata = metrics_calculator.calculate_metrics(train_metrics, test_metrics)
            self.logger.info("Metrics calculation complete.")
            checkpoint.save_calculated_metrics(
                {
                    "raw_metrics": raw_metrics,
                    "static_metadata": static_metadata,
                }
            )
            checkpoint.set_stage(EvaluationCheckpointStore.STAGE_METRICS_COMPLETE)
            gc.collect()
        else:
            self.logger.info("Resuming: static metrics already calculated, restoring from checkpoint.")
            saved = checkpoint.get_calculated_metrics()
            if saved is None:
                raise RuntimeError("Checkpoint claims metrics_complete but no calculated metrics found.")
            raw_metrics = saved["raw_metrics"]
            static_metadata = saved["static_metadata"]

        # Collect runtime metric metadata and merge with static metadata
        runtime_metadata = runtime_metric_registry.get_all_metadata()
        all_metadata = {**static_metadata, **runtime_metadata}

        # --- Format/Prepare Results ---
        self.logger.info("Formatting and structuring results...")
        formatter = MetricsFormatter(self.config, all_metadata)
        formatted_results = formatter.format_results(raw_metrics)
        self.logger.info(f"Formatted {len(formatted_results.evaluations)} evaluation entries.")
        gc.collect()

        # --- Analyze Metrics ---
        self.logger.info("Aggregating and analyzing metrics...")
        analyzer = MetricsAnalyzer(self.config)
        summary = analyzer.analyze(formatted_results)
        self.logger.info("Metrics analysis complete.")

        # --- Generate comparison visualizations ---
        self.logger.info("Generating comparison visualizations...")
        visualizer = ResultsVisualizer(self.config)
        visualizer.generate(formatted_results, summary)
        self.logger.info("Comparison visualizations complete.")
        gc.collect()

        # --- Generate Reports ---
        self.logger.info("Generating reports...")
        writer = ReportWriter(self.config)
        writer.write_ids_report(formatted_results)
        writer.write_summary(summary)
        self.logger.info("Evaluation completed and reports generated successfully.")

        checkpoint.set_stage(EvaluationCheckpointStore.STAGE_COMPLETE)

    def _run_sig_ids_evaluation(
        self, dataset_splits: List[Split], loaded_plugins: list[tuple[AbstractIDSConnector, SignatureModelConfig]]
    ):
        raise NotImplementedError("Signature-based evaluation is not implemented yet.")

    def _get_checkpoint_path(self) -> Path:
        report_dir = ConfigManager.get_report_directory(self.config)
        checkpoint_dir = report_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir / "evaluation_state.pkl"

    def _load_processed_data(self) -> Union[List[Split], List[Folds]]:
        root = ConfigManager.get_processed_data_directory(self.config)
        name = f"{self.config.general.name.replace(' ', '_').lower()}"
        name = name[:200]
        name = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).rstrip()
        name += ".pkl"
        file = root / name
        self.logger.info(f"Loading processed data from {file}...")
        if not file.exists():
            self.logger.error(f"Processed data not found at {file} - Please run the 'dataset' command first.")
            raise FileNotFoundError(f"Processed data not found: {file}\n Please run the 'dataset' command first.")
        try:
            with open(file, "rb") as f:
                datasets = pickle.load(f)

            for ds_idx, splits in enumerate(datasets):
                if isinstance(splits, list):
                    for fold_idx, (X_train, X_test, y_train, y_test) in enumerate(splits):
                        self.logger.info(
                            f"Dataset {ds_idx}, fold {fold_idx}: "
                            f"X_train: {X_train.shape}, y_train: {y_train.shape}, "
                            f"X_test: {X_test.shape}, y_test: {y_test.shape}"
                        )
                else:
                    X_train, X_test, y_train, y_test = splits
                    self.logger.info(
                        f"Dataset {ds_idx}: "
                        f"X_train: {X_train.shape}, y_train: {y_train.shape}, "
                        f"X_test: {X_test.shape}, y_test: {y_test.shape}"
                    )
            return datasets
        except Exception as e:
            self.logger.error(f"Failed to load processed data: {e}")
            raise e
