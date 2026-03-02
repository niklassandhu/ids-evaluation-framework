import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from pandas import DataFrame, Series

from ids_eval.dto.run_config import RunConfig
from ids_eval.run_config_pipeline.config_manager import ConfigManager

# A single split of data into training and testing sets
Split = Tuple[DataFrame, DataFrame, Series, Series]

# A list of splits to represent folds in cross-validation
Folds = List[Split]


class DatasetAnalyser:
    """Analyzes datasets and generates reports on their characteristics."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def report(
        self, all_metadata: dict, datasets: List[DataFrame], splits: Union[List[Split], List[Folds], None] = None
    ) -> None:
        report_root = ConfigManager.get_report_directory(self.config)
        visuals_dir = report_root / "visuals_dataset"
        visuals_dir.mkdir(exist_ok=True)
        dataset_report_file = report_root / "dataset_report.yaml"

        report_data = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "config_hash": all_metadata.get("config_hash", "unknown"),
            },
            "datasets": self._build_datasets_section(datasets, splits, visuals_dir),
            "pipeline_applied": self._build_pipeline_section(all_metadata),
        }

        with open(dataset_report_file, "w") as f:
            yaml.dump(report_data, f, sort_keys=False, default_flow_style=False)

        self.logger.info(f"Dataset report saved to {dataset_report_file}")

    def _build_datasets_section(
        self, datasets: List[DataFrame], splits: Union[List[Split], List[Folds], None], visuals_dir: Path
    ) -> List[Dict[str, Any]]:
        datasets_info = []

        for i, df in enumerate(datasets):
            dataset_name = self.config.data_manager.dataset[i].name
            target_col = self.config.data_manager.split.target_column

            # Basic statistics
            dataset_entry: Dict[str, Any] = {
                "name": dataset_name,
                "statistics": self._summarize_dataframe(df, target_col),
            }

            # Add split information if available
            if splits is not None and i < len(splits):
                split_data = splits[i]
                dataset_entry["splits"] = self._analyze_single_split(split_data, dataset_name, visuals_dir)

            # Generate visualizations
            viz_paths = self._generate_visualizations(df, dataset_name, visuals_dir)
            if viz_paths:
                dataset_entry["visualizations"] = viz_paths

            datasets_info.append(dataset_entry)

        return datasets_info

    def _summarize_dataframe(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        missing_values = df.isnull().sum()
        missing_info = None
        if missing_values.any():
            missing_count = int(missing_values[missing_values > 0].sum())
            missing_info = {"total_missing": missing_count}

        summary: Dict[str, Any] = {
            "total_samples": int(df.shape[0]),
            "features": int(df.shape[1]),
            "numeric_features": len(numeric_cols),
            "categorical_features": len(categorical_cols),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1e6:.2f} MB",
        }

        if missing_info:
            summary["missing_values"] = missing_info

        # Add class distribution if target column exists
        if target_col in df.columns:
            summary["class_distribution"] = self._format_class_distribution(df[target_col])

        return summary

    @staticmethod
    def _format_class_distribution(y: pd.Series) -> Dict[str, Dict[str, Any]]:
        counts = y.value_counts()
        percentages = y.value_counts(normalize=True) * 100

        distribution = {}
        for label in counts.index:
            distribution[str(label)] = {"count": int(counts[label]), "percent": round(float(percentages[label]), 5)}
        return distribution

    def _analyze_single_split(
        self, split_data: Union[Split, Folds], dataset_name: str, visuals_dir: Path
    ) -> Dict[str, Any]:
        is_kfold = isinstance(split_data, list)

        if is_kfold:
            return self._analyze_kfold_splits(split_data, dataset_name)
        else:
            return self._analyze_train_test_split(split_data, dataset_name, visuals_dir)

    def _analyze_train_test_split(self, split: Split, dataset_name: str, visuals_dir: Path) -> Dict[str, Any]:
        _, _, y_train, y_test = split

        # Generate visualization
        viz_path = visuals_dir / f"{dataset_name}_split_dist.png"
        self.plot_split_distribution(y_train, y_test, viz_path)

        return {
            "type": "train-test",
            "train": {"samples": int(len(y_train)), "distribution": self._format_distribution_compact(y_train)},
            "test": {"samples": int(len(y_test)), "distribution": self._format_distribution_compact(y_test)},
            "visualization": str(viz_path),
        }

    @staticmethod
    def _analyze_kfold_splits(folds: Folds, dataset_name: str) -> Dict[str, Any]:
        """Analyzes K-Fold cross-validation splits."""
        fold_stats = []
        for j, fold in enumerate(folds):
            _, _, y_train, y_test = fold
            fold_stats.append({"fold": j, "train_samples": int(len(y_train)), "test_samples": int(len(y_test))})

        # Calculate averages
        avg_train = sum(f["train_samples"] for f in fold_stats) / len(fold_stats)
        avg_test = sum(f["test_samples"] for f in fold_stats) / len(fold_stats)

        return {
            "type": "k-fold",
            "n_folds": len(folds),
            "avg_train_samples": int(avg_train),
            "avg_test_samples": int(avg_test),
            "folds": fold_stats,
        }

    @staticmethod
    def _format_distribution_compact(y: pd.Series) -> Dict[str, str]:
        percentages = y.value_counts(normalize=True) * 100
        return {str(label): f"{pct:.1f}%" for label, pct in percentages.items()}

    def _build_pipeline_section(self, all_metadata: dict) -> Dict[str, Any]:
        pipeline: Dict[str, Any] = {}
        if "constructor" in all_metadata:
            constructor = all_metadata["constructor"]
            pipeline["construction"] = self._compact_constructor_steps(constructor)

        if "preprocessor" in all_metadata:
            preprocessor = all_metadata["preprocessor"]
            pipeline["preprocessing"] = self._compact_preprocessor_steps(preprocessor)

        if "feature_selector" in all_metadata:
            feature_selector = all_metadata["feature_selector"]
            steps = feature_selector.get("steps", [])
            if steps:
                pipeline["feature_selection"] = steps
            else:
                pipeline["feature_selection"] = {"method": "none"}

        if "splitter" in all_metadata:
            splitter = all_metadata["splitter"]
            pipeline["splitting"] = self._compact_splitter_info(splitter)

        return pipeline

    @staticmethod
    def _compact_constructor_steps(constructor: dict) -> List[Dict[str, Any]]:
        steps = constructor.get("steps", [])
        compact_steps = []

        for step in steps:
            action = step.get("action", "unknown")
            compact_step: Dict[str, Any] = {"step": action}

            if "count" in step:
                compact_step["affected_columns"] = step["count"]
            if "removed" in step:
                compact_step["rows_removed"] = step["removed"]
            if "output_shape" in step:
                compact_step["output_shape"] = step["output_shape"]
            if "renamed_count" in step:
                compact_step["columns_renamed"] = step["renamed_count"]

            compact_steps.append(compact_step)

        return compact_steps

    @staticmethod
    def _compact_preprocessor_steps(preprocessor: dict) -> List[Dict[str, Any]]:
        steps = preprocessor.get("steps", [])
        compact_steps = []

        for step in steps:
            action = step.get("action", "unknown")
            compact_step: Dict[str, Any] = {"step": action}

            if "columns" in step:
                columns = step["columns"]
                if isinstance(columns, list):
                    compact_step["affected_columns"] = len(columns)

            compact_steps.append(compact_step)

        return compact_steps

    @staticmethod
    def _compact_splitter_info(splitter: dict) -> Dict[str, Any]:
        steps = splitter.get("steps", [])
        if not steps:
            return {"method": "none"}

        first_step = steps[0]
        action = first_step.get("action", "unknown")

        info: Dict[str, Any] = {"method": action, "datasets_processed": len(steps)}

        if "test_size" in first_step:
            info["test_size"] = first_step["test_size"]

        return info

    def _generate_visualizations(self, df: pd.DataFrame, name: str, out_dir: Path) -> Dict[str, str]:
        paths = {}
        target_col = self.config.data_manager.split.target_column
        if target_col in df.columns:
            path = out_dir / f"{name}_target_dist.png"
            self.plot_class_distribution(df[target_col], path)
            paths["target_distribution"] = str(path)
        return paths

    def plot_class_distribution(self, y: pd.Series, out_path: Path) -> None:
        if y.empty:
            return
        counts = y.value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        counts.plot(kind="bar", ax=ax, colormap="viridis")
        ax.set_xlabel(self.config.data_manager.split.target_column)
        ax.set_ylabel("Frequency")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)

    def plot_split_distribution(self, y_train: pd.Series, y_test: pd.Series, out_path: Path) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        y_train.value_counts().plot(kind="bar", ax=ax1, colormap="plasma", title="Train Distribution")
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.set_ylabel("Proportion")
        ax1.set_xlabel(self.config.data_manager.split.target_column)
        y_test.value_counts().plot(kind="bar", ax=ax2, colormap="cividis", title="Test Distribution")
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
