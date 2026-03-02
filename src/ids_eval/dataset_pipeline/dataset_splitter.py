from __future__ import annotations

import logging
from typing import List, Tuple, Union

import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import StratifiedKFold, train_test_split

from ids_eval.dto.run_config import RunConfig
from ids_eval.enumeration.internal_label import InternalLabel
from ids_eval.enumeration.split_method import SplitMethod

# A single split of data into training and testing sets
Split = Tuple[DataFrame, DataFrame, Series, Series]

# A list of splits to represent folds in cross-validation
Folds = List[Split]


class DatasetSplitter:
    """Handles splitting datasets into training and testing sets based on configured strategies."""

    def __init__(self, config: RunConfig):
        self.config: RunConfig = config
        self.metadata = {"steps": []}
        self.logger = logging.getLogger(__name__)

    def split(self, datasets: list[pd.DataFrame]) -> Union[List[Split], List[Folds]]:
        split_method = self.config.data_manager.split.method

        match split_method:
            case SplitMethod.INTRA:
                self.logger.info("Splitting datasets into intra-dataset splits...")
                return self._split_intra_dataset(datasets)
            case SplitMethod.KFOLDSPLIT:
                self.logger.info("Splitting datasets into k-fold splits...")
                return self._get_k_fold_splits(datasets)
            case SplitMethod.TIMESTAMP:
                self.logger.info("Splitting datasets into timestamp splits...")
                return self._split_by_timestamp(datasets)
            case SplitMethod.CROSS_DATASET:
                self.logger.info("Splitting datasets into cross-dataset splits...")
                return self._split_cross_dataset(datasets)
            case SplitMethod.BENIGN_TRAIN:
                self.logger.info("Splitting datasets for semi-supervised learning (benign-only training)...")
                return self._split_benign_train(datasets)
            case SplitMethod.CROSS_DATASET_BENIGN:
                self.logger.info("Splitting for cross-dataset evaluation with benign-only training...")
                return self._split_cross_dataset_benign(datasets)
            case _:
                self.logger.error(f"Unsupported split method: {split_method}")
                raise ValueError(f"Unsupported split method: {split_method}")

    def _split_intra_dataset(self, datasets: list[pd.DataFrame]) -> List[Split]:
        """Splits each dataset independently into a single training and testing set."""
        target_column = self.config.data_manager.split.target_column
        test_size = self.config.data_manager.split.test_size

        results: List[Split] = []
        for i, df in enumerate(datasets):
            if target_column not in df.columns:
                self.logger.error(f"Target column '{target_column}' not found in dataset index {i}")
                raise ValueError(f"Target column '{target_column}' not found in dataset index {i}")

            y = df[target_column]
            X = df.drop(columns=InternalLabel.__values__() + [target_column], errors="ignore")

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=self.config.general.seed,
                stratify=y,  # Stratify to maintain class distribution
            )

            results.append((X_train, X_test, y_train, y_test))

            # Record metadata for this dataset's split
            self.metadata["steps"].append(
                {
                    "action": "intra_dataset_split",
                    "dataset_index": i,
                    "test_size": test_size,
                    "train_shape": list(X_train.shape),
                    "test_shape": list(X_test.shape),
                    "target_distribution_train": y_train.value_counts(normalize=True).to_dict(),
                    "target_distribution_test": y_test.value_counts(normalize=True).to_dict(),
                }
            )

        self.logger.info(f"Intra-dataset split completed for {len(results)} datasets with test_size={test_size}.")
        return results

    def _get_k_fold_splits(self, datasets: list[pd.DataFrame]) -> List[Folds]:
        """Generates k-fold cross-validation splits for each dataset."""
        params = self.config.data_manager.split.params
        if params is None or not hasattr(params, "n_splits"):
            raise ValueError("KFold split requires 'n_splits' parameter in config")
        n_splits = params.n_splits

        target_column = self.config.data_manager.split.target_column
        results: List[Folds] = []
        for i, df in enumerate(datasets):
            if target_column not in df.columns:
                self.logger.error(f"Target column '{target_column}' not found in dataset index {i}")
                raise ValueError(f"Target column '{target_column}' not found in dataset index {i}")
            y = df[target_column]
            X = df.drop(columns=InternalLabel.__values__() + [target_column], errors="ignore")

            # Use StratifiedKFold for classification tasks to preserve class distribution
            skf = StratifiedKFold(n_splits=n_splits, random_state=self.config.general.seed)
            fold_splits: Folds = []
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                fold_splits.append((X_train, X_test, y_train, y_test))

                self.metadata["steps"].append(
                    {
                        "action": "kfold_split",
                        "dataset_index": i,
                        "fold": fold_idx,
                        "n_splits": n_splits,
                        "train_shape": list(X_train.shape),
                        "test_shape": list(X_test.shape),
                        "target_distribution_train": y_train.value_counts(normalize=True).to_dict(),
                        "target_distribution_test": y_test.value_counts(normalize=True).to_dict(),
                    }
                )
            results.append(fold_splits)
        self.logger.info(f"K-Fold split completed for {len(results)} datasets with n_splits={n_splits}.")
        return results

    def _split_by_timestamp(self, datasets: list[pd.DataFrame]) -> List[Split]:
        """Splits each dataset based on a timestamp column."""
        params = self.config.data_manager.split.params
        if params is None or not hasattr(params, "timestamp_column"):
            raise ValueError("Timestamp split requires 'timestamp_column' parameter in config")
        ts_col = params.timestamp_column
        test_size = self.config.data_manager.split.test_size

        results: List[Split] = []
        for i, df in enumerate(datasets):
            if ts_col not in df.columns:
                raise ValueError(f"Timestamp column '{ts_col}' not found in dataset index {i}")
            target_column = self.config.data_manager.split.target_column
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset index {i}")

            # Sort over temporary copy without changing the original data type
            sort_key = df[ts_col]
            if not pd.api.types.is_datetime64_any_dtype(sort_key):
                numeric = pd.to_numeric(sort_key, errors="coerce")
                valid_mask = numeric.notna()
                df = df.loc[valid_mask].copy()
                sort_key = pd.to_datetime(numeric[valid_mask])
            df_sorted = (
                df.assign(_sort_ts=sort_key).sort_values("_sort_ts").drop(columns="_sort_ts").reset_index(drop=True)
            )

            split_index = int((1 - test_size) * len(df_sorted))

            train_df = df_sorted.iloc[:split_index]
            test_df = df_sorted.iloc[split_index:]

            y_train = train_df[target_column]
            X_train = train_df.drop(columns=InternalLabel.__values__() + [target_column], errors="ignore")
            y_test = test_df[target_column]
            X_test = test_df.drop(columns=InternalLabel.__values__() + [target_column], errors="ignore")

            results.append((X_train, X_test, y_train, y_test))

            self.metadata["steps"].append(
                {
                    "action": "timestamp_split",
                    "dataset_index": i,
                    "timestamp_column": ts_col,
                    "test_size": test_size,
                    "train_shape": list(X_train.shape),
                    "test_shape": list(X_test.shape),
                    "target_distribution_train": y_train.value_counts(normalize=True).to_dict(),
                    "target_distribution_test": y_test.value_counts(normalize=True).to_dict(),
                }
            )

        self.logger.info(f"Timestamp split completed for {len(results)} datasets using column '{ts_col}'.")
        return results

    def _split_cross_dataset(self, datasets: list[pd.DataFrame]) -> List[Split]:
        """Splits datasets ensuring consistent features across all datasets."""
        target_column = self.config.data_manager.split.target_column
        test_size = self.config.data_manager.split.test_size

        if len(datasets) < 2:
            raise ValueError("Cross-dataset split requires at least two datasets")

        # we need to make sure that every column (features) are the same across all datasets
        common_columns = set(datasets[0].columns)
        for df in datasets[1:]:
            common_columns = common_columns.intersection(set(df.columns))

        result_columns = list(common_columns)
        result_columns.extend(InternalLabel.__values__())
        result_columns.append(target_column)
        result_columns = list(set(result_columns))

        filtered_datasets = [df[result_columns] for df in datasets]

        dataset_results: List[Split] = []
        for i, df in enumerate(filtered_datasets):
            if target_column not in df.columns:
                self.logger.error(f"Target column '{target_column}' not found in dataset index {i}")
                raise ValueError(f"Target column '{target_column}' not found in dataset index {i}")

            y = df[target_column]
            X = df.drop(columns=InternalLabel.__values__() + [target_column], errors="ignore")

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=self.config.general.seed,
                stratify=y,  # Stratify to maintain class distribution
            )

            dataset_results.append((X_train, X_test, y_train, y_test))

            # Record metadata for this dataset's split
            self.metadata["steps"].append(
                {
                    "action": "cross_dataset_split",
                    "dataset_index": i,
                    "test_size": test_size,
                    "train_shape": list(X_train.shape),
                    "test_shape": list(X_test.shape),
                    "target_distribution_train": y_train.value_counts(normalize=True).to_dict(),
                    "target_distribution_test": y_test.value_counts(normalize=True).to_dict(),
                }
            )

        self.logger.info(
            f"Cross-dataset split completed for {len(dataset_results)} datasets with test_size={test_size}."
        )

        return dataset_results

    def _split_benign_train(self, datasets: list[pd.DataFrame]) -> List[Split]:
        """
        This split method is designed for anomaly detection models (like Kitsune) that
        should be trained only on normal/benign traffic.

        Note: In a real-world scenario, the training data might also contain unknown attacks (noise),
        we keep the set purely noise free at this point.
        """
        target_column = self.config.data_manager.split.target_column
        test_size = self.config.data_manager.split.test_size

        params = self.config.data_manager.split.params
        benign_label = 0
        if params is not None and hasattr(params, "benign_label"):
            benign_label = params.benign_label

        results: List[Split] = []
        for i, df in enumerate(datasets):
            if target_column not in df.columns:
                self.logger.error(f"Target column '{target_column}' not found in dataset index {i}")
                raise ValueError(f"Target column '{target_column}' not found in dataset index {i}")

            y = df[target_column]
            X = df.drop(columns=InternalLabel.__values__() + [target_column], errors="ignore")

            benign_mask = y == benign_label
            attack_mask = ~benign_mask

            X_benign = X[benign_mask]
            y_benign = y[benign_mask]
            X_attack = X[attack_mask]
            y_attack = y[attack_mask]

            n_benign = len(X_benign)
            n_attack = len(X_attack)

            self.logger.info(f"Dataset {i}: {n_benign} benign samples, {n_attack} attack samples")

            if n_benign == 0:
                raise ValueError(
                    f"No benign samples found in dataset {i} with benign_label={benign_label}. "
                    "Check your label configuration."
                )

            if n_benign > 1:
                X_benign_train, X_benign_test, y_benign_train, y_benign_test = train_test_split(
                    X_benign, y_benign, test_size=test_size, random_state=self.config.general.seed
                )
            else:
                X_benign_train = X_benign
                y_benign_train = y_benign
                X_benign_test = pd.DataFrame(columns=X_benign.columns)
                y_benign_test = pd.Series(dtype=y_benign.dtype)

            X_train = X_benign_train
            y_train = y_benign_train

            X_test = pd.concat([X_benign_test, X_attack], ignore_index=True, copy=False)
            y_test = pd.concat([y_benign_test, y_attack], ignore_index=True, copy=False)

            results.append((X_train, X_test, y_train, y_test))

            self.metadata["steps"].append(
                {
                    "action": "benign_train_split",
                    "dataset_index": i,
                    "benign_label": benign_label,
                    "test_size": test_size,
                    "n_benign_total": n_benign,
                    "n_attack_total": n_attack,
                    "train_shape": list(X_train.shape),
                    "test_shape": list(X_test.shape),
                    "train_benign_count": len(X_train),
                    "test_benign_count": len(X_benign_test),
                    "test_attack_count": n_attack,
                    "target_distribution_train": y_train.value_counts(normalize=True).to_dict(),
                    "target_distribution_test": y_test.value_counts(normalize=True).to_dict(),
                }
            )

            benign_percent_train = (len(X_train) / n_benign) * 100 if n_benign > 0 else 0

            self.logger.info(
                f"Dataset {i}: Train={len(X_train)} ({benign_percent_train} benign), "
                f"Test={len(X_test)} ({len(X_benign_test)} benign + {n_attack} attack)"
            )

        self.logger.info(
            f"Benign-train split completed for {len(results)} datasets. "
            f"Training sets contain only benign samples (label={benign_label})."
        )
        return results

    def _split_cross_dataset_benign(self, datasets: list[pd.DataFrame]) -> List[Split]:
        """Splits datasets for cross-dataset evaluation with benign-only training."""
        target_column = self.config.data_manager.split.target_column
        test_size = self.config.data_manager.split.test_size

        if len(datasets) < 2:
            raise ValueError(
                "Cross-dataset-benign split requires at least two datasets. "
                "Use 'benign_train' for single dataset evaluation."
            )

        params = self.config.data_manager.split.params
        benign_label = 0
        if params is not None and hasattr(params, "benign_label"):
            benign_label = params.benign_label

        # --- Common column logic (identical to _split_cross_dataset) ---
        common_columns = set(datasets[0].columns)
        for df in datasets[1:]:
            common_columns = common_columns.intersection(set(df.columns))

        result_columns = list(common_columns)
        result_columns.extend(InternalLabel.__values__())
        result_columns.append(target_column)
        result_columns = list(set(result_columns))

        filtered_datasets = [df[result_columns] for df in datasets]

        # --- Per-dataset benign-train split (mirrors _split_benign_train) ---
        dataset_results: List[Split] = []
        for i, df in enumerate(filtered_datasets):
            if target_column not in df.columns:
                self.logger.error(f"Target column '{target_column}' not found in dataset index {i}")
                raise ValueError(f"Target column '{target_column}' not found in dataset index {i}")

            y = df[target_column]
            X = df.drop(columns=InternalLabel.__values__() + [target_column], errors="ignore")

            benign_mask = y == benign_label
            attack_mask = ~benign_mask

            X_benign = X[benign_mask]
            y_benign = y[benign_mask]
            X_attack = X[attack_mask]
            y_attack = y[attack_mask]

            n_benign = len(X_benign)
            n_attack = len(X_attack)

            self.logger.info(f"Dataset {i}: {n_benign} benign samples, {n_attack} attack samples")

            if n_benign == 0:
                raise ValueError(
                    f"No benign samples found in dataset {i} with benign_label={benign_label}. "
                    "Check your label configuration."
                )

            if n_benign > 1:
                X_benign_train, X_benign_test, y_benign_train, y_benign_test = train_test_split(
                    X_benign, y_benign, test_size=test_size, random_state=self.config.general.seed
                )
            else:
                X_benign_train = X_benign
                y_benign_train = y_benign
                X_benign_test = pd.DataFrame(columns=X_benign.columns)
                y_benign_test = pd.Series(dtype=y_benign.dtype)

            X_train = X_benign_train
            y_train = y_benign_train

            X_test = pd.concat([X_benign_test, X_attack], ignore_index=True, copy=False)
            y_test = pd.concat([y_benign_test, y_attack], ignore_index=True, copy=False)

            dataset_results.append((X_train, X_test, y_train, y_test))

            # Record metadata for this dataset's split
            self.metadata["steps"].append(
                {
                    "action": "cross_dataset_benign_split",
                    "dataset_index": i,
                    "benign_label": benign_label,
                    "test_size": test_size,
                    "n_benign_total": n_benign,
                    "n_attack_total": n_attack,
                    "train_shape": list(X_train.shape),
                    "test_shape": list(X_test.shape),
                    "train_benign_count": len(X_train),
                    "test_benign_count": len(X_benign_test),
                    "test_attack_count": n_attack,
                    "target_distribution_train": y_train.value_counts(normalize=True).to_dict(),
                    "target_distribution_test": y_test.value_counts(normalize=True).to_dict(),
                }
            )

            benign_percent_train = (len(X_train) / n_benign) * 100 if n_benign > 0 else 0

            self.logger.info(
                f"Dataset {i}: Train={len(X_train)} ({benign_percent_train:.1f}% benign), "
                f"Test={len(X_test)} ({len(X_benign_test)} benign + {n_attack} attack)"
            )

        self.logger.info(
            f"Cross-dataset-benign split completed for {len(dataset_results)} datasets with test_size={test_size}. "
            f"Training sets contain only benign samples (label={benign_label})."
        )

        return dataset_results
