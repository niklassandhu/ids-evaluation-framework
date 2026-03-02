from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.linear_model import LogisticRegression

from ids_eval.dto.run_config import RunConfig
from ids_eval.enumeration.feature_selection_method import FeatureSelectionMethod
from ids_eval.enumeration.internal_label import InternalLabel


class FeatureSelector:
    """Handles the selection of features from datasets based on configured methods."""

    def __init__(self, config: RunConfig):
        self.config: RunConfig = config
        self.metadata = {"steps": []}
        self.logger = logging.getLogger(__name__)

    def select_features(self, datasets: list[pd.DataFrame]) -> list[pd.DataFrame]:
        processed_datasets = []
        for i, dataset in enumerate(datasets):
            if (
                not self.config.data_manager.dataset[i].feature_selector
                or self.config.data_manager.dataset[i].feature_selector.method == FeatureSelectionMethod.NONE
            ):
                self.logger.warning(
                    f"No feature selector defined for dataset {self.config.data_manager.dataset[i].name}. Skipping."
                )
                processed_datasets.append(dataset)
                continue

            config = self.config.data_manager.dataset[i].feature_selector
            self.logger.info(f"Applying feature selection method '{config.method.value}' to dataset {i}.")

            processed_dataset = dataset

            match config.method:
                case FeatureSelectionMethod.LOGISTIC_REGRESSION:
                    processed_dataset = self._logistic_regression(processed_dataset, i)
                case FeatureSelectionMethod.VARIANCE_THRESHOLD:
                    processed_dataset = self._variance_threshold(processed_dataset, i)
                case FeatureSelectionMethod.CORRELATION_THRESHOLD:
                    processed_dataset = self._correlation_threshold(processed_dataset, i)
                case _:
                    self.logger.error(f"Unsupported feature selection method: {config.method}")
                    raise ValueError(f"Unsupported feature selection method: {config.method}")

            processed_datasets.append(processed_dataset)

        self.logger.info("Feature selection completed for all datasets.")
        return processed_datasets

    def _logistic_regression(self, dataset: pd.DataFrame, config_index: int) -> pd.DataFrame:
        self.logger.info("Using Logistic Regression to select features.")

        # LogisticRegression is sensitive to NaNs and infinities
        X = dataset.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

        y = dataset[InternalLabel.TARGET_LABEL.value]
        # Ensure 'attack_category' and 'target_label' are not in X
        X = X.drop(columns=InternalLabel.__values__(), errors="ignore")

        params = self.config.data_manager.dataset[config_index].feature_selector.params
        if params is None:
            raise ValueError("Parameters for logistic regression feature selection are not configured.")

        logreg_model = LogisticRegression(
            C=params.C,
            penalty=params.penalty.value,
            solver=params.solver.value,
            max_iter=params.max_iter,
            random_state=self.config.general.seed,
        )

        selector = SelectFromModel(logreg_model, threshold=params.threshold)

        selector.fit(X, y.values.ravel())

        selected_features_mask = selector.get_support()
        selected_features = X.columns[selected_features_mask]

        self.logger.info(f"Selected {len(selected_features)} features using Logistic Regression.")
        self.metadata["steps"].append(
            {
                "action": "logistic_regression_feature_selection",
                "selected_count": len(selected_features),
                "selected_features": selected_features.tolist(),
            }
        )

        # Reconstruct the DataFrame with selected features and original labels
        selected_df = X[selected_features].copy()
        selected_df[InternalLabel.ATTACK_CATEGORY.value] = dataset[InternalLabel.ATTACK_CATEGORY.value]
        selected_df[InternalLabel.TARGET_LABEL.value] = dataset[InternalLabel.TARGET_LABEL.value]
        return selected_df

    def _variance_threshold(self, dataset: pd.DataFrame, config_index: int) -> pd.DataFrame:
        self.logger.info("Using Variance Threshold to select features.")

        X = dataset.drop(columns=InternalLabel.__values__(), errors="ignore")
        X = X.select_dtypes(include=np.number)  # Variance is only for numeric features

        params = self.config.data_manager.dataset[config_index].feature_selector.params
        threshold = 0.0
        if params and hasattr(params, "threshold"):
            threshold = float(params.threshold)

        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        selected_features = X.columns[selector.get_support()]

        self.logger.info(
            f"Selected {len(selected_features)} features using VarianceThreshold with threshold={threshold}."
        )
        self.metadata["steps"].append(
            {
                "action": "variance_threshold_feature_selection",
                "selected_count": len(selected_features),
                "selected_features": selected_features.tolist(),
                "threshold": threshold,
            }
        )

        # Reconstruct the DataFrame
        selected_df = X[selected_features].copy()
        # Add back non-numeric columns and labels that were dropped
        non_numeric_cols = dataset.select_dtypes(exclude=np.number).columns
        selected_df = pd.concat([selected_df, dataset[non_numeric_cols]], axis=1)
        return selected_df

    def _correlation_threshold(self, dataset: pd.DataFrame, config_index: int) -> pd.DataFrame:
        self.logger.info("Using Correlation Threshold to select features.")
        y = dataset[InternalLabel.TARGET_LABEL.value]
        X = dataset.drop(columns=InternalLabel.__values__(), errors="ignore")
        X_numeric = X.select_dtypes(include=np.number)

        params = self.config.data_manager.dataset[config_index].feature_selector.params
        threshold = 0.95
        if params and hasattr(params, "threshold"):
            threshold = float(params.threshold)

        corr_matrix = X_numeric.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        selected_features = X.columns.drop(to_drop)

        self.logger.info(f"Dropped {len(to_drop)} highly correlated features (threshold={threshold}).")
        self.metadata["steps"].append(
            {
                "action": "correlation_threshold_feature_selection",
                "dropped_count": len(to_drop),
                "dropped_features": to_drop,
                "threshold": threshold,
            }
        )

        selected_df = X[selected_features].copy()
        selected_df[InternalLabel.ATTACK_CATEGORY.value] = dataset[InternalLabel.ATTACK_CATEGORY.value]
        selected_df[InternalLabel.TARGET_LABEL.value] = y
        return selected_df
