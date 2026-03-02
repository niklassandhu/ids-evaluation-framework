import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

from ids_eval.dto.run_config import RunConfig
from ids_eval.enumeration.internal_label import InternalLabel
from ids_eval.enumeration.preprocess_method import PreprocessMethod


class DatasetPreprocessor:
    """Handles all preprocessing of datasets, including scaling, encoding, and imputation."""

    def __init__(self, config: RunConfig):
        self.config: RunConfig = config
        self.metadata = {"steps": []}
        self.scalers = {}
        self.encoders = {}
        self.logger = logging.getLogger(__name__)

    def preprocess(self, datasets: list[pd.DataFrame]) -> list[pd.DataFrame]:
        preprocessed_datasets = []
        for i, dataset in enumerate(datasets):
            self.logger.info(f"Preprocessing dataset {i}...")
            processed_df = dataset

            preprocess_configs = self.config.data_manager.dataset[i].preprocess
            if not preprocess_configs:
                preprocessed_datasets.append(processed_df)
                continue

            for config in preprocess_configs:
                method = config.method
                self.logger.debug(f"Applying preprocessing method: {method.value}")

                # Determine columns to apply the transformation on
                if config.auto_columns:
                    if method in [
                        PreprocessMethod.MIN_MAX,
                        PreprocessMethod.STANDARD,
                        PreprocessMethod.IMPUTE_MEAN,
                        PreprocessMethod.IMPUTE_MEDIAN,
                    ]:
                        cols = processed_df.select_dtypes(include=["number"]).columns.tolist()
                    elif method in [PreprocessMethod.LABEL, PreprocessMethod.ONE_HOT]:
                        cols = processed_df.select_dtypes(include=["object", "category"]).columns.tolist()
                    elif method == PreprocessMethod.IMPUTE_MOST_FREQUENT:
                        cols = processed_df.columns.tolist()
                    else:
                        cols = config.columns  # Fallback to specified columns
                else:
                    cols = config.columns

                # never preprocess "target_label" or "attack_category" remove from cols
                cols = [col for col in cols if col not in InternalLabel.__values__()]

                match method:
                    case PreprocessMethod.REMOVE_DUPLICATE_ROWS:
                        processed_df = self._remove_duplicate_rows(processed_df)
                    case PreprocessMethod.REMOVE_NAN_ROWS:
                        processed_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                        processed_df.dropna(inplace=True)
                    case PreprocessMethod.REMOVE_SINGLE_VALUE_COLUMNS:
                        processed_df = self._remove_single_value_columns(processed_df, cols)
                    case PreprocessMethod.REMOVE_CLASS:
                        processed_df = self._remove_class(processed_df, cols)
                    case PreprocessMethod.REMOVE_ROWS:
                        processed_df = self._remove_rows(processed_df, cols)
                    case PreprocessMethod.MIN_MAX:
                        processed_df = self._scale_min_max(processed_df, cols)
                    case PreprocessMethod.STANDARD:
                        processed_df = self._scale_standard(processed_df, cols)
                    case PreprocessMethod.LABEL:
                        processed_df = self._encode_label(processed_df, cols)
                    case PreprocessMethod.ONE_HOT:
                        processed_df = self._encode_one_hot(processed_df, cols)
                    case PreprocessMethod.IMPUTE_MEAN:
                        processed_df = self._impute(processed_df, cols, strategy="mean")
                    case PreprocessMethod.IMPUTE_MEDIAN:
                        processed_df = self._impute(processed_df, cols, strategy="median")
                    case PreprocessMethod.IMPUTE_MOST_FREQUENT:
                        processed_df = self._impute(processed_df, cols, strategy="most_frequent")
                    case PreprocessMethod.CAST_NUMERIC:
                        processed_df = self._cast_numeric(processed_df, cols)
                    case PreprocessMethod.NONE:
                        self.logger.info("Preprocessing method 'none' specified; no action taken.")
                    case _:
                        self.logger.error(f"Unsupported normalization method: {method}")
                        raise ValueError(f"Unsupported normalization method {method}")

            preprocessed_datasets.append(processed_df)

        self.logger.info("All datasets have been preprocessed.")
        return preprocessed_datasets

    def _remove_class(self, df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
        if not classes:
            self.logger.warning("Class removal skipped: no classes specified.")
            return df

        label_col = InternalLabel.ATTACK_CATEGORY.value

        if label_col not in df.columns:
            self.logger.warning(f"Column '{label_col}' not found. Class removal skipped.")
            return df

        initial_rows = len(df)
        # Remove rows where attack_category is in the list of classes to remove
        df = df[not df[label_col].isin(classes)]
        removed_count = initial_rows - len(df)

        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} rows with classes: {classes}")
        else:
            self.logger.info(f"No rows found with classes: {classes}")

        self.metadata["steps"].append({"action": "remove_class", "classes": classes, "removed_rows": removed_count})

        return df

    def _remove_duplicate_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_rows = len(df)
        df.drop_duplicates(inplace=True, keep="first")
        dedup_count = initial_rows - len(df)
        if dedup_count:
            self.logger.info(f"Removed {dedup_count} duplicated rows.")
        self.metadata["steps"].append({"action": "remove_duplicate_rows", "count": dedup_count})
        return df

    def _remove_single_value_columns(self, df: pd.DataFrame, cols) -> pd.DataFrame:
        single_val_cols = [col for col in cols if df[col].nunique() <= 1]
        if single_val_cols:
            df.drop(columns=single_val_cols, inplace=True)
            self.logger.info(f"Removing {len(single_val_cols)} single-value columns: {single_val_cols}.")
        self.metadata["steps"].append({"action": "remove_single_value_columns", "columns": single_val_cols})
        return df

    def _remove_rows(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        if not columns:
            self.logger.warning("Row removal skipped: no columns specified.")
            return df
        df = df.drop(columns=columns)
        self.logger.info(f"Removed {len(columns)} columns.")
        self.metadata["steps"].append({"action": "remove_rows", "columns": columns})

        return df

    def _scale_min_max(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        if not columns:
            self.logger.warning("Min-Max scaling skipped: no columns specified.")
            return df

        scaler = MinMaxScaler()
        df[columns] = scaler.fit_transform(df[columns])
        self.logger.info(f"Normalized {len(columns)} columns using Min-Max Scaling.")
        self.scalers["min_max"] = scaler
        self.metadata["steps"].append({"action": "min_max_scaling", "columns": columns})

        return df

    def _encode_label(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        if not columns:
            self.logger.warning("Label encoding skipped: no columns specified.")
            return df

        for col in columns:
            le = LabelEncoder()
            df[col] = df[col].astype("string")
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le

        self.logger.info(f"Encoded {len(columns)} columns using label encoding.")
        self.metadata["steps"].append({"action": "encode", "columns": columns, "type": "label_encoding"})

        return df

    def _encode_one_hot(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        if not columns:
            self.logger.warning("One-hot encoding skipped: no columns specified.")
            return df

        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        one_hot_encoded = encoder.fit_transform(df[columns])
        self.encoders["one_hot"] = encoder

        one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(columns), index=df.index)
        df_encoded = pd.concat([df.drop(columns=columns), one_hot_df], axis=1)

        self.logger.info(f"One-hot encoded {len(columns)} columns.")
        self.metadata["steps"].append({"action": "encode", "columns": columns, "type": "one_hot_encoding"})

        return df_encoded

    def _scale_standard(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        if not columns:
            self.logger.warning("Standard scaling skipped: no columns specified.")
            return df

        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
        self.logger.info(f"Standard scaled {len(columns)} columns.")
        self.scalers["standard"] = scaler
        self.metadata["steps"].append({"action": "standard_scaling", "columns": columns})
        return df

    def _impute(
        self, df: pd.DataFrame, columns: list[str], strategy: Literal["mean", "median", "most_frequent"]
    ) -> pd.DataFrame:
        if not columns:
            self.logger.warning(f"Imputation with strategy '{strategy}' skipped: no columns specified.")
            return df

        imputer = SimpleImputer(strategy=strategy)
        df[columns] = imputer.fit_transform(df[columns])
        self.logger.info(f"Imputed {len(columns)} columns using strategy '{strategy}'.")
        self.metadata["steps"].append({"action": "impute", "columns": columns, "strategy": strategy})
        return df

    def _cast_numeric(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        if not columns:
            self.logger.warning("Cast to numeric skipped: no columns specified.")
            return df

        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                self.logger.warning(f"Column '{col}' not found for casting to numeric.")
        self.logger.info(f"Cast {len(columns)} columns to numeric (coerce errors).")
        self.metadata["steps"].append({"action": "cast_numeric", "columns": columns})
        return df
