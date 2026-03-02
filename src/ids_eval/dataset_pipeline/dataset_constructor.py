import gc
import logging
import os

import pandas as pd

from ids_eval.dto.data_manager_config import DatasetConfig, SubfileConfig
from ids_eval.dto.run_config import RunConfig
from ids_eval.enumeration.internal_label import InternalLabel
from ids_eval.exception.no_data_loaded import NoDataLoaded
from ids_eval.exception.pcap_needs_conversion import PcapNeedsConversion


class DatasetConstructor:
    def __init__(self, config: RunConfig):
        self.config: RunConfig = config
        self.metadata = {"sources": [], "steps": []}
        self.logger = logging.getLogger(__name__)

    def _load_single_file(self, file_path: str, use_pyarrow: bool = False) -> pd.DataFrame:
        _, extension = os.path.splitext(file_path)
        if extension == ".csv":
            if use_pyarrow:
                try:
                    # PyArrow backend: faster loading and lower memory usage
                    df = pd.read_csv(file_path, engine="pyarrow", dtype_backend="pyarrow")
                    self.logger.debug(f"Loaded '{file_path}' using PyArrow backend")
                except ImportError:
                    self.logger.warning(
                        "PyArrow not available, falling back to default pandas CSV reader. "
                        "Install pyarrow for better performance: uv add pyarrow"
                    )
                    df = pd.read_csv(file_path, low_memory=False)
                except Exception as e:
                    self.logger.warning(f"PyArrow loading failed ({e}), falling back to default pandas reader")
                    df = pd.read_csv(file_path, low_memory=False)
            else:
                df = pd.read_csv(file_path, low_memory=False)
        elif extension == ".parquet":
            df = pd.read_parquet(file_path)
        elif extension == ".pcap":
            self.logger.error("PCAP file detected while importing dataset. Please convert it before proceeding.")
            raise PcapNeedsConversion
        else:
            self.logger.warning(f"Unsupported file format: {extension}. Skipping file: {file_path}")
            return pd.DataFrame()

        df.columns = df.columns.str.strip()
        return df

    def _apply_label_column(self, df: pd.DataFrame, subfile: SubfileConfig) -> pd.DataFrame:
        """Extracts labels from a CSV column and sets attack_category and target_label."""
        label_col = subfile.label_column

        if label_col not in df.columns:
            available_cols = df.columns.tolist()
            raise ValueError(f"Label column '{label_col}' not found in file. " f"Available columns: {available_cols}")

        df[label_col] = df[label_col].astype(str).str.strip()

        if subfile.label_mapping:
            df[InternalLabel.ATTACK_CATEGORY.value] = df[label_col].map(lambda x: subfile.label_mapping.get(x, x))
        else:
            df[InternalLabel.ATTACK_CATEGORY.value] = df[label_col]

        benign_set = set(subfile.benign_labels or [])
        df[InternalLabel.TARGET_LABEL.value] = df[label_col].apply(lambda x: 0 if x in benign_set else 1)

        label_counts = df[InternalLabel.ATTACK_CATEGORY.value].value_counts()
        self.logger.info(f"  > Label distribution: {label_counts.to_dict()}")

        return df

    def __load_df_and_set_target(
        self, file_path: str, subfile: SubfileConfig, use_pyarrow: bool = False
    ) -> pd.DataFrame:
        df = self._load_single_file(file_path, use_pyarrow=use_pyarrow)
        if df.empty:
            return df

        if subfile.label_column:
            # Dynamic mode: read labels from CSV column
            df = self._apply_label_column(df, subfile)
            self.logger.info(
                f"  > Loaded '{file_path}' with labels from column '{subfile.label_column}' ({df.shape[0]} rows)"
            )
        else:
            # Static mode: use attack_type from config
            df[InternalLabel.ATTACK_CATEGORY.value] = subfile.attack_type
            if subfile.is_benign is not None:
                df[InternalLabel.TARGET_LABEL.value] = int(not subfile.is_benign)
            else:
                df[InternalLabel.TARGET_LABEL.value] = int(subfile.attack_type.lower() != "benign")
            self.logger.info(f"  > Loaded and labeled '{file_path}' as '{subfile.attack_type}' ({df.shape[0]} rows)")

        return df

    def construct(self) -> list[pd.DataFrame]:
        if not self.config.data_manager:
            self.logger.error("No datasets configured to load.")
            raise NoDataLoaded("No datasets configured to load.")
        config_datasets: list[DatasetConfig] = self.config.data_manager.dataset
        constructed_datasets = []
        for i, dataset in enumerate(config_datasets):
            self.logger.info(f"Constructing {dataset.name} dataset.")

            constructed_subfiles = []
            use_pyarrow = dataset.constructor.use_pyarrow
            if use_pyarrow:
                self.logger.info("Using PyArrow backend for CSV loading")

            for subfile in dataset.constructor.subfiles:
                full_path = os.path.join(dataset.constructor.base_path, subfile.subpath)

                if not os.path.exists(full_path):
                    self.logger.warning(f"Path '{full_path}' does not exist. Skipping.")
                    continue

                if os.path.isdir(full_path):
                    # look recursively for files in the directory
                    for root, _, files in os.walk(full_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            loaded_df = self.__load_df_and_set_target(file_path, subfile, use_pyarrow=use_pyarrow)
                            constructed_subfiles.append(loaded_df)
                else:
                    loaded_df = self.__load_df_and_set_target(full_path, subfile, use_pyarrow=use_pyarrow)
                    constructed_subfiles.append(loaded_df)

                del loaded_df

            if not constructed_subfiles:
                self.logger.error("No data loaded. Please check the configuration and paths.")
                raise NoDataLoaded(f"No data loaded for subfile: {str(dataset.constructor.subfiles)}")

            # Concatenate all subfiles into a single dataset
            dataset_df = pd.concat(constructed_subfiles, ignore_index=True, copy=False)

            # Free memory from individual subfile DataFrames
            del constructed_subfiles
            gc.collect()

            # Apply feature mapping if configured (for cross-dataset evaluation)
            # This also drops all columns NOT in the mapping to reduce memory and ensure consistency
            feature_mapping = self.config.data_manager.dataset[i].constructor.feature_mapping
            if feature_mapping:
                # Columns to keep: mapped features + internal labels
                internal_labels = InternalLabel.__values__()
                columns_to_keep = list(feature_mapping.keys()) + internal_labels
                columns_to_keep = [col for col in columns_to_keep if col in dataset_df.columns]

                # Calculate how many columns we need to drop
                original_cols = len(dataset_df.columns)
                dataset_df = dataset_df[columns_to_keep]
                dropped_cols = original_cols - len(dataset_df.columns)

                # Rename the mapped columns
                dataset_df = dataset_df.rename(columns=feature_mapping)

                self.logger.info(
                    f"Applied feature mapping: kept {len(feature_mapping)} features, " f"dropped {dropped_cols} columns"
                )
                self.metadata["steps"].append(
                    {
                        "action": "Feature mapping",
                        "kept_features": len(feature_mapping),
                        "dropped_columns": dropped_cols,
                        "mappings": feature_mapping,
                    }
                )
                gc.collect()

            self.logger.info(f"Dataset loaded. Total size with cleaning: {str(dataset_df.shape)}")
            self.metadata["steps"].append(
                {
                    "action": "Loaded Dataset:" + self.config.data_manager.dataset[i].name,
                    "output_shape": str(dataset_df.shape),
                }
            )

            constructed_datasets.append(dataset_df)

        self.logger.info(f"Total datasets loaded: {len(constructed_datasets)}")

        return constructed_datasets
