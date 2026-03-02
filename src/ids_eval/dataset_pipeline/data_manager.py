from __future__ import annotations

import gc
import logging
import pickle

from ids_eval.dataset_pipeline.dataset_analyser import DatasetAnalyser
from ids_eval.dataset_pipeline.dataset_constructor import DatasetConstructor
from ids_eval.dataset_pipeline.dataset_preprocessor import DatasetPreprocessor
from ids_eval.dataset_pipeline.dataset_splitter import DatasetSplitter
from ids_eval.dataset_pipeline.feature_selector import FeatureSelector
from ids_eval.dto.run_config import RunConfig
from ids_eval.run_config_pipeline.config_manager import ConfigManager


class DataManager:
    """Orchestrates the entire data preparation pipeline."""

    def __init__(self, config: RunConfig) -> None:
        self.config: RunConfig = config
        self.logger = logging.getLogger(__name__)

    def run(self) -> None:
        self.logger.info("Starting data preparation pipeline...")

        # --- 1. Dataset Construction ---
        constructor = DatasetConstructor(self.config)
        datasets = constructor.construct()
        self.logger.info(f"Constructed {len(datasets)} dataset(s).")
        gc.collect()

        # --- 2. Preprocessing ---
        preprocessor = DatasetPreprocessor(self.config)
        datasets = preprocessor.preprocess(datasets)
        self.logger.info("Preprocessing complete.")
        gc.collect()

        # --- 3. Feature Selection ---
        selector = FeatureSelector(self.config)
        datasets = selector.select_features(datasets)
        self.logger.info("Feature selection complete.")
        gc.collect()

        # --- 4. Data Splitting ---
        splitter = DatasetSplitter(self.config)
        splits = splitter.split(datasets)
        self.logger.info("Data splitting complete.")
        gc.collect()

        # --- 5. Save Processed Data ---
        self._save_processed_data(splits)

        # --- 6. Analysis and Reporting ---
        analyser = DatasetAnalyser(self.config)
        all_metadata = {
            "config_hash": self.config.get_config_file_hash(),
            "constructor": constructor.metadata,
            "preprocessor": preprocessor.metadata,
            "feature_selector": selector.metadata,
            "splitter": splitter.metadata,
        }
        analyser.report(all_metadata, datasets, splits)
        self.logger.info("Dataset analysis and reporting complete.")

        self.logger.info("Data preparation pipeline finished successfully.")

    def _save_processed_data(self, splits: list) -> None:
        root = ConfigManager.get_processed_data_directory(self.config)
        name = f"{self.config.general.name.replace(' ', '_').lower()}"
        name = name[:200]
        # make name safe for filesystems
        name = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).rstrip()
        name += ".pkl"
        file = root / name[:200]  # Limit filename length to 200 chars to avoid OS issues
        with open(file, "wb") as f:
            pickle.dump(splits, f)
        self.logger.info(f"Processed and split data saved to: {file}")
