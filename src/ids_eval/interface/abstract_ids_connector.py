import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ids_eval.dto.run_config import RunConfig


class AbstractIDSConnector(ABC):

    def __init__(self):
        self.is_deployed = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config: RunConfig | None = None

    def deploy(self, model_params: dict[str, Any]) -> None:
        if self.is_deployed:
            self.logger.debug("IDS is already deployed. Skipping deployment.")
            return

        try:
            self._ids_deploy(model_params)
        except Exception as e:
            self.logger.error(f"Error in IDS deployment: {e}")
            self.logger.exception(e)
            raise RuntimeError("Error in IDS deployment")
        self.is_deployed = True

    def prepare(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        if not self.is_deployed:
            self.logger.error("IDS is not deployed. Please deploy the IDS first.")
            raise ValueError("IDS is not deployed. Please deploy the IDS first.")

        try:
            self._ids_prepare(x_train, y_train)
        except Exception as e:
            self.logger.error(f"Error in IDS preparation: {e}")
            self.logger.exception(e)
            raise RuntimeError("Error in IDS preparation")

    def detect(self, x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_deployed:
            self.logger.error("IDS is not deployed. Please deploy the IDS first.")
            raise ValueError("IDS is not deployed. Please deploy the IDS first.")
        try:
            output = self._ids_detect(x_test)
        except Exception as e:
            self.logger.error(f"Error in IDS detection: {e}")
            self.logger.exception(e)
            raise RuntimeError("Error in IDS detection")
        return output

    def save(self, path: Path) -> None:
        if not self.is_deployed:
            self.logger.error("IDS is not deployed. Cannot save an undeployed model.")
            raise ValueError("IDS is not deployed. Cannot save an undeployed model.")
        try:
            path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Saving model to {path}...")
            self._ids_save(path)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            self.logger.exception(e)
            raise RuntimeError("Error saving model")

    def load(self, path: Path) -> bool:
        try:
            self.logger.info(f"Loading model from {path}...")
            success = self._ids_load(path)
            if success:
                self.is_deployed = True
                self.logger.info(f"Model loaded from {path}")
            return success
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.logger.exception(e)
            return False

    @abstractmethod
    def _ids_deploy(self, params: dict[str, Any]) -> None:
        """
        Your plugin should perform any necessary setup or initialization here.
        Your plugin MUST set the self.model attribute to the trained IDS model.
        """
        pass

    @abstractmethod
    def _ids_prepare(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        This method will be invoked from the evaluation framework to prepare (train) the IDS.
        Training data is passed and in the format of a list containing two elements:
        X_train: pd.DataFrame - features for training
        y_train: pd.Series - labels for training
        """
        pass

    @abstractmethod
    def _ids_detect(self, x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Testing data is passed and in the format of a list containing two elements:
        X_test: pd.DataFrame - features for testing
        y_test: pd.Series - labels for testing

        Must return a tuple of two elements:
        - 'y_pred': The predicted label of the data point. Either the class label or if binary classification 0 for normal else 1.
        - 'y_proba': The confidence score of the prediction.
        """
        pass

    @abstractmethod
    def _ids_save(self, path: Path) -> None:
        """
        Your plugin should serialize and save all model components to the specified path.
        """
        pass

    @abstractmethod
    def _ids_load(self, path: Path) -> bool:
        """
        Your plugin should deserialize and restore all model components from the specified path.

        Returns:
            True if the model was loaded successfully, False otherwise.
        """
        pass
