from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from ids_eval.interface.abstract_ids_connector import AbstractIDSConnector


class MlDecisionTree(AbstractIDSConnector):
    def __init__(self):
        super().__init__()
        self.model: DecisionTreeClassifier | None = None

    def _ids_deploy(self, params: dict[str, Any]) -> None:
        self.model = DecisionTreeClassifier(**params)
        self.logger.info(f"Deployed Decision Tree classifier with params: {params}")

    def _ids_prepare(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model.fit(x_train, y_train.values.ravel())

    def _ids_detect(self, x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        y_pred = self.model.predict(x_test)
        y_proba = self.model.predict_proba(x_test)

        return y_pred, y_proba

    def _ids_save(self, path: Path) -> None:
        model_file = path / "model.joblib"
        joblib.dump(self.model, model_file)
        self.logger.info(f"Saved Decision Tree model to {model_file}")

    def _ids_load(self, path: Path) -> bool:
        model_file = path / "model.joblib"
        if model_file.exists():
            self.model = joblib.load(model_file)
            self.logger.info(f"Loaded Decision Tree model from {model_file}")
            return True
        return False
