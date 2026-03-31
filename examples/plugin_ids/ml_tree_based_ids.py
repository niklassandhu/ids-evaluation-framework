from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from ids_eval.interface.abstract_ids_connector import AbstractIDSConnector


class MlTreeBasedIds(AbstractIDSConnector):
    """
    Tree-Based Ensemble IDS using stacking of 4 base learners.
    Based on: L. Yang, A. Moubayed, I. Hamieh, und A. Shami,
    „Tree-Based Intelligent Intrusion Detection System in Internet of Vehicles“,
    in 2019 IEEE Global Communications Conference (GLOBECOM),
    Waikoloa, HI, USA: IEEE, Dez. 2019, S. 1–6. doi: 10.1109/GLOBECOM38437.2019.9013892.

    Architecture:
    - Base learners: Decision Tree, Random Forest, Extra Trees, XGBoost
    - Meta-learner: XGBoost (stacking)
    """

    def __init__(self):
        super().__init__()
        self.dt = None
        self.rf = None
        self.et = None
        self.xgb_base = None

        self.stacking_model = None
        self.le = preprocessing.LabelEncoder()
        self.params = {}

    def _ids_deploy(self, params: dict[str, Any]) -> None:
        self.params = params
        self.logger.info(f"Deploying Tree-Based IDS with params: {params}")

    def _ids_prepare(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        n_samples = x_train.shape[0]
        n_features = x_train.shape[1]

        self.logger.info(f"Training Tree-Based IDS on {n_samples} samples with {n_features} features...")

        n_estimators_rf = self.params.get("n_estimators_rf", 100)
        n_estimators_et = self.params.get("n_estimators_et", 100)
        n_estimators_xgb_base = self.params.get("n_estimators_xgb_base", 10)
        n_estimators_xgb_stack = self.params.get("n_estimators_xgb_stack", 100)
        max_depth = self.params.get("max_depth", None)
        random_state = self.params.get("random_state", 0)

        y = self.le.fit_transform(y_train)
        X = x_train

        self.logger.info("Training base learners...")

        self.dt = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        self.dt.fit(X, y)

        self.logger.info("Decision Tree trained...")

        self.rf = RandomForestClassifier(
            n_estimators=n_estimators_rf, max_depth=max_depth, random_state=random_state, n_jobs=-1  # Use all CPU cores
        )
        self.rf.fit(X, y)
        self.logger.info(f"Random Forest trained with {n_estimators_rf} trees")

        self.et = ExtraTreesClassifier(
            n_estimators=n_estimators_et, max_depth=max_depth, random_state=random_state, n_jobs=-1
        )
        self.et.fit(X, y)
        self.logger.info(f"Extra Trees trained with {n_estimators_et} trees")

        self.xgb_base = xgb.XGBClassifier(
            n_estimators=n_estimators_xgb_base, max_depth=max_depth, random_state=random_state, n_jobs=-1, verbosity=0
        )
        self.xgb_base.fit(X, y)
        self.logger.info(f"XGBoost base learner trained with {n_estimators_xgb_base} estimators")

        self.logger.info("Generating meta-features for stacking...")
        meta_features_train = np.column_stack(
            [self.dt.predict(X), self.rf.predict(X), self.et.predict(X), self.xgb_base.predict(X)]
        )

        self.logger.info("Training stacking meta-learner...")
        self.stacking_model = xgb.XGBClassifier(
            n_estimators=n_estimators_xgb_stack, random_state=random_state, n_jobs=-1, verbosity=0
        )
        self.stacking_model.fit(meta_features_train, y)
        self.logger.info(f"Stacking model trained with {n_estimators_xgb_stack} estimators")

        self.logger.info("Training complete!")

    def _ids_detect(self, x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n_samples = x_test.shape[0]
        self.logger.info(f"Detecting on {n_samples} samples...")

        dt_pred = self.dt.predict(x_test)
        rf_pred = self.rf.predict(x_test)
        et_pred = self.et.predict(x_test)
        xgb_pred = self.xgb_base.predict(x_test)

        meta_features_test = np.column_stack([dt_pred, rf_pred, et_pred, xgb_pred])

        y_pred = self.stacking_model.predict(meta_features_test)
        y_pred = self.le.inverse_transform(y_pred)
        y_proba_full = self.stacking_model.predict_proba(meta_features_test)

        self.logger.info(
            f"Detection complete. Predictions shape: {y_pred.shape}, Probabilities shape: {y_proba_full.shape}"
        )

        return y_pred, y_proba_full

    def _ids_save(self, path: Path) -> None:
        compression_level = 3
        joblib.dump(self.dt, path / "dt.joblib", compress=compression_level)
        joblib.dump(self.rf, path / "rf.joblib", compress=compression_level)
        joblib.dump(self.et, path / "et.joblib", compress=compression_level)
        joblib.dump(self.xgb_base, path / "xgb_base.joblib", compress=compression_level)
        joblib.dump(self.stacking_model, path / "stacking_model.joblib", compress=compression_level)
        joblib.dump(self.le, path / "label_encoder.joblib", compress=compression_level)
        joblib.dump(self.params, path / "params.joblib", compress=compression_level)

    def _ids_load(self, path: Path) -> bool:
        required_files = [
            "dt.joblib",
            "rf.joblib",
            "et.joblib",
            "xgb_base.joblib",
            "stacking_model.joblib",
            "label_encoder.joblib",
            "params.joblib",
        ]

        for file_name in required_files:
            if not (path / file_name).exists():
                return False

        self.dt = joblib.load(path / "dt.joblib")
        self.rf = joblib.load(path / "rf.joblib")
        self.et = joblib.load(path / "et.joblib")
        self.xgb_base = joblib.load(path / "xgb_base.joblib")
        self.stacking_model = joblib.load(path / "stacking_model.joblib")
        self.le = joblib.load(path / "label_encoder.joblib")
        self.params = joblib.load(path / "params.joblib")

        return True
