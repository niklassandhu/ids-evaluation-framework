from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from keras import Input
from sklearn.preprocessing import MinMaxScaler

from ids_eval.interface.abstract_ids_connector import AbstractIDSConnector


class MlIdsDnn(AbstractIDSConnector):
    """
    Deep Neural Network (DNN) IDS model based on the ICCCNT paper:
    R. Vigneswaran, V. Ravi, S. Kp, und P. Poornachandran,
    „Evaluating Shallow and Deep Neural Networks for Network Intrusion Detection Systems in Cyber Security“,
    Juli 2018, S. 1–6. doi: 10.1109/ICCCNT.2018.8494096.
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.scaler: MinMaxScaler | None = None
        self.params: dict[str, Any] = {}
        self.n_features: int = 0

    def _ids_deploy(self, params: dict[str, Any]) -> None:
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam

        self.Sequential = Sequential
        self.Dense = Dense
        self.Dropout = Dropout
        self.Adam = Adam
        self.scaler = MinMaxScaler()

        # Store hyperparameters with defaults from the paper
        self.params = {
            "epochs": params.get("epochs", 100),
            "batch_size": params.get("batch_size", 256),
            "dropout_rate": params.get("dropout_rate", 0.01),
            "learning_rate": params.get("learning_rate", 0.01),
            "layer_sizes": params.get("layer_sizes", [1024, 768, 512, 256, 128]),
            "verbose": params.get("verbose", 0),
            "validation_split": params.get("validation_split", 0.0),
        }

    def _build_model(self, n_features: int) -> None:
        self.n_features = n_features

        layer_sizes = self.params["layer_sizes"]
        dropout_rate = self.params["dropout_rate"]
        learning_rate = self.params["learning_rate"]

        self.model = self.Sequential()

        # First layer with input dimension
        self.model.add(Input(shape=(n_features,)))
        self.model.add(self.Dropout(dropout_rate))

        # Additional hidden layers with dropout
        for units in layer_sizes[1:]:
            self.model.add(self.Dense(units, activation="relu"))
            self.model.add(self.Dropout(dropout_rate))

        # Output layer with sigmoid for binary classification
        self.model.add(self.Dense(1, activation="sigmoid"))

        # Compile model with Adam optimizer and custom learning rate
        optimizer = self.Adam(learning_rate=learning_rate)
        self.model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    def _ids_prepare(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        x_normalized = self.scaler.fit_transform(x_train.values)
        y_binary = y_train.values.astype(np.float32).ravel()

        n_features = x_train.shape[1]
        self._build_model(n_features)

        self.logger.info(f"Training model on {len(x_train)} samples for {self.params['epochs']} epochs...")

        self.model.fit(
            x_normalized,
            y_binary,
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            validation_split=self.params["validation_split"],
            verbose=self.params["verbose"],
        )

    def _ids_detect(self, x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x_normalized = self.scaler.transform(x_test.values)
        y_proba = self.model.predict(x_normalized, verbose=0).ravel()

        # Convert probabilities to binary predictions (threshold = 0.5) - can also be parametrized
        y_pred = (y_proba >= 0.5).astype(int)

        return y_pred, y_proba

    def _ids_save(self, path: Path) -> None:
        # Save Keras model
        model_file = path / "model.keras"
        self.model.save(model_file)
        self.logger.info(f"Saved Keras model to {model_file}")

        # Save components
        aux_data = {"scaler": self.scaler, "params": self.params, "n_features": self.n_features}
        aux_file = path / "aux_components.joblib"
        joblib.dump(aux_data, aux_file)
        self.logger.info(f"Saved auxiliary components to {aux_file}")

    def _ids_load(self, path: Path) -> bool:
        from tensorflow.keras.models import load_model

        model_file = path / "model.keras"
        aux_file = path / "aux_components.joblib"

        # Check if both files exist
        if not model_file.exists() or not aux_file.exists():
            self.logger.warning(f"Model files not found at {path}")
            return False

        # Load Keras model
        self.model = load_model(model_file)
        self.logger.info(f"Loaded Keras model from {model_file}")

        # Load components
        aux_data = joblib.load(aux_file)
        self.scaler = aux_data["scaler"]
        self.params = aux_data["params"]
        self.n_features = aux_data["n_features"]
        self.logger.info(f"Loaded auxiliary components from {aux_file}")

        return True
