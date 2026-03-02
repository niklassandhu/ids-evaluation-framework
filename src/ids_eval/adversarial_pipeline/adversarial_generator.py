from __future__ import annotations

import gc
import logging
from typing import Any

import numpy as np
import pandas as pd

from ids_eval.dto.adversarial_config import AdversarialAttackPluginConfig
from ids_eval.dto.run_config import RunConfig
from ids_eval.interface.abstract_adversarial_attack import AbstractAdversarialAttack
from ids_eval.interface.abstract_ids_connector import AbstractIDSConnector
from ids_eval.registry.adversarial_attack_registry import AdversarialAttackRegistry


class AdversarialGenerator:
    def __init__(self, config: RunConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.registry = AdversarialAttackRegistry(config)
        self._surrogate_model: Any = None
        self._surrogate_input_shape: tuple[int, ...] | None = None
        self._surrogate_nb_classes: int | None = None

    def is_enabled(self) -> bool:
        if self.config.evaluation.adversarial_attacks is None:
            return False
        return self.config.evaluation.adversarial_attacks.enabled

    def generate_adversarial_samples(
        self,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        ids_model: AbstractIDSConnector,
        x_train: pd.DataFrame | None = None,
        y_train: pd.Series | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Returns:
            Dictionary mapping attack names to adversarial samples DataFrames.
        """
        if not self.is_enabled():
            return {}

        results: dict[str, pd.DataFrame] = {}

        model_type = self._detect_model_type(ids_model)
        underlying_model = self._extract_model(ids_model)
        scaler = self._extract_scaler(ids_model)

        self.logger.info(f"Detected model type: {model_type}")

        loaded_plugins: list[tuple[AbstractAdversarialAttack, AdversarialAttackPluginConfig]] = (
            self.registry.load_plugins()
        )

        if not loaded_plugins:
            self.logger.warning("No adversarial attack plugins loaded")
            return {}

        for attack, attack_cfg in loaded_plugins:
            try:
                model_to_attack = underlying_model
                model_type_to_use = model_type

                if attack.requires_gradients and model_type not in ["tensorflow", "pytorch"]:
                    # Need a surrogate model for gradient-based attack on a non-differentiable model
                    if self._should_use_surrogate() and x_train is not None and y_train is not None:
                        self.logger.info(f"Creating surrogate model for {attack.name} attack on {model_type} model")
                        model_to_attack = self._get_or_create_surrogate(x_train, y_train)
                        # Surrogate is a TensorFlow/Keras model for gradient support
                        model_type_to_use = "tensorflow"
                    else:
                        self.logger.warning(
                            f"Skipping {attack.name} attack: requires gradients but model type "
                            f"'{model_type}' doesn't support them and surrogate is disabled"
                        )
                        continue
                elif not attack.requires_gradients and model_type in ["xgboost", "custom"]:
                    # For black-box attacks on complex models (like stacking),
                    # use the full IDS connector instead of the internal model
                    # This ensures the entire pipeline is used for predictions
                    model_to_attack = ids_model
                    model_type_to_use = "ids_connector"

                params = attack_cfg.params or {}
                if self._surrogate_nb_classes is not None and model_type_to_use == "tensorflow":
                    params["nb_classes"] = self._surrogate_nb_classes
                else:
                    params["nb_classes"] = len(y_test.unique())
                params["input_shape"] = (x_test.shape[1],)
                params["feature_names"] = list(x_test.columns)

                attack.deploy(model=model_to_attack, model_type=model_type_to_use, params=params, scaler=scaler)

                x_adv = attack.generate(x_test, y_test)
                results[attack.name] = x_adv

                self.logger.info(f"Generated {len(x_adv)} adversarial samples with {attack.name}")
                gc.collect()

            except Exception as e:
                self.logger.error(f"Failed to generate adversarial samples with {attack.name}: {e}")
                continue

        return results

    @staticmethod
    def _detect_model_type(ids_model: AbstractIDSConnector) -> str:
        # Try to get the underlying model
        model = None
        if hasattr(ids_model, "model"):
            model = ids_model.model
        elif hasattr(ids_model, "stacking_model"):
            model = ids_model.stacking_model

        if model is None:
            return "custom"

        # Check for TensorFlow/Keras models
        try:
            import tensorflow as tf

            if isinstance(model, tf.keras.Model):
                return "tensorflow"
        except ImportError:
            pass

        # Check for sklearn models
        if hasattr(model, "fit") and hasattr(model, "predict"):
            if hasattr(model, "predict_proba"):
                return "sklearn"

        return "custom"

    @staticmethod
    def _extract_model(ids_model: AbstractIDSConnector) -> Any:
        if hasattr(ids_model, "model"):
            return ids_model.model

        # For stacking models, try to get the final estimator
        if hasattr(ids_model, "stacking_model"):
            return ids_model.stacking_model

        # Return the connector itself as fallback
        return ids_model

    @staticmethod
    def _extract_scaler(ids_model: AbstractIDSConnector) -> Any:
        if hasattr(ids_model, "scaler"):
            return ids_model.scaler
        return None

    def _should_use_surrogate(self) -> bool:
        if self.config.evaluation is None:
            return False
        if self.config.evaluation.adversarial_attacks is None:
            return False
        return self.config.evaluation.adversarial_attacks.use_surrogate

    def _get_or_create_surrogate(self, x_train: pd.DataFrame, y_train: pd.Series) -> Any:
        if self._surrogate_model is not None:
            return self._surrogate_model

        import tensorflow as tf
        from sklearn.preprocessing import LabelEncoder

        epochs = 50  # default
        if self.config.evaluation is not None and self.config.evaluation.adversarial_attacks is not None:
            epochs = self.config.evaluation.adversarial_attacks.surrogate_epochs

        self.logger.info(f"Training TensorFlow surrogate model with {epochs} epochs...")

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_train.values.ravel())
        nb_classes = len(label_encoder.classes_)

        self._surrogate_nb_classes = nb_classes
        self._surrogate_input_shape = (x_train.shape[1],)

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=self._surrogate_input_shape),
                tf.keras.layers.Dropout(0.01),
                tf.keras.layers.Dense(1024, activation="relu"),
                tf.keras.layers.Dropout(0.01),
                tf.keras.layers.Dense(768, activation="relu"),
                tf.keras.layers.Dropout(0.01),
                tf.keras.layers.Dense(523, activation="relu"),
                tf.keras.layers.Dropout(0.01),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.01),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.01),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

        model.fit(
            x_train.values.astype(np.float32),
            y_encoded,
            epochs=epochs,
            batch_size=256,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=1,
        )

        self.logger.info("Surrogate model training complete")
        self._surrogate_model = model

        return self._surrogate_model
