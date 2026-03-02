from typing import Any

import numpy as np
import pandas as pd

from ids_eval.interface.abstract_adversarial_attack import AbstractAdversarialAttack


class FgsmAttack(AbstractAdversarialAttack):
    def __init__(self) -> None:
        super().__init__()
        self.art_classifier: Any = None
        self.attack: Any = None
        self.eps: float = 0.1
        self.clip_min: float = 0.0
        self.clip_max: float = 1.0
        self.scaler: Any = None

    @property
    def name(self) -> str:
        return "FGSM"

    @property
    def requires_gradients(self) -> bool:
        return True

    def _attack_deploy(
        self,
        model: Any | None,
        model_type: str,
        params: dict[str, Any],
        scaler: Any | None = None,
    ) -> None:
        from art.attacks.evasion import FastGradientMethod

        self.eps = params.get("eps", 0.1)
        self.clip_min = params.get("clip_min", 0.0)
        self.clip_max = params.get("clip_max", 1.0)
        self.scaler = scaler

        self.art_classifier = self._create_art_classifier(model, model_type, params)

        self.attack = FastGradientMethod(
            estimator=self.art_classifier,
            eps=self.eps,
            eps_step=params.get("eps_step", self.eps),
            targeted=params.get("targeted", False),
            num_random_init=params.get("num_random_init", 0),
            batch_size=params.get("batch_size", 32),
        )

        self.logger.info(f"Initialized FGSM attack with eps={self.eps}")

    def _create_art_classifier(
        self,
        model: Any,
        model_type: str,
        params: dict[str, Any],
    ) -> Any:
        if model_type == "tensorflow":
            return self._create_tensorflow_classifier(model, params)
        elif model_type == "sklearn":
            return self._create_sklearn_classifier(model, params)
        else:
            raise ValueError(
                f"Unsupported model type for FGSM: {model_type}. "
                "FGSM requires gradient access (tensorflow or sklearn with surrogate)."
            )

    def _create_tensorflow_classifier(self, model: Any, params: dict[str, Any]) -> Any:
        import tensorflow as tf
        from art.estimators.classification import TensorFlowV2Classifier

        # Get input shape from model or params
        if hasattr(model, "input_shape") and model.input_shape is not None:
            input_shape = model.input_shape[1:]  # Exclude batch dimension
        else:
            input_shape = params.get("input_shape", (params.get("nb_classes", 2),))

        nb_classes = params.get("nb_classes", 2)
        loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        classifier = TensorFlowV2Classifier(
            model=model,
            nb_classes=nb_classes,
            input_shape=input_shape,
            clip_values=(self.clip_min, self.clip_max),
            loss_object=loss_object,
        )

        return classifier

    def _create_sklearn_classifier(self, model: Any, params: dict[str, Any]) -> Any:
        from art.estimators.classification import SklearnClassifier

        classifier = SklearnClassifier(
            model=model,
            clip_values=(self.clip_min, self.clip_max),
        )

        return classifier

    def _attack_generate(self, x_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        if self.scaler is not None:
            x_input = self.scaler.transform(x_test.values)
        else:
            x_input = x_test.values.astype(np.float32)

        x_adv = self.attack.generate(x=x_input)

        if self.scaler is not None:
            x_adv = self.scaler.inverse_transform(x_adv)

        x_adv = np.clip(x_adv, self.clip_min, self.clip_max)

        return pd.DataFrame(x_adv, columns=x_test.columns, index=x_test.index)
