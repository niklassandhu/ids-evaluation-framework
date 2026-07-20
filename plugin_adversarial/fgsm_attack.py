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

        if self.eps <= 0:
            raise ValueError(
                "FGSM requires eps > 0. "
                "Do not generate adversarial samples with epsilon=0. "
                "Compute clean accuracy separately and add it as "
                "the epsilon=0 point in the robustness curve."
            )

        eps_step = params.get("eps_step", self.eps)

        # requires eps_step > 0
        if eps_step <= 0:
            raise ValueError(
                "FGSM requires eps_step > 0."
            )

        self.art_classifier = self._create_art_classifier(
            model,
            model_type,
            params,
        )

        self.attack = FastGradientMethod(
            estimator=self.art_classifier,
            eps=self.eps,
            eps_step=eps_step,
            targeted=params.get("targeted", False),
            num_random_init=params.get("num_random_init", 0),
            batch_size=params.get("batch_size", 32),
        )

        self.logger.info(
            f"Initialized FGSM attack with eps={self.eps}, "
            f"eps_step={eps_step}"
        )

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

        #ART expects one probability column per class
        out_units = None
        if hasattr(model, "output_shape") and model.output_shape is not None:
            out_units = model.output_shape[-1]

        if out_units == 1:
            p = model.outputs[0]  # (batch, 1) sigmoid probability P(class 1)
            two_col = tf.keras.layers.Concatenate(axis=-1)([1.0 - p, p])
            model = tf.keras.Model(inputs=model.inputs, outputs=two_col)
            self.logger.info(
                "FGSM: wrapped single-sigmoid model to 2-column [1-p, p] output for correct gradients"
            )

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
