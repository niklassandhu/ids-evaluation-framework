from typing import Any

import numpy as np
import pandas as pd

from ids_eval.interface.abstract_adversarial_attack import AbstractAdversarialAttack


class JunkDataAttack(AbstractAdversarialAttack):
    def __init__(self) -> None:
        super().__init__()
        self.target_features: list[str] = []
        self.amount: float = 1024.0
        self.mode: str = "constant"
        self.random_min: float = 0.0
        self.random_max: float = 1024.0
        self.percentage: float = 0.1
        self.clip_min: float | None = None
        self.clip_max: float | None = None
        self.seed: int | None = None

    @property
    def name(self) -> str:
        mode = self.mode.capitalize()
        return f"Junk Data ({mode})"

    @property
    def requires_gradients(self) -> bool:
        return False

    def _attack_deploy(
        self, model: Any | None, model_type: str, params: dict[str, Any], scaler: Any | None = None
    ) -> None:
        self.target_features = params.get("target_features", [])
        self.amount = params.get("amount", 1024.0)
        self.mode = params.get("mode", "constant")
        self.random_min = params.get("random_min", 0.0)
        self.random_max = params.get("random_max", 1024.0)
        self.percentage = params.get("percentage", 0.1)
        self.clip_min = params.get("clip_min", None)
        self.clip_max = params.get("clip_max", None)
        self.seed = params.get("seed", None)

        if self.seed is not None:
            np.random.seed(self.seed)

        if not self.target_features:
            self.logger.warning("No target_features specified. Attack will have no effect.")

        self.logger.info(
            f"Configured Junk Data attack: mode={self.mode}, " f"targets={len(self.target_features)} features"
        )

    def _attack_generate(self, x_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        x_adv = x_test.copy()
        modified_features = []

        for feature in self.target_features:
            if feature not in x_adv.columns:
                self.logger.warning(f"Target feature '{feature}' not found in data, skipping")
                continue

            modified_features.append(feature)

            if self.mode == "constant":
                x_adv[feature] = x_adv[feature] + self.amount

            elif self.mode == "random":
                junk = np.random.uniform(self.random_min, self.random_max, len(x_adv))
                x_adv[feature] = x_adv[feature] + junk

            elif self.mode == "percentage":
                x_adv[feature] = x_adv[feature] * (1 + self.percentage)

            else:
                self.logger.warning(f"Unknown mode '{self.mode}', using constant")
                x_adv[feature] = x_adv[feature] + self.amount

        # Apply clipping if specified
        if self.clip_min is not None or self.clip_max is not None:
            for feature in modified_features:
                if self.clip_min is not None:
                    x_adv[feature] = x_adv[feature].clip(lower=self.clip_min)
                if self.clip_max is not None:
                    x_adv[feature] = x_adv[feature].clip(upper=self.clip_max)

        self.logger.info(f"Modified {len(modified_features)} features with Junk Data attack")

        return x_adv
