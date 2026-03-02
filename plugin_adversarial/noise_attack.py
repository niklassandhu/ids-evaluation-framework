from typing import Any

import numpy as np
import pandas as pd

from ids_eval.interface.abstract_adversarial_attack import AbstractAdversarialAttack


class NoiseAttack(AbstractAdversarialAttack):
    def __init__(self) -> None:
        super().__init__()
        self.std: float = 0.1
        self.clip_min: float = 0.0
        self.clip_max: float = 1.0
        self.seed: int | None = None

    @property
    def name(self) -> str:
        return "Gaussian Noise"

    @property
    def requires_gradients(self) -> bool:
        return False

    def _attack_deploy(
        self, model: Any | None, model_type: str, params: dict[str, Any], scaler: Any | None = None
    ) -> None:
        self.std = params.get("std", 0.1)
        self.clip_min = params.get("clip_min", 0.0)
        self.clip_max = params.get("clip_max", 1.0)
        self.seed = params.get("seed", None)

        if self.seed is not None:
            np.random.seed(self.seed)

    def _attack_generate(self, x_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        noise = np.random.normal(0, self.std, x_test.shape)

        x_adv = x_test.values + noise
        x_adv = np.clip(x_adv, self.clip_min, self.clip_max)

        return pd.DataFrame(x_adv, columns=x_test.columns, index=x_test.index)
