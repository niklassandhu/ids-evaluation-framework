from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class AbstractAdversarialAttack(ABC):
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_deployed: bool = False
        self._params: dict[str, Any] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def requires_gradients(self) -> bool:
        return False

    def deploy(self, model: Any | None, model_type: str, params: dict[str, Any], scaler: Any | None = None) -> None:
        self._params = params
        self._attack_deploy(model, model_type, params, scaler)
        self.is_deployed = True
        self.logger.info(f"Deployed {self.name}")

    def generate(self, x_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        if not self.is_deployed:
            raise RuntimeError(f"{self.name} attack has not been deployed. Call deploy() first.")

        self.logger.info(f"Generating adversarial samples with {self.name} for {len(x_test)} samples")
        x_adv = self._attack_generate(x_test, y_test)

        # Ensure output has same structure as input
        if isinstance(x_adv, np.ndarray):
            x_adv = pd.DataFrame(x_adv, columns=x_test.columns, index=x_test.index)

        self.logger.info(f"Generated {len(x_adv)} adversarial samples")
        return x_adv

    @abstractmethod
    def _attack_deploy(
        self, model: Any | None, model_type: str, params: dict[str, Any], scaler: Any | None = None
    ) -> None:
        pass

    @abstractmethod
    def _attack_generate(self, x_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame | np.ndarray:
        pass
