from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
from sklearn.metrics import accuracy_score


AttackGenerateFn = Callable[[float], pd.DataFrame]
PredictFn = Callable[[pd.DataFrame], Any]


@dataclass(frozen=True)
class RobustnessPoint:
    epsilon: float
    accuracy: float


class RobustnessSweepRunner:
    def __init__(self, eps_values: list[float]) -> None:
        self.eps_values = [float(eps) for eps in eps_values]
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(
        self,
        y_true: pd.Series,
        attack_generate: AttackGenerateFn,
        predict_fn: PredictFn,
        x_clean: pd.DataFrame | None = None, #ensuring baseline setup
    ) -> list[dict[str, Any]]:
        curve: list[dict[str, Any]] = []

        if x_clean is not None:
            self.logger.info("Robustness sweep: epsilon=0 (clean baseline)")
            clean_acc = accuracy_score(y_true, predict_fn(x_clean))
            curve.append(
                {
                    "epsilon": 0.0,
                    "accuracy": round(float(clean_acc), 5),
                }
            )

        for eps in self.eps_values:
            self.logger.info("Robustness sweep: epsilon=%s", eps)

            x_adv = attack_generate(eps)
            y_pred = predict_fn(x_adv)

            acc = accuracy_score(y_true, y_pred)
            curve.append(
                {
                    "epsilon": round(float(eps), 6),
                    "accuracy": round(float(acc), 5),
                }
            )

        return curve