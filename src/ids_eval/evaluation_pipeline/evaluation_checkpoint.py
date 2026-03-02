from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any


class EvaluationCheckpointStore:
    """Manages per-step checkpointing for the evaluation pipeline."""

    # Valid pipeline stages in order of progression
    STAGE_EVALUATION_IN_PROGRESS = "evaluation_in_progress"
    STAGE_EVALUATION_COMPLETE = "evaluation_complete"
    STAGE_METRICS_COMPLETE = "metrics_complete"
    STAGE_COMPLETE = "complete"

    def __init__(self, checkpoint_path: Path):
        self._path = checkpoint_path
        self._logger = logging.getLogger(__name__)
        self._stage: str = self.STAGE_EVALUATION_IN_PROGRESS
        self._completed_train: dict[str, list[dict[str, Any]]] = {}
        self._completed_test: dict[str, list[dict[str, Any]]] = {}
        self._calculated_metrics: dict[str, Any] | None = None

    def save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "wb") as f:
                pickle.dump(
                    {
                        "stage": self._stage,
                        "completed_train": self._completed_train,
                        "completed_test": self._completed_test,
                        "calculated_metrics": self._calculated_metrics,
                    },
                    f,
                )
        except Exception as e:
            self._logger.warning(f"Failed to save checkpoint to {self._path}: {e}")

    @classmethod
    def load(cls, path: Path) -> EvaluationCheckpointStore:
        store = cls(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        store._stage = data.get("stage", cls.STAGE_EVALUATION_IN_PROGRESS)
        store._completed_train = data.get("completed_train", {})
        store._completed_test = data.get("completed_test", {})
        store._calculated_metrics = data.get("calculated_metrics", None)
        return store

    @classmethod
    def load_or_create(cls, path: Path) -> EvaluationCheckpointStore:
        if path.exists():
            try:
                store = cls.load(path)
                logger = logging.getLogger(__name__)
                n_train = len(store._completed_train)
                n_test = len(store._completed_test)
                logger.info(
                    f"Checkpoint loaded: stage='{store._stage}', "
                    f"{n_train} train steps, {n_test} test steps completed."
                )
                return store
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to load checkpoint from {path}: {e}. Starting fresh.")
        return cls(path)

    def clear(self) -> None:
        if self._path.exists():
            self._path.unlink()
        self._stage = self.STAGE_EVALUATION_IN_PROGRESS
        self._completed_train = {}
        self._completed_test = {}
        self._calculated_metrics = None
        self._logger.info("Checkpoint cleared.")

    @property
    def stage(self) -> str:
        return self._stage

    def set_stage(self, stage: str) -> None:
        self._stage = stage
        self.save()
        self._logger.info(f"Checkpoint stage updated to '{stage}'.")

    def is_train_completed(self, run_id: str) -> bool:
        return run_id in self._completed_train

    def get_train_metrics(self, run_id: str) -> list[dict[str, Any]]:
        return self._completed_train.get(run_id, [])

    def save_train_step(self, run_id: str, metrics: list[dict[str, Any]]) -> None:
        self._completed_train[run_id] = metrics
        self.save()

    def is_test_completed(self, run_id: str) -> bool:
        return run_id in self._completed_test

    def get_test_metrics(self, run_id: str) -> list[dict[str, Any]]:
        return self._completed_test.get(run_id, [])

    def save_test_step(self, run_id: str, metrics: list[dict[str, Any]]) -> None:
        self._completed_test[run_id] = metrics
        self.save()

    def get_all_train_metrics(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for metrics in self._completed_train.values():
            result.extend(metrics)
        return result

    def get_all_test_metrics(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for metrics in self._completed_test.values():
            result.extend(metrics)
        return result

    def save_calculated_metrics(self, data: dict[str, Any]) -> None:
        self._calculated_metrics = data
        self.save()

    def get_calculated_metrics(self) -> dict[str, Any] | None:
        return self._calculated_metrics
