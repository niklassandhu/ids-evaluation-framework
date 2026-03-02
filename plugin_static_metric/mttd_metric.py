from typing import Any

import numpy as np

from ids_eval.dto.metric_config import MetricMetadata
from ids_eval.interface.abstract_static_metric import AbstractStaticMetric


class MttdMetric(AbstractStaticMetric):
    """
    Mean Time To Detect (MTTD) metric -- measured in samples (packets / CSV rows).

    Measures how many samples elapse between the start of an attack instance and its first correct detection
    by the IDS. Because our framework operates on tabular packet/flow data without guaranteed
    timestamps, the delay is expressed in **number of samples** rather than wall-clock time.

    Formula: MTTD = (1/k) * Σ(idx_first_detection - idx_attack_start)

    Where:
    - k = number of correctly detected attack instances
    - idx_first_detection = index of the first True Positive within the instance
    - idx_attack_start = index of the first sample of the attack instance

    A lower MTTD indicates faster detection of attacks.
    """

    def __init__(self):
        super().__init__()

    def _static_metric_metadata(self, is_multiclass: bool) -> list[MetricMetadata]:
        return [
            MetricMetadata(
                key="test_mttd_samples",
                display_name="MTTD (samples)",
                category="detection",
                unit="samples",
                higher_is_better=False,
                description="Mean samples (packets/rows) from attack start until first correct detection",
            ),
            MetricMetadata(
                key="test_attack_instances_total",
                display_name="Attack Instances (Total)",
                category="detection",
                description="Total number of attack instances in test data",
            ),
            MetricMetadata(
                key="test_attack_instances_detected",
                display_name="Attack Instances (Detected)",
                category="detection",
                higher_is_better=True,
                description="Number of attack instances successfully detected",
            ),
            MetricMetadata(
                key="test_attack_instances_undetected",
                display_name="Attack Instances (Undetected)",
                category="detection",
                higher_is_better=False,
                description="Number of attack instances not detected",
            ),
        ]

    def _static_metric_prepare(self) -> None:
        pass

    def _static_metric_calculate(self, metrics: dict[str, Any], is_multiclass: bool) -> dict[str, Any]:
        y_true = np.asarray(metrics["test_y_true"])
        y_pred = np.asarray(metrics["test_y_pred"])

        # Find all consecutive attack instances
        attack_instances = self._find_attack_instances(y_true)

        detection_delays: list[int] = []
        undetected_count = 0

        for start_idx, end_idx in attack_instances:
            # Find first True Positive within this attack instance
            first_tp_offset = self._find_first_tp(y_true[start_idx : end_idx + 1], y_pred[start_idx : end_idx + 1])
            if first_tp_offset is not None:
                detection_delays.append(first_tp_offset)
            else:
                undetected_count += 1

        # Calculate MTTD
        if detection_delays:
            mttd = float(sum(detection_delays)) / len(detection_delays)
        else:
            mttd = float("inf")

        return {
            "test_mttd_samples": round(mttd, 5),  # Average samples to detect an attack instance
            "test_attack_instances_total": len(attack_instances),
            "test_attack_instances_detected": len(detection_delays),
            "test_attack_instances_undetected": undetected_count,
        }

    @staticmethod
    def _find_attack_instances(y_true: np.ndarray) -> list[tuple[int, int]]:
        instances: list[tuple[int, int]] = []

        # Convert to binary: 0 = benign, 1 = attack
        is_attack = (y_true != 0).astype(int)

        # Find transitions (edges) where attacks start and end
        # Pad with 0 at the beginning and end to detect attacks at boundaries
        padded = np.concatenate([[0], is_attack, [0]])
        diff = np.diff(padded)

        # Start indices: where diff == 1 (transition from 0 to 1)
        starts = np.where(diff == 1)[0]
        # End indices: where diff == -1 (transition from 1 to 0), minus 1 to get last attack index
        ends = np.where(diff == -1)[0] - 1

        for start, end in zip(starts, ends):
            instances.append((int(start), int(end)))

        return instances

    @staticmethod
    def _find_first_tp(y_true_slice: np.ndarray, y_pred_slice: np.ndarray) -> int | None:
        # Convert to binary: 0 = benign, non-zero = attack
        true_attack = y_true_slice != 0
        pred_attack = y_pred_slice != 0

        # Find True Positives (both true and pred indicate attack)
        tp_mask = true_attack & pred_attack
        tp_indices = np.where(tp_mask)[0]

        if len(tp_indices) > 0:
            return int(tp_indices[0])
        return None

    def _static_metric_visualize(self, metrics: dict[str, Any], visual_name_prefix: str, is_multiclass: bool) -> None:
        pass
