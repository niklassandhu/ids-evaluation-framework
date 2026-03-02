from typing import Any

from ids_eval.dto.metric_config import MetricMetadata
from ids_eval.interface.abstract_static_metric import AbstractStaticMetric

"""
Matthews Correlation Coefficient Metric calculation according to:
B. Matthews, Comparison of the predicted and observed secondary structure
of T4 phage lysozyme, en, Biochimica et Biophysica Acta (BBA) - Protein Structure,
vol. 405, no. 2, 442–451, issn: 00052795. doi: 10 . 1016 / 0005 - 2795(75 ) 90109 - 9.
Accessed: Jan. 21, 2026. [Online]. Available: https://linkinghub.elsevier.com/retrieve/pii/0005279575901099

We use the sklearn.metrics matthews_corrcoef builtin function here.
"""


class MatthewsCorrelationCoefficient(AbstractStaticMetric):
    def __init__(self):
        super().__init__()

    def _static_metric_metadata(self, is_multiclass: bool) -> list[MetricMetadata]:
        return [
            MetricMetadata(
                key="test_matthews_correlation_coefficient",
                display_name="MCC",
                category="detection",
                unit="ratio",
                higher_is_better=True,
                description="Matthews Correlation Coefficient (-1 to 1, 1 is perfect)",
                comparison_group="detection_performance",
                comparison_chart_type="grouped_bar",
            )
        ]

    def _static_metric_prepare(self) -> None:
        from sklearn.metrics import matthews_corrcoef

        self.mcc_fc = matthews_corrcoef

    def _static_metric_calculate(self, metrics: dict[str, Any], is_multiclass: bool) -> dict[str, Any]:
        y_true = metrics["test_y_true"]
        y_pred = metrics["test_y_pred"]
        mcc = self.mcc_fc(y_true, y_pred)

        return {"test_matthews_correlation_coefficient": round(float(max(0.000, mcc)), 5)}

    def _static_metric_visualize(self, metrics: dict[str, Any], visual_name_prefix: str, is_multiclass: bool) -> None:
        pass
