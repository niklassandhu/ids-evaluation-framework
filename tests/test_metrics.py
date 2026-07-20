from plugin_static_metric.pr_auc_metric import PrAucMetric
from plugin_static_metric.robustness_index_metric import RobustnessIndexMetric


def test_pr_auc_average_precision():
    metric = PrAucMetric()
    metric._static_metric_prepare()
    result = metric._static_metric_calculate(
        {"test_y_true": [0, 0, 1, 1], "test_y_proba": [0.1, 0.4, 0.35, 0.8]},
        is_multiclass=False,
    )
    assert result["test_pr_auc"] == 0.83333


def test_robustness_index_normalized_area():
    metric = RobustnessIndexMetric()
    result = metric._static_metric_calculate(
        {
            "robustness_curve": [
                {"epsilon": 0.0, "accuracy": 1.0},
                {"epsilon": 0.1, "accuracy": 0.5},
                {"epsilon": 0.2, "accuracy": 0.5},
            ]
        },
        is_multiclass=False,
    )
    assert result["test_robustness_index"] == 0.625
