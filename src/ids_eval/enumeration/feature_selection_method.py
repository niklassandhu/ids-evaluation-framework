from enum import Enum


class FeatureSelectionMethod(Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    VARIANCE_THRESHOLD = "variance_threshold"
    CORRELATION_THRESHOLD = "correlation_threshold"
    NONE = "none"
    OTHER = "other"  # Placeholder for future methods
