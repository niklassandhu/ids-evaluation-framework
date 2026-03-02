from enum import Enum


class PreprocessMethod(Enum):
    MIN_MAX = "min_max"
    STANDARD = "standard"
    LABEL = "label"
    ONE_HOT = "one_hot"

    IMPUTE_MEAN = "impute_mean"
    IMPUTE_MEDIAN = "impute_median"
    IMPUTE_MOST_FREQUENT = "impute_most_frequent"

    CAST_NUMERIC = "cast_numeric"

    REMOVE_DUPLICATE_ROWS = "remove_duplicate_rows"
    REMOVE_NAN_ROWS = "remove_nan_rows"
    REMOVE_SINGLE_VALUE_ROWS = "remove_single_value_rows"
    REMOVE_SINGLE_VALUE_COLUMNS = "remove_single_value_columns"
    REMOVE_ROWS = "remove_rows"
    REMOVE_CLASS = "remove_class"

    NONE = "none"
