from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from ids_eval.dto.feature_selection_config import (
    CorrelationThresholdConfig,
    LogisticRegressionConfig,
    VarianceThresholdConfig,
)
from ids_eval.dto.split_benign_train_config import SplitBenignTrainConfig
from ids_eval.dto.split_kfold_config import SplitKFoldConfig
from ids_eval.dto.split_timestamp_config import SplitTimestampConfig
from ids_eval.enumeration.feature_selection_method import FeatureSelectionMethod
from ids_eval.enumeration.preprocess_method import PreprocessMethod
from ids_eval.enumeration.split_method import SplitMethod


class SubfileConfig(BaseModel):
    """Configuration for a single subfile within a dataset."""

    name: str = Field(..., description="Display name for this subfile")
    subpath: str = Field(..., description="Relative path to the file")

    # --- Static mode fields ---
    attack_type: str | None = Field(
        default=None, description="Type of attack in this file (static mode - all rows get this label)"
    )
    is_benign: bool | None = Field(
        default=None, description="Whether this file contains benign traffic (only used with attack_type)"
    )

    # --- Dynamic mode fields ---
    label_column: str | None = Field(
        default=None, description="Column name containing attack labels (dynamic mode - reads labels from CSV)"
    )
    label_mapping: dict[str, str] | None = Field(
        default=None,
        description="Optional mapping from CSV labels to framework labels (e.g., {'FTP-Patator': 'bruteforce'})",
    )
    benign_labels: list[str] | None = Field(
        default=None,
        description="List of label values considered benign (used with label_column to determine target_label)",
    )

    @model_validator(mode="after")
    def validate_label_mode(self) -> "SubfileConfig":
        """Validate that either attack_type OR label_column is specified, but not both."""
        has_attack_type = self.attack_type is not None
        has_label_column = self.label_column is not None

        if not has_attack_type and not has_label_column:
            raise ValueError("Either 'attack_type' or 'label_column' must be specified")

        if has_attack_type and has_label_column:
            raise ValueError("Cannot specify both 'attack_type' and 'label_column' - choose one mode")

        if has_label_column and self.is_benign is not None:
            raise ValueError("'is_benign' can only be used with 'attack_type' mode, not 'label_column' mode")

        if has_label_column and not self.benign_labels:
            raise ValueError("'benign_labels' must be specified when using 'label_column' mode")

        if has_attack_type and self.label_mapping is not None:
            raise ValueError("'label_mapping' can only be used with 'label_column' mode")

        return self


class ConstructorConfig(BaseModel):
    base_path: str = Field(..., description="Base directory path for dataset files")
    subfiles: list[SubfileConfig] = Field(..., min_length=1, description="List of subfiles to load")
    feature_mapping: dict[str, str] | None = Field(
        default=None,
        description="Optional mapping to rename columns (e.g., {'old_name': 'new_name'}). "
        "Applied after loading, before preprocessing. Useful for cross-dataset evaluation.",
    )
    use_pyarrow: bool = Field(
        default=False,
        description="Use PyArrow backend for CSV loading. Provides faster loading and lower memory usage. "
        "Requires pyarrow to be installed.",
    )


class PreprocessConfig(BaseModel):
    method: PreprocessMethod
    columns: list[str] = Field(default_factory=list, description="Columns to apply preprocessing to")
    auto_columns: bool = Field(default=False, description="If true, numerical columns will be auto-detected")

    @model_validator(mode="after")
    def validate_columns_requirement(self) -> "PreprocessConfig":
        """Validate that columns are provided when required."""
        methods_requiring_columns = {PreprocessMethod.MIN_MAX, PreprocessMethod.LABEL, PreprocessMethod.ONE_HOT}
        if not self.auto_columns and len(self.columns) == 0 and self.method in methods_requiring_columns:
            raise ValueError(f"columns must be non-empty when auto_columns is false for method '{self.method.value}'")
        return self


# Type alias for feature selector params
type FeatureSelectorParams = (LogisticRegressionConfig | VarianceThresholdConfig | CorrelationThresholdConfig | None)


class FeatureSelectorConfig(BaseModel):
    method: FeatureSelectionMethod
    params: FeatureSelectorParams = Field(default=None, description="Method-specific parameters")

    @model_validator(mode="after")
    def validate_params_for_method(self) -> "FeatureSelectorConfig":
        """Validate that required params are provided for certain methods."""
        if self.method == FeatureSelectionMethod.LOGISTIC_REGRESSION:
            if self.params is None:
                raise ValueError("params required for logistic_regression method")
            if not isinstance(self.params, LogisticRegressionConfig):
                raise ValueError("params must be LogisticRegressionConfig for logistic_regression method")
        return self


# Type alias for split params
type SplitParams = SplitTimestampConfig | SplitKFoldConfig | SplitBenignTrainConfig | None


class SplitConfig(BaseModel):
    method: SplitMethod
    test_size: Annotated[float, Field(gt=0, lt=1, description="Test set size ratio (0 < x < 1)")]
    target_column: str = Field(..., description="Name of the target/label column")
    params: SplitParams = Field(default=None, description="Method-specific parameters")

    @model_validator(mode="after")
    def validate_params_for_method(self) -> "SplitConfig":
        """Validate that required params are provided for certain methods."""
        if self.method == SplitMethod.KFOLDSPLIT:
            if self.params is None or not isinstance(self.params, SplitKFoldConfig):
                raise ValueError("params with n_splits required for k_fold_split method")
        elif self.method == SplitMethod.TIMESTAMP:
            if self.params is None or not isinstance(self.params, SplitTimestampConfig):
                raise ValueError("params with timestamp_column required for timestamp method")
        return self


class DatasetConfig(BaseModel):
    name: str = Field(..., description="Display name for this dataset")
    constructor: ConstructorConfig
    preprocess: list[PreprocessConfig] = Field(default_factory=list, description="List of preprocessing steps.")
    feature_selector: FeatureSelectorConfig


class DataManagerConfig(BaseModel):
    dataset: list[DatasetConfig] = Field(..., min_length=1, description="List of datasets to process.")
    split: SplitConfig
