from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from ids_eval.dto.adversarial_config import AdversarialAttacksConfig
from ids_eval.dto.metric_config import MetricPluginConfig


class SignatureModelConfig(BaseModel):
    plugin: str = Field(..., description="Name of the plugin to use")
    params: dict[str, Any] | None = Field(default=None, description="Model parameters")
    save_model: bool = Field(default=False, description="Whether to save the model after training")
    model_path: str | None = Field(default=None, description="Custom path/name for the saved model (optional)")


class MlModelConfig(BaseModel):
    plugin: str = Field(..., description="Name of the plugin to use")
    params: dict[str, Any] | None = Field(default=None, description="Model parameters")
    save_model: bool = Field(default=False, description="Whether to save the model after training")
    model_path: str | None = Field(default=None, description="Custom path/name for the saved model (optional)")


class EvaluationGeneralConfig(BaseModel):
    is_multiclass: bool = Field(..., description="Enable multiclass evaluation")


class EvaluationConfig(BaseModel):
    general: EvaluationGeneralConfig = Field(default_factory=EvaluationGeneralConfig)
    anomaly_models: list[MlModelConfig] | None = Field(
        default=None, description="List of anomaly detection models to evaluate"
    )
    signature_models: list[SignatureModelConfig] | None = Field(
        default=None, description="List of signature-based models to evaluate"
    )
    static_metrics: list[MetricPluginConfig] | None = Field(
        default=None,
        description="List of static metrics to calculate. "
        "Each entry specifies plugin name and optional params. "
        "If not specified, all available plugins are loaded.",
    )
    runtime_metrics: list[MetricPluginConfig] | None = Field(
        default=None,
        description="List of runtime metrics to collect. "
        "Each entry specifies plugin name and optional params. "
        "If not specified, all available plugins are loaded.",
    )
    adversarial_attacks: AdversarialAttacksConfig | None = Field(
        default=None,
        description="Configuration for adversarial attack evaluation. "
        "If enabled, models will be tested against adversarial samples.",
    )

    @model_validator(mode="after")
    def validate_at_least_one_model_type(self) -> "EvaluationConfig":
        """Ensure at least one model type is configured."""
        if self.anomaly_models is None and self.signature_models is None:
            raise ValueError("Evaluation config must contain 'anomaly_models' or 'signature_models'")
        if self.anomaly_models is not None and len(self.anomaly_models) == 0:
            raise ValueError("anomaly_models must contain at least one model")
        if self.signature_models is not None and len(self.signature_models) == 0:
            raise ValueError("signature_models must contain at least one model")
        return self
