from pydantic import BaseModel, Field

from ids_eval.dto.data_manager_config import DataManagerConfig
from ids_eval.dto.evaluation_config import EvaluationConfig


class GeneralConfig(BaseModel):
    name: str = Field(..., description="Name of this evaluation run")
    description: str = Field(..., description="Description of this evaluation run")
    seed: int = Field(..., description="Random seed for reproducibility")
    report_path: str = Field(default="out/reports/", description="Path for output reports")
    processed_data_path: str = Field(
        default="out/processed_datasets/", description="Path for storage processed datasets"
    )
    model_storage_path: str = Field(default="out/saved_models/", description="Base path for storing trained models")
    plugin_adversarial_path: str = Field(
        default="plugin_adversarial/", description="Path to adversarial attack plugins"
    )
    plugin_ids_path: str = Field(default="plugin_ids/", description="Path to IDS model plugins")
    plugin_static_metric_path: str = Field(
        default="plugin_static_metric/", description="Path to static metric calculation plugins"
    )
    plugin_runtime_metric_path: str = Field(
        default="plugin_runtime_metric/", description="Path to runtime metric collection plugins"
    )


class RunConfig(BaseModel):
    general: GeneralConfig
    data_manager: DataManagerConfig | None = None
    evaluation: EvaluationConfig | None = None
    # Internal field set by ConfigManager - hash of the entire config file
    _config_file_hash: str | None = None

    def set_config_file_hash(self, hash_value: str) -> None:
        """Sets the hash of the config file (called by ConfigManager after loading)."""
        object.__setattr__(self, "_config_file_hash", hash_value)

    def get_config_file_hash(self) -> str | None:
        """Returns the hash of the config file."""
        return getattr(self, "_config_file_hash", None)
