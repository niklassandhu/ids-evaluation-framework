import hashlib
import logging
from pathlib import Path
from typing import Any

import yaml
from colorlog.escape_codes import parse_colors
from pydantic import ValidationError

from ids_eval.dto.run_config import RunConfig


class ConfigManager:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)

    def load(self) -> RunConfig:
        self.logger.info(f"Loading config: {self.config_path}")

        data = self._load_yaml()

        try:
            config = RunConfig.model_validate(data)
            config_hash = self._compute_config_hash(config)
            config.set_config_file_hash(config_hash)

            self.logger.info("Successfully loaded and validated config.")
            self.logger.info(
                f"{parse_colors('bold_cyan')}Your config hash is: {parse_colors('bold_green')}{config_hash}{parse_colors('reset')}"
            )
            return config
        except ValidationError as e:
            # Convert Pydantic validation errors to user-friendly messages
            raise ValueError(self._format_validation_error(e)) from e

    def _load_yaml(self) -> dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in config: {exc}") from exc

        if not isinstance(loaded, dict):
            raise ValueError("Top-level YAML must be a mapping (dict)")

        return loaded

    @staticmethod
    def _compute_config_hash(config: RunConfig) -> str:
        content = yaml.dump(
            config.model_dump(mode="json", exclude_none=True, exclude_unset=True),
            sort_keys=True,
            default_flow_style=False,
            allow_unicode=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:8]

    @staticmethod
    def _format_validation_error(error: ValidationError) -> str:
        messages = []
        for err in error.errors():
            location = ".".join(str(loc) for loc in err["loc"])
            msg = err["msg"]
            messages.append(f"  - {location}: {msg}")

        return "Configuration validation failed:\n" + "\n".join(messages)

    @staticmethod
    def get_report_directory(config: RunConfig) -> Path:
        base_path = Path(config.general.report_path)
        config_hash = config.get_config_file_hash()
        if config_hash:
            report_dir = base_path / config_hash
        else:
            report_dir = base_path
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir

    @staticmethod
    def get_processed_data_directory(config: RunConfig) -> Path:
        base_path = Path(config.general.processed_data_path)
        config_hash = config.get_config_file_hash()
        if config_hash:
            data_dir = base_path / config_hash
        else:
            data_dir = base_path
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    def store_config(self, config: RunConfig) -> None:
        report_dir = self.get_report_directory(config)
        config_file = report_dir / "config.yaml"
        with config_file.open("w", encoding="utf-8") as f:
            yaml.dump(
                config.model_dump(mode="json", exclude_none=True, exclude_unset=True),
                f,
                default_flow_style=False,
                sort_keys=True,
                allow_unicode=True,
            )
        self.logger.info(f"Stored current config to: {config_file}")
