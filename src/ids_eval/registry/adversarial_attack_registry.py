from __future__ import annotations

import importlib.util
import inspect
import logging
from pathlib import Path

from ids_eval.dto.adversarial_config import AdversarialAttackPluginConfig
from ids_eval.dto.run_config import RunConfig
from ids_eval.interface.abstract_adversarial_attack import AbstractAdversarialAttack

LoadedAdversarialPlugins = list[tuple[AbstractAdversarialAttack, AdversarialAttackPluginConfig]]


class AdversarialAttackRegistry:
    def __init__(self, config: RunConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.plugin_path = self.config.general.plugin_adversarial_path

    def load_plugins(self) -> LoadedAdversarialPlugins:
        if not self._is_enabled():
            self.logger.info("Adversarial attacks are not enabled in configuration")
            return []

        available_plugins = self._import_plugins()
        loaded_plugins: LoadedAdversarialPlugins = []

        attack_configs = self.config.evaluation.adversarial_attacks.attacks

        for attack_cfg in attack_configs:
            if attack_cfg.plugin is None:
                continue

            # Normalize plugin name for lookup
            plugin_key = attack_cfg.plugin.lower().replace("_", "")

            try:
                AttackClass = available_plugins[plugin_key]
            except KeyError:
                self.logger.error(f"Adversarial attack plugin '{attack_cfg.plugin}' not found. Skipping.")
                continue

            attack_instance: AbstractAdversarialAttack = AttackClass()
            loaded_plugins.append((attack_instance, attack_cfg))
            self.logger.info(f"  > Loaded adversarial attack plugin: {attack_cfg.plugin}")

        return loaded_plugins

    def _is_enabled(self) -> bool:
        if self.config.evaluation.adversarial_attacks is None:
            return False
        if not self.config.evaluation.adversarial_attacks.enabled:
            return False
        if not self.config.evaluation.adversarial_attacks.attacks:
            return False
        return True

    def _import_plugins(self) -> dict[str, type[AbstractAdversarialAttack]]:
        plugin_dir = Path(self.plugin_path)

        if not plugin_dir.is_dir():
            self.logger.error(f"Adversarial plugin directory not found: {self.plugin_path}")
            raise FileNotFoundError(f"Adversarial plugin directory not found: {self.plugin_path}")

        plugins: dict[str, type[AbstractAdversarialAttack]] = {}

        for file in plugin_dir.glob("*.py"):
            if file.name.startswith("_"):
                continue

            module_name = file.stem

            try:
                spec = importlib.util.spec_from_file_location(module_name, file)
                if spec is None or spec.loader is None:
                    self.logger.warning(f"Could not load adversarial plugin {module_name}")
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, AbstractAdversarialAttack) and obj is not AbstractAdversarialAttack:
                        plugin_name = name.lower()
                        plugins[plugin_name] = obj
                        self.logger.debug(f"Discovered adversarial plugin: {name}")

            except Exception as e:
                self.logger.warning(f"Could not load adversarial plugin {module_name}: {e}")

        return plugins
