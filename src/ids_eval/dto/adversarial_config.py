from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AdversarialAttackPluginConfig(BaseModel):
    plugin: str = Field(..., description="Plugin name (e.g., 'fgsm_attack', 'noise_attack')")
    params: dict[str, Any] | None = Field(
        default=None, description="Optional parameters to pass to the attack's deploy() method"
    )


class AdversarialAttacksConfig(BaseModel):
    enabled: bool = Field(default=False, description="Whether to enable adversarial attack evaluation")
    attacks: list[AdversarialAttackPluginConfig] = Field(
        default_factory=list, description="List of adversarial attack plugins to run"
    )
    use_surrogate: bool = Field(
        default=True, description="Use surrogate models for non-differentiable classifiers (tree-based)"
    )
    surrogate_epochs: int = Field(default=50, description="Number of epochs to train surrogate model")
