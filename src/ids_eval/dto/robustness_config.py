from pydantic import BaseModel, Field


class RobustnessConfig(BaseModel):
    enabled: bool = Field(
        default=False,
        description="Enable robustness sweep evaluation",
    )

    eps_values: list[float] = Field(
        default_factory=lambda: [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
        description="Perturbation magnitudes of epsilon, used for robustness sweep (max 0.3 magnitude)",
    )