from pydantic import BaseModel, Field, model_validator

from ids_eval.enumeration.logistic_regression_penalty import LogisticRegressionPenalty
from ids_eval.enumeration.logistic_regression_solver import LogisticRegressionSolver


class LogisticRegressionConfig(BaseModel):
    penalty: LogisticRegressionPenalty
    C: float = Field(..., gt=0, description="Regularization parameter (must be > 0)")
    solver: LogisticRegressionSolver
    max_iter: int = Field(..., gt=0, description="Maximum iterations (must be > 0)")
    threshold: str | float | None = Field(
        default=None,
        description="Feature selection threshold. See: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html",
    )

    @model_validator(mode="after")
    def validate_penalty_solver_compatibility(self) -> "LogisticRegressionConfig":
        """Validate that penalty is compatible with solver."""
        if not self.penalty.validate(self.solver):
            raise ValueError(f"Incompatible penalty '{self.penalty.value}' for solver '{self.solver.value}'")
        return self


class VarianceThresholdConfig(BaseModel):
    threshold: float = Field(default=0.0, ge=0)


class CorrelationThresholdConfig(BaseModel):
    threshold: float = Field(default=0.95, ge=0, le=1)
