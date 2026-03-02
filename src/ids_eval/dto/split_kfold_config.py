from pydantic import BaseModel, Field


class SplitKFoldConfig(BaseModel):
    n_splits: int = Field(..., ge=2, description="Number of folds (must be >= 2)")
