from pydantic import BaseModel, Field


class SplitBenignTrainConfig(BaseModel):
    benign_label: int = Field(
        default=0, description="The label value that indicates benign/normal samples (default: 0)"
    )
