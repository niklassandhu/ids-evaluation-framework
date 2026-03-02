from pydantic import BaseModel, Field


class SplitTimestampConfig(BaseModel):
    timestamp_column: str = Field(..., description="Name of the timestamp column to sort by")
