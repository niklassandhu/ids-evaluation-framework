from pydantic import BaseModel
from typing import List, Optional


class TimeWindow(BaseModel):
    label: str
    start: Optional[str] = None
    end: Optional[str] = None
    src_ips: Optional[List[str]] = []
    dst_ips: Optional[List[str]] = []
    default: Optional[bool] = False