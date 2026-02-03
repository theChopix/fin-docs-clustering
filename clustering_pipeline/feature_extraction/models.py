from pydantic import BaseModel
from typing import Optional


class ExtractedFeatures(BaseModel):
    doc_type: Optional[str]
    issuer: Optional[str]
    account_present: Optional[bool]
    amounts: Optional[list[float]]
    dates: Optional[list[str]]
    layout: Optional[str]
    language: Optional[str]
    summary: Optional[str]