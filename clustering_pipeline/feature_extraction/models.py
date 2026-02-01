from pydantic import BaseModel


class ExtractedFeatures(BaseModel):
    doc_type: str
    issuer: str
    account_present: bool
    amounts: list[float]
    dates: list[str]
    layout: str
    language: str
    summary: str