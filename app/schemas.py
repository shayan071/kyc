from typing import Literal

from pydantic import BaseModel, Field

FIELDS = [
    "name",
    "father_husband_name",
    "cnic_number",
    #"address",
    "date_of_birth",
    "date_of_issue",
    "date_of_expiry",
]


class CnicFields(BaseModel):
    name: str | None = None
    father_husband_name: str | None = None
    cnic_number: str | None = None
    #address: str | None = None
    date_of_birth: str | None = None
    date_of_issue: str | None = None
    date_of_expiry: str | None = None


class OcrResult(BaseModel):
    model: str
    fields: CnicFields
    raw_text: str
    used_fallback: bool = False
    seconds: float


FieldStatus = Literal["match", "partial", "mismatch", "missing_extracted", "missing_db"]


class FieldComparison(BaseModel):
    field: str
    extracted: str | None
    expected: str | None
    status: FieldStatus
    score: float | None = None


class ComparisonResult(BaseModel):
    verdict: Literal["match", "review", "reject", "not_compared"]
    reason: str
    fields: list[FieldComparison] = Field(default_factory=list)


class SignatureResult(BaseModel):
    verdict: Literal["genuine", "uncertain", "forged", "unavailable"]
    score: float | None = None
    threshold: float
    message: str | None = None
    cropped_signature_png_b64: str | None = None


class VerifyResponse(BaseModel):
    ocr: OcrResult | None = None
    db_record: CnicFields | None = None
    comparison: ComparisonResult
    signature: SignatureResult
    warnings: list[str] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    id: str
    loaded_in_lmstudio: bool
    default: bool


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    lmstudio_reachable: bool
