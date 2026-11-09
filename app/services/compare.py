import re
from datetime import date, datetime

from rapidfuzz import fuzz

from app.schemas import (
    ComparisonResult,
    CnicFields,
    FieldComparison,
    FieldStatus,
)

NAME_MATCH, NAME_PARTIAL = 88, 70
ADDR_MATCH, ADDR_PARTIAL = 75, 55

DATE_FORMATS = ["%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%Y.%m.%d"]


def norm_text(value: str | None) -> str:
    if not value:
        return ""
    value = re.sub(r"[^\w\s]", " ", value.lower())
    return re.sub(r"\s+", " ", value).strip()


def norm_cnic(value: str | None) -> str:
    return re.sub(r"\D", "", value or "")


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    value = value.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _text_status(a: str, b: str, match: int, partial: int, scorer) -> tuple[FieldStatus, float]:
    score = float(scorer(norm_text(a), norm_text(b)))
    if score >= match:
        return "match", score
    if score >= partial:
        return "partial", score
    return "mismatch", score


def compare_field(field: str, extracted: str | None, expected: str | None) -> FieldComparison:
    def result(status: FieldStatus, score: float | None = None) -> FieldComparison:
        return FieldComparison(
            field=field, extracted=extracted, expected=expected, status=status, score=score
        )

    if not extracted:
        return result("missing_extracted")
    if not expected:
        return result("missing_db")

    if field == "cnic_number":
        same = norm_cnic(extracted) == norm_cnic(expected)
        return result("match" if same else "mismatch", 100.0 if same else 0.0)

    if field.startswith("date_"):
        d1, d2 = parse_date(extracted), parse_date(expected)
        if d1 is None or d2 is None:
            return result("partial", None)  # unparseable date: needs a human look
        same = d1 == d2
        return result("match" if same else "mismatch", 100.0 if same else 0.0)

#    if field == "address":
#        status, score = _text_status(extracted, expected, ADDR_MATCH, ADDR_PARTIAL, fuzz.token_set_ratio)
    else:
        status, score = _text_status(extracted, expected, NAME_MATCH, NAME_PARTIAL, fuzz.token_sort_ratio)
    return result(status, round(score, 1))


COMPARED_FIELDS = [
    "name",
    "father_husband_name",
    "cnic_number",
    #"address",
    "date_of_birth",
    "date_of_issue",
    "date_of_expiry",
]


def compare_records(extracted: CnicFields, db_record: CnicFields | None) -> ComparisonResult:
    if db_record is None:
        return ComparisonResult(
            verdict="reject", reason="No matching person found in the database.", fields=[]
        )

    #rows = [
    #    compare_field(f, getattr(extracted, f), getattr(db_record, f)) for f in COMPARED_FIELDS
    #]
    rows = [
        compare_field(f, getattr(extracted, f, None), getattr(db_record, f, None))
        for f in COMPARED_FIELDS
    ]
    rows = [r for r in rows if r is not None]


    by_field = {r.field: r for r in rows}

    if by_field["cnic_number"].status == "mismatch":
        return ComparisonResult(
            verdict="reject", reason="CNIC number does not match the database.", fields=rows
        )

    # Only fields present on both sides count towards the verdict.
    judged = [r for r in rows if r.status in ("match", "partial", "mismatch")]
    bad = [r.field for r in judged if r.status == "mismatch"]
    partial = [r.field for r in judged if r.status == "partial"]
    missing = [r.field for r in rows if r.status.startswith("missing")]

    if bad:
        return ComparisonResult(
            verdict="review", reason="Mismatch in: " + ", ".join(bad), fields=rows
        )
    if partial or missing:
        parts = []
        if partial:
            parts.append("close but not exact: " + ", ".join(partial))
        if missing:
            parts.append("not available on one side: " + ", ".join(missing))
        return ComparisonResult(verdict="review", reason="; ".join(parts).capitalize(), fields=rows)
    return ComparisonResult(verdict="match", reason="All fields match.", fields=rows)
