import base64
import io
import json
import re
import time

import httpx
from PIL import Image, ImageOps

from app.config import Settings
from app.schemas import FIELDS, CnicFields, OcrResult

PROMPT = """You are reading a Pakistani CNIC (national identity card) image.
Extract the printed text and return ONLY one JSON object with exactly these keys:
name, father_husband_name, cnic_number, address, date_of_birth, date_of_issue, date_of_expiry.

Rules:
- Use the English text on the card, not the Urdu.
- cnic_number format: 12345-1234567-1
- Dates format: DD.MM.YYYY
- father_husband_name is the "Father Name" or "Husband Name" value.
- If a value is not visible or not readable, use null. Never guess or invent values.
- Output the JSON object only, with no explanation and no markdown."""

CNIC_RE = re.compile(r"(\d{5})[-\s]?(\d{7})[-\s]?(\d)")
DATE_RE = re.compile(r"\b(\d{2})[./\-](\d{2})[./\-](\d{4})\b")


def prepare_image(data: bytes, max_side: int) -> str:
    """Return a base64 JPEG data URL, orientation-fixed and downscaled."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def format_cnic(value: str | None) -> str | None:
    if not value:
        return None
    m = CNIC_RE.search(value)
    if not m:
        return value.strip() or None
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


def parse_json_fields(text: str) -> CnicFields | None:
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    values = {}
    for key in FIELDS:
        v = obj.get(key)
        values[key] = str(v).strip() if v not in (None, "", "null") else None
    return CnicFields(**values)


def regex_fallback(text: str) -> CnicFields:
    """Best-effort extraction when the model did not return valid JSON."""
    cnic = CNIC_RE.search(text)
    dates = []
    for d, m, y in DATE_RE.findall(text):
        dates.append((int(y), int(m), int(d), f"{d}.{m}.{y}"))
    dates = sorted(set(dates))
    dob = issue = expiry = None
    if len(dates) >= 3:
        dob, issue, expiry = dates[0][3], dates[1][3], dates[-1][3]
    return CnicFields(
        cnic_number=format_cnic(cnic.group(0)) if cnic else None,
        date_of_birth=dob,
        date_of_issue=issue,
        date_of_expiry=expiry,
    )


class OcrError(RuntimeError):
    pass


class OcrService:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.s = settings
        self.client = client

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.s.lmstudio_api_key}"}

    async def list_loaded_models(self) -> list[str] | None:
        try:
            r = await self.client.get(
                f"{self.s.lmstudio_base_url}/models", headers=self._headers(), timeout=5
            )
            r.raise_for_status()
            return [m["id"] for m in r.json().get("data", [])]
        except (httpx.HTTPError, KeyError, ValueError):
            return None

    async def extract(self, image_bytes: bytes, model: str) -> OcrResult:
        started = time.perf_counter()
        try:
            data_url = prepare_image(image_bytes, self.s.ocr_max_side)
        except Exception as exc:  # unreadable / unsupported image
            raise OcrError(f"Could not read the CNIC image: {exc}") from exc

        payload = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": self.s.ocr_max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        }
        try:
            r = await self.client.post(
                f"{self.s.lmstudio_base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self.s.ocr_timeout_s,
            )
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"] or ""
        except httpx.ConnectError as exc:
            raise OcrError(
                f"Cannot reach LM Studio at {self.s.lmstudio_base_url}. Is the local server running?"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise OcrError(
                f"LM Studio returned {exc.response.status_code}: {exc.response.text[:300]}"
            ) from exc
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise OcrError(f"OCR request failed: {exc}") from exc

        fields = parse_json_fields(raw)
        used_fallback = fields is None
        if fields is None:
            fields = regex_fallback(raw)
        fields.cnic_number = format_cnic(fields.cnic_number)

        return OcrResult(
            model=model,
            fields=fields,
            raw_text=raw,
            used_fallback=used_fallback,
            seconds=round(time.perf_counter() - started, 2),
        )
