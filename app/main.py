import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.mock_db import router as mock_db_router
from app.schemas import (
    CnicFields,
    ComparisonResult,
    ModelInfo,
    ModelsResponse,
    OcrResult,
    SignatureResult,
    VerifyResponse,
)
from app.services.compare import compare_records
from app.services.db_client import DbClient, DbError
from app.services.mysql_db import MysqlDbClient
from app.services.ocr import OcrError, OcrService
from app.services.signature import SignatureService

settings = get_settings()
STATIC_DIR = Path(__file__).parent / "static"
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with httpx.AsyncClient() as client:
        app.state.ocr = OcrService(settings, client)
        app.state.db = (
            MysqlDbClient(settings) if settings.db_backend == "mysql" else DbClient(settings, client)
        )
        app.state.signature = SignatureService(settings)
        yield


app = FastAPI(title="CNIC OCR & Signature Verification", version="0.1.0", lifespan=lifespan)
app.include_router(mock_db_router)


async def _read_upload(file: UploadFile, label: str) -> bytes:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"{label}: unsupported file type '{file.content_type}'. Use JPG, PNG or WEBP.")
    data = await file.read()
    if not data:
        raise HTTPException(400, f"{label}: file is empty.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"{label}: larger than {settings.max_upload_mb} MB.")
    return data


def _resolve_model(model: str | None) -> str:
    chosen = model or settings.default_ocr_model
    if chosen not in settings.ocr_model_list:
        raise HTTPException(
            400, f"Model '{chosen}' is not enabled. Available: {', '.join(settings.ocr_model_list)}"
        )
    return chosen


@app.get("/api/health")
async def health():
    sig_err = app.state.signature.status
    loaded = await app.state.ocr.list_loaded_models()
    db_status = getattr(app.state.db, "status", None)
    db_message = await db_status() if db_status else None
    return {
        "status": "ok",
        "db_backend": settings.db_backend,
        "db_ready": db_message is None,
        "db_message": db_message,
        "db_columns": getattr(app.state.db, "mapping", None),
        "lmstudio_reachable": loaded is not None,
        "signature_model_ready": sig_err is None,
        "signature_model_message": sig_err,
    }


@app.get("/api/models", response_model=ModelsResponse)
async def models():
    loaded = await app.state.ocr.list_loaded_models()
    return ModelsResponse(
        lmstudio_reachable=loaded is not None,
        models=[
            ModelInfo(
                id=m,
                loaded_in_lmstudio=bool(loaded and m in loaded),
                default=m == settings.default_ocr_model,
            )
            for m in settings.ocr_model_list
        ],
    )


@app.post("/api/ocr", response_model=OcrResult)
async def ocr_only(cnic_image: UploadFile = File(...), model: str | None = Form(None)):
    data = await _read_upload(cnic_image, "CNIC image")
    try:
        return await app.state.ocr.extract(data, _resolve_model(model))
    except OcrError as exc:
        raise HTTPException(502, str(exc)) from exc


@app.post("/api/verify", response_model=VerifyResponse)
async def verify(
    cnic_image: UploadFile = File(...),
    signature_image: UploadFile = File(...),
    model: str | None = Form(None),
    cnic_override: str | None = Form(None),
):
    cnic_bytes = await _read_upload(cnic_image, "CNIC image")
    sig_bytes = await _read_upload(signature_image, "Signature image")
    chosen = _resolve_model(model)

    warnings: list[str] = []
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    async def run_ocr() -> OcrResult | None:
        try:
            return await app.state.ocr.extract(cnic_bytes, chosen)
        except OcrError as exc:
            warnings.append(str(exc))
            return None

    async def run_signature() -> SignatureResult:
        started = time.perf_counter()
        result = await run_in_threadpool(app.state.signature.verify, cnic_bytes, sig_bytes)
        timings["signature_s"] = round(time.perf_counter() - started, 2)
        return result

    ocr_result, signature = await asyncio.gather(run_ocr(), run_signature())
    if ocr_result:
        timings["ocr_s"] = ocr_result.seconds
        if ocr_result.used_fallback:
            warnings.append(
                "The model did not return valid JSON; fields were recovered with a regex fallback and may be incomplete."
            )
    if signature.message and signature.verdict == "unavailable":
        warnings.append(f"Signature check: {signature.message}")

    cnic_key = (cnic_override or "").strip() or (ocr_result.fields.cnic_number if ocr_result else None)

    db_record: CnicFields | None = None
    comparison = ComparisonResult(verdict="not_compared", reason="Comparison was not run.")
    if not ocr_result:
        comparison.reason = "OCR failed, so there was nothing to compare."
    elif not cnic_key:
        comparison.reason = "No CNIC number was read from the card. Enter it manually and retry."
    else:
        started = time.perf_counter()
        try:
            db_record = await app.state.db.get_person(cnic_key)
            comparison = compare_records(ocr_result.fields, db_record)
        except DbError as exc:
            warnings.append(str(exc))
            comparison.reason = "Database lookup failed."
        timings["db_and_compare_s"] = round(time.perf_counter() - started, 2)

    timings["total_s"] = round(time.perf_counter() - t0, 2)
    return VerifyResponse(
        ocr=ocr_result,
        db_record=db_record,
        comparison=comparison,
        signature=signature,
        warnings=warnings,
        timings=timings,
    )


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")