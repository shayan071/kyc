# CNIC OCR + Signature Verification

FastAPI app that reads a CNIC with a vision model in LM Studio, checks the fields against a DB API, and compares the card signature with an uploaded specimen using a SigNet (Siamese CNN) model.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design.

## Run

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # Windows: copy .env.example .env
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 (API docs at `/docs`). If you change the port, also change `KYC_DB_API_URL` while using the built-in mock DB.

## Setup checklist

1. **LM Studio:** load `allenai/olmocr-2-7b`, then start the local server (Developer tab). Check the exact model id LM Studio shows and put it in `KYC_OCR_MODELS` / `KYC_DEFAULT_OCR_MODEL` if it differs.
2. **Signature weights:** download SigNet weights (`signet.pth` from the [sigver](https://github.com/luizgh/sigver) project) into `weights/signet.pth`. Without them the data check still works and the signature result shows "Unavailable".
3. **Database API:** by default the app uses the built-in mock (`/mock-db`). To use the real one, set `KYC_DB_API_URL` and `KYC_DB_API_KEY`. The app calls `GET {url}/persons/{cnic_digits}` and expects JSON or a 404. Key names are mapped in `app/services/db_client.py` (`ALIASES`).

## Switching models

`qwen2.5-coder-7b-instruct` is text-only and cannot read images, so it is not offered. To get a switch in the UI, load another **vision** model in LM Studio (for example a Qwen2.5-VL) and list both:

```
KYC_OCR_MODELS=allenai/olmocr-2-7b,qwen2.5-vl-7b-instruct
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/verify` | `cnic_image`, `signature_image`, optional `model`, `cnic_override`. Full pipeline |
| `POST /api/ocr` | `cnic_image`, optional `model`. OCR only |
| `GET /api/models` | Enabled models and whether LM Studio has them loaded |
| `GET /api/health` | LM Studio and signature model status |

## Tests

```bash
pytest -q
```

## Things to tune on real data

- **Signature ROI** (`KYC_SIGNATURE_ROI`): where the signature sits on your card layout.
- **Signature threshold** (`KYC_SIGNATURE_THRESHOLD`): calibrate on your own genuine and forged pairs.
- **Fuzzy thresholds** in `app/services/compare.py`.
- **Prompt** in `app/services/ocr.py`: olmOCR is tuned for page-to-text, so if JSON output is unreliable, the regex fallback kicks in and the UI warns you.
