from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="KYC_", extra="ignore")

    # LM Studio (OpenAI-compatible server)
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_api_key: str = "lm-studio"
    ocr_models: str = "allenai/olmocr-2-7b"  # comma separated, vision models only
    default_ocr_model: str = "allenai/olmocr-2-7b"
    ocr_timeout_s: float = 300.0
    ocr_max_side: int = 1280
    ocr_max_tokens: int = 600

    # Person database: "mysql" (direct) or "api" (HTTP API / built-in mock)
    db_backend: str = "api"
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "ocr"
    mysql_table: str = "data"
    mysql_columns: dict[str, str] = {}  # optional JSON overrides: {"name": "full_name", ...}
    db_cnic_format: str = "dashed"  # how the CNIC is stored: dashed | digits | any

    # External DB API
    db_api_url: str = "http://127.0.0.1:8000/mock-db"
    db_api_key: str | None = None
    db_timeout_s: float = 15.0

    # Signature verification
    signature_weights: str = "weights/signet.pth"
    signature_threshold: float = 0.80
    signature_margin: float = 0.10
    signature_roi: str = "0.52,0.70,0.99,0.97"

    # Uploads
    max_upload_mb: int = 10

    @property
    def ocr_model_list(self) -> list[str]:
        return [m.strip() for m in self.ocr_models.split(",") if m.strip()]

    @property
    def roi(self) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = (float(v) for v in self.signature_roi.split(","))
        return x0, y0, x1, y1


@lru_cache
def get_settings() -> Settings:
    return Settings()