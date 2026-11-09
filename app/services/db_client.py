import re

import httpx

from app.config import Settings
from app.schemas import CnicFields

# Accepted alternative key names from the external API -> our field names.
ALIASES = {
    "name": ["name", "full_name", "holder_name"],
    "father_husband_name": ["father_husband_name", "father_name", "husband_name", "guardian_name"],
    "cnic_number": ["cnic_number", "cnic", "nic", "id_number"],
    "address": ["address", "permanent_address", "current_address"],
    "date_of_birth": ["date_of_birth", "dob", "birth_date"],
    "date_of_issue": ["date_of_issue", "issue_date", "allocation_date", "date_of_allocation"],
    "date_of_expiry": ["date_of_expiry", "expiry_date", "expiry", "valid_until"],
}


class DbError(RuntimeError):
    pass


def _normalise(raw: dict) -> CnicFields:
    values = {}
    for field, keys in ALIASES.items():
        for k in keys:
            if raw.get(k) not in (None, ""):
                values[field] = str(raw[k]).strip()
                break
    return CnicFields(**values)


class DbClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.s = settings
        self.client = client

    async def get_person(self, cnic: str) -> CnicFields | None:
        """Look up a person by CNIC. Returns None if the API says 404."""
        digits = re.sub(r"\D", "", cnic)
        headers = {"Authorization": f"Bearer {self.s.db_api_key}"} if self.s.db_api_key else {}
        try:
            r = await self.client.get(
                f"{self.s.db_api_url}/persons/{digits}",
                headers=headers,
                timeout=self.s.db_timeout_s,
            )
        except httpx.HTTPError as exc:
            raise DbError(f"Could not reach the database API: {exc}") from exc
        if r.status_code == 404:
            return None
        if r.status_code >= 400:
            raise DbError(f"Database API returned {r.status_code}: {r.text[:200]}")
        try:
            body = r.json()
        except ValueError as exc:
            raise DbError("Database API returned invalid JSON.") from exc
        if isinstance(body, dict) and isinstance(body.get("data"), dict):
            body = body["data"]
        return _normalise(body)
