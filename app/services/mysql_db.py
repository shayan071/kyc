"""MySQL-backed person lookup (database `ocr`, table `data` by default).

Columns are auto-detected by name, or set explicitly with KYC_MYSQL_COLUMNS, e.g.
    KYC_MYSQL_COLUMNS={"name":"full_name","date_of_issue":"allocation_dt"}
"""

import datetime as dt
import re
import threading

import pymysql
from pymysql.cursors import DictCursor
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.schemas import FIELDS, CnicFields
from app.services.db_client import DbError

IDENT_RE = re.compile(r"^[A-Za-z0-9_$]+$")

# Exact (normalised) column names tried first, per field.
EXACT = {
    "name": ["name", "fullname", "holdername", "applicantname", "customername", "personname"],
    "father_husband_name": ["fathername", "fatherhusbandname", "husbandname", "guardianname",
                            "fathersname", "fatherorhusbandname"],
    "cnic_number": ["cnic", "cnicnumber", "cnicno", "nic", "idnumber", "nicnumber"],
    "address": ["address", "permanentaddress", "currentaddress"],
    "date_of_birth": ["dob", "dateofbirth", "birthdate"],
    "date_of_issue": ["dateofissue", "issuedate", "allocationdate", "dateofallocation", "issuedon"],
    "date_of_expiry": ["dateofexpiry", "expirydate", "expiry", "validuntil", "expiredate"],
}
# Substring heuristics when no exact name exists.
KEYWORDS = {
    "father_husband_name": ["father", "husband", "guardian"],
    "cnic_number": ["cnic", "nic"],
    "address": ["address"],
    "date_of_birth": ["dob", "birth"],
    "date_of_issue": ["issue", "allocat"],
    "date_of_expiry": ["expir", "validity", "validuntil"],
}


def _norm(col: str) -> str:
    return re.sub(r"[^a-z0-9]", "", col.lower())


def detect_columns(columns: list[str], overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Map our field names to real column names. Unmatched fields are simply left out."""
    overrides = overrides or {}
    by_lower = {c.lower(): c for c in columns}
    mapping: dict[str, str] = {}

    for field, col in overrides.items():
        if field not in FIELDS:
            raise DbError(f"KYC_MYSQL_COLUMNS: unknown field '{field}'. Valid: {', '.join(FIELDS)}")
        if col.lower() not in by_lower:
            raise DbError(f"KYC_MYSQL_COLUMNS: column '{col}' does not exist. Columns: {', '.join(columns)}")
        mapping[field] = by_lower[col.lower()]

    taken = set(mapping.values())
    normed = {c: _norm(c) for c in columns}

    for field in FIELDS:  # pass 1: exact names
        if field in mapping:
            continue
        for col, n in normed.items():
            if col not in taken and n in EXACT[field]:
                mapping[field] = col
                taken.add(col)
                break

    for field in FIELDS:  # pass 2: keywords
        if field in mapping or field not in KEYWORDS:
            continue
        for col, n in normed.items():
            if col not in taken and any(k in n for k in KEYWORDS[field]):
                mapping[field] = col
                taken.add(col)
                break

    if "name" not in mapping:  # last resort: a single remaining column containing "name"
        rest = [c for c, n in normed.items() if c not in taken and "name" in n]
        if len(rest) == 1:
            mapping["name"] = rest[0]
    return mapping


def format_cnic_for_lookup(value: str, mode: str) -> str:
    digits = re.sub(r"\D", "", value)
    if mode == "dashed" and len(digits) == 13:
        return f"{digits[:5]}-{digits[5:12]}-{digits[12]}"
    return digits if mode in ("digits", "any") else value.strip()


def _cell(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, (dt.date, dt.datetime)):
        return value.strftime("%d.%m.%Y")
    text = str(value).strip()
    return text or None


def row_to_fields(row: dict, mapping: dict[str, str]) -> CnicFields:
    return CnicFields(**{field: _cell(row.get(col)) for field, col in mapping.items()})


class MysqlDbClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self._mapping: dict[str, str] | None = None
        self._lock = threading.Lock()

    # -- internals -------------------------------------------------------------------------
    def _connect(self):
        for ident, label in ((self.s.mysql_database, "database"), (self.s.mysql_table, "table")):
            if not IDENT_RE.match(ident):
                raise DbError(f"Invalid MySQL {label} name '{ident}'.")
        try:
            return pymysql.connect(
                host=self.s.mysql_host,
                port=self.s.mysql_port,
                user=self.s.mysql_user,
                password=self.s.mysql_password,
                database=self.s.mysql_database,
                charset="utf8mb4",
                connect_timeout=5,
                read_timeout=int(self.s.db_timeout_s),
                cursorclass=DictCursor,
            )
        except pymysql.MySQLError as exc:
            raise DbError(f"Cannot connect to MySQL ({self.s.mysql_host}:{self.s.mysql_port}): {exc}") from exc

    def _resolve_columns(self, conn) -> dict[str, str]:
        if self._mapping is not None:
            return self._mapping
        with self._lock:
            if self._mapping is not None:
                return self._mapping
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COLUMN_NAME AS c FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s ORDER BY ORDINAL_POSITION",
                    (self.s.mysql_database, self.s.mysql_table),
                )
                columns = [r["c"] for r in cur.fetchall()]
            if not columns:
                raise DbError(
                    f"Table `{self.s.mysql_database}`.`{self.s.mysql_table}` not found or has no columns."
                )
            mapping = detect_columns(columns, self.s.mysql_columns)
            if "cnic_number" not in mapping:
                raise DbError(
                    "Could not find the CNIC column. Set KYC_MYSQL_COLUMNS, e.g. "
                    f'{{"cnic_number":"your_column"}}. Columns: {", ".join(columns)}'
                )
            self._mapping = mapping
            return mapping

    # -- sync work (run in a thread) -------------------------------------------------------
    def _get_person_sync(self, cnic: str) -> CnicFields | None:
        conn = self._connect()
        try:
            mapping = self._resolve_columns(conn)
            cols = ", ".join(f"`{c}`" for c in dict.fromkeys(mapping.values()))
            cnic_col = mapping["cnic_number"]
            mode = self.s.db_cnic_format
            value = format_cnic_for_lookup(cnic, mode)
            where = f"REPLACE(`{cnic_col}`, '-', '') = %s" if mode == "any" else f"`{cnic_col}` = %s"
            sql = f"SELECT {cols} FROM `{self.s.mysql_table}` WHERE {where} LIMIT 1"
            with conn.cursor() as cur:
                cur.execute(sql, (value,))
                row = cur.fetchone()
            return row_to_fields(row, mapping) if row else None
        except pymysql.MySQLError as exc:
            raise DbError(f"MySQL query failed: {exc}") from exc
        finally:
            conn.close()

    def _status_sync(self) -> str | None:
        try:
            conn = self._connect()
            try:
                self._resolve_columns(conn)
            finally:
                conn.close()
            return None
        except DbError as exc:
            return str(exc)
        except pymysql.MySQLError as exc:
            return str(exc)

    # -- public async API (same shape as DbClient) -----------------------------------------
    async def get_person(self, cnic: str) -> CnicFields | None:
        return await run_in_threadpool(self._get_person_sync, cnic)

    async def status(self) -> str | None:
        return await run_in_threadpool(self._status_sync)

    @property
    def mapping(self) -> dict[str, str] | None:
        return self._mapping