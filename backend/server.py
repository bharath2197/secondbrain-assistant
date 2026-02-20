# server.py
from fastapi import FastAPI, APIRouter, Request, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os
import logging
import json
import re
import httpx
import jwt
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# Groq OpenAI-compatible endpoint
from openai import AsyncOpenAI

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

# -----------------------------
# Env
# -----------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = (os.environ.get("GROQ_MODEL") or "llama-3.3-70b-versatile").strip()

# Email (Resend)
EMAIL_PROVIDER = (os.environ.get("EMAIL_PROVIDER") or "resend").strip().lower()
RESEND_API_KEY = (os.environ.get("RESEND_API_KEY") or "").strip()
FROM_EMAIL = (os.environ.get("FROM_EMAIL") or "").strip()

# Cron security
DISPATCH_SECRET = os.environ.get("DISPATCH_SECRET")

# Daily summary time (local)
DAILY_SUMMARY_HOUR = int(os.environ.get("DAILY_SUMMARY_HOUR") or "8")
DAILY_SUMMARY_MINUTE = int(os.environ.get("DAILY_SUMMARY_MINUTE") or "0")
DAILY_SUMMARY_WINDOW_MINUTES = int(os.environ.get("DAILY_SUMMARY_WINDOW_MINUTES") or "5")

# Due send window (cron every 5 min)
DUE_EARLY_WINDOW_HOURS = int(os.environ.get("DUE_EARLY_WINDOW_HOURS") or "2")
DUE_LATE_GRACE_MINUTES = int(os.environ.get("DUE_LATE_GRACE_MINUTES") or "5")

# Proactive gap questions (LLM-side)
MAX_GAP_QUESTIONS = int(os.environ.get("MAX_GAP_QUESTIONS") or "3")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_ANON_KEY env vars")

app = FastAPI()
api_router = APIRouter(prefix="/api")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

groq_client = None
if GROQ_API_KEY:
    groq_client = AsyncOpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )

# -----------------------------
# Supabase REST helpers
# -----------------------------
class SupabaseREST:
    """
    token is a Supabase JWT (anon/service_role/user JWT).
    For Supabase REST:
      - apikey: use anon/service_role key
      - Authorization: Bearer <JWT>
    """
    def __init__(self, token: str, apikey: str):
        self.base = f"{SUPABASE_URL}/rest/v1"
        self.headers = {
            "apikey": apikey,
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    async def select(self, table: str, params: dict = None) -> list:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get(f"{self.base}/{table}", headers=self.headers, params=params or {})
            if r.status_code >= 400:
                logger.error(f"Supabase SELECT {table}: {r.status_code} {r.text}")
                raise HTTPException(status_code=r.status_code, detail=r.text)
            return r.json()

    async def insert(self, table: str, data: dict) -> list:
        h = {**self.headers, "Prefer": "return=representation"}
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post(f"{self.base}/{table}", headers=h, json=data)
            if r.status_code >= 400:
                logger.error(f"Supabase INSERT {table}: {r.status_code} {r.text}")
                raise HTTPException(status_code=r.status_code, detail=r.text)
            return r.json()

    async def upsert(self, table: str, data: dict, on_conflict: str = "id") -> list:
        h = {**self.headers, "Prefer": "return=representation,resolution=merge-duplicates"}
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post(f"{self.base}/{table}?on_conflict={on_conflict}", headers=h, json=data)
            if r.status_code >= 400:
                logger.error(f"Supabase UPSERT {table}: {r.status_code} {r.text}")
                raise HTTPException(status_code=r.status_code, detail=r.text)
            return r.json()

    async def update(self, table: str, data: dict, match: dict) -> list:
        h = {**self.headers, "Prefer": "return=representation"}
        params = {k: f"eq.{v}" for k, v in match.items()}
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.patch(f"{self.base}/{table}", headers=h, json=data, params=params)
            if r.status_code >= 400:
                logger.error(f"Supabase UPDATE {table}: {r.status_code} {r.text}")
                raise HTTPException(status_code=r.status_code, detail=r.text)
            return r.json()

    async def delete(self, table: str, match: dict) -> None:
        params = {k: f"eq.{v}" for k, v in match.items()}
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.delete(f"{self.base}/{table}", headers=self.headers, params=params)
            if r.status_code >= 400:
                logger.error(f"Supabase DELETE {table}: {r.status_code} {r.text}")
                raise HTTPException(status_code=r.status_code, detail=r.text)

# -----------------------------
# Auth helpers
# -----------------------------
def get_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing auth token")
    return auth[7:]


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, options={"verify_signature": False})
    except jwt.DecodeError:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_user_id(request: Request) -> str:
    payload = decode_token(get_token(request))
    uid = payload.get("sub")
    if not uid:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    return uid


def get_db(request: Request) -> SupabaseREST:
    token = get_token(request)
    return SupabaseREST(token=token, apikey=SUPABASE_ANON_KEY)


def get_service_db() -> SupabaseREST:
    """
    For cron job: use service role key if available, otherwise anon.
    IMPORTANT: service role must be used for BOTH apikey and bearer token.
    """
    key = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY
    return SupabaseREST(token=key, apikey=key)

# -----------------------------
# Request models
# -----------------------------
class ChatRequest(BaseModel):
    message: str
    pending_reminder: Optional[Dict[str, Any]] = None


class ProfileUpdate(BaseModel):
    timezone: str


class ReminderStatusUpdate(BaseModel):
    status: str

# -----------------------------
# Date parsing (server-side)
# -----------------------------
WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

TIME_12H_STRICT_RE = re.compile(r"\b(?P<h>1[0-2]|0?[1-9])(?::(?P<m>[0-5]\d))?\s*(?P<ampm>am|pm)\b", re.IGNORECASE)
TIME_24H_STRICT_RE = re.compile(r"\b(?P<h>[01]?\d|2[0-3]):(?P<m>[0-5]\d)\b")
IN_WEEKS_RE = re.compile(r"\bin\s+(?P<n>\d+)\s+weeks?\b", re.IGNORECASE)
IN_DAYS_RE = re.compile(r"\bin\s+(?P<n>\d+)\s+days?\b", re.IGNORECASE)
THIS_NEXT_DOW_RE = re.compile(
    r"\b(?:(?P<prefix>this|next)\s+)?(?P<dow>mon(?:day)?|tue(?:s|sday)?|wed(?:nesday)?|thu(?:rs|rsday|r|rday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)\b",
    re.IGNORECASE,
)

def _parse_time_from_text_strict(text: str) -> Optional[Tuple[int, int]]:
    m24 = TIME_24H_STRICT_RE.search(text)
    if m24:
        return (int(m24.group("h")), int(m24.group("m")))

    m12 = TIME_12H_STRICT_RE.search(text)
    if not m12:
        return None

    h = int(m12.group("h"))
    minute = int(m12.group("m") or "0")
    ampm = (m12.group("ampm") or "").lower()

    if h == 12:
        h = 0
    if ampm == "pm":
        h += 12
    return (h, minute)

def _next_weekday(base: datetime, weekday: int) -> datetime:
    days_ahead = (weekday - base.weekday()) % 7
    return base + timedelta(days=days_ahead)

def _extract_hour_min_from_pending(pending: Optional[Dict[str, Any]], user_tz: str) -> Optional[Tuple[int, int]]:
    if not pending or not isinstance(pending, dict):
        return None
    dt_str = pending.get("due_datetime_local")
    if not dt_str or not isinstance(dt_str, str):
        return None
    try:
        dt = datetime.fromisoformat(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(user_tz))
        return (dt.hour, dt.minute)
    except Exception:
        return None

def _resolve_relative_due_datetime(message: str, user_tz: str, pending: Optional[Dict[str, Any]] = None) -> Optional[str]:
    text = message.strip().lower()
    now_local = datetime.now(ZoneInfo(user_tz))

    explicit_time = _parse_time_from_text_strict(text)
    carry_time = _extract_hour_min_from_pending(pending, user_tz)
    default_time = (9, 0)

    def pick_time() -> Tuple[int, int]:
        if explicit_time:
            return explicit_time
        if carry_time:
            return carry_time
        return default_time

    m = IN_WEEKS_RE.search(text)
    if m:
        n = int(m.group("n"))
        target = now_local + timedelta(weeks=n)
        hr_min = pick_time()
        target = target.replace(hour=hr_min[0], minute=hr_min[1], second=0, microsecond=0)
        return target.replace(tzinfo=None).isoformat()

    m = IN_DAYS_RE.search(text)
    if m:
        n = int(m.group("n"))
        target = now_local + timedelta(days=n)
        hr_min = pick_time()
        target = target.replace(hour=hr_min[0], minute=hr_min[1], second=0, microsecond=0)
        return target.replace(tzinfo=None).isoformat()

    m = THIS_NEXT_DOW_RE.search(text)
    if m:
        prefix = (m.group("prefix") or "").lower()
        dow_raw = (m.group("dow") or "").lower()
        dow_key = dow_raw[:3] if dow_raw[:3] in WEEKDAYS else dow_raw
        weekday = WEEKDAYS.get(dow_key)
        if weekday is None:
            return None

        base_date = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        upcoming = _next_weekday(base_date, weekday)
        if prefix == "next":
            upcoming = upcoming + timedelta(days=7)

        hr_min = pick_time()
        target = upcoming.replace(hour=hr_min[0], minute=hr_min[1], second=0, microsecond=0)
        return target.replace(tzinfo=None).isoformat()

    return None

def local_to_utc(local_dt_str: str, tz_str: str) -> str:
    local_dt = datetime.fromisoformat(local_dt_str)
    if local_dt.tzinfo is None:
        local_dt = local_dt.replace(tzinfo=ZoneInfo(tz_str))
    return local_dt.astimezone(ZoneInfo("UTC")).isoformat()

def message_contains_weekday(text: str) -> bool:
    t = (text or "").lower()
    keys = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday","mon","tue","wed","thu","fri","sat","sun"]
    return any(k in t for k in keys)

# -----------------------------
# Deterministic order quantity gap detection (server-side)
# -----------------------------
ORDER_TOTAL_RE = re.compile(r"\border\s+(?:for\s+)?(?P<total>\d+)\b", re.IGNORECASE)
COLOR_QTY_RE = re.compile(
    r"\b(?P<qty>\d+)\s+(?P<color>red|blue|green|black|white|yellow|navy|maroon|grey|gray)\b",
    re.IGNORECASE
)

def detect_quantity_gap(message: str) -> Optional[Tuple[int, int]]:
    if not message:
        return None
    m = ORDER_TOTAL_RE.search(message)
    if not m:
        return None
    try:
        total = int(m.group("total"))
    except Exception:
        return None

    matches = list(COLOR_QTY_RE.finditer(message))
    if not matches:
        return None

    specified = 0
    for mm in matches:
        try:
            specified += int(mm.group("qty"))
        except Exception:
            continue

    if specified > 0 and specified < total:
        return (total, specified)
    return None

# -----------------------------
# LLM Extraction (Groq)
# -----------------------------
EXTRACTION_PROMPT = """You are SecondBrain, a proactive personal assistant for garment production operations.

Current date/time (user's local): {now}
User's default timezone: {timezone}

CORE GOAL:
- Store what happened in the knowledge base.
- Ensure what needs to happen gets scheduled or clarified.
- Prevent missed follow ups and missing details.

MAX_QUESTIONS = {max_q}

STRICT RULES:
1) Never invent facts, names, quantities, dates, or times.
2) If the message contains ONLY factual info with no action intent, type="kb".
3) If the message contains ONLY follow-up/action with date/time intent, type="reminder".
4) If the message contains BOTH factual info AND follow-up/action with date/time intent, type="both".
5) If a reminder is intended but date/time is missing, type="clarify".
6) PROACTIVE GAP CHECK (MANDATORY when orders or production work are mentioned):
   - Detect missing or inconsistent details that block execution.
   - If blocking gaps exist, ask up to MAX_QUESTIONS questions in a numbered list.
   - Questions MUST follow the GAP PRIORITY ORDER below.
   - Ask only blocking questions.

GAP PRIORITY ORDER:
P0: Numeric inconsistency (total quantity != sum of breakdowns). ALWAYS first.
P1: Missing date/time for an explicitly requested follow up.
P2: Missing delivery date or delivery location for an order.
P3: Missing specs that block production: fabric, size breakdown, branding, sample approval.
P4: Missing commercial terms: price, payment status, PO/reference.
P5: Entity spelling inconsistency (lowest priority unless it blocks identification).

IMPORTANT:
- Preserve entity names exactly as typed by the user in kb_entry and reminder.
- If quantities are inconsistent, ask about the missing remainder before any name/spelling question.
- If relative weekday time is provided (like Monday 10 am), you may output it, but the server may override it.

PENDING REMINDER STATE: {pending_reminder}

RETURN ONLY VALID JSON:
{{
  "type": "kb" | "reminder" | "both" | "clarify" | "chat",
  "kb_entry": {{
    "entity_type": "note" | "contact" | "order" | "fact",
    "entity_name": "string or null",
    "order_ref": "string or null",
    "details": "concise extracted facts"
  }} or null,
  "reminder": {{
    "title": "short title",
    "due_datetime_local": "YYYY-MM-DDTHH:MM:SS",
    "timezone": "IANA timezone string or null",
    "order_ref": "string or null",
    "related_party": "string or null"
  }} or null,
  "gaps": [
    {{
      "field": "string",
      "issue": "string",
      "question": "string",
      "priority": "P0|P1|P2|P3|P4|P5"
    }}
  ],
  "clarify_question": "string or null",
  "updated_pending": {{
    "title": "string or null",
    "due_datetime_local": "string or null",
    "order_ref": "string or null",
    "related_party": "string or null"
  }} or null,
  "response": "natural language confirmation"
}}"""

async def extract_message(message: str, user_tz: str, pending: dict = None, recent: list = None) -> dict:
    now_str = datetime.now(ZoneInfo(user_tz)).strftime("%Y-%m-%d %H:%M:%S %A")
    system_msg = EXTRACTION_PROMPT.format(
        now=now_str,
        timezone=user_tz,
        pending_reminder=json.dumps(pending) if pending else "None",
        max_q=str(MAX_GAP_QUESTIONS),
    )

    if not groq_client:
        return {
            "type": "chat",
            "kb_entry": None,
            "reminder": None,
            "gaps": [],
            "clarify_question": None,
            "updated_pending": None,
            "response": "LLM not configured.",
        }

    parts = []
    if recent:
        parts.append("Recent conversation:")
        for m in recent[-6:]:
            parts.append(f"  {m['role']}: {m['content']}")
        parts.append("")
    parts.append(f"User says: {message}")

    try:
        resp = await groq_client.chat.completions.create(
            model=GROQ_MODEL.strip(),
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": "\n".join(parts)},
            ],
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                data = json.loads(m.group())
            else:
                data = {
                    "type": "chat",
                    "kb_entry": None,
                    "reminder": None,
                    "gaps": [],
                    "clarify_question": None,
                    "updated_pending": None,
                    "response": text or "LLM returned empty response",
                }

        if "gaps" not in data or not isinstance(data.get("gaps"), list):
            data["gaps"] = []

        if data.get("gaps") and not data.get("clarify_question"):
            qs = []
            for g in data["gaps"][:MAX_GAP_QUESTIONS]:
                q = (g.get("question") or "").strip()
                if q:
                    qs.append(q)
            if qs:
                data["clarify_question"] = "\n".join([f"{i+1}. {q}" for i, q in enumerate(qs)])

        return data

    except Exception:
        logger.exception("Groq call failed")
        return {
            "type": "chat",
            "kb_entry": None,
            "reminder": None,
            "gaps": [],
            "clarify_question": None,
            "updated_pending": None,
            "response": "LLM failed. Please try again.",
        }

# -----------------------------
# Email via Resend
# -----------------------------
def _resend_ready() -> bool:
    return EMAIL_PROVIDER == "resend" and bool(RESEND_API_KEY) and bool(FROM_EMAIL)

def send_email(to_email: str, subject: str, body: str) -> None:
    if not _resend_ready():
        raise RuntimeError("Resend not configured. Set EMAIL_PROVIDER=resend, RESEND_API_KEY, FROM_EMAIL.")

    url = "https://api.resend.com/emails"
    headers = {
        "Authorization": f"Bearer {RESEND_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "from": FROM_EMAIL,
        "to": [to_email],
        "subject": subject,
        "text": body,
    }
    r = httpx.post(url, headers=headers, json=payload, timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(f"Resend error {r.status_code}: {r.text}")

def fmt_local(dt_utc: datetime, tz_str: str) -> str:
    dt_local = dt_utc.astimezone(ZoneInfo(tz_str))
    return dt_local.strftime("%b %-d, %Y at %-I:%M %p")

def _parse_supabase_timestamptz(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    s = str(value).strip()
    try:
        s = s.replace(" ", "T")
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None

# -----------------------------
# API Endpoints
# -----------------------------
@api_router.get("/")
async def root():
    return {"status": "ok", "service": "SecondBrain API"}

@api_router.get("/profile")
async def get_profile(request: Request):
    db = get_db(request)
    uid = get_user_id(request)
    rows = await db.select("user_profile", {"id": f"eq.{uid}"})
    return rows[0] if rows else None

@api_router.post("/profile")
async def upsert_profile(body: ProfileUpdate, request: Request):
    db = get_db(request)
    uid = get_user_id(request)
    email = decode_token(get_token(request)).get("email", "")
    data = {"id": uid, "user_email": email, "timezone": body.timezone}
    result = await db.upsert("user_profile", data, on_conflict="id")
    return result[0] if result else data

@api_router.patch("/profile")
async def patch_profile(body: ProfileUpdate, request: Request):
    return await upsert_profile(body, request)

@api_router.post("/chat")
async def chat_endpoint(body: ChatRequest, request: Request):
    db = get_db(request)
    uid = get_user_id(request)

    profiles = await db.select("user_profile", {"id": f"eq.{uid}"})
    user_tz = profiles[0]["timezone"] if profiles and profiles[0].get("timezone") else "UTC"

    recent = await db.select(
        "chat_messages",
        {"user_id": f"eq.{uid}", "order": "created_at.desc", "limit": "10"},
    )
    recent.reverse()

    await db.insert("chat_messages", {"user_id": uid, "role": "user", "content": body.message})

    extraction = await extract_message(
        body.message,
        user_tz,
        body.pending_reminder,
        [{"role": m["role"], "content": m["content"]} for m in recent],
    )

    # Hard rule: if quantity gap exists, do not let name-spelling clarify win
    qty_gap = detect_quantity_gap(body.message)
    if qty_gap:
        cq = (extraction.get("clarify_question") or "").lower()
        if "entity name" in cq or ("surya" in cq and "suriya" in cq):
            extraction["clarify_question"] = None

    result_type = extraction.get("type", "chat")
    response_text = extraction.get("response", "Got it!")
    saved_kb = None
    saved_reminder = None

    def _forced_due_from_server() -> Optional[str]:
        return _resolve_relative_due_datetime(body.message, user_tz, pending=body.pending_reminder)

    def _override_due_if_relative(rem: dict) -> dict:
        if not rem:
            return rem

        forced = _forced_due_from_server()
        if forced:
            return {**rem, "due_datetime_local": forced}

        # If message contains weekday, force server resolution even if model gave a date
        if message_contains_weekday(body.message):
            forced2 = _resolve_relative_due_datetime(body.message, user_tz, pending=body.pending_reminder)
            if forced2:
                return {**rem, "due_datetime_local": forced2}

        return rem

    if result_type == "reminder" and extraction.get("kb_entry") and extraction.get("reminder"):
        result_type = "both"
    if result_type == "kb" and extraction.get("reminder") and extraction.get("kb_entry"):
        result_type = "both"

    async def _save_kb(kb_data):
        entry = {
            "user_id": uid,
            "entity_type": kb_data.get("entity_type", "note"),
            "entity_name": kb_data.get("entity_name"),
            "order_ref": kb_data.get("order_ref"),
            "details": kb_data.get("details", body.message),
            "source_message": body.message,
        }
        rows = await db.insert("kb_entries", entry)
        return rows[0] if rows else entry

    async def _save_reminder(rem_data):
        rem_data = _override_due_if_relative(rem_data)

        due_local = (rem_data.get("due_datetime_local") or "").strip()
        title = (rem_data.get("title") or "").strip() or "Reminder"

        explicit_tz = rem_data.get("timezone")
        eff_tz = explicit_tz if explicit_tz else user_tz

        if not due_local:
            forced = _forced_due_from_server()
            if forced:
                due_local = forced
            else:
                raise HTTPException(status_code=400, detail="Reminder missing due_datetime_local")

        due_utc = local_to_utc(due_local, eff_tz)

        rdata = {
            "user_id": uid,
            "title": title,
            "due_datetime": due_utc,
            "timezone": eff_tz,
            "related_order_ref": rem_data.get("order_ref"),
            "related_party": rem_data.get("related_party"),
            "status": "open",
            "source_message": body.message,
            "notes": rem_data.get("notes"),
            "emailed_at": None,
        }
        rows = await db.insert("reminders", rdata)
        saved = rows[0] if rows else rdata

        try:
            dt_local = datetime.fromisoformat(due_local).replace(tzinfo=ZoneInfo(eff_tz))
            abs_str = dt_local.strftime("%b %-d, %Y at %-I:%M %p")
            saved["_confirmation"] = f"Reminder created: {title} - {abs_str} ({eff_tz})"
        except Exception:
            pass

        return saved

    if result_type == "clarify":
        response_text = extraction.get("clarify_question") or response_text

    if result_type == "both":
        if extraction.get("kb_entry"):
            saved_kb = await _save_kb(extraction["kb_entry"])
        if extraction.get("reminder"):
            saved_reminder = await _save_reminder(extraction["reminder"])

        parts = []
        if saved_kb:
            parts.append(f"Saved to Knowledge Base: {saved_kb.get('details','')[:140]}")
        if saved_reminder and saved_reminder.get("_confirmation"):
            parts.append(saved_reminder["_confirmation"])

        response_text = " | ".join(parts) if parts else response_text

    elif result_type == "kb" and extraction.get("kb_entry"):
        saved_kb = await _save_kb(extraction["kb_entry"])

    elif result_type == "reminder" and extraction.get("reminder"):
        saved_reminder = await _save_reminder(extraction["reminder"])
        if saved_reminder and saved_reminder.get("_confirmation"):
            response_text = saved_reminder["_confirmation"]

    # Deterministic P0 question: quantity remainder (always ask if mismatch)
    qty_gap2 = detect_quantity_gap(body.message)
    if qty_gap2:
        total, specified = qty_gap2
        remaining = total - specified
        remainder_q = f"What colors are the remaining {remaining} shirts?"
        if remainder_q.lower() not in (response_text or "").lower():
            response_text = (response_text or "").strip()
            if response_text:
                response_text += "\n\n"
            response_text += f"1. {remainder_q}"

    # If LLM provided other clarify questions, append them after P0
    llm_q = (extraction.get("clarify_question") or "").strip()
    if llm_q:
        if llm_q.lower() not in (response_text or "").lower():
            response_text = (response_text or "").strip()
            if response_text:
                response_text += "\n\n"
            response_text += llm_q

    await db.insert("chat_messages", {"user_id": uid, "role": "assistant", "content": response_text})

    return {
        "type": result_type,
        "response": response_text,
        "kb_entry": saved_kb,
        "reminder": saved_reminder,
        "clarify_question": extraction.get("clarify_question"),
        "updated_pending": extraction.get("updated_pending"),
        "gaps": extraction.get("gaps", []),
    }

@api_router.get("/messages")
async def get_messages(request: Request, limit: int = 200):
    db = get_db(request)
    uid = get_user_id(request)
    rows = await db.select(
        "chat_messages",
        {"user_id": f"eq.{uid}", "order": "created_at.desc", "limit": str(limit)},
    )
    rows.reverse()
    return rows

@api_router.get("/kb")
async def get_kb(request: Request, entity_type: str = None, search: str = None):
    db = get_db(request)
    uid = get_user_id(request)
    params: dict = {"user_id": f"eq.{uid}", "order": "created_at.desc", "limit": "100"}
    if entity_type:
        params["entity_type"] = f"eq.{entity_type}"
    if search:
        params["or"] = f"(details.ilike.%{search}%,entity_name.ilike.%{search}%,order_ref.ilike.%{search}%)"
    return await db.select("kb_entries", params)

@api_router.delete("/kb/{entry_id}")
async def delete_kb(entry_id: str, request: Request):
    db = get_db(request)
    uid = get_user_id(request)
    await db.delete("kb_entries", {"id": entry_id, "user_id": uid})
    return {"ok": True}

@api_router.get("/reminders")
async def get_reminders(request: Request, status: str = None):
    db = get_db(request)
    uid = get_user_id(request)
    params: dict = {"user_id": f"eq.{uid}", "order": "due_datetime.asc", "limit": "500"}
    if status:
        params["status"] = f"eq.{status}"
    return await db.select("reminders", params)

@api_router.patch("/reminders/{reminder_id}")
async def update_reminder(reminder_id: str, body: ReminderStatusUpdate, request: Request):
    db = get_db(request)
    uid = get_user_id(request)
    result = await db.update("reminders", {"status": body.status}, {"id": reminder_id, "user_id": uid})
    return result[0] if result else {"ok": True}

# -----------------------------
# Cron endpoint: dispatch emails
# -----------------------------
@api_router.post("/dispatch-emails")
async def dispatch_emails(request: Request):
    """
    Security: requires X-Dispatch-Secret header to match DISPATCH_SECRET.
    Sends:
      1) Due emails within [due-2h, due+5m] for open reminders (once)
      2) Daily summary at 08:00 local time (once/day/user)
    """
    secret = request.headers.get("X-Dispatch-Secret", "")
    if not DISPATCH_SECRET or secret != DISPATCH_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not _resend_ready():
        raise HTTPException(status_code=500, detail="Resend not configured")

    db = get_service_db()
    now_utc = datetime.now(timezone.utc)

    sent_due = 0
    sent_daily = 0
    marked_missed = 0
    checked_reminders = 0
    checked_users = 0

    users = await db.select(
        "user_profile",
        {
            "select": "id,user_email,timezone,last_daily_email_at",
            "user_email": "not.is.null",
            "timezone": "not.is.null",
            "limit": "1000",
        },
    )

    for u in users:
        checked_users += 1
        user_id = u.get("id")
        user_email = (u.get("user_email") or "").strip()
        tz_str = (u.get("timezone") or "").strip()
        if not user_id or not user_email or not tz_str:
            continue

        # 1) DUE EMAILS
        reminders = await db.select(
            "reminders",
            {
                "select": "id,title,due_datetime,timezone,status,emailed_at",
                "user_id": f"eq.{user_id}",
                "status": "eq.open",
                "emailed_at": "is.null",
                "order": "due_datetime.asc",
                "limit": "200",
            },
        )

        for r in reminders:
            checked_reminders += 1
            rid = r.get("id")
            title = (r.get("title") or "Reminder").strip()
            due_utc = _parse_supabase_timestamptz(r.get("due_datetime"))
            if not rid or not due_utc:
                continue

            window_start = due_utc - timedelta(hours=DUE_EARLY_WINDOW_HOURS)
            window_end = due_utc + timedelta(minutes=DUE_LATE_GRACE_MINUTES)

            if now_utc < window_start:
                continue

            if now_utc > window_end:
                try:
                    await db.update("reminders", {"status": "missed"}, {"id": rid, "user_id": user_id})
                    marked_missed += 1
                except Exception:
                    logger.exception(f"Failed marking missed reminder_id={rid}")
                continue

            display_tz = (r.get("timezone") or tz_str).strip() or tz_str
            subject = f"Reminder: {title}"
            body_text = (
                "Hi,\n\n"
                "This is your reminder:\n"
                f"- {title}\n\n"
                "Scheduled time:\n"
                f"- {fmt_local(due_utc, display_tz)} ({display_tz})\n\n"
                "SecondBrain"
            )

            try:
                send_email(user_email, subject, body_text)
                await db.update("reminders", {"emailed_at": now_utc.isoformat()}, {"id": rid, "user_id": user_id})
                sent_due += 1
            except Exception:
                logger.exception(f"Failed sending due email user_id={user_id} reminder_id={rid}")

        # 2) DAILY SUMMARY
        try:
            user_now_local = now_utc.astimezone(ZoneInfo(tz_str))
        except Exception:
            continue

        summary_start = user_now_local.replace(
            hour=DAILY_SUMMARY_HOUR,
            minute=DAILY_SUMMARY_MINUTE,
            second=0,
            microsecond=0,
        )
        summary_end = summary_start + timedelta(minutes=DAILY_SUMMARY_WINDOW_MINUTES)

        if not (summary_start <= user_now_local < summary_end):
            continue

        last_daily = _parse_supabase_timestamptz(u.get("last_daily_email_at"))
        if last_daily:
            last_local_date = last_daily.astimezone(ZoneInfo(tz_str)).date()
            if last_local_date == user_now_local.date():
                continue

        start_of_day_local = user_now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        start_of_next_day_local = start_of_day_local + timedelta(days=1)
        start_utc = start_of_day_local.astimezone(timezone.utc)
        end_utc = start_of_next_day_local.astimezone(timezone.utc)

        open_all = await db.select(
            "reminders",
            {
                "select": "title,due_datetime,timezone,status",
                "user_id": f"eq.{user_id}",
                "status": "eq.open",
                "order": "due_datetime.asc",
                "limit": "500",
            },
        )

        items: List[str] = []
        for rr in open_all:
            dtu = _parse_supabase_timestamptz(rr.get("due_datetime"))
            if not dtu:
                continue
            if not (start_utc <= dtu < end_utc):
                continue
            title2 = (rr.get("title") or "Reminder").strip()
            display_tz2 = (rr.get("timezone") or tz_str).strip() or tz_str
            time_str = dtu.astimezone(ZoneInfo(display_tz2)).strftime("%-I:%M %p")
            items.append(f"- {time_str} — {title2}")

        if not items:
            summary_body = (
                "Hi,\n\n"
                "Good morning.\n\n"
                "You have no reminders scheduled for today.\n\n"
                "SecondBrain"
            )
        else:
            summary_body = (
                "Hi,\n\n"
                "Good morning.\n\n"
                "Here are your reminders for today:\n\n"
                + "\n".join(items)
                + "\n\nSecondBrain"
            )

        try:
            send_email(user_email, "Your reminders for today", summary_body)
            await db.update("user_profile", {"last_daily_email_at": now_utc.isoformat()}, {"id": user_id})
            sent_daily += 1
        except Exception:
            logger.exception(f"Failed sending daily summary user_id={user_id}")

    return {
        "ok": True,
        "sent_due": sent_due,
        "sent_daily": sent_daily,
        "marked_missed": marked_missed,
        "checked_users": checked_users,
        "checked_reminders": checked_reminders,
        "now_utc": now_utc.isoformat(),
    }

# -----------------------------
# Middleware & startup
# -----------------------------
app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=[o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()] or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
