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
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# We use OpenAI SDK with Groq OpenAI-compatible endpoint
from openai import AsyncOpenAI

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = (os.environ.get("GROQ_MODEL") or "llama-3.3-70b-versatile").strip()  # strip whitespace/newlines

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
    def __init__(self, token: str):
        self.base = f"{SUPABASE_URL}/rest/v1"
        self.headers = {
            "apikey": SUPABASE_ANON_KEY,
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
    return SupabaseREST(get_token(request))


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
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "tues": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}

# Bug fix 1: strict time parsing.
# Only parse explicit times:
# - 12h requires am/pm: "3pm", "3 pm", "3:30pm"
# - 24h requires colon: "15:30"
# Never parse naked numbers like "2" in "2 weeks".
TIME_12H_STRICT_RE = re.compile(r"\b(?P<h>1[0-2]|0?[1-9])(?::(?P<m>[0-5]\d))?\s*(?P<ampm>am|pm)\b", re.IGNORECASE)
TIME_24H_STRICT_RE = re.compile(r"\b(?P<h>[01]?\d|2[0-3]):(?P<m>[0-5]\d)\b")

IN_WEEKS_RE = re.compile(r"\bin\s+(?P<n>\d+)\s+weeks?\b", re.IGNORECASE)
IN_DAYS_RE = re.compile(r"\bin\s+(?P<n>\d+)\s+days?\b", re.IGNORECASE)

THIS_NEXT_DOW_RE = re.compile(
    r"\b(?:(?P<prefix>this|next)\s+)?(?P<dow>mon(?:day)?|tue(?:s|sday)?|wed(?:nesday)?|thu(?:rs|rsday|r|rday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)\b",
    re.IGNORECASE,
)

def _parse_time_from_text_strict(text: str) -> Optional[Tuple[int, int]]:
    """
    Bug fix 1: strict time parsing.
    Returns (hour24, minute) only if text contains an explicit time.
    """
    m24 = TIME_24H_STRICT_RE.search(text)
    if m24:
        h = int(m24.group("h"))
        minute = int(m24.group("m"))
        return (h, minute)

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
    """Next occurrence of weekday (could be today if same weekday)."""
    days_ahead = (weekday - base.weekday()) % 7
    return base + timedelta(days=days_ahead)

def _extract_hour_min_from_pending(pending: Optional[Dict[str, Any]], user_tz: str) -> Optional[Tuple[int, int]]:
    """
    Bug fix 3: pending reminder carry-over.
    If pending_reminder has due_datetime_local, reuse its hour/minute when user provides a day but no time.
    """
    if not pending or not isinstance(pending, dict):
        return None
    dt_str = pending.get("due_datetime_local")
    if not dt_str or not isinstance(dt_str, str):
        return None
    try:
        dt = datetime.fromisoformat(dt_str)
        # pending is stored as local naive in this app
        if dt.tzinfo is None:
            # treat as user local
            dt = dt.replace(tzinfo=ZoneInfo(user_tz))
        return (dt.hour, dt.minute)
    except Exception:
        return None

def _resolve_relative_due_datetime(message: str, user_tz: str, pending: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Deterministic parser for key cases:
    - "in 2 weeks", "in 10 days"
    - "Friday at 3pm", "this Friday 3pm", "next Friday 3pm"
    Defaults time to 09:00 if missing, except:
      Bug fix 3: if pending_reminder has a prior due_datetime_local, reuse its hour/minute when user gives a weekday without time.
    Returns ISO local naive string: YYYY-MM-DDTHH:MM:SS
    """
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

    # in N weeks
    m = IN_WEEKS_RE.search(text)
    if m:
        n = int(m.group("n"))
        target = now_local + timedelta(weeks=n)
        hr_min = pick_time()
        target = target.replace(hour=hr_min[0], minute=hr_min[1], second=0, microsecond=0)
        return target.replace(tzinfo=None).isoformat()

    # in N days
    m = IN_DAYS_RE.search(text)
    if m:
        n = int(m.group("n"))
        target = now_local + timedelta(days=n)
        hr_min = pick_time()
        target = target.replace(hour=hr_min[0], minute=hr_min[1], second=0, microsecond=0)
        return target.replace(tzinfo=None).isoformat()

    # weekday patterns
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

        # "next Friday" means next week's Friday
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


# -----------------------------
# LLM Extraction (Groq)
# -----------------------------
EXTRACTION_PROMPT = """You are SecondBrain, a personal assistant. Process the user's message and determine the appropriate action.

Current date/time (user's local): {now}
User's default timezone: {timezone}

RULES:
1. If the message contains ONLY factual information (a fact, note, contact info, order details, preferences) with NO date/time/follow-up intent, classify as "kb".
2. If the message mentions ONLY a future action, deadline, or follow-up with date/time intent, classify as "reminder".
3. If the message contains BOTH factual information AND a follow-up/action with date/time intent, you MUST classify as "both" and fill BOTH kb_entry AND reminder.
4. If a reminder is being built (see PENDING STATE) but required fields are still missing after considering this message, classify as "clarify" and ask exactly ONE specific question for the missing field.
5. For general conversation or greetings, classify as "chat".

IMPORTANT:
- If the user gives relative time like "next Friday at 3pm" or "in 2 weeks", you may set due_datetime_local, BUT the server may override it.
- The ONLY required reminder field is due_datetime_local. If missing, return type "clarify" and ask for date/time.

PENDING REMINDER STATE: {pending_reminder}

RETURN ONLY VALID JSON:
{{
  "type": "kb" | "reminder" | "both" | "clarify" | "chat",
  "kb_entry": {{
    "entity_type": "note" | "contact" | "order" | "fact",
    "entity_name": "string or null",
    "order_ref": "string or null",
    "details": "the key information extracted"
  }} or null,
  "reminder": {{
    "title": "short descriptive title or Reminder if none given",
    "due_datetime_local": "YYYY-MM-DDTHH:MM:SS",
    "timezone": "IANA timezone string or null",
    "order_ref": "string or null",
    "related_party": "string or null"
  }} or null,
  "clarify_question": "string or null",
  "updated_pending": {{
    "title": "string or null",
    "due_datetime_local": "string or null",
    "order_ref": "string or null",
    "related_party": "string or null"
  }} or null,
  "response": "natural language response confirming what you did"
}}"""

async def extract_message(message: str, user_tz: str, pending: dict = None, recent: list = None) -> dict:
    now_str = datetime.now(ZoneInfo(user_tz)).strftime("%Y-%m-%d %H:%M:%S %A")
    system_msg = EXTRACTION_PROMPT.format(
        now=now_str,
        timezone=user_tz,
        pending_reminder=json.dumps(pending) if pending else "None",
    )

    if not groq_client:
        return {
            "type": "chat",
            "kb_entry": None,
            "reminder": None,
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
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
            return {
                "type": "chat",
                "kb_entry": None,
                "reminder": None,
                "clarify_question": None,
                "updated_pending": None,
                "response": text or "LLM returned empty response",
            }
    except Exception:
        logger.exception("Groq call failed")
        return {
            "type": "chat",
            "kb_entry": None,
            "reminder": None,
            "clarify_question": None,
            "updated_pending": None,
            "response": "LLM failed. Please try again.",
        }


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

# Optional: some frontends call PATCH
@api_router.patch("/profile")
async def patch_profile(body: ProfileUpdate, request: Request):
    return await upsert_profile(body, request)


@api_router.post("/chat")
async def chat_endpoint(body: ChatRequest, request: Request):
    db = get_db(request)
    uid = get_user_id(request)

    # Get timezone (auto-default to UTC if missing row)
    profiles = await db.select("user_profile", {"id": f"eq.{uid}"})
    user_tz = profiles[0]["timezone"] if profiles and profiles[0].get("timezone") else "UTC"

    # Recent messages for context
    recent = await db.select(
        "chat_messages",
        {"user_id": f"eq.{uid}", "order": "created_at.desc", "limit": "10"},
    )
    recent.reverse()

    # Save user message
    await db.insert("chat_messages", {"user_id": uid, "role": "user", "content": body.message})

    # LLM extraction
    extraction = await extract_message(
        body.message,
        user_tz,
        body.pending_reminder,
        [{"role": m["role"], "content": m["content"]} for m in recent],
    )

    result_type = extraction.get("type", "chat")
    response_text = extraction.get("response", "Got it!")
    saved_kb = None
    saved_reminder = None

    # Bug fix 2: If LLM returns clarify but server can resolve a due datetime, override clarify and create reminder.
    # Also used to enforce deterministic parsing for relative dates/times.
    def _forced_due_from_server() -> Optional[str]:
        return _resolve_relative_due_datetime(body.message, user_tz, pending=body.pending_reminder)

    def _override_due_if_relative(rem: dict) -> dict:
        if not rem:
            return rem
        forced = _forced_due_from_server()
        if forced:
            rem = {**rem, "due_datetime_local": forced}
        return rem

    # Treat inconsistent LLM outputs as "both"
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
            # Last chance: server-side forced parse using pending carry-over (bug fix 3)
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
        }
        rows = await db.insert("reminders", rdata)
        saved = rows[0] if rows else rdata

        # absolute confirmation text (format must remain)
        try:
            dt_local = datetime.fromisoformat(due_local).replace(tzinfo=ZoneInfo(eff_tz))
            abs_str = dt_local.strftime("%b %-d, %Y at %-I:%M %p")
            saved["_confirmation"] = f"Reminder created: {title} - {abs_str} ({eff_tz})"
        except Exception:
            pass

        return saved

    # Bug fix 2: override clarify if server can resolve due datetime.
    if result_type == "clarify":
        forced_due = _forced_due_from_server()
        if forced_due:
            # Build reminder payload from whatever we have
            base_rem = extraction.get("reminder") or {}
            if not isinstance(base_rem, dict):
                base_rem = {}
            # If LLM didn't provide reminder, try to use updated_pending
            if not base_rem and isinstance(extraction.get("updated_pending"), dict):
                base_rem = dict(extraction["updated_pending"])
            # Carry over a title if present in pending
            if body.pending_reminder and isinstance(body.pending_reminder, dict):
                if body.pending_reminder.get("title") and not base_rem.get("title"):
                    base_rem["title"] = body.pending_reminder.get("title")
            base_rem["due_datetime_local"] = forced_due
            if not base_rem.get("title"):
                base_rem["title"] = "Reminder"
            saved_reminder = await _save_reminder(base_rem)
            if saved_reminder and saved_reminder.get("_confirmation"):
                response_text = saved_reminder["_confirmation"]
            result_type = "reminder"
        else:
            # keep clarify behavior
            response_text = extraction.get("clarify_question") or response_text

    # Process non-clarify
    if result_type == "both":
        if extraction.get("kb_entry"):
            saved_kb = await _save_kb(extraction["kb_entry"])
        if extraction.get("reminder"):
            saved_reminder = await _save_reminder(extraction["reminder"])

        parts = []
        if saved_kb:
            parts.append(f"Saved to Knowledge Base: {saved_kb.get('details','')[:80]}")
        if saved_reminder and saved_reminder.get("_confirmation"):
            parts.append(saved_reminder["_confirmation"])
        if parts:
            response_text = " | ".join(parts)

    elif result_type == "kb" and extraction.get("kb_entry"):
        saved_kb = await _save_kb(extraction["kb_entry"])

    elif result_type == "reminder" and extraction.get("reminder"):
        saved_reminder = await _save_reminder(extraction["reminder"])
        if saved_reminder and saved_reminder.get("_confirmation"):
            response_text = saved_reminder["_confirmation"]

    # Save assistant message
    await db.insert("chat_messages", {"user_id": uid, "role": "assistant", "content": response_text})

    return {
        "type": result_type,
        "response": response_text,
        "kb_entry": saved_kb,
        "reminder": saved_reminder,
        "clarify_question": extraction.get("clarify_question"),
        "updated_pending": extraction.get("updated_pending"),
    }


@api_router.get("/messages")
async def get_messages(request: Request, limit: int = 200):
    db = get_db(request)
    uid = get_user_id(request)

    # fetch newest first then reverse so UI shows chronological
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
