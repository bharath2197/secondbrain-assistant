from fastapi import FastAPI, APIRouter, Request, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os
import logging
import json
import uuid
import re
import httpx
import jwt
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from emergentintegrations.llm.chat import LlmChat, UserMessage

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_ANON_KEY = os.environ.get('SUPABASE_ANON_KEY')
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')

app = FastAPI()
api_router = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---- Supabase REST helpers ----
class SupabaseREST:
    def __init__(self, token: str):
        self.base = f"{SUPABASE_URL}/rest/v1"
        self.headers = {
            "apikey": SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    async def select(self, table: str, params: dict = None) -> list:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"{self.base}/{table}", headers=self.headers, params=params or {})
            if r.status_code >= 400:
                logger.error(f"Supabase SELECT {table}: {r.status_code} {r.text}")
                raise HTTPException(status_code=r.status_code, detail=r.text)
            return r.json()

    async def insert(self, table: str, data: dict) -> list:
        h = {**self.headers, "Prefer": "return=representation"}
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(f"{self.base}/{table}", headers=h, json=data)
            if r.status_code >= 400:
                logger.error(f"Supabase INSERT {table}: {r.status_code} {r.text}")
                raise HTTPException(status_code=r.status_code, detail=r.text)
            return r.json()

    async def upsert(self, table: str, data: dict, on_conflict: str = "id") -> list:
        h = {**self.headers, "Prefer": "return=representation,resolution=merge-duplicates"}
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(f"{self.base}/{table}?on_conflict={on_conflict}", headers=h, json=data)
            if r.status_code >= 400:
                logger.error(f"Supabase UPSERT {table}: {r.status_code} {r.text}")
                raise HTTPException(status_code=r.status_code, detail=r.text)
            return r.json()

    async def update(self, table: str, data: dict, match: dict) -> list:
        h = {**self.headers, "Prefer": "return=representation"}
        params = {k: f"eq.{v}" for k, v in match.items()}
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.patch(f"{self.base}/{table}", headers=h, json=data, params=params)
            if r.status_code >= 400:
                logger.error(f"Supabase UPDATE {table}: {r.status_code} {r.text}")
                raise HTTPException(status_code=r.status_code, detail=r.text)
            return r.json()

    async def delete(self, table: str, match: dict) -> None:
        params = {k: f"eq.{v}" for k, v in match.items()}
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.delete(f"{self.base}/{table}", headers=self.headers, params=params)
            if r.status_code >= 400:
                logger.error(f"Supabase DELETE {table}: {r.status_code} {r.text}")
                raise HTTPException(status_code=r.status_code, detail=r.text)


# ---- Auth helpers ----
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


# ---- Request models ----
class ChatRequest(BaseModel):
    message: str
    pending_reminder: Optional[Dict[str, Any]] = None


class ProfileUpdate(BaseModel):
    timezone: str


class ReminderStatusUpdate(BaseModel):
    status: str


# ---- LLM Extraction ----
EXTRACTION_PROMPT = """You are SecondBrain, a personal assistant. Process the user's message and determine the appropriate action.

Current date/time (user's local): {now}
User's default timezone: {timezone}

RULES:
1. If the message contains ONLY factual information (a fact, note, contact info, order details, preferences) with NO date/time/follow-up intent, classify as "kb".
2. If the message mentions ONLY a future action, deadline, or follow-up with date/time intent, classify as "reminder".
3. If the message contains BOTH factual information AND a follow-up/action with date/time intent, you MUST classify as "both" and fill BOTH kb_entry AND reminder. Examples of mixed input: "PO 600 shipment delayed due to weather. Follow up Friday at 2 PM." — the factual update is kb_entry, the follow-up is a reminder.
4. If a reminder is being built (see PENDING STATE) but required fields are still missing after considering this message, classify as "clarify" and ask exactly ONE specific question for the missing field.
5. For general conversation or greetings, classify as "chat".

REMINDER RULES:
- The ONLY truly required field is due_datetime_local. If the user gives a date/time but no explicit task/subject, set title to "Reminder" and create immediately. Do NOT ask for title.
- If the user provides a relative time like "tomorrow" or "next Monday at 3pm", compute the actual date from the current date/time.
- If no time is given for a reminder, default to 09:00.
- Once due_datetime_local is present, create the reminder immediately. Do NOT ask more questions.
- Extract order_ref/reference numbers (PO, INV, CASE, etc.) when present and attach to BOTH kb_entry and reminder.

TIMEZONE RULE:
- If the user explicitly names a timezone (e.g. "America/New_York", "EST", "PST", "IST", "UTC"), set reminder.timezone to the IANA timezone string and compute due_datetime_local in THAT timezone.
- If the user does NOT mention a timezone, set reminder.timezone to null (the system will use the default).

PENDING REMINDER STATE: {pending_reminder}
If there is a pending reminder, try to fill missing fields from the current message.

RETURN ONLY VALID JSON (no markdown, no code fences):
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
  "response": "friendly natural language response confirming what you did"
}}"""


async def extract_message(message: str, user_tz: str, pending: dict = None, recent: list = None) -> dict:
    now_str = datetime.now(ZoneInfo(user_tz)).strftime("%Y-%m-%d %H:%M:%S %A")
    system_msg = EXTRACTION_PROMPT.format(
        now=now_str,
        timezone=user_tz,
        pending_reminder=json.dumps(pending) if pending else "None"
    )

    chat = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=f"extract-{uuid.uuid4()}",
        system_message=system_msg
    ).with_model("openai", "gpt-4o")

    parts = []
    if recent:
        parts.append("Recent conversation:")
        for m in recent[-6:]:
            parts.append(f"  {m['role']}: {m['content']}")
        parts.append("")
    parts.append(f"User says: {message}")

    response = await chat.send_message(UserMessage(text="\n".join(parts)))

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        m = re.search(r'\{[\s\S]*\}', response)
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
            "response": response
        }


def local_to_utc(local_dt_str: str, tz_str: str) -> str:
    local_dt = datetime.fromisoformat(local_dt_str)
    if local_dt.tzinfo is None:
        local_dt = local_dt.replace(tzinfo=ZoneInfo(tz_str))
    return local_dt.astimezone(ZoneInfo('UTC')).isoformat()


# ---- API Endpoints ----
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


@api_router.post("/chat")
async def chat_endpoint(body: ChatRequest, request: Request):
    db = get_db(request)
    uid = get_user_id(request)

    # Get timezone
    profiles = await db.select("user_profile", {"id": f"eq.{uid}"})
    user_tz = profiles[0]["timezone"] if profiles else "UTC"

    # Recent messages for context
    recent = await db.select("chat_messages", {
        "user_id": f"eq.{uid}",
        "order": "created_at.desc",
        "limit": "10"
    })
    recent.reverse()

    # Save user message
    await db.insert("chat_messages", {"user_id": uid, "role": "user", "content": body.message})

    # LLM extraction
    extraction = await extract_message(
        body.message, user_tz, body.pending_reminder,
        [{"role": m["role"], "content": m["content"]} for m in recent]
    )

    result_type = extraction.get("type", "chat")
    response_text = extraction.get("response", "Got it!")
    saved_kb = None
    saved_reminder = None

    # Server-side fallback: if LLM returned "reminder" but also filled kb_entry, treat as "both"
    if result_type == "reminder" and extraction.get("kb_entry") and extraction.get("reminder"):
        result_type = "both"
    if result_type == "kb" and extraction.get("reminder") and extraction.get("kb_entry"):
        result_type = "both"

    # Helper: save a KB entry
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

    # Helper: save a reminder and build absolute confirmation text
    async def _save_reminder(rem_data):
        due_local = rem_data.get("due_datetime_local", "")
        title = rem_data.get("title") or "Reminder"
        # Issue 1: Use explicit timezone from LLM if provided, else fall back to profile
        explicit_tz = rem_data.get("timezone")
        eff_tz = explicit_tz if explicit_tz else user_tz
        due_utc = local_to_utc(due_local, eff_tz) if due_local else None
        rdata = {
            "user_id": uid,
            "title": title,
            "due_datetime": due_utc,
            "timezone": eff_tz,
            "related_order_ref": rem_data.get("order_ref"),
            "related_party": rem_data.get("related_party"),
            "status": "open",
            "source_message": body.message,
        }
        rows = await db.insert("reminders", rdata)
        saved = rows[0] if rows else rdata
        try:
            dt_local = datetime.fromisoformat(due_local).replace(tzinfo=ZoneInfo(eff_tz)) if due_local else None
            if dt_local:
                abs_str = dt_local.strftime("%b %-d, %Y at %-I:%M %p")
                saved["_confirmation"] = f"Reminder created: {title} \u2014 {abs_str} ({eff_tz})"
        except Exception:
            pass
        return saved

    # Process based on type
    if result_type == "both":
        # Mixed: save BOTH KB + reminder
        if extraction.get("kb_entry"):
            saved_kb = await _save_kb(extraction["kb_entry"])
        if extraction.get("reminder"):
            saved_reminder = await _save_reminder(extraction["reminder"])
        # Build combined confirmation
        parts = []
        if saved_kb:
            parts.append(f"Saved to Knowledge Base: {saved_kb.get('details', '')[:80]}")
        if saved_reminder and saved_reminder.get("_confirmation"):
            parts.append(saved_reminder["_confirmation"])
        if parts:
            response_text = " | ".join(parts)

    elif result_type == "kb" and extraction.get("kb_entry"):
        saved_kb = await _save_kb(extraction["kb_entry"])

    elif result_type == "reminder" and extraction.get("reminder"):
        saved_reminder = await _save_reminder(extraction["reminder"])
        # Override response with absolute confirmation (Issue 3)
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
async def get_messages(request: Request, limit: int = 50):
    db = get_db(request)
    uid = get_user_id(request)
    return await db.select("chat_messages", {
        "user_id": f"eq.{uid}",
        "order": "created_at.asc",
        "limit": str(limit),
    })


@api_router.get("/kb")
async def get_kb(request: Request, entity_type: str = None, search: str = None):
    db = get_db(request)
    uid = get_user_id(request)
    params: dict = {"user_id": f"eq.{uid}", "order": "created_at.desc"}
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
    params: dict = {"user_id": f"eq.{uid}", "order": "due_datetime.asc"}
    if status:
        params["status"] = f"eq.{status}"
    return await db.select("reminders", params)


@api_router.patch("/reminders/{reminder_id}")
async def update_reminder(reminder_id: str, body: ReminderStatusUpdate, request: Request):
    db = get_db(request)
    uid = get_user_id(request)
    result = await db.update("reminders", {"status": body.status}, {"id": reminder_id, "user_id": uid})
    return result[0] if result else {"ok": True}


# ---- Middleware & startup ----
app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)
