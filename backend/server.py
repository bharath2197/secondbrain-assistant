import os
import logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AsyncOpenAI


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_LEVEL = (os.getenv("LOG_LEVEL") or "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("server")


# -----------------------------------------------------------------------------
# App + CORS
# Keep CORS middleware using env var CORS_ORIGINS (comma-separated) with fallback "*".
# -----------------------------------------------------------------------------
app = FastAPI()

cors_origins_raw = (os.getenv("CORS_ORIGINS") or "").strip()
if cors_origins_raw:
    origins = [o.strip() for o in cors_origins_raw.split(",") if o.strip()]
else:
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY") or ""
SUPABASE_REST_URL = f"{SUPABASE_URL}/rest/v1" if SUPABASE_URL else ""
SUPABASE_AUTH_URL = f"{SUPABASE_URL}/auth/v1" if SUPABASE_URL else ""

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = (os.getenv("GROQ_MODEL", "") or "").strip()  # strip whitespace/newlines
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIMEZONE", "America/Los_Angeles")

# Tables (must match exactly)
TABLE_CHAT_MESSAGES = "chat_messages"
TABLE_KB = "kb_entries"
TABLE_REMINDERS = "reminders"
TABLE_PROFILE = "user_profile"

openai_client = AsyncOpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None  # accepted for compatibility, not stored (no column exists)
    metadata: Dict[str, Any] = Field(default_factory=dict)  # accepted for compatibility, not stored


class ChatResponse(BaseModel):
    reply: str
    thread_id: Optional[str] = None


class ProfileUpsertRequest(BaseModel):
    timezone: Optional[str] = None


class KbCreateRequest(BaseModel):
    # Match actual columns for kb_entries
    entity_type: Optional[str] = None
    entity_name: Optional[str] = None
    order_ref: Optional[str] = None
    details: Optional[str] = None
    source_message: Optional[str] = None

    # Compatibility fields (some frontends might still send these)
    content: Optional[str] = None


class ReminderCreateRequest(BaseModel):
    title: str
    due_datetime: str  # ISO string with timezone offset preferred
    timezone: str
    related_order_ref: Optional[str] = None
    related_party: Optional[str] = None
    status: Optional[str] = "open"
    notes: Optional[str] = None
    source_message: Optional[str] = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _extract_bearer_token(authorization: str) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization Bearer token")
    return authorization.split(" ", 1)[1].strip()


def _require_supabase() -> None:
    if not SUPABASE_URL or not SUPABASE_REST_URL or not SUPABASE_AUTH_URL:
        raise HTTPException(status_code=500, detail="Supabase not configured (SUPABASE_URL)")


def _require_groq() -> None:
    if not GROQ_API_KEY or not GROQ_MODEL:
        raise HTTPException(status_code=500, detail="Groq not configured (GROQ_API_KEY / GROQ_MODEL)")


def _sb_headers(user_jwt: str) -> Dict[str, str]:
    # Supabase REST style: pass user JWT, include anon key as apikey.
    return {
        "Authorization": f"Bearer {user_jwt}",
        "apikey": SUPABASE_ANON_KEY,
        "Content-Type": "application/json",
    }


def _log_supabase_failure(resp: httpx.Response, context: str) -> None:
    try:
        body_preview = resp.text[:2000]
    except Exception:
        body_preview = "<unreadable>"
    logger.error(
        "Supabase %s failed: status=%s url=%s body=%s",
        context,
        resp.status_code,
        str(resp.request.url),
        body_preview,
    )


async def _sb_request(
    method: str,
    user_jwt: str,
    table_or_path: str,
    *,
    params: Optional[Dict[str, str]] = None,
    json_body: Any = None,
    prefer: Optional[str] = None,
    timeout_s: float = 25.0,
) -> httpx.Response:
    _require_supabase()
    url = f"{SUPABASE_REST_URL}/{table_or_path.lstrip('/')}"
    headers = _sb_headers(user_jwt)
    if prefer:
        headers["Prefer"] = prefer

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.request(method, url, headers=headers, params=params, json=json_body)
        return resp
    except Exception as e:
        logger.exception("Supabase request exception method=%s url=%s", method, url)
        raise HTTPException(status_code=500, detail=f"Supabase request failed: {str(e)}")


async def _sb_get_user(user_jwt: str) -> Dict[str, Any]:
    _require_supabase()
    url = f"{SUPABASE_AUTH_URL}/user"
    headers = _sb_headers(user_jwt)

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url, headers=headers)
    except Exception as e:
        logger.exception("Supabase auth /user exception")
        raise HTTPException(status_code=500, detail=f"Supabase auth request failed: {str(e)}")

    if resp.status_code != 200:
        _log_supabase_failure(resp, "AUTH /user")
        raise HTTPException(status_code=401, detail="Invalid Supabase token")

    data = resp.json() or {}
    if not data.get("id"):
        raise HTTPException(status_code=401, detail="Invalid Supabase user")
    return data


# -----------------------------------------------------------------------------
# Profile (table: user_profile)
# Actual columns: id (uuid), created_at, user_email, timezone
# Important bug fix based on your schema: the PK is id, not user_id.
# -----------------------------------------------------------------------------
async def _get_or_create_profile(user_jwt: str, user_id: str, user_email: str) -> Dict[str, Any]:
    resp = await _sb_request(
        "GET",
        user_jwt,
        TABLE_PROFILE,
        params={
            "select": "id,created_at,user_email,timezone",
            "id": f"eq.{user_id}",
            "limit": "1",
        },
    )
    if resp.status_code >= 400:
        _log_supabase_failure(resp, "SELECT user_profile")
        raise HTTPException(status_code=500, detail="Failed to load profile")

    rows = resp.json() or []
    if rows:
        prof = rows[0]
        if not prof.get("timezone"):
            prof["timezone"] = DEFAULT_TIMEZONE
        return prof

    # Auto-create missing profile row
    new_row = {
        "id": user_id,
        "user_email": user_email or "",
        "timezone": DEFAULT_TIMEZONE,
    }
    ins = await _sb_request(
        "POST",
        user_jwt,
        TABLE_PROFILE,
        params={"on_conflict": "id"},
        json_body=[new_row],
        prefer="return=representation,resolution=merge-duplicates",
    )
    if ins.status_code >= 400:
        _log_supabase_failure(ins, "UPSERT user_profile autocreate")
        raise HTTPException(status_code=500, detail="Failed to auto-create profile")

    created = ins.json() or []
    return created[0] if created else new_row


async def _upsert_profile(user_jwt: str, user_id: str, user_email: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    row = {"id": user_id, "user_email": user_email or ""}
    row.update(patch)

    resp = await _sb_request(
        "POST",
        user_jwt,
        TABLE_PROFILE,
        params={"on_conflict": "id"},
        json_body=[row],
        prefer="return=representation,resolution=merge-duplicates",
    )
    if resp.status_code >= 400:
        _log_supabase_failure(resp, "UPSERT user_profile")
        raise HTTPException(status_code=500, detail="Failed to save profile")

    rows = resp.json() or []
    return rows[0] if rows else await _get_or_create_profile(user_jwt, user_id, user_email)


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/profile")
async def get_profile(authorization: str = Header(default="")):
    token = _extract_bearer_token(authorization)
    user = await _sb_get_user(token)
    user_id = user["id"]
    user_email = user.get("email") or ""

    prof = await _get_or_create_profile(token, user_id, user_email)
    return {"user": {"id": user_id, "email": user_email}, "profile": prof}


# Support both PATCH and POST to prevent 405 from mismatched frontend method.
@app.patch("/api/profile")
async def patch_profile(payload: ProfileUpsertRequest, authorization: str = Header(default="")):
    token = _extract_bearer_token(authorization)
    user = await _sb_get_user(token)
    user_id = user["id"]
    user_email = user.get("email") or ""

    patch: Dict[str, Any] = {}
    if payload.timezone is not None:
        patch["timezone"] = payload.timezone

    if not patch:
        prof = await _get_or_create_profile(token, user_id, user_email)
        return {"profile": prof}

    prof = await _upsert_profile(token, user_id, user_email, patch)
    return {"profile": prof}


@app.post("/api/profile")
async def post_profile(payload: ProfileUpsertRequest, authorization: str = Header(default="")):
    return await patch_profile(payload, authorization=authorization)


@app.get("/api/messages")
async def get_messages(
    request: Request,
    limit: int = 200,
    authorization: str = Header(default=""),
):
    token = _extract_bearer_token(authorization)
    user = await _sb_get_user(token)
    user_id = user["id"]

    # Fetch desc then reverse to return ascending and remain robust after relogin
    resp = await _sb_request(
        "GET",
        token,
        TABLE_CHAT_MESSAGES,
        params={
            "select": "id,created_at,user_id,role,content",
            "user_id": f"eq.{user_id}",
            "order": "created_at.desc",
            "limit": str(limit),
        },
    )
    if resp.status_code >= 400:
        _log_supabase_failure(resp, "SELECT chat_messages")
        raise HTTPException(status_code=500, detail="Failed to load messages")

    rows = resp.json() or []
    rows.reverse()
    return {"messages": rows}


def _chat_system_prompt(timezone_name: str) -> str:
    return (
        "You are a helpful assistant.\n"
        f"User timezone: {timezone_name}\n"
        "Reply concisely.\n"
    )


async def _insert_chat_message(
    user_jwt: str,
    user_id: str,
    role: str,
    content: str,
) -> Dict[str, Any]:
    row = {
        "user_id": user_id,
        "role": role,
        "content": content,
    }
    resp = await _sb_request(
        "POST",
        user_jwt,
        TABLE_CHAT_MESSAGES,
        json_body=[row],
        prefer="return=representation",
    )
    if resp.status_code >= 400:
        _log_supabase_failure(resp, f"INSERT chat_messages role={role}")
        raise HTTPException(status_code=500, detail="Failed to insert message")

    created = resp.json() or []
    return created[0] if created else row


async def _load_recent_chat_messages(user_jwt: str, user_id: str, limit: int = 30) -> List[Dict[str, str]]:
    resp = await _sb_request(
        "GET",
        user_jwt,
        TABLE_CHAT_MESSAGES,
        params={
            "select": "role,content,created_at",
            "user_id": f"eq.{user_id}",
            "order": "created_at.desc",
            "limit": str(limit),
        },
    )
    if resp.status_code >= 400:
        _log_supabase_failure(resp, "SELECT chat_messages history")
        raise HTTPException(status_code=500, detail="Failed to load chat history")

    rows = resp.json() or []
    rows.reverse()

    msgs: List[Dict[str, str]] = []
    for r in rows:
        role = (r.get("role") or "user").strip().lower()
        if role not in ("user", "assistant", "system"):
            role = "user"
        msgs.append({"role": role, "content": r.get("content") or ""})
    return msgs


@app.post("/api/chat")
async def chat(req: ChatRequest, authorization: str = Header(default="")):
    token = _extract_bearer_token(authorization)
    _require_groq()

    user = await _sb_get_user(token)
    user_id = user["id"]
    user_email = user.get("email") or ""

    # Ensure profile exists so timezone does not break chat
    prof = await _get_or_create_profile(token, user_id, user_email)
    tz_name = (prof.get("timezone") or DEFAULT_TIMEZONE).strip() or DEFAULT_TIMEZONE

    user_text = (req.message or "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Empty message")

    # Insert user message into chat_messages
    await _insert_chat_message(token, user_id, "user", user_text)

    # Load recent messages, call Groq
    history = await _load_recent_chat_messages(token, user_id, limit=30)
    groq_messages = [{"role": "system", "content": _chat_system_prompt(tz_name)}] + history

    try:
        completion = await openai_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=groq_messages,
            temperature=0.2,
        )
        reply_text = (completion.choices[0].message.content or "").strip()
        if not reply_text:
            reply_text = "OK."
    except Exception as e:
        logger.exception("Groq chat completion failed")
        raise HTTPException(status_code=500, detail=f"Groq error: {str(e)}")

    # Insert assistant message into chat_messages
    await _insert_chat_message(token, user_id, "assistant", reply_text)

    return ChatResponse(reply=reply_text, thread_id=req.thread_id)


@app.get("/api/kb")
async def get_kb(limit: int = 200, authorization: str = Header(default="")):
    token = _extract_bearer_token(authorization)
    user = await _sb_get_user(token)
    user_id = user["id"]

    resp = await _sb_request(
        "GET",
        token,
        TABLE_KB,
        params={
            "select": "id,created_at,user_id,entity_type,entity_name,order_ref,details,source_message",
            "user_id": f"eq.{user_id}",
            "order": "created_at.desc",
            "limit": str(limit),
        },
    )
    if resp.status_code >= 400:
        _log_supabase_failure(resp, "SELECT kb_entries")
        raise HTTPException(status_code=500, detail="Failed to load KB")

    return {"kb": resp.json() or []}


@app.post("/api/kb")
async def create_kb(payload: KbCreateRequest, authorization: str = Header(default="")):
    token = _extract_bearer_token(authorization)
    user = await _sb_get_user(token)
    user_id = user["id"]

    details = (payload.details or payload.content or "").strip()
    if not details:
        raise HTTPException(status_code=400, detail="Missing details")

    row = {
        "user_id": user_id,
        "entity_type": (payload.entity_type or "note"),
        "entity_name": (payload.entity_name or ""),
        "order_ref": (payload.order_ref or ""),
        "details": details,
        "source_message": (payload.source_message or ""),
    }

    resp = await _sb_request(
        "POST",
        token,
        TABLE_KB,
        json_body=[row],
        prefer="return=representation",
    )
    if resp.status_code >= 400:
        _log_supabase_failure(resp, "INSERT kb_entries")
        raise HTTPException(status_code=500, detail="Failed to create KB entry")

    created = resp.json() or []
    return {"kb": created[0] if created else row}


@app.get("/api/reminders")
async def get_reminders(
    limit: int = 500,
    status: Optional[str] = Query(default=None),
    authorization: str = Header(default=""),
):
    token = _extract_bearer_token(authorization)
    user = await _sb_get_user(token)
    user_id = user["id"]

    params: Dict[str, str] = {
        "select": "id,created_at,user_id,title,due_datetime,timezone,related_order_ref,related_party,status,emailed_at,notes,source_message",
        "user_id": f"eq.{user_id}",
        "order": "due_datetime.asc",
        "limit": str(limit),
    }
    if status:
        params["status"] = f"eq.{status}"

    resp = await _sb_request("GET", token, TABLE_REMINDERS, params=params)
    if resp.status_code >= 400:
        _log_supabase_failure(resp, "SELECT reminders")
        raise HTTPException(status_code=500, detail="Failed to load reminders")

    return {"reminders": resp.json() or []}


@app.post("/api/reminders")
async def create_reminder(payload: ReminderCreateRequest, authorization: str = Header(default="")):
    token = _extract_bearer_token(authorization)
    user = await _sb_get_user(token)
    user_id = user["id"]

    title = (payload.title or "").strip()
    due_dt = (payload.due_datetime or "").strip()
    tz_name = (payload.timezone or "").strip()
    if not title or not due_dt or not tz_name:
        raise HTTPException(status_code=400, detail="Missing title, due_datetime, or timezone")

    row = {
        "user_id": user_id,
        "title": title,
        "due_datetime": due_dt,
        "timezone": tz_name,
        "related_order_ref": payload.related_order_ref or "",
        "related_party": payload.related_party or "",
        "status": (payload.status or "open"),
        "notes": payload.notes or "",
        "source_message": payload.source_message or "",
    }

    resp = await _sb_request(
        "POST",
        token,
        TABLE_REMINDERS,
        json_body=[row],
        prefer="return=representation",
    )
    if resp.status_code >= 400:
        _log_supabase_failure(resp, "INSERT reminders")
        raise HTTPException(status_code=500, detail="Failed to create reminder")

    created = resp.json() or []
    return {"reminder": created[0] if created else row}


@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": "secondbrain-backend",
        "cors_origins": origins,
        "groq_model": GROQ_MODEL,
    }
