import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, date, time
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AsyncOpenAI


# -----------------------------------------------------------------------------
# App + CORS (keep env var CORS_ORIGINS comma-separated with fallback "*")
# -----------------------------------------------------------------------------
app = FastAPI()

cors_origins_raw = os.getenv("CORS_ORIGINS", "").strip()
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
# Env
# -----------------------------------------------------------------------------
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY") or ""
SUPABASE_REST_URL = f"{SUPABASE_URL}/rest/v1" if SUPABASE_URL else ""
SUPABASE_AUTH_URL = f"{SUPABASE_URL}/auth/v1" if SUPABASE_URL else ""

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = (os.getenv("GROQ_MODEL", "") or "").strip()  # strip whitespace/newlines to avoid model_not_found
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIMEZONE", "America/Los_Angeles")

openai_client = AsyncOpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    reply: str
    actions: Dict[str, Any] = Field(default_factory=dict)


class ProfilePatch(BaseModel):
    timezone: Optional[str] = None
    display_name: Optional[str] = None
    pending_reminder: Optional[Dict[str, Any]] = None


# -----------------------------------------------------------------------------
# Supabase REST helpers (keep Authorization Bearer token from request)
# -----------------------------------------------------------------------------
def _require_supabase():
    if not SUPABASE_URL or not SUPABASE_REST_URL:
        raise HTTPException(status_code=500, detail="SUPABASE_URL not configured")


def _sb_headers(user_jwt: str) -> Dict[str, str]:
    # Keep Supabase REST style as-is: pass through user JWT, include anon key.
    hdrs = {
        "Authorization": f"Bearer {user_jwt}",
        "apikey": SUPABASE_ANON_KEY,
        "Content-Type": "application/json",
    }
    return hdrs


async def _sb_get_user(user_jwt: str) -> Dict[str, Any]:
    _require_supabase()
    if not SUPABASE_AUTH_URL:
        raise HTTPException(status_code=500, detail="Supabase auth URL not configured")

    url = f"{SUPABASE_AUTH_URL}/user"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, headers=_sb_headers(user_jwt))
    if r.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid Supabase token")
    return r.json()


async def _sb_select(
    user_jwt: str,
    table: str,
    select: str = "*",
    filters: Optional[Dict[str, str]] = None,
    order: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    _require_supabase()
    params: Dict[str, str] = {"select": select}
    if filters:
        params.update(filters)
    if order:
        params["order"] = order
    if limit is not None:
        params["limit"] = str(limit)

    url = f"{SUPABASE_REST_URL}/{table}"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, headers=_sb_headers(user_jwt), params=params)
    if r.status_code >= 400:
        raise HTTPException(status_code=500, detail=f"Supabase select failed: {r.text}")
    return r.json() or []


async def _sb_insert(
    user_jwt: str,
    table: str,
    rows: List[Dict[str, Any]],
    returning: str = "representation",
) -> List[Dict[str, Any]]:
    _require_supabase()
    url = f"{SUPABASE_REST_URL}/{table}"
    headers = _sb_headers(user_jwt)
    headers["Prefer"] = f"return={returning}"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, headers=headers, content=json.dumps(rows))
    if r.status_code >= 400:
        raise HTTPException(status_code=500, detail=f"Supabase insert failed: {r.text}")
    return r.json() or []


async def _sb_upsert(
    user_jwt: str,
    table: str,
    rows: List[Dict[str, Any]],
    on_conflict: Optional[str] = None,
    returning: str = "representation",
) -> List[Dict[str, Any]]:
    _require_supabase()
    url = f"{SUPABASE_REST_URL}/{table}"
    headers = _sb_headers(user_jwt)
    prefer_parts = [f"return={returning}", "resolution=merge-duplicates"]
    headers["Prefer"] = ", ".join(prefer_parts)
    params = {}
    if on_conflict:
        params["on_conflict"] = on_conflict
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, headers=headers, params=params, content=json.dumps(rows))
    if r.status_code >= 400:
        raise HTTPException(status_code=500, detail=f"Supabase upsert failed: {r.text}")
    return r.json() or []


async def _sb_patch(
    user_jwt: str,
    table: str,
    filters: Dict[str, str],
    patch: Dict[str, Any],
    returning: str = "representation",
) -> List[Dict[str, Any]]:
    _require_supabase()
    url = f"{SUPABASE_REST_URL}/{table}"
    headers = _sb_headers(user_jwt)
    headers["Prefer"] = f"return={returning}"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.patch(url, headers=headers, params=filters, content=json.dumps(patch))
    if r.status_code >= 400:
        raise HTTPException(status_code=500, detail=f"Supabase patch failed: {r.text}")
    return r.json() or []


# -----------------------------------------------------------------------------
# Time parsing and reminder parsing helpers
# -----------------------------------------------------------------------------
WEEKDAY_MAP = {
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
# ONLY parse explicit times like "3pm", "3 pm", "3:30pm", or "15:30".
# Do not parse naked numbers.
TIME_12H_RE = re.compile(r"\b(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*(am|pm)\b", re.IGNORECASE)
TIME_24H_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")

# Relative duration patterns
IN_DAYS_RE = re.compile(r"\b(in\s+)?(\d+)\s+day(s)?\b", re.IGNORECASE)
IN_WEEKS_RE = re.compile(r"\b(in\s+)?(\d+)\s+week(s)?\b", re.IGNORECASE)
IN_HOURS_RE = re.compile(r"\b(in\s+)?(\d+)\s+hour(s)?\b", re.IGNORECASE)
TOMORROW_RE = re.compile(r"\btomorrow\b", re.IGNORECASE)
TODAY_RE = re.compile(r"\btoday\b", re.IGNORECASE)

NEXT_THIS_WEEKDAY_RE = re.compile(
    r"\b(?:(next|this)\s+)?(mon(day)?|tue(s(day)?)?|wed(nesday)?|thu(r(s(day)?)?)?|fri(day)?|sat(urday)?|sun(day)?)\b",
    re.IGNORECASE,
)

AT_RE = re.compile(r"\bat\b", re.IGNORECASE)

# "follow up" style defaults
DEFAULT_REMINDER_TIME = time(9, 0)  # 09:00 local when no explicit time is provided


def _safe_tz(tz_name: Optional[str]) -> ZoneInfo:
    try:
        return ZoneInfo(tz_name or DEFAULT_TIMEZONE)
    except Exception:
        return ZoneInfo(DEFAULT_TIMEZONE)


def _now_local(tz: ZoneInfo) -> datetime:
    return datetime.now(tz)


def _format_dt_local(dt_local: datetime, tz_name: str) -> str:
    # "Mon D, YYYY at H:MM AM/PM (Timezone)"
    # Example: "Feb 27, 2026 at 3:00 PM (America/Los_Angeles)"
    mon = dt_local.strftime("%b")
    day = dt_local.day
    yr = dt_local.year
    hm = dt_local.strftime("%I:%M %p").lstrip("0")
    return f"{mon} {day}, {yr} at {hm} ({tz_name})"


def _parse_explicit_time(text: str) -> Optional[time]:
    """
    Bug fix 1: strict time parsing that ignores naked numbers.
    """
    m24 = TIME_24H_RE.search(text)
    if m24:
        hh = int(m24.group(1))
        mm = int(m24.group(2))
        return time(hh, mm)

    m12 = TIME_12H_RE.search(text)
    if m12:
        hh = int(m12.group(1))
        mm = int(m12.group(2) or "0")
        ampm = (m12.group(3) or "").lower()
        if ampm == "pm" and hh != 12:
            hh += 12
        if ampm == "am" and hh == 12:
            hh = 0
        return time(hh, mm)

    return None


def _next_weekday(base_date: date, target_weekday: int) -> date:
    days_ahead = (target_weekday - base_date.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return base_date + timedelta(days=days_ahead)


def _upcoming_weekday_including_today(base_date: date, target_weekday: int) -> date:
    days_ahead = (target_weekday - base_date.weekday()) % 7
    return base_date + timedelta(days=days_ahead)


def _resolve_weekday_date(
    base_date: date,
    target_weekday: int,
    qualifier: Optional[str],
) -> date:
    """
    Interprets:
      - "Friday" => upcoming Friday (including today if today is Friday)
      - "this Friday" => upcoming Friday (including today if today is Friday)
      - "next Friday" => Friday of next week (always +7 from upcoming)
    """
    qualifier_norm = (qualifier or "").lower().strip()
    upcoming = _upcoming_weekday_including_today(base_date, target_weekday)

    if qualifier_norm == "next":
        # If upcoming is this week's occurrence (possibly tomorrow), "next" means one week after that.
        return upcoming + timedelta(days=7)
    # "this" or none
    return upcoming


def _parse_due_datetime_local(
    text: str,
    tz_name: str,
    now_local: Optional[datetime] = None,
    pending_reminder: Optional[Dict[str, Any]] = None,
) -> Optional[datetime]:
    """
    Returns a timezone-aware local datetime if parseable, else None.

    Bug fix 3: pending reminder carry-over.
    If user says a weekday without a time and pending_reminder has due_datetime_local,
    reuse its hour/minute rather than defaulting to 09:00.
    """
    tz = _safe_tz(tz_name)
    now = now_local or _now_local(tz)
    text_l = text.strip()

    explicit_t = _parse_explicit_time(text_l)

    # Determine the time to use if missing:
    # Bug fix 3: carry over hour/minute from pending reminder if available.
    carried_time: Optional[time] = None
    if pending_reminder and isinstance(pending_reminder, dict):
        prior = pending_reminder.get("due_datetime_local")
        if isinstance(prior, str) and prior:
            try:
                prior_dt = datetime.fromisoformat(prior)
                if prior_dt.tzinfo is None:
                    prior_dt = prior_dt.replace(tzinfo=tz)
                carried_time = prior_dt.timetz().replace(tzinfo=None)  # time object
            except Exception:
                carried_time = None

    def pick_time() -> time:
        if explicit_t:
            return explicit_t
        if carried_time:
            return time(carried_time.hour, carried_time.minute)
        return DEFAULT_REMINDER_TIME

    # today/tomorrow
    if TOMORROW_RE.search(text_l):
        d = (now + timedelta(days=1)).date()
        return datetime.combine(d, pick_time(), tzinfo=tz)
    if TODAY_RE.search(text_l):
        d = now.date()
        return datetime.combine(d, pick_time(), tzinfo=tz)

    # relative durations (in X days/weeks/hours)
    m_weeks = IN_WEEKS_RE.search(text_l)
    if m_weeks:
        weeks = int(m_weeks.group(2))
        d = (now + timedelta(days=7 * weeks)).date()
        return datetime.combine(d, pick_time(), tzinfo=tz)

    m_days = IN_DAYS_RE.search(text_l)
    if m_days:
        days = int(m_days.group(2))
        d = (now + timedelta(days=days)).date()
        return datetime.combine(d, pick_time(), tzinfo=tz)

    m_hours = IN_HOURS_RE.search(text_l)
    if m_hours:
        hours = int(m_hours.group(2))
        # If user gives hours, keep it as now + hours (but if no explicit minutes, keep minute=0)
        dt = now + timedelta(hours=hours)
        if explicit_t:
            # If they explicitly included a time, treat it as a clock time (rare with hours phrasing)
            dt = datetime.combine(dt.date(), explicit_t, tzinfo=tz)
        else:
            dt = dt.replace(second=0, microsecond=0)
        return dt

    # weekdays (next/this/none)
    m_wd = NEXT_THIS_WEEKDAY_RE.search(text_l)
    if m_wd:
        qualifier = m_wd.group(1)  # next/this/None
        wd_token = (m_wd.group(2) or "").lower()
        wd_token = re.sub(r"[^a-z]", "", wd_token)
        if wd_token in WEEKDAY_MAP:
            target = WEEKDAY_MAP[wd_token]
            resolved_date = _resolve_weekday_date(now.date(), target, qualifier)
            return datetime.combine(resolved_date, pick_time(), tzinfo=tz)

    return None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort: find a JSON object in a model response.
    """
    if not text:
        return None
    text = text.strip()

    # If it's already JSON
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass

    # Try to locate first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        blob = text[start : end + 1]
        try:
            return json.loads(blob)
        except Exception:
            return None
    return None


def _build_system_prompt(tz_name: str) -> str:
    return (
        "You are SecondBrain Assistant. Extract structured actions.\n"
        "Return ONLY a JSON object with keys:\n"
        '  "type": one of ["message","reminder","kb","mixed","clarify"]\n'
        '  "reply": a helpful assistant reply to show the user\n'
        '  "reminder": optional object { "title": string, "due_text": string }\n'
        '  "kb": optional object { "content": string }\n'
        '  "clarify_question": optional string when type="clarify"\n'
        "Rules:\n"
        "1) If user asks to remember/save a fact, use kb.\n"
        "2) If user asks to be reminded at a time/date, use reminder.\n"
        "3) For mixed (both), include both.\n"
        "4) Use the user's timezone: "
        + tz_name
        + "\n"
        "5) If date is ambiguous and cannot be resolved, use type=clarify and ask one question.\n"
    )


# -----------------------------------------------------------------------------
# Profile helpers (including pending reminder storage for bug fix 3)
# -----------------------------------------------------------------------------
async def _get_profile(user_jwt: str, user_id: str) -> Dict[str, Any]:
    rows = await _sb_select(
        user_jwt,
        "profiles",
        select="user_id,timezone,display_name,pending_reminder",
        filters={"user_id": f"eq.{user_id}"},
        limit=1,
    )
    if rows:
        prof = rows[0]
        # Ensure defaults
        prof.setdefault("timezone", DEFAULT_TIMEZONE)
        prof.setdefault("pending_reminder", None)
        return prof

    # Create if missing
    created = await _sb_insert(
        user_jwt,
        "profiles",
        [
            {
                "user_id": user_id,
                "timezone": DEFAULT_TIMEZONE,
                "display_name": None,
                "pending_reminder": None,
            }
        ],
    )
    return created[0] if created else {"user_id": user_id, "timezone": DEFAULT_TIMEZONE, "pending_reminder": None}


async def _set_pending_reminder(user_jwt: str, user_id: str, pending: Optional[Dict[str, Any]]) -> None:
    await _sb_upsert(
        user_jwt,
        "profiles",
        [{"user_id": user_id, "pending_reminder": pending}],
        on_conflict="user_id",
        returning="minimal",
    )


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/profile")
async def get_profile(authorization: str = Header(default="")):
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization Bearer token")
    token = authorization.split(" ", 1)[1].strip()

    user = await _sb_get_user(token)
    user_id = user.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user")

    prof = await _get_profile(token, user_id)
    return {"user": {"id": user_id, "email": user.get("email")}, "profile": prof}


@app.patch("/api/profile")
async def patch_profile(payload: ProfilePatch, authorization: str = Header(default="")):
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization Bearer token")
    token = authorization.split(" ", 1)[1].strip()

    user = await _sb_get_user(token)
    user_id = user.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user")

    patch: Dict[str, Any] = {}
    if payload.timezone is not None:
        patch["timezone"] = payload.timezone
    if payload.display_name is not None:
        patch["display_name"] = payload.display_name
    if payload.pending_reminder is not None:
        patch["pending_reminder"] = payload.pending_reminder

    if not patch:
        prof = await _get_profile(token, user_id)
        return {"profile": prof}

    rows = await _sb_upsert(token, "profiles", [{"user_id": user_id, **patch}], on_conflict="user_id")
    return {"profile": rows[0] if rows else await _get_profile(token, user_id)}


@app.get("/api/messages")
async def get_messages(
    request: Request,
    limit: int = 200,  # Bug fix 4: default limit 200
    authorization: str = Header(default=""),
):
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization Bearer token")
    token = authorization.split(" ", 1)[1].strip()

    user = await _sb_get_user(token)
    user_id = user.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user")

    # Bug fix 4: fetch desc then reverse so latest messages reliably appear after relogin
    rows = await _sb_select(
        token,
        "messages",
        select="id,created_at,role,content,thread_id,metadata",
        filters={"user_id": f"eq.{user_id}"},
        order="created_at.desc",
        limit=limit,
    )
    rows.reverse()
    return {"messages": rows}


@app.get("/api/kb")
async def get_kb(limit: int = 200, authorization: str = Header(default="")):
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization Bearer token")
    token = authorization.split(" ", 1)[1].strip()

    user = await _sb_get_user(token)
    user_id = user.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user")

    rows = await _sb_select(
        token,
        "kb",
        select="id,created_at,content,source,metadata",
        filters={"user_id": f"eq.{user_id}"},
        order="created_at.desc",
        limit=limit,
    )
    return {"kb": rows}


@app.post("/api/kb")
async def add_kb(payload: Dict[str, Any], authorization: str = Header(default="")):
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization Bearer token")
    token = authorization.split(" ", 1)[1].strip()

    user = await _sb_get_user(token)
    user_id = user.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user")

    content = (payload.get("content") or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="Missing content")

    row = {
        "user_id": user_id,
        "content": content,
        "source": payload.get("source") or "user",
        "metadata": payload.get("metadata") or {},
    }
    created = await _sb_insert(token, "kb", [row])
    return {"kb": created[0] if created else row}


@app.get("/api/reminders")
async def get_reminders(limit: int = 200, authorization: str = Header(default="")):
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization Bearer token")
    token = authorization.split(" ", 1)[1].strip()

    user = await _sb_get_user(token)
    user_id = user.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user")

    rows = await _sb_select(
        token,
        "reminders",
        select="id,created_at,title,due_datetime_local,timezone,status,metadata",
        filters={"user_id": f"eq.{user_id}"},
        order="due_datetime_local.asc",
        limit=limit,
    )
    return {"reminders": rows}


@app.post("/api/reminders")
async def create_reminder(payload: Dict[str, Any], authorization: str = Header(default="")):
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization Bearer token")
    token = authorization.split(" ", 1)[1].strip()

    user = await _sb_get_user(token)
    user_id = user.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user")

    prof = await _get_profile(token, user_id)
    tz_name = prof.get("timezone") or DEFAULT_TIMEZONE

    title = (payload.get("title") or "").strip() or "Reminder"
    due_text = (payload.get("due_text") or "").strip()
    due_dt = _parse_due_datetime_local(due_text, tz_name, pending_reminder=prof.get("pending_reminder"))

    if not due_dt:
        raise HTTPException(status_code=400, detail="Unable to parse due date/time")

    row = {
        "user_id": user_id,
        "title": title,
        "due_datetime_local": due_dt.isoformat(),
        "timezone": tz_name,
        "status": "scheduled",
        "metadata": payload.get("metadata") or {},
    }
    created = await _sb_insert(token, "reminders", [row])
    await _set_pending_reminder(token, user_id, None)
    return {"reminder": created[0] if created else row}


# -----------------------------------------------------------------------------
# /api/chat (Groq + reminder + kb)
# -----------------------------------------------------------------------------
@app.post("/api/chat")
async def chat(req: ChatRequest, authorization: str = Header(default="")):
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization Bearer token")
    token = authorization.split(" ", 1)[1].strip()

    if not GROQ_API_KEY or not GROQ_MODEL:
        raise HTTPException(status_code=500, detail="Groq not configured (GROQ_API_KEY / GROQ_MODEL)")

    user = await _sb_get_user(token)
    user_id = user.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user")

    prof = await _get_profile(token, user_id)
    tz_name = prof.get("timezone") or DEFAULT_TIMEZONE
    tz = _safe_tz(tz_name)
    now = _now_local(tz)

    user_text = (req.message or "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Empty message")

    # Store user message
    await _sb_insert(
        token,
        "messages",
        [
            {
                "user_id": user_id,
                "role": "user",
                "content": user_text,
                "thread_id": req.thread_id,
                "metadata": req.metadata or {},
            }
        ],
        returning="minimal",
    )

    # Pull recent messages for context
    history = await _sb_select(
        token,
        "messages",
        select="created_at,role,content",
        filters={"user_id": f"eq.{user_id}"},
        order="created_at.desc",
        limit=30,
    )
    history.reverse()

    system_prompt = _build_system_prompt(tz_name)

    chat_messages = [{"role": "system", "content": system_prompt}]
    for m in history:
        role = m.get("role") or "user"
        content = m.get("content") or ""
        if role not in ("user", "assistant", "system"):
            role = "user"
        chat_messages.append({"role": role, "content": content})

    # If we have a pending reminder (after a clarification question), feed it to the model.
    pending = prof.get("pending_reminder")
    if pending:
        chat_messages.append(
            {
                "role": "system",
                "content": f"Pending reminder context (server-side): {json.dumps(pending)}",
            }
        )

    # Call Groq
    try:
        completion = await openai_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=chat_messages,
            temperature=0.2,
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq error: {str(e)}")

    parsed = _extract_json_object(raw) or {}
    action_type = (parsed.get("type") or "message").strip().lower()
    reply_text = (parsed.get("reply") or "").strip() or "OK."

    reminder_obj = parsed.get("reminder") if isinstance(parsed.get("reminder"), dict) else None
    kb_obj = parsed.get("kb") if isinstance(parsed.get("kb"), dict) else None
    clarify_q = (parsed.get("clarify_question") or "").strip()

    actions: Dict[str, Any] = {"llm_type": action_type}

    created_reminder: Optional[Dict[str, Any]] = None
    created_kb: Optional[Dict[str, Any]] = None

    # Try to create KB if requested
    if action_type in ("kb", "mixed") and kb_obj:
        kb_content = (kb_obj.get("content") or "").strip()
        if kb_content:
            kb_row = {
                "user_id": user_id,
                "content": kb_content,
                "source": "chat",
                "metadata": {"thread_id": req.thread_id, "captured_at": now.isoformat()},
            }
            created = await _sb_insert(token, "kb", [kb_row])
            created_kb = created[0] if created else kb_row
            actions["kb_saved"] = True

    # Reminder flow
    # Bug fix 2: if LLM returns type="clarify" but server-side parser can resolve due_datetime_local, override and create.
    wants_reminder = action_type in ("reminder", "mixed", "clarify") and reminder_obj is not None

    if wants_reminder and reminder_obj:
        title = (reminder_obj.get("title") or "").strip() or "Reminder"
        due_text = (reminder_obj.get("due_text") or "").strip()

        # Server-side parse with strict time parsing (bug fix 1) and pending carry-over (bug fix 3)
        due_dt = _parse_due_datetime_local(
            due_text if due_text else user_text,
            tz_name,
            now_local=now,
            pending_reminder=pending,
        )

        if due_dt:
            # Bug fix 2: override clarify if parseable
            reminder_row = {
                "user_id": user_id,
                "title": title,
                "due_datetime_local": due_dt.isoformat(),
                "timezone": tz_name,
                "status": "scheduled",
                "metadata": {"thread_id": req.thread_id, "source_text": user_text},
            }
            created = await _sb_insert(token, "reminders", [reminder_row])
            created_reminder = created[0] if created else reminder_row
            actions["reminder_created"] = True
            await _set_pending_reminder(token, user_id, None)

            # Confirmation formatting must remain
            confirm = f"Reminder created: {title} - {_format_dt_local(due_dt, tz_name)}"
            reply_text = confirm if action_type != "mixed" else (reply_text + "\n\n" + confirm).strip()
        else:
            # Not parseable. Store pending reminder so a follow-up like "this friday" can keep the time (bug fix 3).
            pending_payload = {
                "title": title,
                "due_text": due_text,
                # store any known time by attempting explicit parse against the original text
                "due_datetime_local": None,
                "timezone": tz_name,
                "source_text": user_text,
                "created_at": now.isoformat(),
            }

            # If original message had an explicit time, save it as "today at that time" placeholder
            explicit_t = _parse_explicit_time(user_text)
            if explicit_t:
                placeholder = datetime.combine(now.date(), explicit_t, tzinfo=tz)
                pending_payload["due_datetime_local"] = placeholder.isoformat()

            await _set_pending_reminder(token, user_id, pending_payload)
            actions["pending_reminder"] = True

            if clarify_q:
                reply_text = clarify_q
            else:
                reply_text = "Which date should I set this reminder for?"

    # Store assistant message
    await _sb_insert(
        token,
        "messages",
        [
            {
                "user_id": user_id,
                "role": "assistant",
                "content": reply_text,
                "thread_id": req.thread_id,
                "metadata": {"actions": actions},
            }
        ],
        returning="minimal",
    )

    if created_reminder:
        actions["reminder"] = created_reminder
    if created_kb:
        actions["kb"] = created_kb

    return ChatResponse(reply=reply_text, actions=actions)


# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": "secondbrain-backend",
        "groq_model": GROQ_MODEL,
        "cors_origins": origins,
    }
