# SecondBrain Assistant

SecondBrain Assistant is an AI-powered personal assistant that combines chat, memory, and reminders in one system. It lets users store notes, ask questions over saved context, set reminders in natural language, and receive daily reminder digests.

Most productivity tools split knowledge, tasks, reminders, and chat across different apps. SecondBrain brings them together into a single assistant.

🔗 **Live app:** https://bharath2197-secondbrain-assistant.vercel.app

---

## What it does

- Store personal notes and structured memory
- Chat with an AI assistant using saved context
- Create reminders using natural language
- Manage reminder timezones correctly
- Receive daily summaries of reminders
- Keep a searchable history of interactions

**Example use cases:**
- *"Remind me tomorrow at 10 AM to call Surya Textiles"*
- *"What reminders do I have today?"*
- *"Remember that I spoke with a vendor about 1000 shirts"*
- *"Summarize my pending follow-ups"*

---

## Core features

### 1. AI Chat Assistant
Users interact in natural language. The assistant responds using stored data and past user inputs.

### 2. Memory & Knowledge Storage
Stores notes, reminders, and conversation context — making it retrievable and reusable across sessions.

### 3. Natural Language Reminder Creation
Users create reminders in plain English. The system extracts date, time, and intent, then stores the reminder in the database.

### 4. Daily Reminder Digest
Sends a morning summary of all reminders scheduled for the day.

### 5. Timezone Support
Each user has a saved timezone so reminders trigger at the correct local time.

### 6. Cloud Deployment
Backend is deployed and accessible through a live hosted API.

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Frontend | Vercel-hosted web app |
| Backend | FastAPI + Python |
| Database | Supabase |
| AI Layer | Groq LLM API |
| Deployment | Render |

---

## System architecture

```
User → Frontend (Vercel)
         ↓
   FastAPI Backend (Render)
         ↓
   ┌─────────────────────┐
   │  Supabase (DB)      │  ← reminders, memory, user settings
   │  Groq LLM API       │  ← AI response generation
   │  Scheduler          │  ← daily digest jobs
   └─────────────────────┘
```

**Basic flow:**
1. User sends a message
2. Backend classifies intent: reminder / memory entry / general chat
3. Relevant data is stored or retrieved from Supabase
4. LLM generates a contextual response
5. Scheduled jobs check reminders and prepare daily summaries

---

## Example user flows

**Reminder flow**
```
Input: "Remind me Monday at 10 AM to follow up with Surya Textiles"
→ Extracts reminder text, date, time
→ Applies user timezone
→ Stores in database
→ Returns confirmation
```

**Knowledge flow**
```
Input: "Today I spoke with Surya Textiles. Order is 1000 shirts, 100 red and 100 blue."
→ Stores as memory/note
→ Retrievable in future conversations
```

**Chat retrieval flow**
```
Input: "What was the Surya Textiles order?"
→ Searches stored context
→ Returns saved order details
```

---

## Project structure

```
secondbrain-assistant/
│
├── backend/
│   ├── server.py
│   ├── routes/
│   ├── services/
│   ├── models/
│   └── utils/
│
├── frontend/
│   ├── pages/
│   ├── components/
│   └── lib/
│
├── requirements.txt
├── README.md
└── .env
```

---

## To run locally

```bash
git clone https://github.com/bharath2197/secondbrain-assistant.git
cd secondbrain-assistant
pip install -r requirements.txt
# Add your .env with Supabase, Groq, and other API keys
uvicorn backend.server:app --reload
```
