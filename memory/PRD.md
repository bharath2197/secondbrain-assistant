# SecondBrain - Personal Assistant PWA

## Problem Statement
Build a production-ready Personal Assistant web app (mobile-first, installable PWA) that captures unstructured text/voice and turns it into structured knowledge + reminders using AI extraction.

## Architecture
- **Frontend**: React + Tailwind CSS + shadcn/ui, mobile-first PWA
- **Backend**: FastAPI (Python) REST API with `/api` prefix
- **Auth + DB**: Supabase (PostgreSQL + RLS + email/password auth)
- **AI**: OpenAI GPT-4o via Emergent LLM key for message extraction
- **Voice**: Browser-native Web Speech API + speechSynthesis

## User Personas
1. **Individual**: Wants to quickly capture notes, facts, and reminders via chat
2. **Small business owner**: Tracks orders, contacts, and follow-up reminders

## Core Requirements
- Single chat interface for all input
- AI classifies messages as KB entries, reminders, or conversation
- Multi-turn slot filling for reminders (no clarification loops)
- Voice input (Web Speech API) and TTS (speechSynthesis)
- Knowledge Base viewer with search/filter
- Reminders with Overdue/Today/Upcoming sections
- Supabase RLS for user data isolation
- Timezone-aware reminders (UTC storage, local display)
- PWA installable with manifest

## What's Been Implemented (Feb 2026)
- [x] FastAPI backend with Supabase REST proxy (httpx)
- [x] JWT token extraction and user-scoped queries
- [x] LLM extraction endpoint (POST /api/chat) with GPT-4o
- [x] CRUD endpoints: profile, kb, reminders, messages
- [x] React frontend with 4 pages: Chat, KB, Reminders, Settings
- [x] Supabase email/password auth (signup + login)
- [x] Timezone detection + confirmation dialog on first login
- [x] Bottom navigation (mobile-first)
- [x] Voice input (mic button, tap-to-toggle)
- [x] Text-to-speech (speaker button on messages, auto-speak toggle)
- [x] PWA manifest
- [x] Auto theme (system preference dark/light)
- [x] SQL setup script for Supabase tables + RLS policies

## Prioritized Backlog
### P0 (User Must Do)
- [ ] Run `supabase_setup.sql` in Supabase SQL Editor
- [ ] Verify Supabase project URL and anon key are correct

### P1 (Next Phase)
- [ ] Service worker for offline support
- [ ] Push notification reminders
- [ ] PWA icon generation (192x192, 512x512)

### P2 (Enhancement)
- [ ] Chat message editing/deletion
- [ ] Reminder snooze functionality
- [ ] KB entry editing
- [ ] Export data (JSON/CSV)

## Key Files
- Backend: `/app/backend/server.py`
- SQL Setup: `/app/backend/supabase_setup.sql`
- Frontend Entry: `/app/frontend/src/App.js`
- Auth: `/app/frontend/src/contexts/AuthContext.js`
- Chat: `/app/frontend/src/pages/ChatPage.js`
- Layout: `/app/frontend/src/layouts/AppLayout.js`
