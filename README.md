# SecondBrain Assistant

SecondBrain Assistant is an AI-powered personal assistant that combines chat, memory, and reminders in one system. It lets users store notes, ask questions over saved context, set reminders in natural language, and receive daily reminder digests.

This project was built to solve a simple problem: most productivity tools split knowledge, tasks, reminders, and chat across different apps. SecondBrain brings them together into a single assistant.

---

## What it does

SecondBrain Assistant helps users:

- store personal notes and structured memory
- chat with an AI assistant using saved context
- create reminders using natural language
- manage reminder timezones correctly
- receive daily summaries of reminders
- keep a searchable history of interactions

Example use cases:

- “Remind me tomorrow at 10 AM to call Surya Textiles”
- “What reminders do I have today?”
- “Remember that I spoke with a vendor about 1000 shirts”
- “Summarize my pending follow-ups”

---

## Core features

### 1. AI Chat Assistant
Users can interact with the assistant in natural language. The assistant can respond using stored data and past user inputs.

### 2. Memory and Knowledge Storage
The app stores useful user information such as notes, reminders, and conversation context, making it possible to retrieve and reuse information later.

### 3. Natural Language Reminder Creation
Users can create reminders by typing plain English. The system extracts date, time, and intent, then stores the reminder in the database.

### 4. Daily Reminder Digest
The assistant can send a morning summary of reminders scheduled for the day.

### 5. Timezone Support
Each user can have a saved timezone so reminders trigger at the correct local time.

### 6. Cloud Deployment
The backend is deployed and accessible through a live hosted API.

---

## Why this project matters

This project demonstrates:

- full-stack product thinking
- LLM-powered assistant workflows
- backend API design
- database integration
- reminder and scheduling logic
- timezone-aware user systems
- deployment and debugging in production

It is not just a chatbot. It is a practical AI assistant with persistent memory and real user workflows.

---

## Tech stack

### Frontend
- Vercel-hosted web app

### Backend
- FastAPI
- Python

### Database
- Supabase

### AI Layer
- Groq LLM API

### Deployment
- Render

---

## System architecture

User interacts with the frontend, which sends requests to the FastAPI backend.  
The backend:

- processes chat messages
- stores and retrieves reminders
- saves user-specific settings such as timezone
- communicates with the LLM for AI responses
- reads and writes structured data in Supabase

Basic flow:

1. User sends a message
2. Backend determines whether it is:
   - a reminder request
   - a memory/knowledge entry
   - a general chat query
3. Relevant data is stored or retrieved from Supabase
4. LLM generates a contextual response
5. Scheduled jobs check reminders and prepare daily summaries

---

## Example user flows

### Reminder flow
Input:
`Remind me Monday at 10 AM to follow up with Surya Textiles`

System behavior:
- extracts reminder text
- parses due date and time
- applies the user’s timezone
- stores reminder in database
- returns confirmation

### Knowledge flow
Input:
`Today I spoke with Surya Textiles. Order is 1000 shirts, 100 red and 100 blue.`

System behavior:
- stores the information as a memory or note
- makes it retrievable in future conversations

### Chat retrieval flow
Input:
`What was the Surya Textiles order?`

System behavior:
- searches stored context
- returns the saved order details

---

## Project structure

Example high-level structure:

```bash
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
