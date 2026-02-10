-- SecondBrain: Run this SQL in your Supabase SQL Editor (Dashboard > SQL Editor > New query)

-- 1. user_profile
CREATE TABLE IF NOT EXISTS user_profile (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  user_email TEXT,
  timezone TEXT DEFAULT 'UTC',
  created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE user_profile ENABLE ROW LEVEL SECURITY;

CREATE POLICY "user_profile_select" ON user_profile
  FOR SELECT USING (auth.uid() = id);
CREATE POLICY "user_profile_insert" ON user_profile
  FOR INSERT WITH CHECK (auth.uid() = id);
CREATE POLICY "user_profile_update" ON user_profile
  FOR UPDATE USING (auth.uid() = id);

-- 2. kb_entries
CREATE TABLE IF NOT EXISTS kb_entries (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  entity_type TEXT DEFAULT 'note',
  entity_name TEXT,
  order_ref TEXT,
  details TEXT NOT NULL,
  source_message TEXT
);

ALTER TABLE kb_entries ENABLE ROW LEVEL SECURITY;

CREATE POLICY "kb_entries_select" ON kb_entries
  FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "kb_entries_insert" ON kb_entries
  FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "kb_entries_delete" ON kb_entries
  FOR DELETE USING (auth.uid() = user_id);

-- 3. reminders
CREATE TABLE IF NOT EXISTS reminders (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  title TEXT NOT NULL,
  due_datetime TIMESTAMPTZ NOT NULL,
  timezone TEXT DEFAULT 'UTC',
  related_order_ref TEXT,
  related_party TEXT,
  status TEXT DEFAULT 'open' CHECK (status IN ('open', 'done')),
  emailed_at TIMESTAMPTZ,
  notes TEXT,
  source_message TEXT
);

ALTER TABLE reminders ENABLE ROW LEVEL SECURITY;

CREATE POLICY "reminders_select" ON reminders
  FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "reminders_insert" ON reminders
  FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "reminders_update" ON reminders
  FOR UPDATE USING (auth.uid() = user_id);

-- 4. chat_messages
CREATE TABLE IF NOT EXISTS chat_messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content TEXT NOT NULL
);

ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;

CREATE POLICY "chat_messages_select" ON chat_messages
  FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "chat_messages_insert" ON chat_messages
  FOR INSERT WITH CHECK (auth.uid() = user_id);
