-- Migration 003: In-app episode listen tracking
-- Enables digest filtering for users without Spotify

CREATE TABLE IF NOT EXISTS user_episode_listens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    episode_id UUID NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,

    -- Listen data
    listened_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    progress_seconds INTEGER, -- How far they got (NULL = unknown)
    fully_played BOOLEAN DEFAULT false,
    source TEXT DEFAULT 'app' CHECK (source IN ('app', 'import')),

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(user_id, episode_id)
);

CREATE INDEX idx_user_episode_listens_user ON user_episode_listens(user_id);
CREATE INDEX idx_user_episode_listens_episode ON user_episode_listens(episode_id);

-- RLS: users can only see and manage their own listen records
ALTER TABLE user_episode_listens ENABLE ROW LEVEL SECURITY;

CREATE POLICY user_episode_listens_select ON user_episode_listens
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY user_episode_listens_insert ON user_episode_listens
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY user_episode_listens_update ON user_episode_listens
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY user_episode_listens_delete ON user_episode_listens
    FOR DELETE USING (auth.uid() = user_id);

