-- Parakeet Podcast Processor (P3) - Multi-Tenant Schema with Deduplication
-- Phase 1: Foundation for centralized podcast subscription service
--
-- Key Design Principles:
-- 1. DEDUPLICATION: Podcasts and episodes are shared entities
-- 2. MULTI-TENANT: Users have isolated subscriptions and preferences
-- 3. SPOTIFY INTEGRATION: Track user listening history to avoid duplicates
-- 4. CATALOG: Admin-curated + user-custom podcasts

-- ============================================================================
-- CORE SHARED ENTITIES (Processed once, shared across all users)
-- ============================================================================

-- Podcasts: Shared catalog available to all users
CREATE TABLE IF NOT EXISTS podcasts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    rss_url TEXT UNIQUE NOT NULL,
    description TEXT,
    category TEXT,
    author TEXT,
    image_url TEXT,
    website_url TEXT,

    -- Catalog management
    is_curated BOOLEAN DEFAULT false, -- Admin-curated vs user-added
    added_by_user_id UUID REFERENCES auth.users(id), -- NULL if admin-curated

    -- Spotify integration
    spotify_show_id TEXT UNIQUE, -- Spotify show ID for linking

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    last_fetched_at TIMESTAMPTZ,

    -- Indexing
    CONSTRAINT valid_rss_url CHECK (rss_url ~* '^https?://.*')
);

CREATE INDEX idx_podcasts_curated ON podcasts(is_curated);
CREATE INDEX idx_podcasts_spotify ON podcasts(spotify_show_id) WHERE spotify_show_id IS NOT NULL;
CREATE INDEX idx_podcasts_category ON podcasts(category);

-- Episodes: Shared across all users (processed ONCE)
CREATE TABLE IF NOT EXISTS episodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    podcast_id UUID NOT NULL REFERENCES podcasts(id) ON DELETE CASCADE,

    -- Episode metadata
    title TEXT NOT NULL,
    description TEXT,
    published_date TIMESTAMPTZ NOT NULL,
    audio_url TEXT UNIQUE NOT NULL,
    duration_seconds INTEGER,
    file_size_bytes BIGINT,

    -- Local storage
    file_path TEXT, -- Downloaded audio file path

    -- Processing status (CRITICAL for deduplication)
    status TEXT DEFAULT 'pending' CHECK (status IN (
        'pending',      -- Not yet downloaded
        'downloaded',   -- Audio file downloaded
        'transcribing', -- Currently being transcribed
        'transcribed',  -- Transcription complete
        'processing',   -- LLM processing (summary/clips)
        'completed',    -- Fully processed
        'failed'        -- Processing failed
    )),

    -- Spotify integration
    spotify_episode_id TEXT UNIQUE, -- Spotify episode ID for linking

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),

    CONSTRAINT valid_audio_url CHECK (audio_url ~* '^https?://.*')
);

CREATE INDEX idx_episodes_podcast ON episodes(podcast_id);
CREATE INDEX idx_episodes_status ON episodes(status);
CREATE INDEX idx_episodes_published ON episodes(published_date DESC);
CREATE INDEX idx_episodes_spotify ON episodes(spotify_episode_id) WHERE spotify_episode_id IS NOT NULL;
CREATE UNIQUE INDEX idx_episodes_podcast_published ON episodes(podcast_id, published_date);

-- Transcripts: Shared across all users
CREATE TABLE IF NOT EXISTS transcripts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id UUID NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,

    -- Transcript segments
    speaker TEXT,
    timestamp_start REAL NOT NULL, -- Seconds from start
    timestamp_end REAL NOT NULL,
    text TEXT NOT NULL,
    confidence REAL, -- Transcription confidence score

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_transcripts_episode ON transcripts(episode_id);
CREATE INDEX idx_transcripts_timestamp ON transcripts(episode_id, timestamp_start);

-- Summaries: Shared across all users (LLM processing happens ONCE)
CREATE TABLE IF NOT EXISTS summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id UUID UNIQUE NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,

    -- Structured summary data (JSON for flexibility)
    key_topics JSONB, -- Array of topics
    themes JSONB, -- Array of themes
    quotes JSONB, -- Array of {speaker, text, timestamp}
    companies JSONB, -- Array of mentioned companies/startups
    key_insights JSONB, -- Array of insights
    actionable_takeaways JSONB, -- Array of takeaways

    -- Full text summary
    full_summary TEXT,

    -- Processing metadata
    llm_model TEXT, -- Which model generated this
    processing_time_seconds REAL,

    -- Digest grouping
    digest_date DATE, -- Which daily digest this belongs to

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_summaries_episode ON summaries(episode_id);
CREATE INDEX idx_summaries_digest_date ON summaries(digest_date);

-- Audio Clips: Shared across all users (clip selection happens ONCE)
CREATE TABLE IF NOT EXISTS audio_clips (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id UUID NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,

    -- Clip details
    start_sec REAL NOT NULL,
    end_sec REAL NOT NULL,
    duration_sec REAL NOT NULL,
    text TEXT NOT NULL, -- Transcript text for this clip
    topic TEXT, -- What this clip is about

    -- Selection scoring
    score REAL, -- LLM-assigned importance score

    -- File storage
    file_path TEXT, -- Extracted audio clip file

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_audio_clips_episode ON audio_clips(episode_id);
CREATE INDEX idx_audio_clips_score ON audio_clips(episode_id, score DESC);

-- Audio Digests: Shared playlists (could be per-user in future)
CREATE TABLE IF NOT EXISTS audio_digests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Digest info
    digest_date DATE NOT NULL,
    title TEXT,
    description TEXT,

    -- Audio file
    file_path TEXT NOT NULL,
    duration_sec REAL NOT NULL,
    target_duration_sec REAL,

    -- Stats
    clip_count INTEGER,
    episode_count INTEGER,

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_audio_digests_date ON audio_digests(digest_date DESC);

-- ============================================================================
-- USER-SPECIFIC ENTITIES (Multi-tenant, isolated per user)
-- ============================================================================

-- User Subscriptions: Which podcasts each user follows
CREATE TABLE IF NOT EXISTS user_subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    podcast_id UUID NOT NULL REFERENCES podcasts(id) ON DELETE CASCADE,

    -- Subscription settings
    is_active BOOLEAN DEFAULT true,
    auto_added_from_spotify BOOLEAN DEFAULT false, -- Was this auto-synced from Spotify?

    -- Metadata
    subscribed_at TIMESTAMPTZ DEFAULT now(),
    unsubscribed_at TIMESTAMPTZ,

    UNIQUE(user_id, podcast_id)
);

CREATE INDEX idx_user_subscriptions_user ON user_subscriptions(user_id) WHERE is_active = true;
CREATE INDEX idx_user_subscriptions_podcast ON user_subscriptions(podcast_id);

-- User Digest Preferences: How/when users want their digests
CREATE TABLE IF NOT EXISTS user_digest_preferences (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,

    -- Frequency
    frequency TEXT DEFAULT 'daily' CHECK (frequency IN ('daily', 'weekly', 'custom')),
    delivery_time TIME DEFAULT '08:00:00', -- What time to generate digest
    delivery_timezone TEXT DEFAULT 'UTC',

    -- Content filters
    min_episode_duration_sec INTEGER DEFAULT 300, -- Skip very short episodes
    max_episodes_per_digest INTEGER DEFAULT 10,
    include_categories JSONB, -- Array of categories to include (NULL = all)
    exclude_categories JSONB, -- Array of categories to exclude

    -- Spotify integration
    exclude_listened_on_spotify BOOLEAN DEFAULT true, -- Filter out episodes user already heard
    spotify_sync_enabled BOOLEAN DEFAULT true,

    -- Email delivery
    email_enabled BOOLEAN DEFAULT false,
    email_address TEXT,

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- User Spotify Tracking: Track what users have listened to on Spotify
CREATE TABLE IF NOT EXISTS user_spotify_tracking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    episode_id UUID NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,

    -- Spotify data
    spotify_episode_id TEXT NOT NULL,
    listened_at TIMESTAMPTZ NOT NULL, -- When they listened
    fully_played BOOLEAN, -- Did they listen to completion?
    progress_ms BIGINT, -- How much they listened

    -- Metadata
    synced_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(user_id, episode_id)
);

CREATE INDEX idx_user_spotify_tracking_user ON user_spotify_tracking(user_id);
CREATE INDEX idx_user_spotify_tracking_episode ON user_spotify_tracking(episode_id);

-- User Spotify Shows: Track which Spotify shows users follow (for auto-subscription)
CREATE TABLE IF NOT EXISTS user_spotify_shows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

    -- Spotify show data
    spotify_show_id TEXT NOT NULL,
    show_name TEXT,
    show_publisher TEXT,

    -- Auto-subscription
    auto_subscribed BOOLEAN DEFAULT false, -- Did we auto-create subscription?
    matched_podcast_id UUID REFERENCES podcasts(id), -- Which podcast in our catalog matches

    -- Metadata
    followed_at TIMESTAMPTZ,
    synced_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(user_id, spotify_show_id)
);

CREATE INDEX idx_user_spotify_shows_user ON user_spotify_shows(user_id);
CREATE INDEX idx_user_spotify_shows_spotify ON user_spotify_shows(spotify_show_id);

-- User Digests: Personalized digest for each user
CREATE TABLE IF NOT EXISTS user_digests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

    -- Digest info
    digest_date DATE NOT NULL,
    title TEXT,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'generating', 'completed', 'failed')),

    -- Content (references to shared summaries)
    episode_ids JSONB, -- Array of episode UUIDs included
    summary_ids JSONB, -- Array of summary UUIDs included

    -- Generated content
    markdown_path TEXT,
    json_path TEXT,
    audio_digest_id UUID REFERENCES audio_digests(id),

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT now(),
    generated_at TIMESTAMPTZ,

    UNIQUE(user_id, digest_date)
);

CREATE INDEX idx_user_digests_user ON user_digests(user_id);
CREATE INDEX idx_user_digests_date ON user_digests(digest_date DESC);
CREATE INDEX idx_user_digests_status ON user_digests(status);

-- ============================================================================
-- ADMIN & CATALOG MANAGEMENT
-- ============================================================================

-- Admin Users: Track who has admin privileges for catalog curation
CREATE TABLE IF NOT EXISTS admin_users (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    role TEXT DEFAULT 'curator' CHECK (role IN ('curator', 'moderator', 'super_admin')),
    granted_at TIMESTAMPTZ DEFAULT now(),
    granted_by UUID REFERENCES auth.users(id)
);

-- Podcast Catalog Requests: Users can suggest podcasts for curation
CREATE TABLE IF NOT EXISTS podcast_catalog_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

    -- Requested podcast
    rss_url TEXT NOT NULL,
    title TEXT,
    description TEXT,
    reason TEXT, -- Why they want this in the catalog

    -- Review status
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
    reviewed_by UUID REFERENCES auth.users(id),
    reviewed_at TIMESTAMPTZ,
    review_notes TEXT,

    -- If approved, which podcast was created
    approved_podcast_id UUID REFERENCES podcasts(id),

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_podcast_catalog_requests_status ON podcast_catalog_requests(status);
CREATE INDEX idx_podcast_catalog_requests_user ON podcast_catalog_requests(user_id);

-- ============================================================================
-- PROCESSING QUEUE (For background job management)
-- ============================================================================

-- Processing Jobs: Track what needs to be processed
CREATE TABLE IF NOT EXISTS processing_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Job details
    job_type TEXT NOT NULL CHECK (job_type IN (
        'fetch_feed',           -- Fetch new episodes from RSS
        'download_episode',     -- Download audio file
        'transcribe_episode',   -- Transcribe audio to text
        'summarize_episode',    -- LLM summarization
        'generate_clips',       -- Audio clip generation
        'generate_digest',      -- User digest generation
        'sync_spotify'          -- Sync user's Spotify data
    )),

    -- Target resources
    podcast_id UUID REFERENCES podcasts(id) ON DELETE CASCADE,
    episode_id UUID REFERENCES episodes(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,

    -- Status
    status TEXT DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    priority INTEGER DEFAULT 5, -- 1-10, higher = more important

    -- Execution
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_processing_jobs_status ON processing_jobs(status, priority DESC);
CREATE INDEX idx_processing_jobs_type ON processing_jobs(job_type);
CREATE INDEX idx_processing_jobs_episode ON processing_jobs(episode_id);

-- ============================================================================
-- OAUTH INTEGRATIONS
-- ============================================================================

-- User OAuth Connections: Store OAuth tokens for external services
CREATE TABLE IF NOT EXISTS user_oauth_connections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

    -- Provider
    provider TEXT NOT NULL CHECK (provider IN ('spotify', 'google')),
    provider_user_id TEXT, -- User's ID on the provider

    -- Tokens (encrypted at application layer)
    access_token TEXT NOT NULL,
    refresh_token TEXT,
    token_expires_at TIMESTAMPTZ,

    -- Scopes
    scopes JSONB, -- Array of granted scopes

    -- Metadata
    connected_at TIMESTAMPTZ DEFAULT now(),
    last_refreshed_at TIMESTAMPTZ,

    UNIQUE(user_id, provider)
);

CREATE INDEX idx_user_oauth_connections_user ON user_oauth_connections(user_id);

-- ============================================================================
-- FUNCTIONS & TRIGGERS
-- ============================================================================

-- Update updated_at timestamp on row update
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at trigger to relevant tables
CREATE TRIGGER update_podcasts_updated_at BEFORE UPDATE ON podcasts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_episodes_updated_at BEFORE UPDATE ON episodes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_summaries_updated_at BEFORE UPDATE ON summaries
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_digest_preferences_updated_at BEFORE UPDATE ON user_digest_preferences
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- ============================================================================

-- Enable RLS on user-specific tables
ALTER TABLE user_subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_digest_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_spotify_tracking ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_spotify_shows ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_digests ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_oauth_connections ENABLE ROW LEVEL SECURITY;
ALTER TABLE podcast_catalog_requests ENABLE ROW LEVEL SECURITY;

-- User Subscriptions: Users can only see/modify their own
CREATE POLICY user_subscriptions_select ON user_subscriptions
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY user_subscriptions_insert ON user_subscriptions
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY user_subscriptions_update ON user_subscriptions
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY user_subscriptions_delete ON user_subscriptions
    FOR DELETE USING (auth.uid() = user_id);

-- User Digest Preferences: Users can only see/modify their own
CREATE POLICY user_digest_preferences_select ON user_digest_preferences
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY user_digest_preferences_insert ON user_digest_preferences
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY user_digest_preferences_update ON user_digest_preferences
    FOR UPDATE USING (auth.uid() = user_id);

-- User Spotify Tracking: Users can only see their own
CREATE POLICY user_spotify_tracking_select ON user_spotify_tracking
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY user_spotify_tracking_insert ON user_spotify_tracking
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- User Spotify Shows: Users can only see their own
CREATE POLICY user_spotify_shows_select ON user_spotify_shows
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY user_spotify_shows_insert ON user_spotify_shows
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- User Digests: Users can only see their own
CREATE POLICY user_digests_select ON user_digests
    FOR SELECT USING (auth.uid() = user_id);

-- User OAuth Connections: Users can only see their own
CREATE POLICY user_oauth_connections_select ON user_oauth_connections
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY user_oauth_connections_insert ON user_oauth_connections
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY user_oauth_connections_update ON user_oauth_connections
    FOR UPDATE USING (auth.uid() = user_id);

-- Podcast Catalog Requests: Users can see their own requests
CREATE POLICY podcast_catalog_requests_select ON podcast_catalog_requests
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY podcast_catalog_requests_insert ON podcast_catalog_requests
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Shared tables (podcasts, episodes, transcripts, summaries) are readable by all authenticated users
-- No RLS needed as they're shared resources

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Composite indexes for common queries
CREATE INDEX idx_episodes_podcast_status ON episodes(podcast_id, status);
CREATE INDEX idx_user_subscriptions_active ON user_subscriptions(user_id, is_active, podcast_id);
CREATE INDEX idx_processing_jobs_pending ON processing_jobs(status, created_at)
    WHERE status IN ('queued', 'running');

-- Full-text search indexes (for podcast/episode search)
CREATE INDEX idx_podcasts_title_search ON podcasts USING gin(to_tsvector('english', title || ' ' || coalesce(description, '')));
CREATE INDEX idx_episodes_title_search ON episodes USING gin(to_tsvector('english', title || ' ' || coalesce(description, '')));

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Create a system user for background jobs (optional)
-- INSERT INTO auth.users (id, email) VALUES ('00000000-0000-0000-0000-000000000000', 'system@example.invalid')
--     ON CONFLICT (id) DO NOTHING;

COMMENT ON TABLE podcasts IS 'Shared podcast catalog - processed once, available to all users';
COMMENT ON TABLE episodes IS 'Shared episodes - transcribed and processed once, shared across all users';
COMMENT ON TABLE user_subscriptions IS 'User-specific podcast subscriptions - which podcasts each user follows';
COMMENT ON TABLE user_spotify_tracking IS 'Track which episodes users listened to on Spotify to avoid duplicates in digests';
COMMENT ON TABLE processing_jobs IS 'Background job queue for episode processing - ensures deduplication';

