-- Migration 002: Add Performance Indexes
-- Adds composite indexes to improve query performance for common access patterns

-- User subscriptions: frequently queried by user_id and is_active together
CREATE INDEX IF NOT EXISTS idx_user_subscriptions_user_active
    ON user_subscriptions(user_id, is_active)
    WHERE is_active = true;

-- Episodes: frequently queried by podcast with ordering by date
CREATE INDEX IF NOT EXISTS idx_episodes_podcast_published
    ON episodes(podcast_id, published_date DESC);

-- Episodes: status-based queries for processing pipeline
CREATE INDEX IF NOT EXISTS idx_episodes_status
    ON episodes(status)
    WHERE status IN ('pending', 'downloaded', 'transcribing', 'transcribed', 'processing');

-- Transcripts: ordered retrieval for episode processing
CREATE INDEX IF NOT EXISTS idx_transcripts_episode_timestamp
    ON transcripts(episode_id, timestamp_start);

-- Summaries: quick lookup by episode
CREATE INDEX IF NOT EXISTS idx_summaries_episode
    ON summaries(episode_id);

-- User digests: querying digests by user and date
CREATE INDEX IF NOT EXISTS idx_user_digests_user_date
    ON user_digests(user_id, digest_date DESC);

-- User digests: filtering by status
CREATE INDEX IF NOT EXISTS idx_user_digests_status
    ON user_digests(status)
    WHERE status IN ('pending', 'generating');

-- OAuth connections: provider lookup
CREATE INDEX IF NOT EXISTS idx_user_oauth_provider
    ON user_oauth_connections(user_id, provider);

-- OAuth connections: token expiration for refresh jobs
-- Note: Cannot use NOW() in partial index (not IMMUTABLE), so index all rows
CREATE INDEX IF NOT EXISTS idx_user_oauth_expired
    ON user_oauth_connections(token_expires_at);

-- Spotify shows: matching with catalog
CREATE INDEX IF NOT EXISTS idx_user_spotify_shows_show_id
    ON user_spotify_shows(spotify_show_id);

-- Spotify tracking: episode lookup
CREATE INDEX IF NOT EXISTS idx_user_spotify_tracking_episode
    ON user_spotify_tracking(user_id, episode_id);

-- Spotify tracking: recently listened
CREATE INDEX IF NOT EXISTS idx_user_spotify_tracking_listened
    ON user_spotify_tracking(user_id, listened_at DESC);

-- Podcasts: Spotify show ID matching
CREATE INDEX IF NOT EXISTS idx_podcasts_spotify_show_id
    ON podcasts(spotify_show_id)
    WHERE spotify_show_id IS NOT NULL;

-- Podcasts: catalog requests by status
CREATE INDEX IF NOT EXISTS idx_podcast_catalog_requests_status
    ON podcast_catalog_requests(status)
    WHERE status = 'pending';

-- Admin users: quick admin check
CREATE INDEX IF NOT EXISTS idx_admin_users_user_id
    ON admin_users(user_id);

-- Podcast feed fetching: last_fetched_at for scheduled jobs
CREATE INDEX IF NOT EXISTS idx_podcasts_last_fetched
    ON podcasts(last_fetched_at NULLS FIRST);

-- Note: Index comments removed - COMMENT ON INDEX doesn't support IF EXISTS
-- and some indexes may already exist from the first migration

-- Analyze tables to update query planner statistics
ANALYZE user_subscriptions;
ANALYZE episodes;
ANALYZE transcripts;
ANALYZE summaries;
ANALYZE user_digests;
ANALYZE user_oauth_connections;
ANALYZE user_spotify_shows;
ANALYZE user_spotify_tracking;
ANALYZE podcasts;
ANALYZE podcast_catalog_requests;
ANALYZE admin_users;

