-- Migration 004: Newsletter feed support + email delivery groundwork
--
-- Allows the system to ingest newsletter RSS feeds (Substack, Ghost, Beehiiv, etc.)
-- alongside podcast feeds, using the same summarization pipeline.

-- Add feed type to podcasts table so the pipeline knows how to process entries
ALTER TABLE podcasts
    ADD COLUMN IF NOT EXISTS feed_type TEXT DEFAULT 'podcast'
        CHECK (feed_type IN ('podcast', 'newsletter'));

CREATE INDEX IF NOT EXISTS idx_podcasts_feed_type ON podcasts(feed_type);

-- Make audio_url nullable: newsletters have no audio file
ALTER TABLE episodes
    ALTER COLUMN audio_url DROP NOT NULL;

-- Store raw article text/HTML for newsletter entries (replaces transcription step)
ALTER TABLE episodes
    ADD COLUMN IF NOT EXISTS article_content TEXT;

-- Handy index for filtering by feed type via join
CREATE INDEX IF NOT EXISTS idx_episodes_has_audio
    ON episodes((audio_url IS NOT NULL));

