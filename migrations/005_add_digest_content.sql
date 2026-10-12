-- Migration 002: Add content column to user_digests for storing rendered markdown
-- Run this on your Supabase project via SQL editor or CLI:
--   supabase db push  (if using supabase CLI)
-- Or paste into Supabase dashboard > SQL editor

ALTER TABLE user_digests ADD COLUMN IF NOT EXISTS content TEXT;

COMMENT ON COLUMN user_digests.content IS
  'Rendered markdown content of the digest (stored inline for fast retrieval). '
  'markdown_path / json_path remain available for large/file-based storage.';

