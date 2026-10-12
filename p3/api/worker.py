"""
Background worker for processing podcasts
Uses arq for async job queue with Redis
Fixed: All async/sync issues resolved, proper logging added
"""

import asyncio
import base64
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from pathlib import Path

from arq import create_pool, cron as arq_cron
from arq.connections import RedisSettings
from supabase import create_client
import httpx
import feedparser
import spotipy

from p3.api.config import get_settings
from p3.api.emailer import render_digest_html, send_email
from p3.api.security import decrypt_token, encrypt_token
from p3.api.storage import download_episode_temp, delete_episode_temp


settings = get_settings()
logger = logging.getLogger(__name__)

# Redis settings for arq
redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)

# Download concurrency limiter
download_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_DOWNLOADS)


async def get_supabase():
    """Get Supabase client"""
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)


def parse_duration_to_seconds(duration_str: str) -> Optional[int]:
    """
    Parse iTunes duration string to seconds.
    Handles formats like: "3:14:26", "14:26", "326", "1:02:03"

    Args:
        duration_str: Duration string from iTunes RSS feed

    Returns:
        Duration in seconds, or None if parsing fails
    """
    if not duration_str:
        return None

    try:
        # If it's already an integer (seconds), return it
        return int(duration_str)
    except ValueError:
        pass

    try:
        # Parse time format (HH:MM:SS or MM:SS)
        parts = duration_str.strip().split(':')
        if len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:  # MM:SS
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        else:
            logger.warning(f"Unexpected duration format: {duration_str}")
            return None
    except (ValueError, AttributeError) as e:
        logger.warning(f"Failed to parse duration '{duration_str}': {e}")
        return None


def _generate_structured_summary(transcript: str, provider: str, model: str) -> Optional[dict]:
    """
    Generate structured summary from transcript using LLM.

    Args:
        transcript: Full transcript text
        provider: LLM provider (openai, groq, anthropic, etc.)
        model: LLM model name

    Returns:
        Dict with structured summary data or None if failed
    """
    import json

    # Get API key for provider
    api_key = None
    provider_lower = provider.lower()
    if provider_lower == "groq":
        api_key = settings.GROQ_API_KEY
    elif provider_lower == "openai":
        api_key = settings.OPENAI_API_KEY
    elif provider_lower == "anthropic":
        api_key = settings.ANTHROPIC_API_KEY
    elif provider_lower == "gemini":
        api_key = settings.GEMINI_API_KEY
    elif provider_lower == "cerebras":
        api_key = settings.CEREBRAS_API_KEY
    elif provider_lower == "together":
        api_key = settings.TOGETHER_API_KEY
    elif provider_lower == "fireworks":
        api_key = settings.FIREWORKS_API_KEY
    elif provider_lower == "openrouter":
        api_key = settings.OPENROUTER_API_KEY

    if not api_key:
        logger.error(f"No API key configured for provider: {provider}")
        return None

    # Prepare prompt for structured extraction
    prompt = f"""Analyze this podcast transcript and extract structured information in JSON format.

Return a valid JSON object with this exact structure:
{{
  "key_topics": ["topic1", "topic2", "topic3"],
  "themes": ["theme1", "theme2"],
  "quotes": ["quote1", "quote2", "quote3"],
  "companies": ["company1", "company2"],
  "key_insights": ["insight1", "insight2", "insight3"],
  "actionable_takeaways": ["takeaway1", "takeaway2"],
  "summary": "A detailed 4-6 sentence summary of the episode"
}}

Guidelines:
- key_topics: Main subjects discussed (4-7 items)
- themes: Broader patterns or themes (3-5 items)
- quotes: Notable or insightful quotes (5-8 items)
- companies: Any companies or startups mentioned
- key_insights: Important insights or learnings (4-6 items)
- actionable_takeaways: Practical takeaways (3-5 items)
- summary: Comprehensive overview in 4-6 sentences

Transcript:
{transcript[:15000]}

Return ONLY the JSON object, no other text."""

    try:
        # Use OpenAI-compatible API for most providers
        if provider_lower in ["openai", "groq", "together", "fireworks", "openrouter", "cerebras"]:
            # Determine base URL
            base_urls = {
                "groq": "https://api.groq.com/openai/v1",
                "openai": "https://api.openai.com/v1",
                "together": "https://api.together.xyz/v1",
                "fireworks": "https://api.fireworks.ai/inference/v1",
                "openrouter": "https://openrouter.ai/api/v1",
                "cerebras": "https://api.cerebras.ai/v1",
            }
            base_url = base_urls.get(provider_lower, "https://api.openai.com/v1")

            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "max_tokens": 2000,
                    }
                )
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                # Parse JSON from response
                # Sometimes LLMs wrap JSON in markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                summary_data = json.loads(content)
                logger.info(f"Generated structured summary with {len(summary_data.get('key_topics', []))} topics")
                return summary_data

        elif provider_lower == "anthropic":
            # Use Claude API
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                    },
                    json={
                        "model": model,
                        "max_tokens": 2000,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                )
                response.raise_for_status()
                result = response.json()
                content = result["content"][0]["text"]

                # Parse JSON from response
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                summary_data = json.loads(content)
                logger.info(f"Generated structured summary with {len(summary_data.get('key_topics', []))} topics")
                return summary_data
        elif provider_lower == "ollama":
            # Local Ollama (no API key needed, runs on localhost)
            with httpx.Client(timeout=300.0) as client:
                response = client.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                    }
                )
                response.raise_for_status()
                result = response.json()
                content = result["message"]["content"]

                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                summary_data = json.loads(content)
                logger.info(f"Generated structured summary via Ollama/{model}")
                return summary_data

        elif provider_lower == "gemini":
            # Google Gemini API (free tier: 60 RPM / 1500 RPD on gemini-1.5-flash)
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    params={"key": api_key},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2000},
                    }
                )
                response.raise_for_status()
                result = response.json()
                content = result["candidates"][0]["content"]["parts"][0]["text"]

                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                summary_data = json.loads(content)
                logger.info(f"Generated structured summary via Gemini with {len(summary_data.get('key_topics', []))} topics")
                return summary_data

        else:
            logger.error(f"Unsupported LLM provider: {provider}")
            return None

    except Exception as e:
        logger.error(f"Failed to generate summary with {provider}: {e}")
        return None


async def _get_spotify_token(supabase, user_id: str, oauth_data: dict) -> Optional[str]:
    """
    Return a valid Spotify access token for the user.
    Automatically refreshes the token if it is expired or within 5 minutes of expiry.
    Updates the database with the new token on refresh.
    """
    access_token_enc = oauth_data.get("access_token", "")
    token_expires_at = oauth_data.get("token_expires_at")
    refresh_token_enc = oauth_data.get("refresh_token", "")

    # Determine whether a refresh is needed
    needs_refresh = False
    if token_expires_at:
        try:
            expires_dt = datetime.fromisoformat(str(token_expires_at).replace("Z", "+00:00"))
            if expires_dt.tzinfo is None:
                expires_dt = expires_dt.replace(tzinfo=timezone.utc)
            if expires_dt <= datetime.now(timezone.utc) + timedelta(minutes=5):
                needs_refresh = True
        except Exception:
            needs_refresh = True  # Treat unparseable expiry as expired

    if needs_refresh and refresh_token_enc:
        refresh_token = decrypt_token(refresh_token_enc)
        if refresh_token and settings.SPOTIFY_CLIENT_ID and settings.SPOTIFY_CLIENT_SECRET:
            try:
                credentials = base64.b64encode(
                    f"{settings.SPOTIFY_CLIENT_ID}:{settings.SPOTIFY_CLIENT_SECRET}".encode()
                ).decode()
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        "https://accounts.spotify.com/api/token",
                        headers={
                            "Authorization": f"Basic {credentials}",
                            "Content-Type": "application/x-www-form-urlencoded",
                        },
                        data={"grant_type": "refresh_token", "refresh_token": refresh_token},
                    )

                if response.status_code == 200:
                    token_data = response.json()
                    new_access_token = token_data["access_token"]
                    expires_in = token_data.get("expires_in", 3600)
                    new_expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

                    update_fields = {
                        "access_token": encrypt_token(new_access_token),
                        "token_expires_at": new_expires_at.isoformat(),
                    }
                    # Spotify may return a new refresh token; update if so
                    if "refresh_token" in token_data:
                        update_fields["refresh_token"] = encrypt_token(token_data["refresh_token"])

                    supabase.table("user_oauth_connections").update(update_fields).eq(
                        "user_id", user_id
                    ).eq("provider", "spotify").execute()

                    logger.info(f"Refreshed Spotify token for user {user_id}")
                    return new_access_token
                else:
                    logger.error(
                        f"Spotify token refresh failed for user {user_id}: "
                        f"{response.status_code} {response.text[:200]}"
                    )
            except Exception as e:
                logger.error(f"Error refreshing Spotify token for user {user_id}: {e}")

    # Fall back to the existing (possibly still valid) access token
    return decrypt_token(access_token_enc) or None


def _render_digest_markdown(episodes: list, digest_date: str) -> str:
    """
    Render a list of episodes with their summaries into a readable markdown digest.

    Args:
        episodes: List of episode dicts with nested 'summaries' list
        digest_date: ISO date string for the digest

    Returns:
        Markdown string
    """
    lines = [
        f"# Your Podcast Digest — {digest_date}",
        "",
        f"*{len(episodes)} episode{'s' if len(episodes) != 1 else ''} curated for you*",
        "",
        "---",
        "",
    ]

    for i, episode in enumerate(episodes, 1):
        title = episode.get("title", "Untitled Episode")
        podcast = (episode.get("podcasts") or {}).get("title", "Unknown Podcast")
        published = (episode.get("published_date") or "")[:10]
        duration_sec = episode.get("duration_seconds")
        duration_str = ""
        if duration_sec:
            h, m = divmod(duration_sec // 60, 60)
            duration_str = f" · {h}h {m}m" if h else f" · {m}m"

        lines += [f"## {i}. {podcast} — {title}", ""]
        if published:
            lines.append(f"**Published:** {published}{duration_str}  ")

        # Get the first summary for this episode
        summaries = episode.get("summaries") or []
        if isinstance(summaries, dict):
            summaries = [summaries]
        summary = summaries[0] if summaries else {}

        full_summary = summary.get("full_summary") or summary.get("summary", "")
        if full_summary:
            lines += ["", "### Summary", "", full_summary, ""]

        key_insights = summary.get("key_insights") or []
        if key_insights:
            lines += ["### Key Insights", ""]
            for insight in key_insights:
                lines.append(f"- {insight}")
            lines.append("")

        takeaways = summary.get("actionable_takeaways") or []
        if takeaways:
            lines += ["### Actionable Takeaways", ""]
            for t in takeaways:
                lines.append(f"- {t}")
            lines.append("")

        key_topics = summary.get("key_topics") or []
        if key_topics:
            lines.append(f"**Topics:** {', '.join(key_topics)}  ")

        quotes = summary.get("quotes") or []
        if quotes:
            lines += ["", "### Notable Quotes", ""]
            for q in quotes[:3]:
                lines.append(f"> {q}")
            lines.append("")

        lines += ["---", ""]

    return "\n".join(lines)


def _get_ordered_llm_providers() -> list:
    """
    Return ordered list of (provider, model) tuples to try for LLM summarization.
    Tries configured provider first, then free alternatives, then local Ollama.
    Only includes providers with configured API keys.
    """
    candidates = [
        (settings.LLM_PROVIDER, settings.LLM_MODEL),
        ("gemini", "gemini-1.5-flash"),
        ("cerebras", "llama-3.3-70b"),
        ("groq", "llama-3.1-8b-instant"),
        ("together", "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"),
        ("openrouter", "meta-llama/llama-3.1-8b-instruct:free"),
        ("fireworks", "accounts/fireworks/models/llama-v3p1-8b-instruct"),
    ]

    key_map = {
        "groq": settings.GROQ_API_KEY,
        "openai": settings.OPENAI_API_KEY,
        "anthropic": settings.ANTHROPIC_API_KEY,
        "gemini": settings.GEMINI_API_KEY,
        "cerebras": settings.CEREBRAS_API_KEY,
        "together": settings.TOGETHER_API_KEY,
        "fireworks": settings.FIREWORKS_API_KEY,
        "openrouter": settings.OPENROUTER_API_KEY,
        "ollama": "local",  # no key needed
    }

    seen = set()
    result = []
    for provider, model in candidates:
        p = provider.lower()
        if p in seen:
            continue
        seen.add(p)
        if key_map.get(p):
            result.append((provider, model))

    # Always append local Ollama as final fallback if available
    if "ollama" not in seen:
        result.append(("ollama", "llama3.2:3b"))

    return result


async def fetch_feed_job(ctx, podcast_id: str) -> dict:
    """
    Background job: Fetch new episodes from podcast RSS feed

    Fixed: Using asyncio.to_thread for feedparser (sync library)

    Args:
        podcast_id: UUID of the podcast to fetch

    Returns:
        dict with stats: episodes_found, episodes_new
    """
    supabase = await get_supabase()
    logger.info(f"Fetching feed for podcast {podcast_id}")

    # Get podcast info
    podcast_result = supabase.table("podcasts").select("*").eq("id", podcast_id).execute()

    if not podcast_result.data:
        logger.error(f"Podcast {podcast_id} not found")
        return {"error": "Podcast not found"}

    podcast = podcast_result.data[0]
    rss_url = podcast["rss_url"]

    # Parse RSS feed asynchronously (feedparser is sync, so wrap in thread)
    try:
        feed = await asyncio.to_thread(feedparser.parse, rss_url)
    except Exception as e:
        logger.error(f"Failed to parse RSS feed {rss_url}: {e}")
        return {"error": f"Failed to parse RSS feed: {str(e)}"}

    feed_type = podcast.get("feed_type", "podcast")
    is_newsletter = feed_type == "newsletter"

    episodes_found = len(feed.entries)
    episodes_new = 0
    new_podcast_ids = []    # → download_episode_job
    new_newsletter_ids = [] # → summarize_episode_job directly

    # First pass: Create all new episodes in database
    for entry in feed.entries[:settings.MAX_EPISODES_PER_FETCH]:
        audio_url = entry.enclosures[0].href if entry.enclosures else None

        # Skip audio entries for newsletter feeds, skip non-audio for podcast feeds
        if is_newsletter and audio_url:
            continue  # Newsletter feed shouldn't process audio
        if not is_newsletter and not audio_url:
            continue  # Podcast feed requires audio

        # Deduplication key
        dedup_key = audio_url if audio_url else entry.get("link") or entry.get("id", "")
        if not dedup_key:
            continue

        if audio_url:
            existing = supabase.table("episodes").select("id").eq("audio_url", audio_url).execute()
        else:
            existing = (
                supabase.table("episodes")
                .select("id")
                .eq("podcast_id", podcast_id)
                .eq("title", entry.title or "")
                .execute()
            )

        if existing.data:
            continue  # Already exists

        # For newsletters: extract article text (strip enclosures, use content/summary)
        article_content = None
        if is_newsletter:
            if hasattr(entry, "content") and entry.content:
                article_content = entry.content[0].get("value", "")
            elif hasattr(entry, "summary"):
                article_content = entry.summary or ""

        published_date = (
            datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()
            if hasattr(entry, "published_parsed") and entry.published_parsed
            else datetime.now(timezone.utc).isoformat()
        )

        episode_data = {
            "podcast_id": podcast_id,
            "title": entry.get("title", "Untitled"),
            "description": entry.get("description") or entry.get("summary", ""),
            "published_date": published_date,
            # audio_url is nullable (migration 004); newsletters set it to None
            "audio_url": audio_url,
            "article_content": article_content,
            "duration_seconds": (
                parse_duration_to_seconds(entry.itunes_duration)
                if hasattr(entry, "itunes_duration")
                else None
            ),
            # Newsletters skip download+transcribe — go straight to summarize
            "status": "transcribed" if is_newsletter else "pending",
        }

        result = supabase.table("episodes").insert(episode_data).execute()

        if result.data:
            episodes_new += 1
            ep_id = result.data[0]["id"]
            if is_newsletter:
                new_newsletter_ids.append(ep_id)
            else:
                new_podcast_ids.append(ep_id)
            logger.info(f"Created new {'article' if is_newsletter else 'episode'} {ep_id}")

    # Second pass: Queue downloads for podcasts in batches
    batch_size = settings.DOWNLOAD_BATCH_SIZE
    for i in range(0, len(new_podcast_ids), batch_size):
        batch = new_podcast_ids[i:i + batch_size]

        logger.info(f"Queueing batch {i//batch_size + 1}/{(len(new_podcast_ids) + batch_size - 1)//batch_size} "
                   f"({len(batch)} episodes)")

        for episode_id in batch:
            await ctx["redis"].enqueue_job("download_episode_job", episode_id)

        if i + batch_size < len(new_podcast_ids):
            await asyncio.sleep(0.5)

    # Queue summarization directly for newsletters (no download/transcribe step)
    for ep_id in new_newsletter_ids:
        await ctx["redis"].enqueue_job("summarize_episode_job", ep_id)
        logger.info(f"Queued direct summarization for newsletter article {ep_id}")

    # Update last_fetched_at
    supabase.table("podcasts").update({"last_fetched_at": datetime.now(timezone.utc).isoformat()}).eq(
        "id", podcast_id
    ).execute()

    logger.info(
        f"Fetched {episodes_found} entries, {episodes_new} new "
        f"({len(new_newsletter_ids)} articles, {len(new_podcast_ids)} podcast episodes) "
        f"for {feed_type} {podcast_id}"
    )
    return {
        "episodes_found": episodes_found,
        "episodes_new": episodes_new,
        "feed_type": feed_type,
    }


async def download_episode_job(ctx, episode_id: str) -> dict:
    """
    Background job: Download episode audio file to temporary storage

    Features:
    - Semaphore-based concurrency limiting
    - Automatic retry with exponential backoff
    - Proper error handling and cleanup

    Args:
        episode_id: UUID of the episode to download

    Returns:
        dict with download info
    """
    supabase = await get_supabase()

    # Get episode info
    episode_result = supabase.table("episodes").select("*").eq("id", episode_id).execute()

    if not episode_result.data:
        logger.error(f"Episode {episode_id} not found")
        return {"error": "Episode not found"}

    episode = episode_result.data[0]
    audio_url = episode["audio_url"]

    # Use semaphore to limit concurrent downloads
    async with download_semaphore:
        logger.info(f"[{episode_id[:8]}] Starting download (semaphore acquired)")

        # Retry logic with exponential backoff
        last_error = None
        for attempt in range(1, settings.DOWNLOAD_RETRY_ATTEMPTS + 1):
            try:
                logger.info(f"[{episode_id[:8]}] Download attempt {attempt}/{settings.DOWNLOAD_RETRY_ATTEMPTS}")

                # Download audio file to temporary storage
                file_path = await download_episode_temp(episode_id, audio_url)
                logger.info(f"[{episode_id[:8]}] Downloaded successfully to {file_path}")

                # Update episode with temp file path (use absolute path for reliability)
                supabase.table("episodes").update(
                    {"status": "downloaded", "file_path": str(Path(file_path).resolve())}
                ).eq("id", episode_id).execute()

                # Queue transcription job
                await ctx["redis"].enqueue_job("transcribe_episode_job", episode_id)

                return {"success": True, "file_path": str(Path(file_path).resolve()), "attempts": attempt}

            except Exception as e:
                last_error = e
                logger.warning(f"[{episode_id[:8]}] Attempt {attempt} failed: {e}")

                # Clean up partial download if it exists
                try:
                    delete_episode_temp(episode_id)
                except Exception:
                    pass  # Ignore cleanup errors

                # If not the last attempt, wait before retrying (exponential backoff)
                if attempt < settings.DOWNLOAD_RETRY_ATTEMPTS:
                    wait_time = 2 ** attempt  # 2s, 4s, 8s
                    logger.info(f"[{episode_id[:8]}] Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)

        # All retries failed
        logger.error(f"[{episode_id[:8]}] All {settings.DOWNLOAD_RETRY_ATTEMPTS} attempts failed: {last_error}")

        # Mark as failed
        supabase.table("episodes").update({"status": "failed"}).eq("id", episode_id).execute()

        return {"error": str(last_error), "attempts": settings.DOWNLOAD_RETRY_ATTEMPTS}


async def transcribe_episode_job(ctx, episode_id: str) -> dict:
    """
    Background job: Transcribe episode audio to text

    Args:
        episode_id: UUID of the episode to transcribe

    Returns:
        dict with transcription stats
    """
    supabase = await get_supabase()
    logger.info(f"Transcribing episode {episode_id}")

    # Get episode info
    episode_result = supabase.table("episodes").select("*").eq("id", episode_id).execute()

    if not episode_result.data:
        logger.error(f"Episode {episode_id} not found")
        return {"error": "Episode not found"}

    episode = episode_result.data[0]

    # Newsletter articles have no audio — skip straight to summarization
    if episode.get("article_content") and not episode.get("audio_url"):
        logger.info(f"[{episode_id[:8]}] Newsletter article — skipping transcription")
        supabase.table("episodes").update({"status": "transcribed"}).eq("id", episode_id).execute()
        await ctx["redis"].enqueue_job("summarize_episode_job", episode_id)
        return {"skipped": True, "reason": "newsletter_article"}

    if not episode.get("file_path"):
        logger.error(f"Episode {episode_id} has no audio file")
        return {"error": "Audio file not downloaded"}

    # Update status
    try:
        supabase.table("episodes").update({"status": "transcribing"}).eq("id", episode_id).execute()
    except Exception as db_error:
        logger.error(f"Failed to update episode status to transcribing: {db_error}")
        return {"error": f"Database error: {db_error}"}

    try:
        # Transcribe using Groq API or other provider
        audio_path = episode["file_path"]
        audio_path_obj = Path(audio_path)

        # Check file exists
        if not audio_path_obj.exists():
            logger.error(f"[{episode_id[:8]}] Audio file not found: {audio_path}")
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Get file size
        file_size_bytes = audio_path_obj.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)

        logger.info(f"[{episode_id[:8]}] Transcribing with {settings.TRANSCRIPTION_PROVIDER}")
        logger.info(f"[{episode_id[:8]}] Audio path: {audio_path}")
        logger.info(f"[{episode_id[:8]}] File size: {file_size_mb:.1f} MB")

        # Warn if file is too large for Groq API (25MB limit on free tier)
        provider_lower = settings.TRANSCRIPTION_PROVIDER.lower()
        if provider_lower == "groq" and file_size_mb > 25:
            logger.warning(
                f"[{episode_id[:8]}] File size ({file_size_mb:.1f} MB) exceeds Groq free tier limit (25MB). "
                f"Will fall back to faster-whisper for local transcription."
            )

        # Create transcriber instance (uses configured transcription provider)
        from p3.transcriber import AudioTranscriber

        # Get API key for the configured provider
        api_key = None
        provider_lower = settings.TRANSCRIPTION_PROVIDER.lower()
        if provider_lower == "groq":
            api_key = settings.GROQ_API_KEY
        elif provider_lower == "openai":
            api_key = settings.OPENAI_API_KEY
        elif provider_lower == "assemblyai":
            api_key = settings.ASSEMBLYAI_API_KEY
        elif provider_lower == "deepgram":
            api_key = settings.DEEPGRAM_API_KEY

        # Fall back to Groq if configured provider has no key
        if not api_key and provider_lower not in ("local", "faster-whisper"):
            if settings.GROQ_API_KEY:
                logger.warning(f"[{episode_id[:8]}] No key for {settings.TRANSCRIPTION_PROVIDER}, falling back to Groq")
                provider_lower = "groq"
                api_key = settings.GROQ_API_KEY
            elif settings.ASSEMBLYAI_API_KEY:
                logger.warning(f"[{episode_id[:8]}] No key for {settings.TRANSCRIPTION_PROVIDER}, falling back to AssemblyAI")
                provider_lower = "assemblyai"
                api_key = settings.ASSEMBLYAI_API_KEY
            elif settings.DEEPGRAM_API_KEY:
                logger.warning(f"[{episode_id[:8]}] No key for {settings.TRANSCRIPTION_PROVIDER}, falling back to Deepgram")
                provider_lower = "deepgram"
                api_key = settings.DEEPGRAM_API_KEY

        # Initialize transcriber with API settings
        # Enable faster-whisper fallback (4x faster than standard Whisper)
        transcriber = AudioTranscriber(
            db=None,  # We're not using the old database class
            use_api=True,
            api_provider=provider_lower,
            api_model=settings.TRANSCRIPTION_MODEL,
            api_key=api_key,
            use_faster_whisper=True,  # Use faster-whisper when API fails (4x faster than standard Whisper)
        )

        # Run transcription in thread (API calls can block)
        result = await asyncio.to_thread(
            transcriber.transcribe_with_api,
            audio_path
        )

        if not result or not result.get('segments'):
            logger.error(f"[{episode_id[:8]}] Transcription returned no segments")
            raise Exception("Transcription failed - no segments returned")

        segments = result['segments']
        logger.info(f"[{episode_id[:8]}] Transcribed {len(segments)} segments")

        # Store transcripts
        segment_count = 0
        failed_segments = 0
        for segment in segments:
            try:
                # Validate segment has required fields
                if not all(key in segment for key in ["start", "end", "text"]):
                    logger.warning(f"Segment missing required fields: {segment.keys()}")
                    failed_segments += 1
                    continue

                transcript_data = {
                    "episode_id": episode_id,
                    "speaker": segment.get("speaker"),
                    "timestamp_start": segment["start"],
                    "timestamp_end": segment["end"],
                    "text": segment["text"],
                    "confidence": segment.get("confidence"),
                }

                supabase.table("transcripts").insert(transcript_data).execute()
                segment_count += 1
            except Exception as seg_error:
                logger.error(f"Failed to insert segment {segment_count}: {seg_error}")
                failed_segments += 1
                # Continue processing other segments

        if failed_segments > 0:
            logger.warning(f"Failed to insert {failed_segments} segments out of {len(segments)}")

        if segment_count == 0:
            raise Exception("No segments were successfully inserted")

        logger.info(f"Transcribed episode {episode_id} ({segment_count} segments)")

        # Update episode status
        try:
            supabase.table("episodes").update({"status": "transcribed"}).eq("id", episode_id).execute()
        except Exception as db_error:
            logger.error(f"Failed to update episode status to transcribed: {db_error}")
            raise

        # Queue summarization job
        await ctx["redis"].enqueue_job("summarize_episode_job", episode_id)

        return {"success": True, "segment_count": segment_count}

    except Exception as e:
        logger.error(f"Failed to transcribe episode {episode_id}: {e}", exc_info=True)
        # Mark as failed
        supabase.table("episodes").update({"status": "failed"}).eq("id", episode_id).execute()

        # Clean up temp file on failure
        try:
            delete_episode_temp(episode_id)
        except Exception:
            pass  # Ignore cleanup errors

        return {"error": str(e)}


async def summarize_episode_job(ctx, episode_id: str) -> dict:
    """
    Background job: Generate LLM summary from transcript

    Args:
        episode_id: UUID of the episode to summarize

    Returns:
        dict with summary info
    """
    supabase = await get_supabase()
    logger.info(f"Summarizing episode {episode_id}")

    # Get episode and transcripts
    episode_result = supabase.table("episodes").select("*").eq("id", episode_id).execute()

    if not episode_result.data:
        logger.error(f"Episode {episode_id} not found")
        return {"error": "Episode not found"}

    episode = episode_result.data[0]

    # Get transcripts
    transcripts_result = (
        supabase.table("transcripts")
        .select("*")
        .eq("episode_id", episode_id)
        .order("timestamp_start")
        .execute()
    )

    # Update status
    supabase.table("episodes").update({"status": "processing"}).eq("id", episode_id).execute()

    try:
        if transcripts_result.data:
            # Podcast episode — build transcript from segments
            full_transcript = "\n".join(
                [f"[{t['timestamp_start']:.1f}s] {t['text']}" for t in transcripts_result.data]
            )
        elif episode.get("article_content"):
            # Newsletter article — strip HTML and use article text directly
            import re as _re
            raw_html = episode["article_content"]
            full_transcript = _re.sub(r"<[^>]+>", " ", raw_html)
            full_transcript = _re.sub(r"\s+", " ", full_transcript).strip()
            logger.info(f"[{episode_id[:8]}] Using article_content ({len(full_transcript)} chars) for summarization")
        else:
            logger.error(f"No transcripts or article_content found for episode {episode_id}")
            supabase.table("episodes").update({"status": "failed"}).eq("id", episode_id).execute()
            return {"error": "No content to summarize"}

        # Generate structured summary using LLM with provider fallback chain
        summary = None
        providers_tried = []
        for llm_provider, llm_model in _get_ordered_llm_providers():
            providers_tried.append(llm_provider)
            logger.info(f"[{episode_id[:8]}] Trying LLM provider: {llm_provider}/{llm_model}")
            summary = await asyncio.to_thread(
                _generate_structured_summary,
                full_transcript,
                llm_provider,
                llm_model
            )
            if summary:
                logger.info(f"[{episode_id[:8]}] Summary generated with {llm_provider}")
                break
            logger.warning(f"[{episode_id[:8]}] Provider {llm_provider} failed, trying next...")

        if not summary:
            raise Exception(f"Failed to generate summary with all providers: {providers_tried}")

        # Store summary
        summary_data = {
            "episode_id": episode_id,
            "key_topics": summary.get("key_topics", []),
            "themes": summary.get("themes", []),
            "quotes": summary.get("quotes", []),
            "companies": summary.get("companies", []),
            "key_insights": summary.get("key_insights", []),
            "actionable_takeaways": summary.get("actionable_takeaways", []),
            "full_summary": summary.get("full_summary", summary.get("summary", "")),
            "llm_model": f"{llm_provider}/{llm_model}",
            "digest_date": datetime.now(timezone.utc).date().isoformat(),
        }

        try:
            supabase.table("summaries").insert(summary_data).execute()
        except Exception as db_error:
            logger.error(f"Failed to insert summary into database: {db_error}")
            raise

        logger.info(f"Summarized episode {episode_id}")

        # Update episode status
        supabase.table("episodes").update({"status": "completed"}).eq("id", episode_id).execute()

        # Clean up temporary audio file (no longer needed)
        try:
            deleted = delete_episode_temp(episode_id)
            if deleted:
                logger.info(f"Deleted temporary audio file for episode {episode_id}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to delete temp file for episode {episode_id}: {cleanup_error}")
            # Don't fail the job if cleanup fails

        return {"success": True}

    except Exception as e:
        logger.error(f"Failed to summarize episode {episode_id}: {e}", exc_info=True)
        # Mark as failed
        supabase.table("episodes").update({"status": "failed"}).eq("id", episode_id).execute()

        # Clean up temp file on failure too
        try:
            delete_episode_temp(episode_id)
        except Exception:
            pass  # Ignore cleanup errors

        return {"error": str(e)}


async def generate_user_digest_job(ctx, user_id: str, digest_id: str) -> dict:
    """
    Background job: Generate personalized digest for user

    Args:
        user_id: UUID of the user
        digest_id: UUID of the digest to generate

    Returns:
        dict with generation info
    """
    supabase = await get_supabase()
    logger.info(f"Generating digest {digest_id} for user {user_id}")

    # Get user digest
    digest_result = supabase.table("user_digests").select("*").eq("id", digest_id).execute()

    if not digest_result.data:
        logger.error(f"Digest {digest_id} not found")
        return {"error": "Digest not found"}

    digest = digest_result.data[0]

    # Update status
    supabase.table("user_digests").update({"status": "generating"}).eq("id", digest_id).execute()

    try:
        # Get user subscriptions
        subs_result = (
            supabase.table("user_subscriptions")
            .select("podcast_id")
            .eq("user_id", user_id)
            .eq("is_active", True)
            .execute()
        )

        podcast_ids = [sub["podcast_id"] for sub in subs_result.data]

        if not podcast_ids:
            logger.warning(f"User {user_id} has no active subscriptions")
            return {"error": "No active subscriptions"}

        # Get user preferences
        prefs_result = (
            supabase.table("user_digest_preferences")
            .select("*")
            .eq("user_id", user_id)
            .execute()
        )

        prefs = prefs_result.data[0] if prefs_result.data else {}

        # Get recent completed episodes from subscribed podcasts
        episodes_result = (
            supabase.table("episodes")
            .select("*, summaries(*)")
            .in_("podcast_id", podcast_ids)
            .eq("status", "completed")
            .order("published_date", desc=True)
            .limit(prefs.get("max_episodes_per_digest", 10))
            .execute()
        )

        episodes = episodes_result.data

        # Filter out episodes user has already listened to
        # Covers both Spotify history and in-app listens
        if prefs.get("exclude_listened_on_spotify", True):
            listened_episode_ids = set()

            # Spotify listening history
            spotify_result = (
                supabase.table("user_spotify_tracking")
                .select("episode_id")
                .eq("user_id", user_id)
                .execute()
            )
            listened_episode_ids.update(r["episode_id"] for r in spotify_result.data)

            # In-app listen tracking (migration 003)
            inapp_result = (
                supabase.table("user_episode_listens")
                .select("episode_id")
                .eq("user_id", user_id)
                .execute()
            )
            listened_episode_ids.update(r["episode_id"] for r in inapp_result.data)

            episodes = [e for e in episodes if e["id"] not in listened_episode_ids]

        # Collect episode and summary IDs
        episode_ids = [e["id"] for e in episodes]
        summary_ids = []
        for e in episodes:
            sums = e.get("summaries") or []
            if isinstance(sums, dict):
                sums = [sums]
            if sums:
                summary_ids.append(sums[0]["id"])

        # Render markdown digest content
        digest_date = digest.get("digest_date") or datetime.now(timezone.utc).date().isoformat()
        content = _render_digest_markdown(episodes, digest_date)

        # Update digest record — 'content' column added in migration 002
        update_payload = {
            "status": "completed",
            "episode_ids": episode_ids,
            "summary_ids": summary_ids,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "content": content,
        }
        supabase.table("user_digests").update(update_payload).eq("id", digest_id).execute()

        logger.info(f"Generated digest {digest_id} with {len(episodes)} episodes")

        # Enqueue email delivery if user has it enabled
        if prefs.get("email_enabled") and prefs.get("email_address"):
            await ctx["redis"].enqueue_job("send_digest_email_job", user_id, digest_id)
            logger.info(f"Enqueued email delivery for digest {digest_id} → {prefs['email_address']}")

        return {
            "success": True,
            "episode_count": len(episodes),
            "filtered_by_spotify": len(episodes_result.data) - len(episodes),
        }

    except Exception as e:
        logger.error(f"Failed to generate digest {digest_id}: {e}", exc_info=True)
        # Mark as failed
        supabase.table("user_digests").update({"status": "failed"}).eq("id", digest_id).execute()

        return {"error": str(e)}


async def sync_spotify_job(ctx, user_id: str) -> dict:
    """
    Background job: Sync user's Spotify followed shows and listening history.

    Mirrors the logic in routers/spotify.py but runs without an HTTP request
    context, using the stored OAuth tokens directly from the database.
    Automatically refreshes the Spotify access token if it has expired.
    """
    supabase = await get_supabase()
    logger.info(f"[spotify-sync] Starting for user {user_id}")

    # Fetch stored OAuth connection
    conn_result = (
        supabase.table("user_oauth_connections")
        .select("*")
        .eq("user_id", user_id)
        .eq("provider", "spotify")
        .execute()
    )

    if not conn_result.data:
        logger.warning(f"[spotify-sync] No Spotify connection found for user {user_id}")
        return {"error": "Spotify not connected"}

    oauth_data = conn_result.data[0]
    access_token = await _get_spotify_token(supabase, user_id, oauth_data)

    if not access_token:
        logger.error(f"[spotify-sync] Could not obtain access token for user {user_id}")
        return {"error": "Failed to get Spotify access token"}

    spotify_client = spotipy.Spotify(auth=access_token)
    sync_stats = {"shows_synced": 0, "episodes_synced": 0, "new_subscriptions": 0}

    try:
        # --- 1. Sync followed shows ---
        followed_shows = await asyncio.to_thread(
            spotify_client.current_user_saved_shows, limit=50
        )

        for item in followed_shows["items"]:
            show = item["show"]
            spotify_show_id = show["id"]

            supabase.table("user_spotify_shows").upsert({
                "user_id": user_id,
                "spotify_show_id": spotify_show_id,
                "show_name": show["name"],
                "show_publisher": show.get("publisher"),
                "followed_at": item.get("added_at"),
            }).execute()
            sync_stats["shows_synced"] += 1

            # Match to our catalog and auto-subscribe if found
            podcast_match = (
                supabase.table("podcasts")
                .select("id")
                .eq("spotify_show_id", spotify_show_id)
                .execute()
            )
            if podcast_match.data:
                podcast_id = podcast_match.data[0]["id"]

                supabase.table("user_spotify_shows").update(
                    {"matched_podcast_id": podcast_id}
                ).eq("user_id", user_id).eq("spotify_show_id", spotify_show_id).execute()

                existing_sub = (
                    supabase.table("user_subscriptions")
                    .select("id")
                    .eq("user_id", user_id)
                    .eq("podcast_id", podcast_id)
                    .execute()
                )
                if not existing_sub.data:
                    supabase.table("user_subscriptions").insert({
                        "user_id": user_id,
                        "podcast_id": podcast_id,
                        "is_active": True,
                        "auto_added_from_spotify": True,
                    }).execute()
                    supabase.table("user_spotify_shows").update(
                        {"auto_subscribed": True}
                    ).eq("user_id", user_id).eq("spotify_show_id", spotify_show_id).execute()
                    sync_stats["new_subscriptions"] += 1

        # --- 2. Sync recently played podcast episodes ---
        recently_played = await asyncio.to_thread(
            spotify_client.current_user_recently_played, limit=50
        )

        for item in recently_played["items"]:
            if item["track"]["type"] == "episode":
                episode = item["track"]
                spotify_episode_id = episode["id"]

                episode_match = (
                    supabase.table("episodes")
                    .select("id")
                    .eq("spotify_episode_id", spotify_episode_id)
                    .execute()
                )
                if episode_match.data:
                    episode_id = episode_match.data[0]["id"]
                    supabase.table("user_spotify_tracking").upsert({
                        "user_id": user_id,
                        "episode_id": episode_id,
                        "spotify_episode_id": spotify_episode_id,
                        "listened_at": item["played_at"],
                        "fully_played": True,
                    }).execute()
                    sync_stats["episodes_synced"] += 1

        logger.info(f"[spotify-sync] Completed for user {user_id}: {sync_stats}")
        return {"success": True, "stats": sync_stats}

    except Exception as e:
        logger.error(f"[spotify-sync] Failed for user {user_id}: {e}", exc_info=True)
        return {"error": str(e)}


async def scheduled_daily_digests_job(ctx) -> dict:
    """
    Cron job: Run every hour and enqueue digest generation for users
    whose delivery_time falls within the current UTC hour.

    user_digest_preferences.delivery_time is stored as a TIME value.
    We match users whose delivery hour (in delivery_timezone) equals now.
    For simplicity, delivery_timezone is respected by converting to UTC.
    """
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    supabase = await get_supabase()
    now_utc = datetime.now(timezone.utc)
    current_hour = now_utc.hour
    current_day_of_week = now_utc.strftime("%A").lower()  # 'monday', etc.

    logger.info(f"[scheduled-digests] Running for UTC hour {current_hour:02d}:00")

    # Fetch all active digest preferences
    prefs_result = supabase.table("user_digest_preferences").select(
        "user_id, delivery_time, delivery_timezone, frequency"
    ).execute()

    enqueued = 0
    for pref in prefs_result.data:
        try:
            user_id = pref["user_id"]
            frequency = pref.get("frequency", "daily")
            delivery_time_str = pref.get("delivery_time", "08:00:00")
            tz_name = pref.get("delivery_timezone", "UTC")

            # Parse delivery hour in user's timezone
            try:
                user_tz = ZoneInfo(tz_name)
            except (ZoneInfoNotFoundError, Exception):
                user_tz = ZoneInfo("UTC")

            delivery_hour = int(str(delivery_time_str).split(":")[0])

            # Convert delivery hour to UTC for comparison
            sample_dt = datetime.now(user_tz).replace(
                hour=delivery_hour, minute=0, second=0, microsecond=0
            )
            delivery_hour_utc = sample_dt.astimezone(timezone.utc).hour

            if delivery_hour_utc != current_hour:
                continue

            # Skip weekly digests on non-Monday
            if frequency == "weekly" and current_day_of_week != "monday":
                continue

            # Check if a digest already exists for today to avoid duplicates
            today = now_utc.date().isoformat()
            existing = (
                supabase.table("user_digests")
                .select("id")
                .eq("user_id", user_id)
                .eq("digest_date", today)
                .execute()
            )
            if existing.data:
                continue

            # Create digest record and enqueue job
            digest_record = supabase.table("user_digests").insert({
                "user_id": user_id,
                "digest_date": today,
                "title": f"Digest for {today}",
                "status": "pending",
            }).execute()

            if digest_record.data:
                digest_id = digest_record.data[0]["id"]
                await ctx["redis"].enqueue_job(
                    "generate_user_digest_job", user_id, digest_id
                )
                enqueued += 1
                logger.info(f"[scheduled-digests] Enqueued digest {digest_id} for user {user_id}")

        except Exception as e:
            logger.error(f"[scheduled-digests] Error processing user {pref.get('user_id')}: {e}")

    logger.info(f"[scheduled-digests] Enqueued {enqueued} digests for UTC hour {current_hour:02d}:00")
    return {"enqueued": enqueued, "utc_hour": current_hour}


async def send_digest_email_job(ctx, user_id: str, digest_id: str) -> dict:
    """
    Background job: Send a completed digest to the user's email inbox.

    Fetches the digest + all episode summaries, renders HTML and plain-text
    versions, then delivers via Resend (primary) or SMTP (fallback).
    """
    supabase = await get_supabase()
    logger.info(f"[email] Sending digest {digest_id} for user {user_id}")

    # Fetch user preferences (email address, enabled flag)
    prefs_result = (
        supabase.table("user_digest_preferences")
        .select("email_enabled, email_address")
        .eq("user_id", user_id)
        .execute()
    )
    if not prefs_result.data:
        return {"error": "No preferences found"}

    prefs = prefs_result.data[0]
    if not prefs.get("email_enabled"):
        return {"skipped": True, "reason": "email_disabled"}
    to_address = prefs.get("email_address")
    if not to_address:
        return {"error": "No email address configured"}

    # Fetch digest record
    digest_result = (
        supabase.table("user_digests")
        .select("*")
        .eq("id", digest_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not digest_result.data:
        return {"error": "Digest not found"}
    digest = digest_result.data[0]

    if digest["status"] != "completed":
        return {"error": f"Digest not ready (status={digest['status']})"}

    # Fetch episodes with summaries
    episodes = []
    episode_ids = digest.get("episode_ids") or []
    if episode_ids:
        ep_result = (
            supabase.table("episodes")
            .select("*, podcasts(*), summaries(*)")
            .in_("id", episode_ids)
            .execute()
        )
        episodes = ep_result.data

    if not episodes:
        return {"skipped": True, "reason": "no_episodes"}

    digest_date = str(digest.get("digest_date", ""))
    subject = f"Your Parakeet Digest — {digest_date}"

    html, text = render_digest_html(episodes, digest_date, digest.get("title", ""))

    ok = await send_email(
        to=to_address,
        subject=subject,
        html=html,
        text=text,
        resend_api_key=settings.RESEND_API_KEY,
        smtp_host=settings.SMTP_HOST,
        smtp_port=settings.SMTP_PORT,
        smtp_user=settings.SMTP_USER,
        smtp_password=settings.SMTP_PASSWORD,
        from_address=settings.EMAIL_FROM,
        from_name=settings.EMAIL_FROM_NAME,
    )

    if ok:
        logger.info(f"[email] Delivered digest {digest_id} to {to_address}")
        return {"success": True, "to": to_address}

    logger.error(f"[email] Failed to deliver digest {digest_id} to {to_address}")
    return {"error": "Email delivery failed"}


async def refresh_all_feeds_job(ctx) -> dict:
    """
    Cron job: Refresh RSS feeds for all podcasts with active subscribers.

    Runs every 6 hours. Skips podcasts fetched within the last 6 hours to
    avoid hammering feeds. Enqueues fetch_feed_job for each qualifying podcast.
    """
    supabase = await get_supabase()
    now_utc = datetime.now(timezone.utc)
    stale_threshold = now_utc - timedelta(hours=6)
    logger.info(f"[feed-refresh] Starting scheduled refresh at {now_utc.isoformat()}")

    # Find all podcasts that have at least one active subscriber
    subs_result = (
        supabase.table("user_subscriptions")
        .select("podcast_id")
        .eq("is_active", True)
        .execute()
    )

    if not subs_result.data:
        logger.info("[feed-refresh] No active subscriptions found")
        return {"enqueued": 0}

    # Deduplicate podcast IDs
    podcast_ids = list({row["podcast_id"] for row in subs_result.data})

    # Fetch last_fetched_at for those podcasts to filter out recently refreshed ones
    podcasts_result = (
        supabase.table("podcasts")
        .select("id, title, last_fetched_at")
        .in_("id", podcast_ids)
        .execute()
    )

    enqueued = 0
    skipped = 0
    for podcast in podcasts_result.data:
        last_fetched = podcast.get("last_fetched_at")
        if last_fetched:
            try:
                fetched_dt = datetime.fromisoformat(str(last_fetched).replace("Z", "+00:00"))
                if fetched_dt.tzinfo is None:
                    fetched_dt = fetched_dt.replace(tzinfo=timezone.utc)
                if fetched_dt >= stale_threshold:
                    skipped += 1
                    continue
            except (ValueError, TypeError):
                pass  # Unknown format — proceed with refresh

        await ctx["redis"].enqueue_job("fetch_feed_job", podcast["id"])
        enqueued += 1
        logger.info(f"[feed-refresh] Queued refresh for podcast {podcast['id']} ({podcast.get('title', '')})")

    logger.info(f"[feed-refresh] Enqueued {enqueued} refreshes, skipped {skipped} (recently fetched)")
    return {"enqueued": enqueued, "skipped": skipped}


# Worker class definition
class WorkerSettings:
    """arq worker settings"""

    functions = [
        fetch_feed_job,
        download_episode_job,
        transcribe_episode_job,
        summarize_episode_job,
        generate_user_digest_job,
        send_digest_email_job,
        sync_spotify_job,
        scheduled_daily_digests_job,
        refresh_all_feeds_job,
    ]

    # Cron schedule:
    #   - scheduled_daily_digests_job: every hour on the hour
    #   - refresh_all_feeds_job: every 6 hours (00:00, 06:00, 12:00, 18:00 UTC)
    cron_jobs = [
        arq_cron(scheduled_daily_digests_job, hour=None, minute=0, run_at_startup=False),
        arq_cron(refresh_all_feeds_job, hour={0, 6, 12, 18}, minute=5, run_at_startup=True),
    ]

    redis_settings = redis_settings

    # Worker settings - Conservative to prevent Redis timeouts with Upstash
    max_jobs = 1  # Process ONE job at a time to prevent connection pool exhaustion
    job_timeout = 7200  # 2 hours for very large file downloads/transcriptions (handles 3-hour podcasts)
    keep_result = 3600  # Keep results for 1 hour
    poll_delay = 2.0  # Poll every 2 seconds (reduced Redis load)
    max_tries = 3  # Retry failed jobs up to 3 times


# CLI entry point for worker
def start_worker():
    """Start the arq worker"""
    from arq import run_worker
    import redis.exceptions

    # Configure logging - reduce Redis connection spam
    logging.basicConfig(
        level=logging.INFO if settings.DEBUG else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Suppress verbose Redis connection logs
    logging.getLogger("redis.asyncio.connection").setLevel(logging.ERROR)
    logging.getLogger("redis.asyncio.retry").setLevel(logging.ERROR)

    logger.info("🚀 Starting P3 background worker...")
    logger.info(f"📁 Working directory: {os.getcwd()}")
    logger.info(f"📊 Redis: {settings.REDIS_URL[:50]}...")
    logger.info(f"⚙️  Max concurrent jobs: {WorkerSettings.max_jobs}")
    logger.info(f"⏱️  Poll delay: {WorkerSettings.poll_delay}s")
    logger.info(f"🔄 Max retries: {WorkerSettings.max_tries}")

    try:
        run_worker(WorkerSettings)
    except redis.exceptions.TimeoutError:
        logger.error("\n❌ Redis Connection Timeout")
        logger.error("The worker lost connection to Redis (Upstash).")
        logger.error("This is usually temporary. Simply restart the worker:")
        logger.error("  python -m p3.api.worker")
        logger.error("\nJobs in progress were saved and will resume automatically.\n")
    except redis.exceptions.ConnectionError as e:
        logger.error("\n❌ Redis Connection Error")
        logger.error(f"Could not connect to Redis: {e}")
        logger.error("Check your REDIS_URL in .env and network connection.\n")
    except KeyboardInterrupt:
        logger.info("\n👋 Worker stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}", exc_info=True)


if __name__ == "__main__":
    start_worker()

