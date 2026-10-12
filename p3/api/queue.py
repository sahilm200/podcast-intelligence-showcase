"""
Job queue utilities
Helper functions to enqueue background jobs
"""

import logging
from typing import Optional
from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from p3.api.config import get_settings


settings = get_settings()
logger = logging.getLogger(__name__)

# Redis settings
redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)

# Global pool (singleton)
_redis_pool: Optional[ArqRedis] = None


async def get_redis_pool() -> ArqRedis:
    """Get Redis connection pool for job queue"""
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = await create_pool(redis_settings)
    return _redis_pool


async def enqueue_job(job_name: str, *args, **kwargs) -> Optional[str]:
    """
    Enqueue a background job

    Args:
        job_name: Name of the job function
        *args: Positional arguments for the job
        **kwargs: Keyword arguments for the job

    Returns:
        Job ID if queued successfully, None otherwise
    """
    try:
        redis = await get_redis_pool()
        job = await redis.enqueue_job(job_name, *args, **kwargs)
        return job.job_id if job else None
    except Exception as e:
        logger.error(f"Failed to enqueue job {job_name}: {e}")
        return None


async def fetch_podcast_feed(podcast_id: str) -> Optional[str]:
    """Queue job to fetch new episodes from podcast RSS feed"""
    return await enqueue_job("fetch_feed_job", podcast_id)


async def download_episode(episode_id: str) -> Optional[str]:
    """Queue job to download episode audio file"""
    return await enqueue_job("download_episode_job", episode_id)


async def transcribe_episode(episode_id: str) -> Optional[str]:
    """Queue job to transcribe episode audio"""
    return await enqueue_job("transcribe_episode_job", episode_id)


async def summarize_episode(episode_id: str) -> Optional[str]:
    """Queue job to generate episode summary"""
    return await enqueue_job("summarize_episode_job", episode_id)


async def generate_user_digest(user_id: str, digest_id: str) -> Optional[str]:
    """Queue job to generate personalized user digest"""
    return await enqueue_job("generate_user_digest_job", user_id, digest_id)


async def sync_spotify(user_id: str) -> Optional[str]:
    """Queue job to sync user's Spotify data"""
    return await enqueue_job("sync_spotify_job", user_id)


async def send_digest_email(user_id: str, digest_id: str) -> Optional[str]:
    """Queue job to email a completed digest to the user"""
    return await enqueue_job("send_digest_email_job", user_id, digest_id)


async def close_redis_pool():
    """Close the Redis connection pool (for cleanup)"""
    global _redis_pool
    if _redis_pool is not None:
        await _redis_pool.close()
        _redis_pool = None

