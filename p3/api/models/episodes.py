"""
Episode models
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel


class EpisodeDetailResponse(BaseModel):
    """Full episode detail including podcast info"""
    id: UUID
    podcast_id: UUID
    title: str
    description: Optional[str] = None
    published_date: datetime
    audio_url: Optional[str] = None  # nullable for newsletter articles
    duration_seconds: Optional[int] = None
    status: str
    spotify_episode_id: Optional[str] = None
    created_at: datetime
    # Embedded podcast info
    podcast: Optional[Dict[str, Any]] = None


class SummaryResponse(BaseModel):
    """Episode AI summary"""
    id: UUID
    episode_id: UUID
    llm_model: Optional[str] = None
    # Structured summary fields stored as JSONB in DB
    full_summary: Optional[str] = None
    key_topics: Optional[List[str]] = None
    themes: Optional[List[str]] = None
    quotes: Optional[List[str]] = None
    companies: Optional[List[str]] = None
    key_insights: Optional[List[str]] = None
    actionable_takeaways: Optional[List[str]] = None
    created_at: datetime


class TranscriptSegmentResponse(BaseModel):
    """Single transcript segment"""
    id: UUID
    episode_id: UUID
    speaker: Optional[str] = None
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None
    text: str
    confidence: Optional[float] = None


class MarkListenedRequest(BaseModel):
    """Request to record an in-app listen"""
    progress_seconds: Optional[int] = None
    fully_played: bool = False

