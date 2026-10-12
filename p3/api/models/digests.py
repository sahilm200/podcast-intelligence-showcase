"""
Digest models
"""

from datetime import date, datetime, time
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel


class DigestResponse(BaseModel):
    """Digest information"""
    id: UUID
    user_id: UUID
    digest_date: date
    title: Optional[str] = None
    status: str
    created_at: datetime
    generated_at: Optional[datetime] = None


class DigestDetailResponse(DigestResponse):
    """Detailed digest with episodes"""
    episode_ids: Optional[List[str]] = None
    summary_ids: Optional[List[str]] = None
    markdown_path: Optional[str] = None
    json_path: Optional[str] = None
    content: Optional[str] = None
    audio_digest_id: Optional[UUID] = None
    episodes: List[dict] = []


class DigestPreferencesResponse(BaseModel):
    """User digest preferences"""
    user_id: UUID
    frequency: str = "daily"
    delivery_time: time = time(8, 0)
    delivery_timezone: str = "UTC"
    min_episode_duration_sec: int = 300
    max_episodes_per_digest: int = 10
    include_categories: Optional[List[str]] = None
    exclude_categories: Optional[List[str]] = None
    exclude_listened_on_spotify: bool = True
    spotify_sync_enabled: bool = True
    email_enabled: bool = False
    email_address: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class UpdatePreferencesRequest(BaseModel):
    """Request to update digest preferences"""
    frequency: Optional[str] = None
    delivery_time: Optional[time] = None
    delivery_timezone: Optional[str] = None
    min_episode_duration_sec: Optional[int] = None
    max_episodes_per_digest: Optional[int] = None
    include_categories: Optional[List[str]] = None
    exclude_categories: Optional[List[str]] = None
    exclude_listened_on_spotify: Optional[bool] = None
    spotify_sync_enabled: Optional[bool] = None
    email_enabled: Optional[bool] = None
    email_address: Optional[str] = None

