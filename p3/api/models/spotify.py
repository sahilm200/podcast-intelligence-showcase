"""
Spotify integration models
"""

from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel


class SpotifyShowResponse(BaseModel):
    """Spotify show information"""
    id: UUID
    user_id: UUID
    spotify_show_id: str
    show_name: str
    show_publisher: Optional[str] = None
    followed_at: Optional[datetime] = None
    auto_subscribed: bool = False
    matched_podcast_id: Optional[UUID] = None
    matched_podcast: Optional[dict] = None
    synced_at: datetime


class SpotifyListeningHistoryResponse(BaseModel):
    """Spotify listening history entry"""
    id: UUID
    user_id: UUID
    episode_id: UUID
    spotify_episode_id: str
    listened_at: datetime
    fully_played: Optional[bool] = None
    progress_ms: Optional[int] = None
    episodes: Optional[dict] = None


class SyncSpotifyRequest(BaseModel):
    """Request to sync Spotify data (empty for now)"""
    pass

