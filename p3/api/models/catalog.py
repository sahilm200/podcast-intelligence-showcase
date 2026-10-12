"""
Catalog models
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel, HttpUrl


class PodcastResponse(BaseModel):
    """Podcast information"""
    id: UUID
    title: str
    rss_url: str
    description: Optional[str] = None
    category: Optional[str] = None
    author: Optional[str] = None
    image_url: Optional[str] = None
    website_url: Optional[str] = None
    is_curated: bool
    feed_type: str = "podcast"
    spotify_show_id: Optional[str] = None
    created_at: datetime


class EpisodeResponse(BaseModel):
    """Episode information"""
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


class PodcastDetailResponse(PodcastResponse):
    """Detailed podcast information with episodes"""
    recent_episodes: List[EpisodeResponse] = []
    is_subscribed: bool = False


class AddPodcastRequest(BaseModel):
    """Request to add a custom podcast or newsletter feed"""
    title: str
    rss_url: str
    description: Optional[str] = None
    category: Optional[str] = None
    feed_type: str = "podcast"  # "podcast" or "newsletter"

