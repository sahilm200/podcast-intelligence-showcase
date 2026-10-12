"""
Subscription models
"""

from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel

from p3.api.models.catalog import PodcastResponse


class SubscriptionResponse(BaseModel):
    """User subscription information"""
    id: UUID
    user_id: UUID
    podcast_id: UUID
    is_active: bool
    auto_added_from_spotify: bool
    subscribed_at: datetime
    unsubscribed_at: Optional[datetime] = None


class SubscriptionWithPodcastResponse(SubscriptionResponse):
    """Subscription with podcast details"""
    podcast: PodcastResponse


class SubscribeRequest(BaseModel):
    """Request to subscribe to a podcast"""
    podcast_id: UUID


class UnsubscribeRequest(BaseModel):
    """Request to unsubscribe from a podcast"""
    podcast_id: UUID

