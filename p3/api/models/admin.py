"""
Admin models
"""

from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel


class CatalogRequestResponse(BaseModel):
    """Catalog request information"""
    id: UUID
    user_id: UUID
    rss_url: str
    title: Optional[str] = None
    description: Optional[str] = None
    reason: Optional[str] = None
    status: str
    reviewed_by: Optional[UUID] = None
    reviewed_at: Optional[datetime] = None
    review_notes: Optional[str] = None
    approved_podcast_id: Optional[UUID] = None
    created_at: datetime


class ApproveCatalogRequestRequest(BaseModel):
    """Request to approve a catalog request"""
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    notes: Optional[str] = None


class RejectCatalogRequestRequest(BaseModel):
    """Request to reject a catalog request"""
    reason: str


class CuratePodcastRequest(BaseModel):
    """Request to directly add a curated podcast"""
    title: str
    rss_url: str
    description: Optional[str] = None
    category: Optional[str] = None
    author: Optional[str] = None
    website_url: Optional[str] = None
    spotify_show_id: Optional[str] = None

