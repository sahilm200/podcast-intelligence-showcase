"""
Podcast catalog endpoints
Browse curated and user-added podcasts
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from supabase import Client

from p3.api.deps import get_supabase, get_current_user, get_optional_user
from p3.api.models.catalog import (
    PodcastResponse,
    PodcastDetailResponse,
    EpisodeResponse,
    AddPodcastRequest,
)
from p3.api import queue


router = APIRouter()


@router.get("/podcasts", response_model=List[PodcastResponse])
async def list_podcasts(
    category: Optional[str] = Query(None, description="Filter by category"),
    curated_only: bool = Query(False, description="Show only curated podcasts"),
    search: Optional[str] = Query(None, description="Search by title/description"),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0),
    supabase: Client = Depends(get_supabase),
    current_user: Optional[dict] = Depends(get_optional_user),
):
    """
    List podcasts in the catalog

    Returns curated and user-added podcasts. Can filter by category.
    """
    query = supabase.table("podcasts").select("*")

    if curated_only:
        query = query.eq("is_curated", True)

    if category:
        query = query.eq("category", category)

    if search:
        query = query.ilike("title", f"%{search}%")

    query = query.order("created_at", desc=True).range(offset, offset + limit - 1)

    result = query.execute()

    return [PodcastResponse(**podcast) for podcast in result.data]


@router.get("/podcasts/{podcast_id}", response_model=PodcastDetailResponse)
async def get_podcast(
    podcast_id: UUID,
    supabase: Client = Depends(get_supabase),
    current_user: Optional[dict] = Depends(get_optional_user),
):
    """
    Get detailed podcast information including recent episodes
    """
    # Get podcast
    podcast_result = supabase.table("podcasts").select("*").eq("id", str(podcast_id)).execute()

    if not podcast_result.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Podcast not found",
        )

    podcast = podcast_result.data[0]

    # Get recent episodes
    episodes_result = (
        supabase.table("episodes")
        .select("*")
        .eq("podcast_id", str(podcast_id))
        .order("published_date", desc=True)
        .limit(10)
        .execute()
    )

    # Check if current user is subscribed (if authenticated)
    is_subscribed = False
    if current_user:
        sub_result = (
            supabase.table("user_subscriptions")
            .select("*")
            .eq("user_id", current_user.id)
            .eq("podcast_id", str(podcast_id))
            .eq("is_active", True)
            .execute()
        )
        is_subscribed = len(sub_result.data) > 0

    return PodcastDetailResponse(
        **podcast,
        recent_episodes=[EpisodeResponse(**ep) for ep in episodes_result.data],
        is_subscribed=is_subscribed,
    )


@router.post("/podcasts", response_model=PodcastResponse, status_code=status.HTTP_201_CREATED)
async def add_custom_podcast(
    request: AddPodcastRequest,
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """
    Add a custom podcast to the catalog

    Users can add podcasts not in the curated catalog.
    The podcast becomes available for subscription.
    """
    # Check if podcast already exists
    existing = supabase.table("podcasts").select("*").eq("rss_url", request.rss_url).execute()

    if existing.data:
        # Already exists, just return it
        return PodcastResponse(**existing.data[0])

    feed_type = request.feed_type if request.feed_type in ("podcast", "newsletter") else "podcast"

    # Create new podcast
    podcast_data = {
        "title": request.title,
        "rss_url": request.rss_url,
        "description": request.description,
        "category": request.category,
        "is_curated": False,
        "feed_type": feed_type,
        "added_by_user_id": current_user.id,
    }

    result = supabase.table("podcasts").insert(podcast_data).execute()

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add podcast",
        )

    # Queue background job to fetch episodes
    await queue.fetch_podcast_feed(result.data[0]["id"])

    return PodcastResponse(**result.data[0])


@router.get("/categories")
async def list_categories(supabase: Client = Depends(get_supabase)):
    """
    List all available podcast categories
    """
    result = (
        supabase.table("podcasts")
        .select("category")
        .not_.is_("category", "null")
        .execute()
    )

    categories = list(set(podcast["category"] for podcast in result.data if podcast.get("category")))
    categories.sort()

    return {"categories": categories}

