"""
User subscription management endpoints
"""

from datetime import datetime, timezone
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client

from p3.api.deps import get_supabase, get_current_user
from p3.api.models.subscriptions import (
    SubscriptionResponse,
    SubscribeRequest,
    UnsubscribeRequest,
    SubscriptionWithPodcastResponse,
)


router = APIRouter()


@router.get("/", response_model=List[SubscriptionWithPodcastResponse])
async def list_subscriptions(
    active_only: bool = True,
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """
    List user's podcast subscriptions

    Returns all subscriptions with podcast details.
    """
    query = (
        supabase.table("user_subscriptions")
        .select("*, podcasts(*)")
        .eq("user_id", current_user.id)
    )

    if active_only:
        query = query.eq("is_active", True)

    result = query.order("subscribed_at", desc=True).execute()

    return [
        SubscriptionWithPodcastResponse(
            **{**sub, "podcast": sub["podcasts"]}
        )
        for sub in result.data
    ]


@router.post("/subscribe", response_model=SubscriptionResponse, status_code=status.HTTP_201_CREATED)
async def subscribe_to_podcast(
    request: SubscribeRequest,
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """
    Subscribe to a podcast

    Creates a new subscription if it doesn't exist, or reactivates if previously unsubscribed.
    """
    # Check if podcast exists
    podcast_result = supabase.table("podcasts").select("*").eq("id", str(request.podcast_id)).execute()

    if not podcast_result.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Podcast not found",
        )

    # Check if subscription already exists
    existing_sub = (
        supabase.table("user_subscriptions")
        .select("*")
        .eq("user_id", current_user.id)
        .eq("podcast_id", str(request.podcast_id))
        .execute()
    )

    if existing_sub.data:
        # Reactivate if exists
        sub = existing_sub.data[0]
        if not sub["is_active"]:
            result = (
                supabase.table("user_subscriptions")
                .update({"is_active": True, "unsubscribed_at": None})
                .eq("id", sub["id"])
                .execute()
            )
            return SubscriptionResponse(**result.data[0])
        return SubscriptionResponse(**sub)

    # Create new subscription
    sub_data = {
        "user_id": current_user.id,
        "podcast_id": str(request.podcast_id),
        "is_active": True,
        "auto_added_from_spotify": False,
    }

    result = supabase.table("user_subscriptions").insert(sub_data).execute()

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create subscription",
        )

    return SubscriptionResponse(**result.data[0])


@router.post("/unsubscribe")
async def unsubscribe_from_podcast(
    request: UnsubscribeRequest,
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """
    Unsubscribe from a podcast

    Marks the subscription as inactive but keeps the record.
    """
    result = (
        supabase.table("user_subscriptions")
        .update({"is_active": False, "unsubscribed_at": datetime.now(timezone.utc).isoformat()})
        .eq("user_id", current_user.id)
        .eq("podcast_id", str(request.podcast_id))
        .eq("is_active", True)
        .execute()
    )

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Active subscription not found",
        )

    return {"message": "Unsubscribed successfully"}


@router.get("/stats")
async def get_subscription_stats(
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """
    Get subscription statistics for the user
    """
    # Count active subscriptions
    active_subs = (
        supabase.table("user_subscriptions")
        .select("*", count="exact")
        .eq("user_id", current_user.id)
        .eq("is_active", True)
        .execute()
    )

    # Count by category
    subs_with_podcasts = (
        supabase.table("user_subscriptions")
        .select("podcasts(category)")
        .eq("user_id", current_user.id)
        .eq("is_active", True)
        .execute()
    )

    categories = {}
    for sub in subs_with_podcasts.data:
        if sub.get("podcasts") and sub["podcasts"].get("category"):
            cat = sub["podcasts"]["category"]
            categories[cat] = categories.get(cat, 0) + 1

    return {
        "total_active_subscriptions": active_subs.count,
        "subscriptions_by_category": categories,
    }

