"""
User digest endpoints
View and generate personalized digests
"""

import logging
from datetime import date, time, datetime, timezone
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from supabase import Client

from p3.api.deps import get_supabase, get_current_user
from p3.api.models.digests import (
    DigestResponse,
    DigestDetailResponse,
    DigestPreferencesResponse,
    UpdatePreferencesRequest,
)

logger = logging.getLogger(__name__)


router = APIRouter()


@router.get("/", response_model=List[DigestResponse])
async def list_digests(
    limit: int = Query(30, le=100),
    offset: int = Query(0, ge=0),
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """
    List user's digests

    Returns all generated digests ordered by date.
    """
    result = (
        supabase.table("user_digests")
        .select("*")
        .eq("user_id", current_user.id)
        .order("digest_date", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )

    return [DigestResponse(**digest) for digest in result.data]


# Preferences routes must come BEFORE /{digest_id} so FastAPI doesn't try
# to parse "preferences" as a UUID and return 422.

@router.get("/preferences", response_model=DigestPreferencesResponse)
async def get_digest_preferences(
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """
    Get user's digest preferences
    """
    result = (
        supabase.table("user_digest_preferences")
        .select("*")
        .eq("user_id", current_user.id)
        .execute()
    )

    if not result.data:
        # Return default preferences
        return DigestPreferencesResponse(
            user_id=current_user.id,
            frequency="daily",
            delivery_time=time(8, 0),
            delivery_timezone="UTC",
            min_episode_duration_sec=300,
            max_episodes_per_digest=10,
            exclude_listened_on_spotify=True,
            spotify_sync_enabled=True,
            email_enabled=False,
        )

    return DigestPreferencesResponse(**result.data[0])


@router.put("/preferences", response_model=DigestPreferencesResponse)
async def update_digest_preferences(
    request: UpdatePreferencesRequest,
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """
    Update user's digest preferences
    """
    prefs_data = request.model_dump(exclude_unset=True)
    prefs_data["user_id"] = current_user.id

    result = supabase.table("user_digest_preferences").upsert(prefs_data).execute()

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update preferences",
        )

    return DigestPreferencesResponse(**result.data[0])


def _build_digest_markdown(episodes_with_summaries: list, target_date: date) -> str:
    """Build a markdown digest from a list of episodes with their summaries."""
    lines = [
        f"# Podcast Digest — {target_date.strftime('%B %d, %Y')}",
        "",
        f"*Generated from {len(episodes_with_summaries)} episode(s)*",
        "",
        "---",
        "",
    ]

    for ep_data in episodes_with_summaries:
        episode = ep_data["episode"]
        summary = ep_data["summary"]
        podcast_title = ep_data.get("podcast_title", "Unknown Podcast")

        lines.append(f"## {episode['title']}")
        lines.append(f"*{podcast_title}*")
        lines.append("")

        if summary.get("full_summary"):
            lines.append(summary["full_summary"])
            lines.append("")

        if summary.get("key_topics"):
            topics = summary["key_topics"]
            if isinstance(topics, list) and topics:
                lines.append("**Key Topics:** " + ", ".join(str(t) for t in topics[:5]))
                lines.append("")

        if summary.get("key_insights"):
            insights = summary["key_insights"]
            if isinstance(insights, list) and insights:
                lines.append("**Key Insights:**")
                for insight in insights[:3]:
                    lines.append(f"- {insight}")
                lines.append("")

        if summary.get("actionable_takeaways"):
            takeaways = summary["actionable_takeaways"]
            if isinstance(takeaways, list) and takeaways:
                lines.append("**Takeaways:**")
                for t in takeaways[:3]:
                    lines.append(f"- {t}")
                lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


@router.post("/generate")
async def generate_digest(
    digest_date: Optional[date] = Query(None, description="Date for digest (default: today)"),
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """
    Generate a digest for a specific date inline (no background worker needed).

    Looks for completed episodes from the user's subscriptions in the last 7 days,
    builds a markdown digest from available summaries, and returns immediately.
    """
    target_date = digest_date or date.today()

    # Check if digest already exists for this date
    existing = (
        supabase.table("user_digests")
        .select("id, status, content")
        .eq("user_id", current_user.id)
        .eq("digest_date", target_date.isoformat())
        .execute()
    )

    if existing.data:
        existing_digest = existing.data[0]
        # Regenerate if previously failed or pending with no content
        if existing_digest["status"] in ("completed",):
            return {
                "message": "Digest already exists",
                "digest_id": existing_digest["id"],
                "status": existing_digest["status"],
            }
        digest_id = existing_digest["id"]
        supabase.table("user_digests").update({"status": "generating"}).eq("id", digest_id).execute()
    else:
        result = supabase.table("user_digests").insert({
            "user_id": current_user.id,
            "digest_date": target_date.isoformat(),
            "title": f"Digest — {target_date.strftime('%B %d, %Y')}",
            "status": "generating",
        }).execute()

        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create digest record")
        digest_id = result.data[0]["id"]

    try:
        # Get user's active subscriptions
        subs = (
            supabase.table("user_subscriptions")
            .select("podcast_id")
            .eq("user_id", current_user.id)
            .eq("is_active", True)
            .execute()
        )
        podcast_ids = [s["podcast_id"] for s in (subs.data or [])]

        episodes_with_summaries = []

        if podcast_ids:
            # Find completed episodes from those podcasts in last 7 days
            from datetime import timedelta
            since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

            episodes_result = (
                supabase.table("episodes")
                .select("id, title, podcast_id, published_date")
                .in_("podcast_id", podcast_ids)
                .eq("status", "completed")
                .gte("published_date", since)
                .order("published_date", desc=True)
                .limit(20)
                .execute()
            )

            # Fetch podcasts for titles
            podcasts_result = (
                supabase.table("podcasts")
                .select("id, title")
                .in_("id", podcast_ids)
                .execute()
            )
            podcast_title_map = {p["id"]: p["title"] for p in (podcasts_result.data or [])}

            for ep in (episodes_result.data or []):
                summary_result = (
                    supabase.table("summaries")
                    .select("*")
                    .eq("episode_id", ep["id"])
                    .execute()
                )
                if summary_result.data:
                    episodes_with_summaries.append({
                        "episode": ep,
                        "summary": summary_result.data[0],
                        "podcast_title": podcast_title_map.get(ep["podcast_id"], "Unknown"),
                    })

        if episodes_with_summaries:
            content = _build_digest_markdown(episodes_with_summaries, target_date)
            episode_ids = [e["episode"]["id"] for e in episodes_with_summaries]
            supabase.table("user_digests").update({
                "status": "completed",
                "content": content,
                "title": f"Digest — {target_date.strftime('%B %d, %Y')} ({len(episodes_with_summaries)} episodes)",
                "episode_ids": episode_ids,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", digest_id).execute()
            final_status = "completed"
            message = f"Digest generated with {len(episodes_with_summaries)} episode(s)"
        else:
            # No processed episodes — fall back to a library overview from subscribed shows
            if podcast_ids:
                podcasts_result = (
                    supabase.table("podcasts")
                    .select("title, author, description")
                    .in_("id", podcast_ids)
                    .order("title")
                    .execute()
                )
                podcasts = podcasts_result.data or []
            else:
                podcasts = []

            lines = [
                f"# Your Podcast Library — {target_date.strftime('%B %d, %Y')}",
                "",
                f"You are subscribed to **{len(podcasts)} podcast{'s' if len(podcasts) != 1 else ''}**. "
                "Episode-level digests with AI summaries will appear here once episodes have been processed.",
                "",
                "---",
                "",
                "## Subscribed Shows",
                "",
            ]
            for p in podcasts:
                lines.append(f"### {p['title']}")
                if p.get("author"):
                    lines.append(f"*{p['author']}*")
                    lines.append("")
                if p.get("description"):
                    # Trim long descriptions
                    desc = p["description"]
                    if len(desc) > 300:
                        desc = desc[:297] + "…"
                    lines.append(desc)
                    lines.append("")
            content = "\n".join(lines)

            supabase.table("user_digests").update({
                "status": "completed",
                "content": content,
                "title": f"Your Library — {target_date.strftime('%B %d, %Y')}",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", digest_id).execute()
            final_status = "completed"
            message = "Library overview digest generated"

    except Exception as e:
        logger.error(f"Digest generation failed for user {current_user.id}: {e}", exc_info=True)
        supabase.table("user_digests").update({"status": "failed"}).eq("id", digest_id).execute()
        raise HTTPException(status_code=500, detail=f"Digest generation failed: {str(e)}")

    return {
        "message": message,
        "digest_id": digest_id,
        "status": final_status,
    }


@router.get("/{digest_id}", response_model=DigestDetailResponse)
async def get_digest(
    digest_id: UUID,
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """
    Get detailed digest information

    Returns the digest with all included episodes and summaries.
    """
    # Get digest
    digest_result = (
        supabase.table("user_digests")
        .select("*")
        .eq("id", str(digest_id))
        .eq("user_id", current_user.id)
        .execute()
    )

    if not digest_result.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Digest not found",
        )

    digest = digest_result.data[0]

    # Get episodes (if episode_ids is populated)
    episodes = []
    if digest.get("episode_ids"):
        episode_ids = digest["episode_ids"]
        if episode_ids:
            episodes_result = (
                supabase.table("episodes")
                .select("*, podcasts(*), summaries(*)")
                .in_("id", episode_ids)
                .execute()
            )
            episodes = episodes_result.data

    return DigestDetailResponse(**digest, episodes=episodes)


@router.post("/{digest_id}/send-email")
async def send_digest_email(
    digest_id: UUID,
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """
    Manually trigger email delivery for a completed digest

    Requires email_enabled=true and email_address set in digest preferences.
    Enqueues a background job — delivery is not instant.
    """
    result = (
        supabase.table("user_digests")
        .select("id, status")
        .eq("id", str(digest_id))
        .eq("user_id", current_user.id)
        .execute()
    )

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Digest not found",
        )

    digest = result.data[0]
    if digest["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail=f"Digest is not ready yet (status: {digest['status']})",
        )

    # Verify user has email configured
    prefs_result = (
        supabase.table("user_digest_preferences")
        .select("email_enabled, email_address")
        .eq("user_id", current_user.id)
        .execute()
    )
    prefs = prefs_result.data[0] if prefs_result.data else {}

    if not prefs.get("email_enabled"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email delivery is not enabled. Enable it in digest preferences.",
        )

    if not prefs.get("email_address"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No email address configured in digest preferences.",
        )

    job_id = await queue.send_digest_email(current_user.id, str(digest_id))

    return {
        "message": "Email delivery queued",
        "digest_id": str(digest_id),
        "to": prefs["email_address"],
        "job_id": job_id,
    }


@router.get("/{digest_id}/content")
async def get_digest_content(
    digest_id: UUID,
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """
    Get rendered markdown content for a digest

    Returns the pre-rendered markdown string directly.
    Useful for displaying digest content in a client application.
    """
    result = (
        supabase.table("user_digests")
        .select("id, status, content")
        .eq("id", str(digest_id))
        .eq("user_id", current_user.id)
        .execute()
    )

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Digest not found",
        )

    digest = result.data[0]

    if digest["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail=f"Digest is not ready yet (status: {digest['status']})",
        )

    if not digest.get("content"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Digest content not available",
        )

    return {"digest_id": str(digest_id), "content": digest["content"]}

