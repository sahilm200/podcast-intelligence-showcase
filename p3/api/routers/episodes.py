"""
Episode endpoints
Access episode detail, AI summaries, and transcripts
"""

from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from supabase import Client

from p3.api.deps import get_supabase, get_current_user, get_optional_user
from p3.api.models.episodes import (
    EpisodeDetailResponse,
    SummaryResponse,
    TranscriptSegmentResponse,
    MarkListenedRequest,
)


router = APIRouter()


@router.get("/{episode_id}", response_model=EpisodeDetailResponse)
async def get_episode(
    episode_id: UUID,
    supabase: Client = Depends(get_supabase),
    current_user: Optional[dict] = Depends(get_optional_user),
):
    """
    Get episode detail

    Returns episode metadata including podcast info.
    Authentication optional — public content.
    """
    result = (
        supabase.table("episodes")
        .select("*, podcasts(*)")
        .eq("id", str(episode_id))
        .execute()
    )

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Episode not found",
        )

    episode = result.data[0]
    podcast = episode.pop("podcasts", None)
    return EpisodeDetailResponse(**episode, podcast=podcast)


@router.get("/{episode_id}/summary", response_model=SummaryResponse)
async def get_episode_summary(
    episode_id: UUID,
    supabase: Client = Depends(get_supabase),
    current_user: Optional[dict] = Depends(get_optional_user),
):
    """
    Get AI-generated summary for an episode

    Returns structured summary with key topics, themes, quotes, insights,
    and actionable takeaways. Returns 404 if not yet generated.
    """
    # Verify episode exists and check status
    ep_result = (
        supabase.table("episodes")
        .select("id, status")
        .eq("id", str(episode_id))
        .execute()
    )

    if not ep_result.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Episode not found",
        )

    episode_status = ep_result.data[0]["status"]
    if episode_status != "completed":
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail=f"Summary not ready yet (episode status: {episode_status})",
        )

    summary_result = (
        supabase.table("summaries")
        .select("*")
        .eq("episode_id", str(episode_id))
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    if not summary_result.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Summary not available for this episode",
        )

    return SummaryResponse(**summary_result.data[0])


@router.get("/{episode_id}/transcript", response_model=List[TranscriptSegmentResponse])
async def get_episode_transcript(
    episode_id: UUID,
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
    supabase: Client = Depends(get_supabase),
    current_user: Optional[dict] = Depends(get_optional_user),
):
    """
    Get transcript segments for an episode

    Returns segments ordered by start timestamp. Paginate with limit/offset.
    Transcripts can have 500+ segments for a long episode.
    """
    ep_result = (
        supabase.table("episodes")
        .select("id, status")
        .eq("id", str(episode_id))
        .execute()
    )

    if not ep_result.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Episode not found",
        )

    episode_status = ep_result.data[0]["status"]
    if episode_status in ("pending", "downloaded"):
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail=f"Transcript not ready yet (episode status: {episode_status})",
        )

    result = (
        supabase.table("transcripts")
        .select("*")
        .eq("episode_id", str(episode_id))
        .order("timestamp_start", desc=False)
        .range(offset, offset + limit - 1)
        .execute()
    )

    return [TranscriptSegmentResponse(**seg) for seg in result.data]


@router.post("/{episode_id}/listened", status_code=status.HTTP_204_NO_CONTENT)
async def mark_episode_listened(
    episode_id: UUID,
    body: MarkListenedRequest = MarkListenedRequest(),
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """
    Mark an episode as listened (in-app tracking)

    Records listen progress so the digest generator can exclude already-heard
    episodes even for users who don't use Spotify.
    """
    ep_result = (
        supabase.table("episodes")
        .select("id")
        .eq("id", str(episode_id))
        .execute()
    )

    if not ep_result.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Episode not found",
        )

    listen_data = {
        "user_id": current_user.id,
        "episode_id": str(episode_id),
        "listened_at": datetime.now(timezone.utc).isoformat(),
        "progress_seconds": body.progress_seconds,
        "fully_played": body.fully_played,
        "source": "app",
    }

    supabase.table("user_episode_listens").upsert(
        listen_data,
        on_conflict="user_id,episode_id",
    ).execute()


@router.delete("/{episode_id}/listened", status_code=status.HTTP_204_NO_CONTENT)
async def unmark_episode_listened(
    episode_id: UUID,
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """
    Remove an in-app listen record for an episode

    Allows users to correct accidental marks or reset an episode for re-inclusion
    in future digests.
    """
    supabase.table("user_episode_listens").delete().eq(
        "user_id", current_user.id
    ).eq("episode_id", str(episode_id)).execute()

