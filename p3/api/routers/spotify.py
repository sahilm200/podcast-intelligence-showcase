"""
Spotify integration endpoints
Sync user's Spotify listening history and followed shows
Fixed: Async/sync issues resolved, token decryption added
"""

import asyncio
import base64
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from supabase import Client
import httpx
import spotipy

from p3.api.config import get_settings
from p3.api.deps import get_supabase, get_current_user
from p3.api.limiter import limiter
from p3.api.models.spotify import (
    SpotifyShowResponse,
    SpotifyListeningHistoryResponse,
    SyncSpotifyRequest,
)
from p3.api.security import decrypt_token, encrypt_token


router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()


async def _refresh_spotify_token(supabase: Client, user_id: str, oauth_data: dict) -> Optional[str]:
    """
    Attempt to refresh the Spotify access token using the stored refresh token.
    Returns the new access token on success, or None on failure.
    Updates the database with the new token.
    """
    refresh_token_enc = oauth_data.get("refresh_token", "")
    if not refresh_token_enc:
        return None

    refresh_token = decrypt_token(refresh_token_enc)
    if not refresh_token or not settings.SPOTIFY_CLIENT_ID or not settings.SPOTIFY_CLIENT_SECRET:
        return None

    try:
        credentials = base64.b64encode(
            f"{settings.SPOTIFY_CLIENT_ID}:{settings.SPOTIFY_CLIENT_SECRET}".encode()
        ).decode()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://accounts.spotify.com/api/token",
                headers={
                    "Authorization": f"Basic {credentials}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={"grant_type": "refresh_token", "refresh_token": refresh_token},
            )

        if response.status_code != 200:
            logger.error(f"Spotify token refresh failed: {response.status_code}")
            return None

        token_data = response.json()
        new_access_token = token_data["access_token"]
        expires_in = token_data.get("expires_in", 3600)
        new_expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

        update_fields = {
            "access_token": encrypt_token(new_access_token),
            "token_expires_at": new_expires_at.isoformat(),
        }
        if "refresh_token" in token_data:
            update_fields["refresh_token"] = encrypt_token(token_data["refresh_token"])

        supabase.table("user_oauth_connections").update(update_fields).eq(
            "user_id", user_id
        ).eq("provider", "spotify").execute()

        logger.info(f"Refreshed Spotify token for user {user_id}")
        return new_access_token

    except Exception as e:
        logger.error(f"Error refreshing Spotify token: {e}")
        return None


async def get_spotify_client(
    current_user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase),
) -> spotipy.Spotify:
    """
    Get an authenticated Spotify client for the current user.
    Automatically refreshes the access token if it is expired or near expiry.
    """
    result = (
        supabase.table("user_oauth_connections")
        .select("*")
        .eq("user_id", current_user.id)
        .eq("provider", "spotify")
        .execute()
    )

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Spotify not connected. Please authenticate with Spotify first.",
        )

    oauth_data = result.data[0]

    # Check token expiry and refresh if needed
    access_token = None
    token_expires_at = oauth_data.get("token_expires_at")
    needs_refresh = False
    if token_expires_at:
        try:
            expires_dt = datetime.fromisoformat(str(token_expires_at).replace("Z", "+00:00"))
            if expires_dt.tzinfo is None:
                expires_dt = expires_dt.replace(tzinfo=timezone.utc)
            if expires_dt <= datetime.now(timezone.utc) + timedelta(minutes=5):
                needs_refresh = True
        except Exception:
            needs_refresh = True

    if needs_refresh:
        access_token = await _refresh_spotify_token(supabase, current_user.id, oauth_data)

    if not access_token:
        access_token = decrypt_token(oauth_data.get("access_token", ""))

    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to get Spotify token. Please re-authenticate.",
        )

    return spotipy.Spotify(auth=access_token)


@router.post("/sync")
@limiter.limit("10/minute")
async def sync_spotify_data(
    request: Request,
    body: Optional[SyncSpotifyRequest] = None,
    current_user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase),
    spotify_client: spotipy.Spotify = Depends(get_spotify_client),
):
    """
    Sync user's Spotify data

    Fixed: All spotipy calls wrapped in asyncio.to_thread for async execution

    Fetches and stores:
    - Followed podcast shows
    - Recently played podcast episodes
    - Listening history and progress
    """
    logger.info(f"Syncing Spotify data for user {current_user.id}")

    sync_stats = {
        "shows_synced": 0,
        "episodes_synced": 0,
        "new_subscriptions": 0,
    }

    try:
        # 1. Sync followed shows
        # Wrap synchronous spotipy call in thread
        followed_shows = await asyncio.to_thread(
            spotify_client.current_user_saved_shows,
            limit=50
        )

        for item in followed_shows["items"]:
            show = item["show"]
            spotify_show_id = show["id"]
            image_url = show["images"][0]["url"] if show.get("images") else None

            # Upsert into podcasts catalog (creates it if it doesn't exist yet)
            podcast_result = supabase.table("podcasts").upsert(
                {
                    "title": show["name"],
                    "description": show.get("description"),
                    "author": show.get("publisher"),
                    "image_url": image_url,
                    "spotify_show_id": spotify_show_id,
                    "rss_url": f"https://open.spotify.com/show/{spotify_show_id}",
                    "feed_type": "podcast",
                    "added_by_user_id": current_user.id,
                },
                on_conflict="spotify_show_id",
            ).execute()

            podcast_id = podcast_result.data[0]["id"]

            # Store in user_spotify_shows (upsert handles re-syncs)
            supabase.table("user_spotify_shows").upsert(
                {
                    "user_id": current_user.id,
                    "spotify_show_id": spotify_show_id,
                    "show_name": show["name"],
                    "show_publisher": show.get("publisher"),
                    "followed_at": item.get("added_at"),
                    "matched_podcast_id": podcast_id,
                },
                on_conflict="user_id,spotify_show_id",
            ).execute()
            sync_stats["shows_synced"] += 1

            # Auto-subscribe if not already subscribed
            existing_sub = (
                supabase.table("user_subscriptions")
                .select("id")
                .eq("user_id", current_user.id)
                .eq("podcast_id", podcast_id)
                .execute()
            )

            if not existing_sub.data:
                supabase.table("user_subscriptions").insert({
                    "user_id": current_user.id,
                    "podcast_id": podcast_id,
                    "is_active": True,
                    "auto_added_from_spotify": True,
                }).execute()
                sync_stats["new_subscriptions"] += 1
                logger.info(f"Auto-subscribed user {current_user.id} to podcast {podcast_id}")

        # 2. Sync recently played episodes (for filtering digests)
        # Wrap synchronous spotipy call in thread
        recently_played = await asyncio.to_thread(
            spotify_client.current_user_recently_played,
            limit=50
        )

        for item in recently_played["items"]:
            # Check if it's a podcast episode
            if item["track"]["type"] == "episode":
                episode = item["track"]
                spotify_episode_id = episode["id"]

                # Find episode in our database
                episode_match = (
                    supabase.table("episodes")
                    .select("*")
                    .eq("spotify_episode_id", spotify_episode_id)
                    .execute()
                )

                if episode_match.data:
                    episode_id = episode_match.data[0]["id"]

                    # Store listening history
                    tracking_data = {
                        "user_id": current_user.id,
                        "episode_id": episode_id,
                        "spotify_episode_id": spotify_episode_id,
                        "listened_at": item["played_at"],
                        "fully_played": True,  # Spotify API doesn't provide this
                    }

                    supabase.table("user_spotify_tracking").upsert(tracking_data).execute()
                    sync_stats["episodes_synced"] += 1

        logger.info(f"Spotify sync completed for user {current_user.id}: {sync_stats}")

        return {
            "message": "Spotify data synced successfully",
            "stats": sync_stats,
        }

    except Exception as e:
        logger.error(f"Failed to sync Spotify data for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to sync Spotify data: {str(e)}",
        )


@router.get("/shows", response_model=List[SpotifyShowResponse])
async def list_spotify_shows(
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """
    List user's followed Spotify shows

    Returns all shows the user follows on Spotify, with matching info.
    """
    result = (
        supabase.table("user_spotify_shows")
        .select("*, podcasts(*)")
        .eq("user_id", current_user.id)
        .order("followed_at", desc=True)
        .execute()
    )

    return [
        SpotifyShowResponse(
            **{
                **show,
                "matched_podcast": show.get("podcasts"),
            }
        )
        for show in result.data
    ]


@router.get("/listening-history", response_model=List[SpotifyListeningHistoryResponse])
async def get_listening_history(
    limit: int = 50,
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """
    Get user's Spotify listening history

    Returns episodes the user has listened to on Spotify.
    """
    result = (
        supabase.table("user_spotify_tracking")
        .select("*, episodes(*, podcasts(*))")
        .eq("user_id", current_user.id)
        .order("listened_at", desc=True)
        .limit(limit)
        .execute()
    )

    return [
        SpotifyListeningHistoryResponse(**tracking)
        for tracking in result.data
    ]

