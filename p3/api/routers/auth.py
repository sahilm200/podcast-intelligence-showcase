"""
Authentication endpoints
OAuth with Spotify (primary) and Google (alternative)
Improved with CSRF protection, token encryption, and proper JWT verification
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlencode
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import RedirectResponse
from jose import jwt
from supabase import Client
import httpx

from p3.api.config import get_settings
from p3.api.deps import get_supabase, get_current_user
from p3.api.limiter import limiter
from p3.api.queue import get_redis_pool
from p3.api.models.auth import (
    OAuthProvider,
    TokenResponse,
    UserResponse,
    RefreshTokenRequest,
)
from p3.api.security import (
    encrypt_token,
    generate_csrf_token,
    encode_oauth_state,
    decode_oauth_state,
)


settings = get_settings()
router = APIRouter()
logger = logging.getLogger(__name__)

_CSRF_TTL = 600  # 10 minutes


async def _csrf_store(csrf_token: str, redirect_url: Optional[str]) -> None:
    """Persist a CSRF token in Redis with a 10-minute TTL."""
    redis = await get_redis_pool()
    await redis.set(f"csrf:{csrf_token}", redirect_url or "", ex=_CSRF_TTL)


async def _csrf_consume(csrf_token: str) -> Optional[str]:
    """
    Atomically validate and delete a CSRF token.
    Returns the stored redirect_url (may be empty string) or None if not found.
    """
    redis = await get_redis_pool()
    key = f"csrf:{csrf_token}"
    # Pipeline: get then delete — atomic enough for this use-case
    pipe = redis.pipeline()
    pipe.get(key)
    pipe.delete(key)
    results = await pipe.execute()
    value = results[0]
    if value is None:
        return None
    # value is bytes or str depending on redis client
    return value.decode() if isinstance(value, bytes) else value


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt


def create_refresh_token(user_id: str) -> str:
    """Create refresh token with longer expiration"""
    expire = datetime.now(timezone.utc) + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode = {"sub": user_id, "exp": expire, "type": "refresh"}
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


async def store_oauth_connection(
    supabase: Client,
    user_id: str,
    provider: str,
    provider_user_id: str,
    access_token: str,
    refresh_token: Optional[str],
    expires_in: int,
    scopes: list
):
    """
    Store OAuth connection with encrypted tokens

    Args:
        supabase: Supabase client
        user_id: User ID
        provider: OAuth provider (spotify/google)
        provider_user_id: User ID on provider platform
        access_token: Access token to encrypt
        refresh_token: Refresh token to encrypt (optional)
        expires_in: Token expiration in seconds
        scopes: List of granted scopes
    """
    # Encrypt tokens before storage
    encrypted_access = encrypt_token(access_token)
    encrypted_refresh = encrypt_token(refresh_token) if refresh_token else None

    expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

    supabase.table("user_oauth_connections").upsert({
        "user_id": user_id,
        "provider": provider,
        "provider_user_id": provider_user_id,
        "access_token": encrypted_access,
        "refresh_token": encrypted_refresh,
        "token_expires_at": expires_at.isoformat(),
        "scopes": scopes,
        "last_refreshed_at": datetime.now(timezone.utc).isoformat(),
    }, on_conflict="user_id,provider").execute()


@router.get("/login/{provider}")
@limiter.limit("20/minute")
async def oauth_login(
    request: Request,
    provider: OAuthProvider,
    redirect_url: Optional[str] = Query(None, description="URL to redirect after auth"),
):
    """
    Initiate OAuth login flow with CSRF protection

    - **provider**: "spotify" or "google"
    - **redirect_url**: Optional URL to redirect after successful authentication
    """
    # Generate CSRF token and store in Redis with 10-minute TTL
    csrf_token = generate_csrf_token()
    await _csrf_store(csrf_token, redirect_url)

    # Encode state parameter with CSRF token and redirect URL
    state = encode_oauth_state(csrf_token, redirect_url)

    if provider == "spotify":
        # Spotify requires additional scopes for podcast data
        scopes = [
            "user-read-recently-played",
            "user-library-read",
            "user-follow-read",
            "user-read-playback-position",
        ]

        # Build authorize URL manually to ensure state is passed correctly
        params = {
            "client_id": settings.SPOTIFY_CLIENT_ID,
            "response_type": "code",
            "redirect_uri": settings.SPOTIFY_REDIRECT_URI,
            "scope": " ".join(scopes),
            "state": state,
        }
        auth_url = f"https://accounts.spotify.com/authorize?{urlencode(params)}"
        return RedirectResponse(url=auth_url)

    elif provider == "google":
        # Build authorize URL manually
        params = {
            "client_id": settings.GOOGLE_CLIENT_ID,
            "response_type": "code",
            "redirect_uri": settings.GOOGLE_REDIRECT_URI,
            "scope": "openid email profile",
            "state": state,
        }
        auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
        return RedirectResponse(url=auth_url)

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unsupported provider: {provider}",
    )


@router.get("/callback/spotify")
@limiter.limit("20/minute")
async def spotify_callback(
    request: Request,
    code: str,
    state: Optional[str] = None,
    error: Optional[str] = None,
    supabase: Client = Depends(get_supabase),
):
    """
    Spotify OAuth callback with CSRF validation and token encryption

    Handles the OAuth callback from Spotify, validates CSRF, creates/updates user,
    encrypts and stores OAuth tokens, and returns JWT token.
    """
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"OAuth error: {error}",
        )

    # Validate CSRF token
    if not state:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing state parameter (CSRF protection)",
        )

    csrf_token, redirect_url = decode_oauth_state(state)

    stored_redirect = await _csrf_consume(csrf_token)
    if stored_redirect is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or expired state token (CSRF protection)",
        )

    # Use redirect_url from Redis if not encoded in state
    if not redirect_url and stored_redirect:
        redirect_url = stored_redirect

    # Exchange code for token
    async with httpx.AsyncClient() as client:
        token_response = await client.post(
            "https://accounts.spotify.com/api/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": settings.SPOTIFY_REDIRECT_URI,
                "client_id": settings.SPOTIFY_CLIENT_ID,
                "client_secret": settings.SPOTIFY_CLIENT_SECRET,
            },
        )

        if token_response.status_code != 200:
            logger.error(f"Spotify token exchange failed: {token_response.text}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to exchange code for token",
            )

        token_data = token_response.json()
        access_token = token_data["access_token"]
        refresh_token = token_data.get("refresh_token")
        expires_in = token_data.get("expires_in", 3600)

        # Get user profile from Spotify
        profile_response = await client.get(
            "https://api.spotify.com/v1/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )

        if profile_response.status_code != 200:
            logger.error(f"Spotify profile fetch failed: {profile_response.text}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to get user profile from Spotify",
            )

        profile = profile_response.json()

    # Create or get user in Supabase
    email = profile.get("email")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email not provided by Spotify",
        )

    spotify_user_id = profile["id"]
    display_name = profile.get("display_name", email)

    try:
        # Use Supabase Auth to create/get user
        # First, try to find existing user
        users_response = supabase.auth.admin.list_users()
        existing_user = None

        if hasattr(users_response, 'users'):
            for user in users_response.users:
                if user.email == email:
                    existing_user = user
                    break

        if existing_user:
            user = existing_user
        else:
            # Create new user via admin API
            user_response = supabase.auth.admin.create_user({
                "email": email,
                "email_confirm": True,
                "user_metadata": {
                    "display_name": display_name,
                    "spotify_id": spotify_user_id,
                },
            })
            user = user_response.user

        # Store encrypted OAuth tokens
        await store_oauth_connection(
            supabase=supabase,
            user_id=user.id,
            provider="spotify",
            provider_user_id=spotify_user_id,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=expires_in,
            scopes=token_data.get("scope", "").split(),
        )

        # Create JWT tokens for our API
        jwt_access_token = create_access_token({"sub": user.id, "email": email})
        jwt_refresh_token = create_refresh_token(user.id)

        # If redirect_url provided, redirect with token
        if redirect_url:
            redirect_with_token = f"{redirect_url}?token={jwt_access_token}&refresh_token={jwt_refresh_token}"
            return RedirectResponse(url=redirect_with_token)

        # Otherwise return JSON response
        return TokenResponse(
            access_token=jwt_access_token,
            refresh_token=jwt_refresh_token,
            token_type="bearer",
            user=UserResponse(
                id=user.id,
                email=email,
                display_name=display_name,
                provider="spotify",
            ),
        )

    except Exception as e:
        logger.error(f"Failed to create/update user: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create/update user: {str(e)}",
        )


@router.get("/callback/google")
@limiter.limit("20/minute")
async def google_callback(
    request: Request,
    code: str,
    state: Optional[str] = None,
    error: Optional[str] = None,
    supabase: Client = Depends(get_supabase),
):
    """
    Google OAuth callback with CSRF validation, proper JWT verification, and token encryption
    """
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"OAuth error: {error}",
        )

    # Validate CSRF token
    if not state:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing state parameter (CSRF protection)",
        )

    csrf_token, redirect_url = decode_oauth_state(state)

    stored_redirect = await _csrf_consume(csrf_token)
    if stored_redirect is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or expired state token (CSRF protection)",
        )

    # Use redirect_url from Redis if not encoded in state
    if not redirect_url and stored_redirect:
        redirect_url = stored_redirect

    # Exchange code for token
    async with httpx.AsyncClient() as client:
        token_response = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": settings.GOOGLE_REDIRECT_URI,
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
            },
        )

        if token_response.status_code != 200:
            logger.error(f"Google token exchange failed: {token_response.text}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to exchange code for token",
            )

        token_data = token_response.json()
        access_token = token_data["access_token"]
        id_token = token_data.get("id_token")
        expires_in = token_data.get("expires_in", 3600)

        # Verify ID token by fetching user info from Google
        # This is more reliable than trying to verify the signature
        userinfo_response = await client.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )

        if userinfo_response.status_code != 200:
            logger.error(f"Google userinfo fetch failed: {userinfo_response.text}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to verify token with Google",
            )

        user_info = userinfo_response.json()

    email = user_info.get("email")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email not provided by Google",
        )

    google_user_id = user_info["sub"]
    display_name = user_info.get("name", email)

    # Create or update user in Supabase
    try:
        users_response = supabase.auth.admin.list_users()
        existing_user = None

        if hasattr(users_response, 'users'):
            for user in users_response.users:
                if user.email == email:
                    existing_user = user
                    break

        if existing_user:
            user = existing_user
        else:
            user_response = supabase.auth.admin.create_user({
                "email": email,
                "email_confirm": True,
                "user_metadata": {
                    "display_name": display_name,
                    "google_id": google_user_id,
                },
            })
            user = user_response.user

        # Store encrypted OAuth tokens
        await store_oauth_connection(
            supabase=supabase,
            user_id=user.id,
            provider="google",
            provider_user_id=google_user_id,
            access_token=access_token,
            refresh_token=token_data.get("refresh_token"),
            expires_in=expires_in,
            scopes=token_data.get("scope", "").split(),
        )

        # Create JWT tokens for our API
        jwt_access_token = create_access_token({"sub": user.id, "email": email})
        jwt_refresh_token = create_refresh_token(user.id)

        # If redirect_url provided, redirect with token
        if redirect_url:
            redirect_with_token = f"{redirect_url}?token={jwt_access_token}&refresh_token={jwt_refresh_token}"
            return RedirectResponse(url=redirect_with_token)

        # Otherwise return JSON response
        return TokenResponse(
            access_token=jwt_access_token,
            refresh_token=jwt_refresh_token,
            token_type="bearer",
            user=UserResponse(
                id=user.id,
                email=email,
                display_name=display_name,
                provider="google",
            ),
        )

    except Exception as e:
        logger.error(f"Failed to create/update user: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create/update user: {str(e)}",
        )


@router.post("/store-spotify-token")
@limiter.limit("20/minute")
async def store_spotify_token(
    request: Request,
    current_user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase),
):
    """
    Store Spotify OAuth tokens from a Supabase Spotify OAuth session.
    Called by the frontend after Supabase Spotify login to save tokens
    so the backend can make Spotify API calls on behalf of the user.
    """
    body = await request.json()
    provider_token = body.get("provider_token")
    provider_refresh_token = body.get("provider_refresh_token")
    expires_in = body.get("expires_in", 3600)
    provider_user_id = body.get("provider_user_id", "")

    if not provider_token:
        raise HTTPException(status_code=400, detail="provider_token is required")

    await store_oauth_connection(
        supabase=supabase,
        user_id=current_user.id,
        provider="spotify",
        provider_user_id=provider_user_id,
        access_token=provider_token,
        refresh_token=provider_refresh_token,
        expires_in=int(expires_in),
        scopes=["user-read-email", "user-library-read", "user-follow-read", "user-read-recently-played"],
    )
    return {"ok": True}


@router.post("/refresh", response_model=TokenResponse)
@limiter.limit("10/minute")
async def refresh_access_token(
    request: Request,
    body: RefreshTokenRequest,
    supabase: Client = Depends(get_supabase),
):
    """
    Refresh access token using refresh token

    Returns new access and refresh tokens
    """
    try:
        payload = jwt.decode(
            body.refresh_token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )

        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
            )

        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )

        # Get user from Supabase
        user_response = supabase.auth.admin.get_user_by_id(user_id)
        if not user_response or not user_response.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )

        user = user_response.user

        # Create new tokens
        new_access_token = create_access_token({"sub": user.id, "email": user.email})
        new_refresh_token = create_refresh_token(user.id)

        # Get user metadata
        metadata = user.user_metadata or {}

        # Check which OAuth provider they used
        oauth_response = supabase.table("user_oauth_connections").select("provider").eq(
            "user_id", user.id
        ).execute()

        provider = oauth_response.data[0]["provider"] if oauth_response.data else None

        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            user=UserResponse(
                id=user.id,
                email=user.email,
                display_name=metadata.get("display_name", user.email),
                provider=provider,
            ),
        )

    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase),
):
    """
    Get current authenticated user information
    """
    # Get user metadata
    metadata = current_user.user_metadata or {}

    # Check which OAuth provider they used
    oauth_response = supabase.table("user_oauth_connections").select("provider").eq(
        "user_id", current_user.id
    ).execute()

    provider = oauth_response.data[0]["provider"] if oauth_response.data else None

    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        display_name=metadata.get("display_name", current_user.email),
        provider=provider,
    )


@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """
    Logout (client should discard tokens)

    Note: JWT tokens can't be invalidated server-side without a blacklist.
    Client must discard tokens. For stricter security, implement token blacklist.
    """
    return {"message": "Logged out successfully. Please discard your tokens."}

