"""
Dependency injection for FastAPI
Provides reusable dependencies for authentication, database access, etc.
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from supabase import create_client, Client

from p3.api.config import get_settings


settings = get_settings()
security = HTTPBearer()


# Supabase client (singleton)
_supabase_client: Optional[Client] = None


def get_supabase() -> Client:
    """Get Supabase client (singleton)"""
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    return _supabase_client


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    supabase: Client = Depends(get_supabase),
) -> dict:
    """
    Get current authenticated user from JWT token

    Validates the JWT token and returns the user data.
    Raises HTTPException if token is invalid or user not found.
    """
    token = credentials.credentials

    # First try: validate via Supabase API (handles ES256 and HS256 Supabase JWTs)
    try:
        response = supabase.auth.get_user(token)
        if response.user:
            return response.user
    except Exception:
        pass

    # Fallback: validate app-issued HS256 JWT
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={"verify_aud": False},
        )
        user_id: str = payload.get("sub")
        if user_id:
            response = supabase.auth.admin.get_user_by_id(user_id)
            if response.user:
                return response.user
    except JWTError:
        pass

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication token",
    )


async def get_current_active_user(
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    Get current active user (can add additional checks here)
    """
    # Add any additional user validation here (e.g., check if user is banned)
    return current_user


async def get_current_admin_user(
    current_user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase),
) -> dict:
    """
    Get current user if they have admin privileges
    Raises 403 if user is not an admin
    """
    # Check if user is in admin_users table
    result = supabase.table("admin_users").select("*").eq("user_id", current_user.id).execute()

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )

    return current_user


def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
    supabase: Client = Depends(get_supabase),
) -> Optional[dict]:
    """
    Get current user if authenticated, otherwise None
    Useful for endpoints that work for both authenticated and anonymous users
    """
    if not credentials:
        return None

    try:
        token = credentials.credentials
        # Try Supabase API first (handles ES256)
        try:
            response = supabase.auth.get_user(token)
            if response.user:
                return response.user
        except Exception:
            pass
        # Fallback: app-issued HS256
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={"verify_aud": False},
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        response = supabase.auth.admin.get_user_by_id(user_id)
        return response.user if response.user else None
    except Exception:
        return None

