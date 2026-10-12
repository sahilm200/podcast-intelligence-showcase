"""
Authentication models
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, EmailStr


class OAuthProvider(str, Enum):
    """Supported OAuth providers"""
    SPOTIFY = "spotify"
    GOOGLE = "google"


class UserResponse(BaseModel):
    """User information response"""
    id: str
    email: EmailStr
    display_name: str
    provider: Optional[str] = None


class TokenResponse(BaseModel):
    """OAuth token response"""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    user: UserResponse


class RefreshTokenRequest(BaseModel):
    """Request to refresh access token"""
    refresh_token: str

