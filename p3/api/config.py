"""
Configuration management for P3 API
Uses pydantic-settings for environment variable management
"""

from functools import lru_cache
from typing import List
import secrets

from pydantic import Field, field_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        # Don't try to parse CORS_ORIGINS as JSON
        env_parse_enums=True,
    )

    # Environment
    ENVIRONMENT: str = Field(default="development", description="Environment: development, staging, production")
    DEBUG: bool = Field(default=False, description="Debug mode")

    # Server
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")

    # CORS - stored as string, converted to list
    cors_origins_str: str = Field(
        default="http://localhost:3000,http://localhost:8000",
        alias="CORS_ORIGINS",
        description="Allowed CORS origins (comma-separated)",
    )

    @computed_field
    @property
    def CORS_ORIGINS(self) -> List[str]:
        """Parse CORS origins from comma-separated string"""
        return [origin.strip() for origin in self.cors_origins_str.split(",")]

    @field_validator("JWT_SECRET_KEY")
    @classmethod
    def validate_jwt_secret(cls, v):
        if not v or v == "your-secret-key-change-this-in-production":
            raise ValueError(
                "JWT_SECRET_KEY must be set to a secure random value. "
                "Generate one with: python -c 'import secrets; print(secrets.token_hex(32))'"
            )
        if len(v) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters long")
        return v

    # Supabase
    SUPABASE_URL: str = Field(..., description="Supabase project URL")
    SUPABASE_KEY: str = Field(..., description="Supabase anon/service key")
    SUPABASE_JWT_SECRET: str = Field(..., description="Supabase JWT secret")

    # OAuth - Spotify
    SPOTIFY_CLIENT_ID: str = Field(default="", description="Spotify OAuth client ID")
    SPOTIFY_CLIENT_SECRET: str = Field(default="", description="Spotify OAuth client secret")
    SPOTIFY_REDIRECT_URI: str = Field(
        default="http://localhost:8000/auth/callback/spotify",
        description="Spotify OAuth redirect URI",
    )

    # OAuth - Google
    GOOGLE_CLIENT_ID: str = Field(default="", description="Google OAuth client ID")
    GOOGLE_CLIENT_SECRET: str = Field(default="", description="Google OAuth client secret")
    GOOGLE_REDIRECT_URI: str = Field(
        default="http://localhost:8000/auth/callback/google",
        description="Google OAuth redirect URI",
    )

    # JWT
    JWT_SECRET_KEY: str = Field(..., description="JWT secret key")
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="Access token expiration (minutes)")
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=30, description="Refresh token expiration (days)")

    # Redis (for background jobs)
    REDIS_URL: str = Field(default="redis://localhost:6379", description="Redis URL for job queue")

    # Processing
    ENABLE_BACKGROUND_PROCESSING: bool = Field(default=True, description="Enable background job processing")
    MAX_EPISODES_PER_FETCH: int = Field(default=50, description="Max episodes to fetch per podcast")
    MAX_CONCURRENT_DOWNLOADS: int = Field(default=5, description="Max concurrent episode downloads")
    DOWNLOAD_RETRY_ATTEMPTS: int = Field(default=3, description="Number of retry attempts for failed downloads")
    DOWNLOAD_BATCH_SIZE: int = Field(default=5, description="Number of downloads to queue at once")

    # Spotify API
    SPOTIFY_API_CACHE_TTL: int = Field(default=3600, description="Spotify API cache TTL (seconds)")

    # Storage
    AUDIO_RETENTION_DAYS: int = Field(default=7, description="Days to retain audio files")
    BASE_DATA_DIR: str = Field(default="./data", description="Base directory for data storage")
    BASE_OUTPUT_DIR: str = Field(default="./outputs", description="Base directory for outputs")

    # External APIs (from existing P3 config)
    GROQ_API_KEY: str = Field(default="", description="Groq API key")
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    ANTHROPIC_API_KEY: str = Field(default="", description="Anthropic API key")

    # Additional free/cheap LLM providers
    GEMINI_API_KEY: str = Field(default="", description="Google Gemini API key (free tier: 60 RPM)")
    CEREBRAS_API_KEY: str = Field(default="", description="Cerebras API key (free tier)")
    TOGETHER_API_KEY: str = Field(default="", description="Together.ai API key")
    FIREWORKS_API_KEY: str = Field(default="", description="Fireworks.ai API key")
    OPENROUTER_API_KEY: str = Field(default="", description="OpenRouter API key (access free models)")

    # Additional free transcription providers
    ASSEMBLYAI_API_KEY: str = Field(default="", description="AssemblyAI API key (5h free/month)")
    DEEPGRAM_API_KEY: str = Field(default="", description="Deepgram API key (12k free minutes)")

    # LLM Provider (default from existing config)
    LLM_PROVIDER: str = Field(default="groq", description="LLM provider: ollama, groq, openai, anthropic, gemini, cerebras, together, fireworks, openrouter")
    LLM_MODEL: str = Field(default="llama-3.1-8b-instant", description="LLM model name")

    # Transcription Provider
    TRANSCRIPTION_PROVIDER: str = Field(default="groq", description="Transcription provider: groq, openai, assemblyai, deepgram, local")
    TRANSCRIPTION_MODEL: str = Field(default="whisper-large-v3", description="Transcription model")

    # TTS Provider
    TTS_PROVIDER: str = Field(default="groq", description="TTS provider: groq, openai, local")
    TTS_MODEL: str = Field(default="canopylabs/orpheus-v1-english", description="TTS model")

    # Cloudflare R2 Storage (for clips)
    R2_ACCOUNT_ID: str = Field(default="", description="Cloudflare R2 account ID")
    R2_ACCESS_KEY_ID: str = Field(default="", description="R2 access key ID")
    R2_SECRET_ACCESS_KEY: str = Field(default="", description="R2 secret access key")
    R2_BUCKET_NAME: str = Field(default="parakeet-clips", description="R2 bucket name for clips")
    R2_PUBLIC_URL: str = Field(default="", description="R2 public URL (optional CDN)")

    # Email delivery — Resend (primary, free tier: 3,000 emails/month)
    RESEND_API_KEY: str = Field(default="", description="Resend API key (resend.com free tier)")
    EMAIL_FROM: str = Field(default="digest@example.com", description="Sender email address")
    EMAIL_FROM_NAME: str = Field(default="Parakeet Digest", description="Sender display name")

    # Email delivery — SMTP fallback (Gmail, SendGrid, Mailgun, etc.)
    SMTP_HOST: str = Field(default="", description="SMTP server hostname")
    SMTP_PORT: int = Field(default=587, description="SMTP port (587=STARTTLS, 465=SSL)")
    SMTP_USER: str = Field(default="", description="SMTP username/email")
    SMTP_PASSWORD: str = Field(default="", description="SMTP password or app password")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

