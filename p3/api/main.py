"""
Parakeet Podcast Processor (P3) - FastAPI Web Service
Multi-tenant podcast subscription service with smart deduplication
Fixed: Added rate limiting, structured logging, proper error handling
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import uvicorn

from p3.api.config import get_settings
from p3.api.limiter import limiter
from p3.api.storage import cleanup_temp_files
from p3.api.routers import (
    auth,
    catalog,
    subscriptions,
    digests,
    episodes,
    spotify,
    admin,
    health,
)
from p3.api import queue


settings = get_settings()

# Configure structured logging
# Note: `limiter` is imported from p3.api.limiter and shared with routers
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Lifecycle manager for startup and shutdown"""
    # Startup
    logger.info("🚀 P3 API starting...")
    logger.info(f"📊 Environment: {settings.ENVIRONMENT}")
    logger.info(f"🔐 Auth: Spotify + Google OAuth")
    logger.info(f"💾 Database: Supabase (PostgreSQL)")
    logger.info(f"🔴 Redis: {settings.REDIS_URL}")
    logger.info(f"🌐 CORS Origins: {', '.join(settings.CORS_ORIGINS)}")

    if settings.DEBUG:
        logger.warning("⚠️  DEBUG MODE ENABLED - Not for production!")

    # Clean up temp files left over from any previous crash (older than 2h)
    try:
        deleted = await cleanup_temp_files(max_age_hours=2)
        if deleted:
            logger.info(f"🧹 Cleaned up {deleted} stale temp audio files on startup")
    except Exception as e:
        logger.warning(f"Startup temp file cleanup failed (non-fatal): {e}")

    yield

    # Shutdown
    logger.info("👋 P3 API shutting down...")

    # Clean up recent temp files (anything older than 1h)
    try:
        deleted = await cleanup_temp_files(max_age_hours=1)
        if deleted:
            logger.info(f"🧹 Cleaned up {deleted} temp audio files on shutdown")
    except Exception as e:
        logger.warning(f"Shutdown temp file cleanup failed (non-fatal): {e}")

    # Clean up Redis connection pool
    try:
        await queue.close_redis_pool()
        logger.info("✅ Redis connection pool closed")
    except Exception as e:
        logger.error(f"❌ Failed to close Redis pool: {e}")


# Create FastAPI application
app = FastAPI(
    title="Parakeet Podcast Processor API",
    description="Multi-tenant podcast subscription service with smart deduplication",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests"""
    logger.info(f"{request.method} {request.url.path} - Client: {request.client.host}")

    try:
        response = await call_next(request)
        logger.info(f"{request.method} {request.url.path} - Status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"{request.method} {request.url.path} - Error: {str(e)}", exc_info=True)
        raise


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with logging"""
    logger.error(
        f"Unhandled exception: {type(exc).__name__}: {str(exc)}",
        exc_info=True,
        extra={
            "path": request.url.path,
            "method": request.method,
            "client": request.client.host if request.client else "unknown",
        }
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred",
        },
    )


# Include routers with rate limiting
# Health endpoint - no rate limiting
app.include_router(health.router, prefix="/health", tags=["Health"])

# Auth endpoints - moderate rate limiting (20/minute)
app.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"],
)

# Catalog endpoints - standard rate limiting
app.include_router(catalog.router, prefix="/catalog", tags=["Podcast Catalog"])

# Subscription endpoints - standard rate limiting
app.include_router(subscriptions.router, prefix="/subscriptions", tags=["Subscriptions"])

# Digest endpoints - standard rate limiting
app.include_router(digests.router, prefix="/digests", tags=["Digests"])

# Episode endpoints - standard rate limiting
app.include_router(episodes.router, prefix="/episodes", tags=["Episodes"])

# Spotify endpoints - more restrictive (10/minute) to respect Spotify API limits
app.include_router(spotify.router, prefix="/spotify", tags=["Spotify Integration"])

# Admin endpoints - restrictive (5/minute)
app.include_router(admin.router, prefix="/admin", tags=["Admin"])


@app.get("/")
@limiter.limit("100/minute")
async def root(request: Request):
    """Root endpoint with API information"""
    return {
        "service": "Parakeet Podcast Processor API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health",
        "environment": settings.ENVIRONMENT,
    }


def start_server():
    """Start the FastAPI server (for p3-api command)"""
    log_level = "debug" if settings.DEBUG else "info"

    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}")

    uvicorn.run(
        "p3.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=log_level,
        access_log=True,
    )


if __name__ == "__main__":
    start_server()

