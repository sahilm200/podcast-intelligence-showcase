"""
Health check endpoints
"""

from fastapi import APIRouter, Depends
from supabase import Client

from p3.api.deps import get_supabase


router = APIRouter()


@router.get("/")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "service": "p3-api"}


@router.get("/ready")
async def readiness_check(supabase: Client = Depends(get_supabase)):
    """
    Readiness check - verifies database connection
    """
    try:
        # Test Supabase connection
        supabase.table("podcasts").select("count", count="exact").limit(1).execute()
        return {"status": "ready", "database": "connected"}
    except Exception as e:
        return {"status": "not_ready", "database": "disconnected", "error": str(e)}

