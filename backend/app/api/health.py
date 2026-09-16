"""
Health check router for Sentinel API.
"""
from fastapi import APIRouter
try:
    from app.core.config import settings
except ImportError:
    from backend.app.core.config import settings

router = APIRouter()


@router.get("/health", tags=["Health"])
def get_health():
    """
    Health check endpoint to verify backend service availability.
    """
    return {
        "status": "ok",
        "service": "Sentinel Backend",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "message": "Sentinel backend is running successfully.",
    }
