"""API route definitions."""
from .health import router as health_router
from .videos import router as videos_router

__all__ = ["health_router", "videos_router"]
