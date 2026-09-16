"""
Sentinel Backend Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
try:
    from app.api.health import router as health_router
    from app.api.videos import router as videos_router
    from app.api.evidence import router as evidence_router
    from app.api.reports import router as reports_router
    from app.api.sessions import router as sessions_router
    from app.api.cases import router as cases_router
    from app.api.incidents import router as incidents_router
    from app.api.analytics import router as analytics_router
    from app.api.cameras import router as cameras_router
    from app.core.config import settings
except ImportError:
    from backend.app.api.health import router as health_router
    from backend.app.api.videos import router as videos_router
    from backend.app.api.evidence import router as evidence_router
    from backend.app.api.reports import router as reports_router
    from backend.app.api.sessions import router as sessions_router
    from backend.app.api.cases import router as cases_router
    from backend.app.api.incidents import router as incidents_router
    from backend.app.api.analytics import router as analytics_router
    from backend.app.api.cameras import router as cameras_router
    from backend.app.core.config import settings

import json
from typing import Any
from fastapi.responses import JSONResponse
from ai.common.numeric import sanitize_for_json


class SafeJSONResponse(JSONResponse):
    """Guarantees RFC 8259 compliance: no NaN or Infinity is ever emitted to clients."""
    def render(self, content: Any) -> bytes:
        clean_content = sanitize_for_json(content, non_finite_replacement=None)
        return json.dumps(
            clean_content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")


app = FastAPI(
    title="Sentinel Security Video Intelligence Platform API",
    description="Backend API services for video ingestion, timeline event querying, evidence reporting, and security operations.",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    default_response_class=SafeJSONResponse,
)

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routers
app.include_router(health_router, prefix="/api")
app.include_router(videos_router, prefix="/api")
app.include_router(evidence_router, prefix="/api")
app.include_router(reports_router, prefix="/api")
app.include_router(sessions_router, prefix="/api")
app.include_router(cases_router, prefix="/api")
app.include_router(incidents_router, prefix="/api")
app.include_router(analytics_router, prefix="/api")
app.include_router(cameras_router, prefix="/api")


@app.get("/")
def root():
    """Root entry point providing basic API identification."""
    return {
        "service": "Sentinel — AI-Powered Security Video Investigation Platform",
        "status": "online",
        "docs": "/docs",
        "health": "/api/health",
        "videos_upload": "/api/videos/upload",
    }


if __name__ == "__main__":
    import uvicorn

    try:
        import app.main
        app_target = "app.main:app"
    except ImportError:
        app_target = "backend.app.main:app"

    uvicorn.run(
        app_target,
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
