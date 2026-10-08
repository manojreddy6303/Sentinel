"""
Health check router for Sentinel API.
"""
from fastapi import APIRouter
from typing import Any, Dict
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


@router.get("/health/system", tags=["Health"])
def get_system_diagnostics():
    """
    Production diagnostics endpoint to verify persistent volume mounts,
    resolved storage paths, and FFmpeg/ffprobe availability.
    """
    import sys
    import os
    import shutil
    import subprocess
    try:
        from database.session import DB_URL
    except ImportError:
        DB_URL = "unknown"

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        try:
            import imageio_ffmpeg
            cand = imageio_ffmpeg.get_ffmpeg_exe()
            if cand and os.path.exists(cand):
                ffmpeg_bin = cand
        except Exception:
            pass
    ffmpeg_ver = "unavailable"
    if ffmpeg_bin:
        try:
            res = subprocess.run([ffmpeg_bin, "-version"], capture_output=True, text=True, timeout=5)
            ffmpeg_ver = res.stdout.split("\n")[0] if res.returncode == 0 else f"error (code {res.returncode})"
        except Exception as e:
            ffmpeg_ver = str(e)

    ffprobe_bin = shutil.which("ffprobe")
    ffprobe_ver = "unavailable"
    if ffprobe_bin:
        try:
            res = subprocess.run([ffprobe_bin, "-version"], capture_output=True, text=True, timeout=5)
            ffprobe_ver = res.stdout.split("\n")[0] if res.returncode == 0 else f"error (code {res.returncode})"
        except Exception as e:
            ffprobe_ver = str(e)

    return {
        "status": "ok",
        "service": "Sentinel Backend",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "database_url": DB_URL,
        "resolved_sqlite_path": DB_URL.replace("sqlite:///", "") if DB_URL.startswith("sqlite:///") else DB_URL,
        "storage_base_dir": str(settings.STORAGE_BASE_DIR),
        "storage_uploads_dir": str(settings.STORAGE_UPLOADS_DIR),
        "storage_playback_dir": str(settings.STORAGE_PLAYBACK_DIR),
        "storage_evidence_dir": str(settings.STORAGE_EVIDENCE_DIR),
        "storage_reports_dir": str(settings.STORAGE_REPORTS_DIR),
        "ffmpeg": {
            "binary": ffmpeg_bin,
            "version": ffmpeg_ver,
        },
        "ffprobe": {
            "binary": ffprobe_bin,
            "version": ffprobe_ver,
        },
        "python_version": sys.version.split()[0],
    }


@router.get("/health/storage")
def storage_health() -> Dict[str, Any]:
    """
    Storage capacity and integrity health diagnostic endpoint.
    """
    try:
        from app.services.storage_service import storage_service
    except ImportError:
        from backend.app.services.storage_service import storage_service
    total, used, free = storage_service.get_disk_usage()
    integrity = storage_service.audit_storage_integrity()

    free_gb = free / (1024 ** 3)
    total_gb = total / (1024 ** 3)
    is_healthy = free_gb >= settings.SENTINEL_MIN_FREE_DISK_GB

    return {
        "status": "ok" if is_healthy else "warning",
        "disk_free_gb": round(free_gb, 2),
        "disk_total_gb": round(total_gb, 2),
        "min_reserve_gb": settings.SENTINEL_MIN_FREE_DISK_GB,
        "playback_cache_max_gb": settings.SENTINEL_PLAYBACK_CACHE_MAX_GB,
        "integrity": integrity,
    }

