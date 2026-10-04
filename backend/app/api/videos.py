"""
Video Management and Upload API Router
"""
import json
import logging
import math
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse
try:
    from app.core.config import settings
    from app.services.playback_service import (
        ensure_playback_file,
        get_playback_status,
        stream_video_file_with_ranges,
        PlaybackError,
        PlaybackNotFoundError,
        PlaybackConversionError,
        PlaybackMemoryPressureError,
    )
except ImportError:
    from backend.app.core.config import settings
    from backend.app.services.playback_service import (
        ensure_playback_file,
        get_playback_status,
        stream_video_file_with_ranges,
        PlaybackError,
        PlaybackNotFoundError,
        PlaybackConversionError,
        PlaybackMemoryPressureError,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase 3: AI module imports with graceful fallback
# ---------------------------------------------------------------------------
# Add project root to path so AI modules can be imported from both
# `python backend/app/...` and `uvicorn backend.app.main:app` invocations.
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from ai.video.processor import VideoProcessor, VideoNotFoundError, VideoCorruptedError, VideoEmptyError
    from ai.detection.detector import YOLODetector
    from ai.events.generator import EventGenerator
    from ai.events.repository import get_event_repository
    from ai.validation import DetectionValidator
    _AI_AVAILABLE = True
except ImportError as _ai_import_err:
    logger.warning(f"AI modules not available: {_ai_import_err}")
    _AI_AVAILABLE = False

# Module-level singleton detector — loaded once per process, not per request.
_detector: Optional["YOLODetector"] = None

# Active processing locks per video_id to enforce idempotency across concurrent requests
import threading
_processing_locks: Dict[str, threading.Lock] = {}
_processing_locks_guard = threading.Lock()

# Phase 20.3: Global bounded concurrency lock for heavy video processing per worker
_heavy_processing_semaphore = threading.Semaphore(settings.MAX_CONCURRENT_HEAVY_JOBS)

def _get_video_lock(video_id: str) -> threading.Lock:
    with _processing_locks_guard:
        if video_id not in _processing_locks:
            _processing_locks[video_id] = threading.Lock()
        return _processing_locks[video_id]


def recover_stale_processing_jobs() -> int:
    """
    Phase 20.3 Crash-Safe Video Processing State:
    Inspect database on server startup for videos left in 'processing' status
    (e.g., due to an ungraceful container termination or OOM kill) and safely
    transition them to 'interrupted' / 'failed', preserving source files and evidence.
    """
    recovered_count = 0
    try:
        from database.session import SessionLocal
        from database.models import VideoModel
        db = SessionLocal()
        try:
            stale_videos = db.query(VideoModel).filter(VideoModel.status == "processing").all()
            for v in stale_videos:
                v.status = "interrupted"
                recovered_count += 1
                try:
                    meta_path = settings.STORAGE_UPLOADS_DIR / f"{v.id}.json"
                    if meta_path.exists():
                        with open(meta_path, "r", encoding="utf-8") as f_meta:
                            meta = json.load(f_meta)
                        meta["status"] = "interrupted"
                        meta["error"] = "Processing was interrupted by a server restart. Video can be safely reprocessed."
                        meta["interrupted_at"] = datetime.now(timezone.utc).isoformat()
                        with open(meta_path, "w", encoding="utf-8") as f_meta:
                            json.dump(meta, f_meta, indent=2)
                except Exception as meta_exc:
                    logger.warning(f"Could not update sidecar metadata for interrupted video {v.id}: {meta_exc}")
            if recovered_count > 0:
                db.commit()
                logger.info(f"Phase 20.3 Crash Recovery: Safely marked {recovered_count} interrupted video(s) as 'interrupted'.")
        finally:
            db.close()
    except Exception as exc:
        logger.warning(f"Could not check/recover stale video processing jobs: {exc}")
    return recovered_count


def _get_detector() -> "YOLODetector":
    """Return (and lazily initialise) the singleton YOLODetector."""
    global _detector
    if _detector is None:
        _detector = YOLODetector(
            model_name=settings.YOLO_MODEL_NAME,
            confidence_threshold=settings.YOLO_CONFIDENCE_THRESHOLD,
            surveillance_classes=settings.SURVEILLANCE_CLASSES,
        )
    return _detector

router = APIRouter(prefix="/videos", tags=["Videos"])

# Magic byte signatures for video validation
KNOWN_VIDEO_SIGNATURES = [
    # MP4 / QuickTime (ftyp box at byte offset 4)
    (4, b"ftyp"),
    # QuickTime (moov, mdat, wide)
    (4, b"moov"),
    (4, b"mdat"),
    (4, b"wide"),
    # AVI / RIFF (RIFF at offset 0 and AVI  at offset 8)
    (0, b"RIFF"),
    # Matroska / WebM EBML
    (0, b"\x1a\x45\xdf\xa3"),
]

# Magic signatures of disallowed files that might masquerade as videos
DISALLOWED_SIGNATURES = [
    (0, b"MZ"),          # Windows PE Executable / DLL
    (0, b"\x7fELF"),     # Linux ELF Executable
    (0, b"PK\x03\x04"),  # ZIP archive (unless valid container)
    (0, b"<!DOCTYPE"),  # HTML
    (0, b"<html"),       # HTML
    (0, b"<?php"),      # PHP script
]


def sanitize_filename(filename: str) -> str:
    """Strip path components and restrict characters to safe ASCII set."""
    base = os.path.basename(filename)
    # Remove any directory traversal sequences and non-safe chars
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "_", base)
    # Avoid empty names or names starting with dot
    sanitized = sanitized.lstrip(".")
    return sanitized if sanitized else "video"


def validate_video_content(header_bytes: bytes, extension: str) -> bool:
    """
    Validate that binary content matches expected video signatures
    and does not match malicious executable/script headers.
    """
    # Reject known malicious/executable signatures
    for offset, sig in DISALLOWED_SIGNATURES:
        if len(header_bytes) >= offset + len(sig) and header_bytes[offset : offset + len(sig)] == sig:
            return False

    # Check for known video signatures
    for offset, sig in KNOWN_VIDEO_SIGNATURES:
        if len(header_bytes) >= offset + len(sig) and header_bytes[offset : offset + len(sig)] == sig:
            return True

    # If extension is valid video and header does not violate any rule, allow as fallback
    # (some video encoders use uncommon atom ordering)
    return extension in settings.ALLOWED_VIDEO_EXTENSIONS


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload a security surveillance video to Sentinel storage.
    
    Validates:
    - File presence
    - Non-empty payload
    - File extension (.mp4, .mov, .avi, etc.)
    - Binary video content integrity
    - Path safety (prevents path traversal)
    """
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No video file was provided in the request.",
        )

    # 1. Validate file extension
    original_filename = file.filename
    clean_base = sanitize_filename(original_filename)
    extension = Path(clean_base).suffix.lower()

    if not extension or extension not in settings.ALLOWED_VIDEO_EXTENSIONS:
        allowed = ", ".join(sorted(settings.ALLOWED_VIDEO_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{extension or 'unknown'}'. Supported formats are: {allowed}.",
        )

    # 2. Inspect initial chunk for non-empty and signature validation
    chunk_size = 1024 * 64  # 64 KB
    try:
        first_chunk = await file.read(chunk_size)
    except Exception as exc:
        logger.error(f"Error reading uploaded file chunk: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read uploaded file content.",
        )

    if not first_chunk or len(first_chunk) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty (0 bytes).",
        )

    if not validate_video_content(first_chunk, extension):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file '{original_filename}' is not a valid video file.",
        )

    # 3. Generate unique video ID and secure target path
    video_id = str(uuid.uuid4())
    saved_filename = f"{video_id}_{clean_base}"
    uploads_dir = settings.STORAGE_UPLOADS_DIR
    target_path = (uploads_dir / saved_filename).resolve()

    # Prevent path traversal: ensure target_path is strictly inside uploads_dir
    try:
        resolved_uploads = uploads_dir.resolve()
        if not str(target_path).startswith(str(resolved_uploads)):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Security violation: Invalid file storage path.",
            )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file path: {exc}",
        )

    # 4. Stream remainder of the file to disk in chunks
    total_bytes = len(first_chunk)
    try:
        with open(target_path, "wb") as f_out:
            f_out.write(first_chunk)
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > settings.MAX_UPLOAD_SIZE_BYTES:
                    # Exceeded max size limit
                    f_out.close()
                    if target_path.exists():
                        target_path.unlink()
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Video file exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB.",
                    )
                f_out.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to write video file '{target_path}': {exc}")
        if target_path.exists():
            try:
                target_path.unlink()
            except Exception:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store video file on server: {str(exc)}",
        )

    # 5. Persist video metadata sidecar (for fast lookup and streaming)
    metadata = {
        "video_id": video_id,
        "filename": original_filename,
        "saved_filename": saved_filename,
        "storage_path": str(target_path),
        "file_size_bytes": total_bytes,
        "content_type": file.content_type or f"video/{extension.lstrip('.')}",
        "status": "uploaded",
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = uploads_dir / f"{video_id}.json"
    try:
        with open(meta_path, "w", encoding="utf-8") as f_meta:
            json.dump(metadata, f_meta, indent=2)
    except Exception as exc:
        logger.warning(f"Could not write metadata sidecar for {video_id}: {exc}")

    # Synchronize VideoModel into database immediately
    try:
        from database.session import SessionLocal
        from database.models import VideoModel
        db = SessionLocal()
        try:
            vid_record = db.query(VideoModel).filter(VideoModel.id == video_id).first()
            if not vid_record:
                vid_record = VideoModel(
                    id=video_id,
                    original_filename=original_filename,
                    saved_filename=saved_filename,
                    storage_path=str(target_path),
                    file_size_bytes=total_bytes,
                    status="uploaded",
                    uploaded_at=datetime.now(timezone.utc),
                )
                db.add(vid_record)
            else:
                vid_record.original_filename = original_filename
                vid_record.saved_filename = saved_filename
                vid_record.storage_path = str(target_path)
                vid_record.file_size_bytes = total_bytes
            db.commit()
        except Exception as db_exc:
            db.rollback()
            logger.warning(f"Could not persist VideoModel on upload for {video_id}: {db_exc}")
        finally:
            db.close()
    except Exception as exc:
        logger.warning(f"Error opening DB session for video upload sync: {exc}")

    return {
        "video_id": video_id,
        "filename": original_filename,
        "status": "uploaded",
        "message": "Video uploaded successfully.",
        "file_size": total_bytes,
        "stream_url": f"/api/videos/{video_id}/stream",
        "playback_url": f"/api/videos/{video_id}/playback",
    }


def _validate_video_id(video_id: str) -> None:
    """Validate video_id to prevent directory traversal or injection attacks."""
    if not video_id or not re.match(r"^[a-zA-Z0-9_\-]+$", video_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid video ID format '{video_id}'. Only alphanumeric characters, hyphens, and underscores are allowed.",
        )


def _resolve_original_video(video_id: str) -> Path:
    """Safely resolve the original uploaded video path on disk."""
    _validate_video_id(video_id)

    meta_path = settings.STORAGE_UPLOADS_DIR / f"{video_id}.json"
    target_file: Optional[Path] = None

    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                target_file = Path(meta["storage_path"])
        except Exception:
            target_file = None

    if not target_file or not target_file.exists():
        matches = [m for m in settings.STORAGE_UPLOADS_DIR.glob(f"{video_id}_*") if m.suffix != ".json"]
        if matches:
            target_file = matches[0]

    if not target_file or not target_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with ID '{video_id}' not found.",
        )

    # Prevent path traversal
    resolved_uploads = settings.STORAGE_UPLOADS_DIR.resolve()
    try:
        target_file.resolve().relative_to(resolved_uploads)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: file path outside storage directory.",
        )

    return target_file


@router.get("", summary="List all security videos in Sentinel storage")
def list_videos(
    search: Optional[str] = Query(None, description="Search by original filename"),
    status: Optional[str] = Query(None, description="Filter by status (uploaded, processing, processed, failed)"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """
    List all uploaded and ingested surveillance videos with metadata,
    detection statistics, and playback availability.
    """
    from database.session import SessionLocal
    from database.models import VideoModel, EventModel, CorrelatedIncidentModel, EvidenceModel

    db = SessionLocal()
    try:
        query = db.query(VideoModel)
        if search:
            query = query.filter(VideoModel.original_filename.ilike(f"%{search.strip()}%"))
        if status:
            query = query.filter(VideoModel.status == status.strip())

        total = query.count()
        videos = query.order_by(VideoModel.uploaded_at.desc()).offset(offset).limit(limit).all()

        results = []
        for v in videos:
            validated_count = db.query(EventModel).filter(
                EventModel.video_id == v.id, EventModel.validation_status == "VALID"
            ).count()
            raw_count = db.query(EventModel).filter(EventModel.video_id == v.id).count()
            incidents_count = db.query(CorrelatedIncidentModel).filter(
                CorrelatedIncidentModel.video_id == v.id
            ).count()
            evidence_count = db.query(EvidenceModel).filter(
                EvidenceModel.video_id == v.id
            ).count()

            results.append({
                "id": v.id,
                "video_id": v.id,
                "filename": v.original_filename,
                "original_filename": v.original_filename,
                "file_size_bytes": v.file_size_bytes or 0,
                "duration_seconds": v.duration_seconds,
                "fps": v.fps,
                "frame_count": v.frame_count,
                "status": v.status,
                "camera_id": getattr(v, "camera_id", None),
                "uploaded_at": v.uploaded_at.isoformat() if v.uploaded_at else None,
                "processed_at": v.processed_at.isoformat() if v.processed_at else None,
                "detections_count": validated_count,
                "raw_detections_count": raw_count,
                "incidents_count": incidents_count,
                "evidence_count": evidence_count,
                "playback_url": f"/api/videos/{v.id}/playback",
                "stream_url": f"/api/videos/{v.id}/stream",
            })

        return {
            "total": total,
            "videos": results,
            "limit": limit,
            "offset": offset,
        }
    finally:
        db.close()


@router.get("/{video_id}")
def get_video_info(video_id: str) -> Dict[str, Any]:
    """Retrieve metadata information for an uploaded video."""
    _validate_video_id(video_id)
    meta_path = settings.STORAGE_UPLOADS_DIR / f"{video_id}.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["playback_url"] = f"/api/videos/{video_id}/playback"
                return data
        except Exception as exc:
            logger.error(f"Error reading metadata {meta_path}: {exc}")

    # Fallback: scan uploads dir for files matching video_id
    for item in settings.STORAGE_UPLOADS_DIR.glob(f"{video_id}_*"):
        if item.suffix == ".json":
            continue
        return {
            "video_id": video_id,
            "filename": item.name.replace(f"{video_id}_", ""),
            "file_size_bytes": item.stat().st_size,
            "status": "uploaded",
            "stream_url": f"/api/videos/{video_id}/stream",
            "playback_url": f"/api/videos/{video_id}/playback",
        }

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Video with ID '{video_id}' not found.",
    )


@router.get("/{video_id}/playback-status")
def get_video_playback_status_endpoint(video_id: str) -> Dict[str, Any]:
    """
    Check the browser playback status for a given video ID.
    Returns whether the video is ready, converting, or requires conversion.
    """
    _validate_video_id(video_id)
    try:
        original_file = _resolve_original_video(video_id)
    except HTTPException:
        original_file = None

    return get_playback_status(video_id, original_file)


@router.api_route("/{video_id}/playback", methods=["GET", "HEAD"])
def playback_video(video_id: str, request: Request):
    """
    Stream a browser-compatible video for HTML5 playback with RFC 7233 HTTP Range support.

    - If the original uploaded video is already browser compatible (H.264/AVC in MP4),
      it is streamed directly without re-encoding overhead.
    - If the video is encoded in an unsupported codec (e.g. HEVC/H.265, AVI), it is
      transcoded once to H.264 (yuv420p + AAC) and cached in storage/playback/.
    - Supports full range requests for fast, scrubbable seeking across the timeline.
    """
    _validate_video_id(video_id)
    original_path = _resolve_original_video(video_id)

    try:
        playback_path = ensure_playback_file(video_id, original_path)
    except PlaybackNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video file for '{video_id}' not found.",
        )
    except PlaybackMemoryPressureError as err:
        logger.warning(f"Playback transcode deferred for {video_id} due to memory bounds: {err}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "playback_transcode_deferred",
                "message": f"Playback conversion deferred to protect container memory: {str(err)}",
                "video_id": video_id,
                "analysis_preserved": True,
                "retry_after_seconds": 10,
            },
            headers={"Retry-After": "10"},
        )
    except PlaybackConversionError as err:
        logger.error(f"Playback conversion error for {video_id}: {err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to prepare browser-compatible playback stream: {str(err)}",
        )

    range_header = request.headers.get("range")
    return stream_video_file_with_ranges(playback_path, range_header)


@router.api_route("/{video_id}/stream", methods=["GET", "HEAD"])
def stream_video(video_id: str, request: Request):
    """
    Stream or download the original uploaded video file with HTTP Range support.
    """
    target_file = _resolve_original_video(video_id)
    range_header = request.headers.get("range")
    return stream_video_file_with_ranges(target_file, range_header)


def _correlate_security_evidence(video_id: str, security_events: List[Any]) -> None:
    """
    Automatically generate forensic evidence (snapshot + clip) for high-priority
    security events such as POTENTIAL_THEFT.
    """
    try:
        from backend.app.services.evidence_service import EvidenceService
        from database.session import SessionLocal
        from database.models import SecurityEventModel, EvidenceModel
        ev_svc = EvidenceService()
        for s_ev in security_events:
            ev_type = getattr(s_ev, "event_type", None) or (s_ev.get("event_type") if isinstance(s_ev, dict) else None)
            if ev_type == "POTENTIAL_THEFT":
                ev_id = getattr(s_ev, "event_id", None) or (s_ev.get("event_id") or s_ev.get("id") if isinstance(s_ev, dict) else None)
                ts = getattr(s_ev, "timestamp", None) or (s_ev.get("timestamp") if isinstance(s_ev, dict) else 0.0)
                desc = getattr(s_ev, "description", "") or (s_ev.get("description", "") if isinstance(s_ev, dict) else "")

                # Check if evidence candidate provides canonical non-zero timestamp
                meta = getattr(s_ev, "incident_metadata", None) or (s_ev.get("incident_metadata") if isinstance(s_ev, dict) else {}) or {}
                ev_cands = meta.get("evidence_candidates") or []
                if (ts is None or float(ts) <= 0.05) and ev_cands:
                    cand_ts = ev_cands[0].get("timestamp")
                    if cand_ts is not None and float(cand_ts) > 0.0:
                        ts = float(cand_ts)

                # Idempotency check: don't create duplicate evidence if one already exists
                db_dup = SessionLocal()
                existing_ev = None
                try:
                    existing_ev = db_dup.query(EvidenceModel).filter(
                        EvidenceModel.video_id == video_id,
                        EvidenceModel.timestamp_seconds == float(ts),
                    ).first()
                finally:
                    db_dup.close()
                if existing_ev:
                    continue

                try:
                    ev_res = ev_svc.create_evidence(
                        video_id=video_id,
                        timestamp=float(ts),
                        event_id=ev_id,
                        evidence_type="snapshot_and_clip",
                        pre_seconds=3.0,
                        post_seconds=4.0,
                        notes=f"Automated forensic evidence for {ev_type}: {desc}",
                    )
                    new_ev_id = ev_res.get("evidence_id") or ev_res.get("id")
                    if new_ev_id:
                        if hasattr(s_ev, "evidence_id"):
                            s_ev.evidence_id = new_ev_id
                        elif isinstance(s_ev, dict):
                            s_ev["evidence_id"] = new_ev_id
                        db_s = SessionLocal()
                        try:
                            db_rec = db_s.query(SecurityEventModel).filter(
                                (SecurityEventModel.id == ev_id) |
                                ((SecurityEventModel.video_id == video_id) & (SecurityEventModel.event_type == "POTENTIAL_THEFT"))
                            ).first()
                            if db_rec:
                                db_rec.evidence_id = new_ev_id
                                db_s.commit()
                        finally:
                            db_s.close()
                except Exception as inner_err:
                    logger.warning(f"Could not extract auto-evidence for {ev_id}: {inner_err}")
    except Exception as ev_err:
        logger.warning(f"Could not run security evidence correlation for {video_id}: {ev_err}")


def _mark_video_failed(video_id: str, error_detail: str) -> None:
    """Helper to update database and sidecar metadata when video processing fails."""
    try:
        from database.session import SessionLocal
        from database.models import VideoModel
        db = SessionLocal()
        try:
            v_rec = db.query(VideoModel).filter(VideoModel.id == video_id).first()
            if v_rec:
                v_rec.status = "failed"
                db.commit()
        finally:
            db.close()
    except Exception as exc:
        logger.warning(f"Could not update VideoModel status to failed for {video_id}: {exc}")

    try:
        meta_path = settings.STORAGE_UPLOADS_DIR / f"{video_id}.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["status"] = "failed"
            meta["error"] = error_detail
            meta["failed_at"] = datetime.now(timezone.utc).isoformat()
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
    except Exception as exc:
        logger.warning(f"Could not update sidecar metadata to failed for {video_id}: {exc}")


@router.post("/{video_id}/process")
def process_video(
    video_id: str,
    force: bool = Query(False, description="Force re-analysis even if already processed"),
) -> Dict[str, Any]:
    """
    Trigger the OpenCV + YOLO detection pipeline for an uploaded video.

    Steps:
    1. Validate video_id and locate uploaded video on disk.
    2. Open video with OpenCV, read metadata.
    3. Sample frames at the configured rate.
    4. Run YOLO object detection on each sampled frame.
    5. Generate structured timestamped detection events.
    6. Persist events via the event repository.
    7. Return a processing summary.

    Returns:
        Summary dict: video_id, status, duration_seconds, fps,
                      frames_processed, detections_count.
    """
    if not _AI_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI processing modules are not available. Ensure opencv-python and ultralytics are installed.",
        )

    # 1. Locate the video file
    meta_path = settings.STORAGE_UPLOADS_DIR / f"{video_id}.json"
    video_file_path: Optional[Path] = None

    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            video_file_path = Path(meta["storage_path"])
            original_filename = meta.get("filename", "video.mp4")
        except Exception as exc:
            logger.error(f"Error reading video metadata {meta_path}: {exc}")

    if not video_file_path or not video_file_path.exists():
        # Fallback: scan uploads directory
        matches = list(settings.STORAGE_UPLOADS_DIR.glob(f"{video_id}_*"))
        # Exclude .json sidecars
        matches = [m for m in matches if m.suffix != ".json"]
        if matches:
            video_file_path = matches[0]
            original_filename = video_file_path.name.replace(f"{video_id}_", "")
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video with ID '{video_id}' not found. Please upload the video first.",
            )

    # 1.5 Idempotency guard: prevent duplicate concurrent processing jobs for the same video
    v_lock = _get_video_lock(video_id)
    if not v_lock.acquire(blocking=False):
        return {
            "video_id": video_id,
            "status": "processing",
            "message": "Video is already being processed. Concurrent processing prevented.",
        }

    # Phase 20.3: Prevent concurrent heavy video processing across the worker
    heavy_lock_acquired = _heavy_processing_semaphore.acquire(blocking=False)
    if not heavy_lock_acquired:
        v_lock.release()
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "video_id": video_id,
                "status": "busy",
                "message": "Server is currently processing another video. Concurrent heavy processing is serialized to protect system memory.",
            },
        )

    try:
        from database.session import SessionLocal
        from database.models import VideoModel
        db_check = SessionLocal()
        try:
            vid_rec = db_check.query(VideoModel).filter(VideoModel.id == video_id).first()
            if vid_rec:
                if vid_rec.status == "processing":
                    return {
                        "video_id": video_id,
                        "status": "processing",
                        "message": "Video is already being processed. Concurrent processing prevented.",
                    }
                if not force and vid_rec.status in ("processed", "completed", "analyzed"):
                    # Video is already processed. Return existing results idempotently without re-running detection.
                    repo = get_event_repository()
                    existing_events = repo.get_events(video_id, validation_status="ALL")
                    grouped = repo.get_grouped_events(video_id)
                    validated_count = len([
                        e for e in existing_events
                        if (e.get("validation_status") if isinstance(e, dict) else getattr(e, "validation_status", "VALID")) == "VALID"
                    ])
                    logger.info(f"Video {video_id} already processed. Returning cached summary.")
                    return {
                        "video_id": video_id,
                        "status": "completed",
                        "duration_seconds": round(vid_rec.duration_seconds or 0.0, 2),
                        "fps": round(vid_rec.fps or 30.0, 2),
                        "frames_processed": vid_rec.frame_count or 0,
                        "detections_count": validated_count,
                        "raw_detections_count": len(existing_events),
                        "grouped_events_count": len(grouped),
                    }
                vid_rec.status = "processing"
                db_check.commit()
        except Exception as db_lock_err:
            logger.warning(f"Could not check/set processing state for {video_id}: {db_lock_err}")
        finally:
            db_check.close()

        # 2. Open video and extract metadata
        processor = VideoProcessor(
            video_path=str(video_file_path),
            sample_rate_fps=settings.VIDEO_SAMPLE_RATE_FPS,
        )

        try:
            metadata = processor.get_metadata()
        except VideoNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            )
        except (VideoCorruptedError, VideoEmptyError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            )
        except Exception as exc:
            logger.error(f"Unexpected error reading video {video_id}: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to read video metadata: {str(exc)}",
            )

        fps = metadata["fps"]
        duration_seconds = metadata["duration_seconds"]
        frame_w = metadata.get("width")
        frame_h = metadata.get("height")

        # Resolution-Aware Inference Policy (Phase 20)
        from ai.detection.inference_policy import InferenceResolutionPolicy
        effective_imgsz = InferenceResolutionPolicy.select_inference_size(frame_w, frame_h)

        # 3. Get detector (loaded once per process)
        try:
            detector = _get_detector()
        except RuntimeError as exc:
            logger.error(f"YOLO model failed to load: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load YOLO model: {str(exc)}",
            )

        # 4. Sample frames and run detection with mini-batching for maximum throughput
        frame_detections: List[Dict[str, Any]] = []
        from ai.video.frame_cache import BoundedFrameCache
        sampled_frames_cache = BoundedFrameCache(
            max_frames=1200,
            max_memory_mb=settings.FRAME_CACHE_MAX_MEMORY_MB,
            video_path=str(video_file_path),
        )
        frames_processed = 0

        # Phase 20.3: Memory-safe batch size (configurable, default 4)
        BATCH_SIZE = settings.YOLO_BATCH_SIZE
        batch_frames: List[Any] = []
        batch_meta: List[tuple] = []  # (frame_number, timestamp)

        def _flush_batch():
            nonlocal frames_processed
            if not batch_frames:
                return
            if hasattr(detector, "detect_batch"):
                results = detector.detect_batch(
                    batch_frames,
                    [m[1] for m in batch_meta],
                    [m[0] for m in batch_meta],
                    video_id=video_id,
                    imgsz=effective_imgsz,
                )
            else:
                results = [
                    detector.detect(f, m[1], frame_idx=m[0], video_id=video_id, imgsz=effective_imgsz)
                    for f, m in zip(batch_frames, batch_meta)
                ]
            for (f_num, ts), raw_dets in zip(batch_meta, results):
                frame_detections.append({
                    "frame_number": f_num,
                    "detections": raw_dets,
                })
                frames_processed += 1
            batch_frames.clear()
            batch_meta.clear()

        from ai.enhancement import SceneConditionAnalyzer, AdaptiveLowLightEnhancer
        enhancer = AdaptiveLowLightEnhancer()
        scene_report = None

        try:
            for frame_number, timestamp, frame_bgr in processor.sample_frames():
                # Analyze scene lighting condition on initial frames
                if scene_report is None:
                    scene_report = SceneConditionAnalyzer.analyze(frame_bgr)

                # Store original, authoritative frame into memory-safe bounded cache
                sampled_frames_cache[timestamp] = frame_bgr

                # Conditionally enhance analytical proxy for YOLO inference if scene is low-light
                inference_frame = frame_bgr
                if scene_report and scene_report.needs_enhancement:
                    enhanced_frame, was_enhanced, _ = enhancer.enhance_if_needed(frame_bgr, scene_report)
                    if was_enhanced:
                        inference_frame = enhanced_frame

                batch_frames.append(inference_frame)
                batch_meta.append((frame_number, timestamp))

                if len(batch_frames) >= BATCH_SIZE:
                    _flush_batch()

            # Flush any trailing frames in final batch
            _flush_batch()
        except (VideoCorruptedError, VideoEmptyError) as exc:
            _mark_video_failed(video_id, str(exc))
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            )
        except RuntimeError as exc:
            # YOLO inference error — don't crash the server
            logger.error(f"Detection error for video {video_id}: {exc}")
            _mark_video_failed(video_id, str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Detection failed: {str(exc)}",
            )
        except Exception as exc:
            logger.error(f"Unexpected processing error for video {video_id}: {exc}")
            _mark_video_failed(video_id, str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Video processing error: {str(exc)}",
            )

        # 4.5 Multi-signal Detection Validation across frame sequence
        validator = DetectionValidator()
        frame_w = metadata.get("width")
        frame_h = metadata.get("height")
        validator.validate_sequence(
            frame_detections=frame_detections,
            frame_width=float(frame_w) if frame_w else None,
            frame_height=float(frame_h) if frame_h else None,
        )
    
        # 5. Generate structured raw events
        generator = EventGenerator()
        events = generator.generate_events(
            video_id=video_id,
            video_filename=original_filename,
            fps=fps,
            video_duration=duration_seconds,
            frame_detections=frame_detections,
        )
    
        # 6. Run Security Intelligence & Tracking on validated detections
        intel_tracks_count = 0
        intel_events_count = 0
        try:
            from ai.intelligence_pipeline import SecurityIntelligencePipeline
            from ai.intelligence_repository import SecurityIntelligenceRepository
            from ai.zones.zone_manager import ZoneManager
            from ai.schemas import ZoneDefinition
    
            intel_repo = SecurityIntelligenceRepository()
            existing_zones = intel_repo.get_zones(video_id)
            zone_defs = [
                ZoneDefinition(
                    zone_id=z["zone_id"],
                    name=z["name"],
                    polygon=[(pt[0], pt[1]) for pt in z["polygon"]],
                    target_classes=z.get("target_classes") or ["person", "car"],
                    alert_on_entry=z.get("alert_on_entry", True),
                    loitering_threshold_seconds=z.get("loitering_threshold_seconds", 30.0),
                )
                for z in existing_zones
            ]
            intel_pipe = SecurityIntelligencePipeline(zones=zone_defs)
            # Pass all raw events and in-memory sampled frames to eliminate redundant disk re-reads.
            # Pipeline internally filters to validated events before feeding tracker.
            intel_res = intel_pipe.process_video_intelligence(
                video_id=video_id,
                video_path=str(video_file_path),
                raw_events=events,
                fps=fps,
                duration_seconds=duration_seconds,
                sample_rate_fps=settings.VIDEO_SAMPLE_RATE_FPS,
                sampled_frames=sampled_frames_cache if sampled_frames_cache else None,
                frame_width=float(frame_w) if frame_w else None,
                frame_height=float(frame_h) if frame_h else None,
            )
            intel_repo.save_intelligence_results(
                video_id=video_id,
                tracks=intel_res["tracks"],
                vehicle_attributes=intel_res["vehicle_attributes"],
                face_detections=intel_res["face_detections"],
                security_events=intel_res["security_events"],
                specialized_observations=intel_res.get("specialized_observations", []),
                correlated_incidents=intel_res.get("correlated_incidents", []),
            )
            _correlate_security_evidence(video_id, intel_res["security_events"])
            try:
                from backend.app.services.evidence_service import EvidenceService
                EvidenceService().reconcile_evidence_validation(video_id)
            except Exception as rec_err:
                logger.warning(f"Could not reconcile evidence validation for {video_id}: {rec_err}")
            intel_tracks_count = len(intel_res["tracks"])
            intel_events_count = len(intel_res["security_events"])
            logger.info(
                f"Security intelligence generated for {video_id}: "
                f"{intel_tracks_count} tracks, {intel_events_count} security events."
            )
        except Exception as intel_err:
            logger.warning(f"Security intelligence deferred or failed for {video_id}: {intel_err}")
    
        # 7. Group events temporally with full track awareness
        grouped_events = generator.group_events(video_id, events)
    
        # 8. Persist events (both raw and grouped)
        try:
            repo = get_event_repository()
            repo.save_events(video_id, events, grouped_events=grouped_events)
        except Exception as exc:
            logger.error(f"Failed to save events for {video_id}: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to persist detection events: {str(exc)}",
            )
    
        # 7. Update video metadata status to 'processed'
        validated_count = len([
            e for e in events
            if (e.get("validation_status") if isinstance(e, dict) else getattr(e, "validation_status", "VALID")) == "VALID"
        ])
        try:
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["status"] = "processed"
                meta["processed_at"] = datetime.now(timezone.utc).isoformat()
                meta["duration_seconds"] = duration_seconds
                meta["fps"] = fps
                meta["frames_processed"] = frames_processed
                meta["detections_count"] = validated_count
                meta["raw_detections_count"] = len(events)
                meta["grouped_events_count"] = len(grouped_events)
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
        except Exception as exc:
            logger.warning(f"Could not update metadata sidecar for {video_id}: {exc}")
    
        # Synchronize VideoModel record in DB for database parity
        try:
            from database.session import SessionLocal
            from database.models import VideoModel
            db = SessionLocal()
            try:
                vid_record = db.query(VideoModel).filter(VideoModel.id == video_id).first()
                file_size = video_file_path.stat().st_size if video_file_path and video_file_path.exists() else 0
                if not vid_record:
                    vid_record = VideoModel(
                        id=video_id,
                        original_filename=original_filename,
                        saved_filename=video_file_path.name if video_file_path else None,
                        storage_path=str(video_file_path) if video_file_path else "",
                        file_size_bytes=file_size,
                        duration_seconds=duration_seconds,
                        fps=fps,
                        frame_count=frames_processed,
                        status="processed",
                        processed_at=datetime.now(timezone.utc),
                    )
                    db.add(vid_record)
                else:
                    if original_filename and (not vid_record.original_filename or vid_record.original_filename == f"{video_id}.mp4"):
                        vid_record.original_filename = original_filename
                    if video_file_path:
                        vid_record.saved_filename = video_file_path.name
                        vid_record.storage_path = str(video_file_path)
                    if file_size:
                        vid_record.file_size_bytes = file_size
                    vid_record.duration_seconds = duration_seconds
                    vid_record.fps = fps
                    vid_record.frame_count = frames_processed
                    vid_record.status = "processed"
                    vid_record.processed_at = datetime.now(timezone.utc)
                db.commit()
            except Exception as db_exc:
                db.rollback()
                logger.warning(f"Could not sync VideoModel for {video_id}: {db_exc}")
            finally:
                db.close()
        except Exception as exc:
            logger.warning(f"Error initializing DB session for VideoModel sync: {exc}")
    
        logger.info(
            f"Processed video {video_id}: {frames_processed} frames, {validated_count} validated detections ({len(events)} raw), {len(grouped_events)} events grouped."
        )
    
        return {
            "video_id": video_id,
            "status": "completed",
            "duration_seconds": round(duration_seconds, 2),
            "fps": round(fps, 2),
            "frames_processed": frames_processed,
            "detections_count": validated_count,
            "raw_detections_count": len(events),
            "grouped_events_count": len(grouped_events),
        }
    finally:
        try:
            v_lock.release()
        except Exception:
            pass
        if heavy_lock_acquired:
            try:
                _heavy_processing_semaphore.release()
            except Exception:
                pass

        # Phase 20.3: Explicit Scoped Memory Cleanup
        try:
            if 'batch_frames' in locals() and batch_frames is not None:
                batch_frames.clear()
                del batch_frames
            if 'batch_meta' in locals() and batch_meta is not None:
                batch_meta.clear()
                del batch_meta
            if 'sampled_frames_cache' in locals() and sampled_frames_cache is not None:
                if hasattr(sampled_frames_cache, 'clear'):
                    sampled_frames_cache.clear()
                del sampled_frames_cache
            if 'frame_detections' in locals() and frame_detections is not None:
                frame_detections.clear()
                del frame_detections
            if 'processor' in locals() and processor is not None:
                del processor

            import gc
            gc.collect()

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception as _clean_err:
            logger.debug("Post-processing memory cleanup notice: %s", _clean_err)


@router.get("/{video_id}/events")
def get_video_events(
    video_id: str,
    object_class: Optional[str] = Query(None, description="Filter by object class (e.g. 'person', 'car')"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    include_unvalidated: bool = Query(False, description="Include unvalidated/rejected detections for auditing"),
) -> Dict[str, Any]:
    """
    Retrieve detection events for a specific video.
    By default returns only VALID detections. Set include_unvalidated=true to view raw detections.

    Query parameters:
        object_class: Filter results to a specific object class.
        min_confidence: Only return detections above this confidence level.
        include_unvalidated: If True, returns all detections including REJECTED/UNCERTAIN.

    Returns:
        JSON with video_id, total_events count, and list of event dicts.
    """
    if not _AI_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI modules unavailable.",
        )

    # Verify video exists
    meta_path = settings.STORAGE_UPLOADS_DIR / f"{video_id}.json"
    video_exists = meta_path.exists() or bool(
        list(settings.STORAGE_UPLOADS_DIR.glob(f"{video_id}_*"))
    )
    if not video_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with ID '{video_id}' not found.",
        )

    try:
        repo = get_event_repository()
        val_status = "ALL" if include_unvalidated else "VALID"
        events = repo.get_events(
            video_id=video_id,
            object_class=object_class,
            min_confidence=min_confidence,
            validation_status=val_status,
        )
        from database.session import SessionLocal
        from database.models import EventModel
        db = SessionLocal()
        try:
            raw_count = db.query(EventModel).filter(EventModel.video_id == video_id).count()
            val_count = db.query(EventModel).filter(EventModel.video_id == video_id, EventModel.validation_status == "VALID").count()
            rej_count = db.query(EventModel).filter(EventModel.video_id == video_id, EventModel.validation_status == "REJECTED").count()
        finally:
            db.close()
    except Exception as exc:
        logger.error(f"Failed to retrieve events for {video_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve events: {str(exc)}",
        )

    return {
        "video_id": video_id,
        "total_events": len(events),
        "raw_events_count": raw_count,
        "validated_events_count": val_count,
        "rejected_events_count": rej_count,
        "filters": {
            "object_class": object_class,
            "min_confidence": min_confidence,
            "include_unvalidated": include_unvalidated,
        },
        "events": events,
    }


@router.get("/{video_id}/timeline")
def get_video_timeline(
    video_id: str,
    object_class: Optional[str] = Query(None, description="Filter by object class (e.g. 'person', 'car')"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    start_time: Optional[float] = Query(None, ge=0.0, description="Start timestamp in seconds"),
    end_time: Optional[float] = Query(None, ge=0.0, description="End timestamp in seconds"),
) -> Dict[str, Any]:
    """
    Retrieve structured, grouped investigation events for the video timeline.

    Returns:
        JSON with video_id, total_events, and grouped timeline events.
    """
    if not _AI_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI modules unavailable.",
        )

    meta_path = settings.STORAGE_UPLOADS_DIR / f"{video_id}.json"
    video_exists = meta_path.exists() or bool(
        list(settings.STORAGE_UPLOADS_DIR.glob(f"{video_id}_*"))
    )
    if not video_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with ID '{video_id}' not found.",
        )

    try:
        repo = get_event_repository()
        grouped = repo.get_grouped_events(
            video_id=video_id,
            object_class=object_class,
            min_confidence=min_confidence,
            start_time=start_time,
            end_time=end_time,
        )
    except Exception as exc:
        logger.error(f"Failed to retrieve timeline for {video_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve timeline events: {str(exc)}",
        )

    return {
        "video_id": video_id,
        "total_events": len(grouped),
        "filters": {
            "object_class": object_class,
            "min_confidence": min_confidence,
            "start_time": start_time,
            "end_time": end_time,
        },
        "events": grouped,
    }


class InvestigationRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Natural-language inquiry about video detections")


@router.post("/{video_id}/investigate")
def investigate_video(
    video_id: str,
    payload: InvestigationRequest,
) -> Dict[str, Any]:
    """
    Execute natural-language investigation query against the video's indexed detection database.

    Ground truth is sourced exclusively from actual database detection and event records.
    """
    # 1. Validate video_id format to prevent directory traversal
    if not re.match(r"^[a-zA-Z0-9_\-]+$", video_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid video ID format.",
        )

    # 2. Validate video existence
    meta_path = settings.STORAGE_UPLOADS_DIR / f"{video_id}.json"
    video_exists = meta_path.exists() or bool(
        list(settings.STORAGE_UPLOADS_DIR.glob(f"{video_id}_*"))
    )
    if not video_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with ID '{video_id}' not found.",
        )

    # 2. Execute query via InvestigationService
    try:
        from backend.app.services.investigation_service import InvestigationService
        service = InvestigationService()
        result = service.investigate(video_id=video_id, query_text=payload.query)
        result["video_id"] = video_id
        return result
    except Exception as exc:
        logger.error(f"Failed to execute investigation for {video_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Investigation service error: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Phase 7: LLM-Assisted Evidence-Grounded Investigation Endpoint
# ---------------------------------------------------------------------------

class AIInvestigationRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Natural-language inquiry about video detections")
    history: Optional[List[Dict[str, str]]] = Field(default=None, description="Optional conversational history context")


@router.post("/{video_id}/ai-investigate")
def ai_investigate_video(
    video_id: str,
    payload: AIInvestigationRequest,
) -> Dict[str, Any]:
    """
    Execute LLM-assisted, evidence-grounded investigation against the video's indexed detections.

    Ground truth is sourced exclusively from actual database detection and event records.
    The LLM never directly queries the database or invents facts.
    """
    # 1. Validate video_id format to prevent directory traversal
    if not re.match(r"^[a-zA-Z0-9_\-]+$", video_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid video ID format.",
        )

    # 2. Validate video existence
    meta_path = settings.STORAGE_UPLOADS_DIR / f"{video_id}.json"
    video_exists = meta_path.exists() or bool(
        list(settings.STORAGE_UPLOADS_DIR.glob(f"{video_id}_*"))
    )
    if not video_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with ID '{video_id}' not found.",
        )

    # 3. Execute query via InvestigationOrchestrator
    try:
        from ai.investigation.orchestrator import InvestigationOrchestrator
        orchestrator = InvestigationOrchestrator()
        result = orchestrator.process_investigation(
            video_id=video_id,
            user_query=payload.query,
            history=payload.history,
        )
        return result
    except Exception as exc:
        logger.error(f"Failed to execute AI investigation for {video_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI investigation service error: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Phase 6: Evidence Extraction & Management Endpoints
# ---------------------------------------------------------------------------

class CreateEvidenceRequest(BaseModel):
    timestamp: float = Field(..., ge=0.0, description="Timestamp in seconds for the evidence capture")
    event_id: Optional[str] = Field(None, max_length=64, description="Optional detection event ID or grouped event ID")
    evidence_type: str = Field("snapshot_and_clip", description="snapshot_and_clip, snapshot_only, or clip_only")
    pre_seconds: float = Field(3.0, ge=0.5, le=30.0, description="Pre-roll duration in seconds")
    post_seconds: float = Field(3.0, ge=0.5, le=30.0, description="Post-roll duration in seconds")
    notes: Optional[str] = Field(None, max_length=500, description="Optional investigator notes")


@router.post("/{video_id}/evidence")
def create_video_evidence(
    video_id: str,
    payload: CreateEvidenceRequest,
) -> Dict[str, Any]:
    """
    Capture and extract evidence (snapshot image, annotated image, and/or short video clip)
    for an individual detection or grouped event.
    """
    # 1. Validate video_id format to prevent directory traversal
    if not re.match(r"^[a-zA-Z0-9_\-]+$", video_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid video ID format.",
        )

    # 2. Validate video existence
    meta_path = settings.STORAGE_UPLOADS_DIR / f"{video_id}.json"
    video_exists = meta_path.exists() or bool(
        list(settings.STORAGE_UPLOADS_DIR.glob(f"{video_id}_*"))
    )
    if not video_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with ID '{video_id}' not found.",
        )

    # 3. Delegate to EvidenceService
    try:
        from backend.app.services.evidence_service import EvidenceService
        service = EvidenceService()
        result = service.create_evidence(
            video_id=video_id,
            timestamp=payload.timestamp,
            event_id=payload.event_id,
            evidence_type=payload.evidence_type,
            pre_seconds=payload.pre_seconds,
            post_seconds=payload.post_seconds,
            notes=payload.notes,
        )
        return result
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except FileNotFoundError as fnf_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(fnf_err),
        )
    except Exception as exc:
        logger.error(f"Failed to capture evidence for video {video_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evidence extraction error: {str(exc)}",
        )


@router.get("/{video_id}/evidence")
def list_video_evidence(video_id: str) -> Dict[str, Any]:
    """
    Retrieve all captured evidence records associated with a video.
    """
    if not re.match(r"^[a-zA-Z0-9_\-]+$", video_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid video ID format.",
        )

    meta_path = settings.STORAGE_UPLOADS_DIR / f"{video_id}.json"
    video_exists = meta_path.exists() or bool(
        list(settings.STORAGE_UPLOADS_DIR.glob(f"{video_id}_*"))
    )
    if not video_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with ID '{video_id}' not found.",
        )

    try:
        from backend.app.services.evidence_service import EvidenceService
        service = EvidenceService()
        items = service.get_video_evidence(video_id)
        return {
            "video_id": video_id,
            "total_evidence": len(items),
            "evidence": items,
        }
    except Exception as exc:
        logger.error(f"Failed to retrieve evidence for video {video_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list evidence: {str(exc)}",
        )


# ===========================================================================
# Phase 8: Advanced Security Intelligence Endpoints
# ===========================================================================

class ZoneCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    polygon: List[List[float]] = Field(..., min_length=3)
    target_classes: Optional[List[str]] = Field(default=["person", "car"])
    alert_on_entry: bool = True
    loitering_threshold_seconds: float = 30.0


class RunSecurityAnalysisRequest(BaseModel):
    zones: Optional[List[Dict[str, Any]]] = None


def _verify_video_exists_or_raise(video_id: str) -> Path:
    if not re.match(r"^[a-zA-Z0-9_\-]+$", video_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid video ID format.",
        )
    meta_path = settings.STORAGE_UPLOADS_DIR / f"{video_id}.json"
    matches = list(settings.STORAGE_UPLOADS_DIR.glob(f"{video_id}_*"))
    video_matches = [m for m in matches if m.suffix != ".json"]
    if not meta_path.exists() and not video_matches:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with ID '{video_id}' not found.",
        )
    if video_matches:
        return video_matches[0]
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return Path(meta.get("storage_path", ""))
    except Exception:
        return meta_path


@router.get("/{video_id}/tracks")
def get_video_tracks(
    video_id: str,
    object_class: Optional[str] = Query(None, description="Filter tracks by object class (e.g. 'person', 'car')"),
) -> Dict[str, Any]:
    """Retrieve multi-frame tracked objects for a video."""
    _verify_video_exists_or_raise(video_id)
    try:
        from ai.intelligence_repository import SecurityIntelligenceRepository
        repo = SecurityIntelligenceRepository()
        tracks = repo.get_tracks(video_id, object_class=object_class)
        return {
            "video_id": video_id,
            "total_tracks": len(tracks),
            "tracks": tracks,
        }
    except Exception as exc:
        logger.error(f"Error fetching tracks for {video_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch tracking records: {str(exc)}",
        )


@router.get("/{video_id}/attributes")
def get_vehicle_attributes(
    video_id: str,
    color: Optional[str] = Query(None, description="Filter vehicle attributes by color"),
) -> Dict[str, Any]:
    """Retrieve visual vehicle color attributes for a video."""
    _verify_video_exists_or_raise(video_id)
    try:
        from ai.intelligence_repository import SecurityIntelligenceRepository
        repo = SecurityIntelligenceRepository()
        attributes = repo.get_vehicle_attributes(video_id, color=color)
        return {
            "video_id": video_id,
            "total_attributes": len(attributes),
            "attributes": attributes,
        }
    except Exception as exc:
        logger.error(f"Error fetching vehicle attributes for {video_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch vehicle attributes: {str(exc)}",
        )


@router.get("/{video_id}/faces")
def get_face_detections(video_id: str) -> Dict[str, Any]:
    """
    Retrieve face visual region detections.
    STRICT SAFETY CONSTRAINT: Observational region detections only.
    No facial recognition, identity matching, or person identification.
    """
    _verify_video_exists_or_raise(video_id)
    try:
        from ai.intelligence_repository import SecurityIntelligenceRepository
        repo = SecurityIntelligenceRepository()
        faces = repo.get_face_detections(video_id)
        return {
            "video_id": video_id,
            "safety_notice": "Observational visual face regions only. No biometric identity inference or database matching.",
            "total_faces": len(faces),
            "faces": faces,
        }
    except Exception as exc:
        logger.error(f"Error fetching face detections for {video_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch face detections: {str(exc)}",
        )


@router.get("/{video_id}/security-events")
def get_security_events(
    video_id: str,
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    severity: Optional[str] = Query(None, description="Filter by severity level"),
) -> Dict[str, Any]:
    """Retrieve advanced security intelligence events (intrusions, prolonged presence, abandoned objects, activity peaks)."""
    _verify_video_exists_or_raise(video_id)
    try:
        from ai.intelligence_repository import SecurityIntelligenceRepository
        repo = SecurityIntelligenceRepository()
        events = repo.get_security_events(video_id, event_type=event_type, severity=severity)
        return {
            "video_id": video_id,
            "total_events": len(events),
            "events": events,
        }
    except Exception as exc:
        logger.error(f"Error fetching security events for {video_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch security events: {str(exc)}",
        )


@router.get("/{video_id}/specialized")
def get_specialized_observations(
    video_id: str,
    class_name: Optional[str] = Query(None, description="Filter by specialized class name (fire, smoke, weapon, pose, etc.)"),
) -> Dict[str, Any]:
    """Retrieve specialized visual detection observations (fire, smoke, weapons, pose)."""
    _verify_video_exists_or_raise(video_id)
    try:
        from ai.intelligence_repository import SecurityIntelligenceRepository
        repo = SecurityIntelligenceRepository()
        observations = repo.get_specialized_observations(video_id, class_name=class_name)
        return {
            "status": "success",
            "video_id": video_id,
            "total_observations": len(observations),
            "count": len(observations),
            "observations": observations,
        }
    except Exception as exc:
        logger.error(f"Error fetching specialized observations for {video_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch specialized observations: {str(exc)}",
        )


@router.get("/{video_id}/detector-health")
def get_video_detector_health(video_id: str) -> Dict[str, Any]:
    """Retrieve operational health, model provenance, and diagnostic metrics for all Sentinel detectors."""
    _verify_video_exists_or_raise(video_id)
    try:
        from ai.common.detector_health import DetectorHealthRegistry
        registry = DetectorHealthRegistry.get_instance()
        report = registry.get_health_report()
        report["video_id"] = video_id
        return {
            "status": "success",
            "video_id": video_id,
            "health": report,
        }
    except Exception as exc:
        logger.error(f"Error fetching detector health for {video_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch detector health: {str(exc)}",
        )


@router.get("/{video_id}/zones")
def get_video_zones(video_id: str) -> Dict[str, Any]:
    """Retrieve restricted security zones defined for a video."""
    _verify_video_exists_or_raise(video_id)
    try:
        from ai.intelligence_repository import SecurityIntelligenceRepository
        repo = SecurityIntelligenceRepository()
        zones = repo.get_zones(video_id)
        return {
            "video_id": video_id,
            "total_zones": len(zones),
            "zones": zones,
        }
    except Exception as exc:
        logger.error(f"Error fetching zones for {video_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch security zones: {str(exc)}",
        )


@router.post("/{video_id}/zones")
def create_video_zone(video_id: str, payload: ZoneCreateRequest) -> Dict[str, Any]:
    """Create a new restricted security zone for a video."""
    _verify_video_exists_or_raise(video_id)

    # Validate polygon vertices
    for idx, vertex in enumerate(payload.polygon):
        if not isinstance(vertex, (list, tuple)) or len(vertex) != 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Vertex {idx} must be an [x, y] coordinate pair.",
            )
        try:
            float(vertex[0])
            float(vertex[1])
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Vertex {idx} coordinates must be numeric.",
            )

    try:
        from ai.intelligence_repository import SecurityIntelligenceRepository
        repo = SecurityIntelligenceRepository()
        saved = repo.save_zone(
            name=payload.name,
            polygon=[[float(p[0]), float(p[1])] for p in payload.polygon],
            target_classes=payload.target_classes or ["person", "car"],
            video_id=video_id,
        )
        return {
            "video_id": video_id,
            "status": "created",
            "zone": saved,
        }
    except Exception as exc:
        logger.error(f"Error creating zone for {video_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create security zone: {str(exc)}",
        )


@router.delete("/{video_id}/zones/{zone_id}")
def delete_video_zone(video_id: str, zone_id: str) -> Dict[str, Any]:
    """Delete a restricted security zone."""
    _verify_video_exists_or_raise(video_id)
    try:
        from ai.intelligence_repository import SecurityIntelligenceRepository
        repo = SecurityIntelligenceRepository()
        deleted = repo.delete_zone(zone_id=zone_id, video_id=video_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Zone '{zone_id}' not found for video '{video_id}'.",
            )
        return {
            "video_id": video_id,
            "zone_id": zone_id,
            "status": "deleted",
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error deleting zone {zone_id} for {video_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete zone: {str(exc)}",
        )


@router.post("/{video_id}/run-security-analysis")
def run_security_analysis(
    video_id: str,
    payload: Optional[RunSecurityAnalysisRequest] = None,
) -> Dict[str, Any]:
    """
    Execute on-demand Phase 8 Security Intelligence Analysis on a processed video.
    Extracts multi-frame tracks, vehicle colors, anonymous face regions, zone intrusions,
    loitering, abandoned objects, and activity peaks.
    """
    video_file_path = _verify_video_exists_or_raise(video_id)

    # Fetch stored detection events
    from ai.events.repository import get_event_repository
    event_repo = get_event_repository()
    raw_events = event_repo.get_events(video_id)
    if not raw_events:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No detection events found for video '{video_id}'. Process the video first.",
        )

    # Read video metadata for fps and duration
    meta_path = settings.STORAGE_UPLOADS_DIR / f"{video_id}.json"
    fps = 30.0
    duration_seconds = 0.0
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            fps = float(meta.get("fps", 30.0))
            duration_seconds = float(meta.get("duration_seconds", 0.0))
        except Exception:
            pass

    try:
        from ai.intelligence_pipeline import SecurityIntelligencePipeline
        from ai.intelligence_repository import SecurityIntelligenceRepository
        from ai.schemas import ZoneDefinition

        intel_repo = SecurityIntelligenceRepository()

        # Build zones from payload or DB
        zones_data = []
        if payload and payload.zones:
            zones_data = payload.zones
        else:
            zones_data = intel_repo.get_zones(video_id)

        zone_defs = [
            ZoneDefinition(
                zone_id=z.get("zone_id") or f"zone-{uuid.uuid4().hex[:8]}",
                name=z["name"],
                polygon=[(float(p[0]), float(p[1])) for p in z["polygon"]],
                target_classes=z.get("target_classes") or ["person", "car"],
                alert_on_entry=z.get("alert_on_entry", True),
                loitering_threshold_seconds=float(z.get("loitering_threshold_seconds", 30.0)),
            )
            for z in zones_data
        ]

        pipe = SecurityIntelligencePipeline(zones=zone_defs)
        results = pipe.process_video_intelligence(
            video_id=video_id,
            video_path=str(video_file_path),
            raw_events=[e if isinstance(e, dict) else e.to_dict() for e in raw_events],
            fps=fps,
            duration_seconds=duration_seconds,
            sample_rate_fps=settings.VIDEO_SAMPLE_RATE_FPS,
        )

        intel_repo.save_intelligence_results(
            video_id=video_id,
            tracks=results["tracks"],
            vehicle_attributes=results["vehicle_attributes"],
            face_detections=results["face_detections"],
            security_events=results["security_events"],
            specialized_observations=results.get("specialized_observations", []),
            correlated_incidents=results.get("correlated_incidents", []),
        )
        _correlate_security_evidence(video_id, results["security_events"])
        try:
            from backend.app.services.evidence_service import EvidenceService
            EvidenceService().reconcile_evidence_validation(video_id)
        except Exception as rec_err:
            logger.warning(f"Could not reconcile evidence validation for {video_id}: {rec_err}")

        incidents_list = [inc.to_dict() if hasattr(inc, "to_dict") else dict(inc) for inc in results.get("incidents", [])]
        diagnostics = results.get("incident_diagnostics", {})
        corr_incidents = [ci.to_dict() if hasattr(ci, "to_dict") else dict(ci) for ci in results.get("correlated_incidents", [])]
        corr_diagnostics = results.get("correlation_diagnostics", {})

        return {
            "video_id": video_id,
            "status": "completed",
            "tracks_count": len(results["tracks"]),
            "vehicle_attributes_count": len(results["vehicle_attributes"]),
            "face_detections_count": len(results["face_detections"]),
            "security_events_count": len(results["security_events"]),
            "correlated_incidents_count": len(corr_incidents),
            "summary": {
                "tracks": [t.to_dict() for t in results["tracks"][:10]],
                "security_events": [e.to_dict() for e in results["security_events"][:10]],
            },
            "incident_intelligence": {
                "active_detectors": pipe.incident_engine.registry.list_detectors(),
                "incidents_count": len(incidents_list),
                "incidents": incidents_list[:20],
                "diagnostics": diagnostics,
            },
            "correlated_intelligence": {
                "correlated_incidents_count": len(corr_incidents),
                "correlated_incidents": corr_incidents,
                "diagnostics": corr_diagnostics,
            },
        }
    except Exception as exc:
        logger.error(f"Security analysis failed for video {video_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Security analysis error: {str(exc)}",
        )


@router.get("/{video_id}/correlated-incidents", summary="Get canonical correlated incidents")
async def get_video_correlated_incidents(
    video_id: str,
    category: Optional[str] = Query(None, description="Filter by incident category (e.g. VEHICLE, PROPERTY, PERSON)"),
    severity: Optional[str] = Query(None, description="Filter by reliability rating (HIGH, MEDIUM, LOW)"),
    validation_decision: Optional[str] = Query(None, description="Filter by validation decision (ACCEPTED, REVIEW_REQUIRED)"),
):
    """
    Retrieve Phase 16 canonical correlated incidents, supporting tracks, signals,
    deterministic storylines, and complete evidence provenance.
    """
    _verify_video_exists_or_raise(video_id)
    try:
        from ai.intelligence_repository import SecurityIntelligenceRepository
        repo = SecurityIntelligenceRepository()
        incidents = repo.get_correlated_incidents(
            video_id=video_id,
            category=category,
            severity=severity,
            validation_decision=validation_decision,
        )
        return {
            "video_id": video_id,
            "total_count": len(incidents),
            "correlated_incidents": incidents,
        }
    except Exception as exc:
        logger.error(f"Failed to fetch correlated incidents for video {video_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve correlated incidents: {str(exc)}",
        )


# ===========================================================================
# Phase 17: Advanced Forensic Investigation API Endpoints
# ===========================================================================

class Phase17InvestigationQueryRequest(BaseModel):
    """Pydantic model for POST /investigation/query — structured InvestigationQuery body."""
    time_start: Optional[float] = Field(None, ge=0.0, description="Start of temporal window (seconds)")
    time_end: Optional[float] = Field(None, ge=0.0, description="End of temporal window (seconds)")
    incident_categories: Optional[List[str]] = Field(None, description="Filter by incident category")
    validation_decisions: Optional[List[str]] = Field(None, description="Filter by validation decision")
    min_assessment_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_assessment_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    reliability_levels: Optional[List[str]] = Field(None)
    track_ids: Optional[List[str]] = Field(None, description="Filter by track IDs (video-scoped)")
    object_classes: Optional[List[str]] = Field(None, description="Filter by object class")
    zone_ids: Optional[List[str]] = Field(None)
    evidence_required: Optional[bool] = Field(None)
    correlated_only: Optional[bool] = Field(None)
    review_required_only: bool = Field(False)
    rejected_only: bool = Field(False)
    search_text: Optional[str] = Field(None, max_length=500)
    sort_order: str = Field("timestamp_asc")
    result_limit: int = Field(50, ge=1, le=200)
    result_offset: int = Field(0, ge=0)


class EvidenceBundleCreateRequest(BaseModel):
    """Pydantic model for POST /investigation/bundle."""
    bundle_name: str = Field("Investigation Bundle", max_length=255)
    selected_incident_ids: Optional[List[str]] = Field(None)
    selected_event_ids: Optional[List[str]] = Field(None)
    selected_track_ids: Optional[List[str]] = Field(None)
    selected_evidence_ids: Optional[List[str]] = Field(None)
    storyline_text: Optional[str] = Field(None, max_length=5000)
    notes: Optional[str] = Field(None, max_length=2000)
    provenance: Optional[Dict[str, Any]] = Field(None)


def _require_video(video_id: str) -> None:
    """Validate video_id format and ensure it exists. Reuses _verify_video_exists_or_raise."""
    _verify_video_exists_or_raise(video_id)


@router.get("/{video_id}/investigation/timeline", summary="Phase 17: Unified forensic timeline")
def get_investigation_timeline(
    video_id: str,
    start_time: Optional[float] = Query(None, ge=0.0, description="Start of temporal window (seconds)"),
    end_time: Optional[float] = Query(None, ge=0.0, description="End of temporal window (seconds)"),
    include_rejected: bool = Query(False, description="Include REJECTED incidents in timeline"),
) -> Dict[str, Any]:
    """
    Phase 17: Retrieve a unified forensic timeline merging all Sentinel intelligence layers:
    detection events, security events, correlated incidents, and evidence markers.
    Each entry carries a playback-seekable timestamp and layer classification.
    """
    _require_video(video_id)

    # Safe timestamp clamping
    if start_time is not None:
        start_time = max(0.0, start_time)
    if end_time is not None and start_time is not None and end_time < start_time:
        start_time, end_time = end_time, start_time

    try:
        from backend.app.services.investigation_service import InvestigationService
        svc = InvestigationService()
        return svc.get_unified_timeline(
            video_id=video_id,
            start_time=start_time,
            end_time=end_time,
            include_rejected=include_rejected,
        )
    except Exception as exc:
        logger.error(f"Failed to build unified timeline for {video_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Timeline retrieval failed: {str(exc)}",
        )


@router.get("/{video_id}/investigation/search", summary="Phase 17: Natural-language investigation search")
def investigation_search(
    video_id: str,
    q: str = Query(..., min_length=1, max_length=500, description="Natural-language investigation query"),
    video_duration: Optional[float] = Query(None, ge=0.0, description="Video duration (seconds) for temporal clamping"),
) -> Dict[str, Any]:
    """
    Phase 17: Natural-language investigation search with extended temporal and structured filters.

    Extends the existing /investigate endpoint with:
    - around/immediately before|after temporal modes
    - review-required / accepted / rejected validation filters
    - assessment score filters
    - correlated incident filters
    - evidence requirement filters
    - track ID extraction

    Returns an InvestigationResult with unified incidents, events, tracks, evidence, and timeline.
    """
    _require_video(video_id)

    if not re.match(r"^[a-zA-Z0-9_\-]+$", video_id):
        raise HTTPException(status_code=400, detail="Invalid video ID format.")

    try:
        from backend.app.services.investigation_service import InvestigationService
        svc = InvestigationService()
        return svc.investigate_structured(
            video_id=video_id,
            query_text=q,
            video_duration_seconds=video_duration or 0.0,
        )
    except Exception as exc:
        logger.error(f"Phase 17 investigation search failed for {video_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Investigation search failed: {str(exc)}",
        )


@router.post("/{video_id}/investigation/query", summary="Phase 17: Structured investigation query")
def investigation_structured_query(
    video_id: str,
    payload: Phase17InvestigationQueryRequest,
) -> Dict[str, Any]:
    """
    Phase 17: Execute a structured InvestigationQuery directly.

    Accepts explicit filter parameters without natural-language parsing.
    All inputs are validated before database access. No raw SQL is generated.
    Results are scoped exclusively to the specified video_id.
    """
    _require_video(video_id)

    if not re.match(r"^[a-zA-Z0-9_\-]+$", video_id):
        raise HTTPException(status_code=400, detail="Invalid video ID format.")

    try:
        from backend.app.services.investigation_query import InvestigationQuery, InvestigationQueryValidationError
        from backend.app.services.investigation_query_executor import InvestigationQueryExecutor

        iq = InvestigationQuery(
            video_id=video_id,
            time_start=payload.time_start,
            time_end=payload.time_end,
            incident_categories=payload.incident_categories or [],
            validation_decisions=payload.validation_decisions or [],
            min_assessment_score=payload.min_assessment_score,
            max_assessment_score=payload.max_assessment_score,
            reliability_levels=payload.reliability_levels or [],
            track_ids=payload.track_ids or [],
            object_classes=payload.object_classes or [],
            zone_ids=payload.zone_ids or [],
            evidence_required=payload.evidence_required,
            correlated_only=payload.correlated_only,
            review_required_only=payload.review_required_only,
            rejected_only=payload.rejected_only,
            search_text=payload.search_text,
            sort_order=payload.sort_order,
            result_limit=payload.result_limit,
            result_offset=payload.result_offset,
        )
        iq.validate()

        executor = InvestigationQueryExecutor()
        result = executor.execute(iq)
        return result.to_dict()

    except Exception as exc:
        from backend.app.services.investigation_query import InvestigationQueryValidationError
        if isinstance(exc, (ValueError, InvestigationQueryValidationError)):
            raise HTTPException(status_code=400, detail=str(exc))
        logger.error(f"Phase 17 structured query failed for {video_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Structured query execution failed: {str(exc)}",
        )


@router.get("/{video_id}/investigation/tracks/{track_id}", summary="Phase 17: Track investigation")
def investigation_track(
    video_id: str,
    track_id: str,
) -> Dict[str, Any]:
    """
    Phase 17: Full forensic investigation of a single anonymous track.

    Returns track metadata, associated security events, correlated incidents,
    linked evidence, and an observational object lifecycle summary.

    Track IDs are strictly video-scoped — this endpoint rejects track IDs
    not belonging to the specified video.

    No identity, biometric, or personal attribute inference is performed.
    """
    _require_video(video_id)

    # Validate track_id format (prevent path traversal / injection)
    if not re.match(r"^[a-zA-Z0-9_\-]+$", track_id):
        raise HTTPException(status_code=400, detail="Invalid track ID format.")

    try:
        from backend.app.services.investigation_service import InvestigationService
        svc = InvestigationService()
        return svc.investigate_track(video_id=video_id, track_id=track_id)
    except Exception as exc:
        logger.error(f"Phase 17 track investigation failed for {video_id}/{track_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Track investigation failed: {str(exc)}",
        )


@router.get("/{video_id}/investigation/evidence", summary="Phase 17: Investigation evidence retrieval")
def investigation_evidence(
    video_id: str,
    start_time: Optional[float] = Query(None, ge=0.0),
    end_time: Optional[float] = Query(None, ge=0.0),
    event_id: Optional[str] = Query(None, max_length=64),
    incident_id: Optional[str] = Query(None, max_length=64),
) -> Dict[str, Any]:
    """
    Phase 17: Evidence retrieval with optional temporal and event/incident filtering.
    Returns video-isolated validated evidence items with snapshot/clip availability flags.
    """
    _require_video(video_id)

    if start_time is not None:
        start_time = max(0.0, start_time)
    if end_time is not None and start_time is not None and end_time < start_time:
        start_time, end_time = end_time, start_time

    try:
        from database.session import SessionLocal
        from database.models import EvidenceModel
        db = SessionLocal()
        try:
            q = db.query(EvidenceModel).filter(
                EvidenceModel.video_id == video_id,
                EvidenceModel.validation_status == "VALID",
            )
            if start_time is not None:
                q = q.filter(EvidenceModel.timestamp_seconds >= start_time)
            if end_time is not None:
                q = q.filter(EvidenceModel.timestamp_seconds <= end_time)
            if event_id:
                q = q.filter(EvidenceModel.event_id == event_id)

            rows = q.order_by(EvidenceModel.timestamp_seconds.asc()).all()

            # If incident_id provided, also include evidence linked via event_id
            if incident_id:
                incident_ev = db.query(EvidenceModel).filter(
                    EvidenceModel.video_id == video_id,
                    EvidenceModel.validation_status == "VALID",
                    EvidenceModel.event_id == incident_id,
                ).all()
                seen = {r.id for r in rows}
                rows = list(rows) + [r for r in incident_ev if r.id not in seen]
                rows.sort(key=lambda r: r.timestamp_seconds)

            evidence = [
                {
                    "evidence_id": r.id,
                    "video_id": r.video_id,
                    "event_id": r.event_id,
                    "evidence_type": r.evidence_type,
                    "timestamp": round(r.timestamp_seconds, 2),
                    "object_class": r.object_class,
                    "confidence": round(r.confidence, 4) if r.confidence else None,
                    "has_snapshot": bool(r.snapshot_path),
                    "has_annotated": bool(r.annotated_snapshot_path),
                    "has_clip": bool(r.clip_path),
                    "start_time": r.start_time,
                    "end_time": r.end_time,
                    "duration_seconds": r.duration_seconds,
                }
                for r in rows
            ]
        finally:
            db.close()

        return {
            "video_id": video_id,
            "total_count": len(evidence),
            "evidence": evidence,
        }
    except Exception as exc:
        logger.error(f"Phase 17 evidence retrieval failed for {video_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evidence retrieval failed: {str(exc)}",
        )


@router.post("/{video_id}/investigation/bundle", summary="Phase 17: Create evidence bundle")
def create_investigation_bundle(
    video_id: str,
    payload: EvidenceBundleCreateRequest,
) -> Dict[str, Any]:
    """
    Phase 17: Create a persistent evidence bundle for the specified video.

    Bundles collect selected incident IDs, event IDs, track IDs, and evidence IDs
    into a reproducible, video-isolated record. Physical evidence files are
    referenced by ID only — never duplicated.

    All referenced IDs are verified to belong to video_id before saving.
    """
    _require_video(video_id)

    try:
        from backend.app.services.evidence_bundle_service import EvidenceBundleService, EvidenceBundleError
        svc = EvidenceBundleService()
        bundle = svc.create_bundle(
            video_id=video_id,
            bundle_name=payload.bundle_name,
            selected_incident_ids=payload.selected_incident_ids,
            selected_event_ids=payload.selected_event_ids,
            selected_track_ids=payload.selected_track_ids,
            selected_evidence_ids=payload.selected_evidence_ids,
            storyline_text=payload.storyline_text,
            notes=payload.notes,
            provenance=payload.provenance,
        )
        return {"status": "created", "bundle": bundle}
    except Exception as exc:
        from backend.app.services.evidence_bundle_service import EvidenceBundleError
        if isinstance(exc, EvidenceBundleError):
            raise HTTPException(status_code=400, detail=str(exc))
        logger.error(f"Phase 17 bundle creation failed for {video_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bundle creation failed: {str(exc)}",
        )


@router.get("/{video_id}/investigation/bundle/{bundle_id}", summary="Phase 17: Retrieve evidence bundle")
def get_investigation_bundle(
    video_id: str,
    bundle_id: str,
) -> Dict[str, Any]:
    """
    Phase 17: Retrieve an evidence bundle by ID.

    Video isolation is enforced — bundles from other videos are not accessible
    via this endpoint even if the bundle_id is known.
    """
    _require_video(video_id)

    if not re.match(r"^[a-zA-Z0-9_\-]+$", bundle_id):
        raise HTTPException(status_code=400, detail="Invalid bundle ID format.")

    try:
        from backend.app.services.evidence_bundle_service import EvidenceBundleService, EvidenceBundleError
        svc = EvidenceBundleService()
        bundle = svc.get_bundle(video_id=video_id, bundle_id=bundle_id)
        return {"video_id": video_id, "bundle": bundle}
    except Exception as exc:
        from backend.app.services.evidence_bundle_service import EvidenceBundleError
        if isinstance(exc, EvidenceBundleError):
            raise HTTPException(status_code=404, detail=str(exc))
        logger.error(f"Phase 17 bundle retrieval failed for {video_id}/{bundle_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bundle retrieval failed: {str(exc)}",
        )


@router.get("/{video_id}/investigation/bundles", summary="Phase 17: List evidence bundles")
def list_investigation_bundles(video_id: str) -> Dict[str, Any]:
    """Phase 17: List all evidence bundles for the specified video."""
    _require_video(video_id)
    try:
        from backend.app.services.evidence_bundle_service import EvidenceBundleService
        svc = EvidenceBundleService()
        bundles = svc.list_bundles(video_id=video_id)
        return {"video_id": video_id, "total_count": len(bundles), "bundles": bundles}
    except Exception as exc:
        logger.error(f"Phase 17 bundle list failed for {video_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bundle list failed: {str(exc)}",
        )








