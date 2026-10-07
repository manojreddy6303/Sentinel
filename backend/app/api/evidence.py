"""
Evidence Serving API Router
Provides secure metadata lookup and media streaming for captured evidence.
"""

import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse

from backend.app.core.config import settings
from backend.app.services.evidence_service import EvidenceService
from backend.app.services.playback_service import (
    ensure_evidence_clip_playback,
    get_evidence_playback_status,
    stream_video_file_with_ranges,
    PlaybackError,
    PlaybackNotFoundError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evidence", tags=["evidence"])


def _validate_evidence_id(evidence_id: str) -> None:
    """Ensure evidence_id is well-formed to prevent path injection."""
    if not evidence_id or not re.match(r"^[a-zA-Z0-9_\-]+$", evidence_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid evidence ID format.",
        )


def _resolve_evidence_path(evidence_id: str, stored_path: Optional[str], filename_suffix: str) -> Optional[Path]:
    """
    Resolve an evidence media path on disk.
    If the database contains a legacy Windows path (e.g. C:\\Sentinel\\...) that does not
    exist on Linux, seamlessly fall back to resolving against settings.STORAGE_EVIDENCE_DIR.
    """
    if stored_path:
        p = Path(stored_path)
        if p.exists() and p.is_file():
            return p
    # Fallback to active Railway storage evidence directory
    cand = settings.STORAGE_EVIDENCE_DIR / f"{evidence_id}_{filename_suffix}"
    if cand.exists() and cand.is_file():
        return cand
    return None


def _safe_serve_file(file_path_str: str, media_type: str, filename_hint: str) -> FileResponse:
    """Safely verify that the file exists and is strictly within storage boundaries."""
    if not file_path_str:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media file not available for this evidence record.",
        )

    file_path = Path(file_path_str).resolve()
    storage_root = settings.STORAGE_DIR.resolve()

    # Prevent directory traversal outside storage
    try:
        file_path.relative_to(storage_root)
    except ValueError:
        logger.error(f"Directory traversal attempt detected: '{file_path}' not inside '{storage_root}'")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to requested file path.",
        )

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evidence media file not found on server.",
        )

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename_hint,
    )


@router.get("", summary="List all preserved evidence across videos or filtered")
def list_all_evidence(
    video_id: Optional[str] = Query(None, description="Filter by source video ID"),
    object_class: Optional[str] = Query(None, description="Filter by detected object class"),
    validation_status: Optional[str] = Query(None, description="Filter by validation status (VALID, REJECTED, UNCERTAIN, or all)"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Filter by minimum confidence"),
    search: Optional[str] = Query(None, description="Search in notes or source video name"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """
    List all preserved forensic evidence items with snapshots, video clips,
    confidence metrics, and review status.
    """
    import os
    from database.session import SessionLocal
    from database.models import EvidenceModel

    db = SessionLocal()
    try:
        query = db.query(EvidenceModel)
        if video_id:
            query = query.filter(EvidenceModel.video_id == video_id)
        if object_class and object_class != "all":
            query = query.filter(EvidenceModel.object_class == object_class.lower())
        if validation_status and validation_status != "all":
            query = query.filter(EvidenceModel.validation_status == validation_status)
        if min_confidence is not None:
            query = query.filter(EvidenceModel.confidence >= min_confidence)
        if search:
            query = query.filter(
                (EvidenceModel.source_video_name.ilike(f"%{search.strip()}%")) |
                (EvidenceModel.notes.ilike(f"%{search.strip()}%")) |
                (EvidenceModel.object_class.ilike(f"%{search.strip()}%"))
            )

        total = query.count()
        items = query.order_by(EvidenceModel.created_at.desc()).offset(offset).limit(limit).all()

        results = []
        for e in items:
            has_snap = _resolve_evidence_path(e.id, e.snapshot_path, "snapshot.jpg") is not None
            has_ann = _resolve_evidence_path(e.id, e.annotated_snapshot_path, "annotated.jpg") is not None
            has_clip = _resolve_evidence_path(e.id, e.clip_path, "clip.mp4") is not None

            results.append({
                "id": e.id,
                "evidence_id": e.id,
                "video_id": e.video_id,
                "source_video_name": e.source_video_name,
                "evidence_type": e.evidence_type,
                "validation_status": e.validation_status or "VALID",
                "timestamp_seconds": e.timestamp_seconds,
                "object_class": e.object_class,
                "confidence": e.confidence,
                "bounding_box": e.bounding_box,
                "start_time": e.start_time,
                "end_time": e.end_time,
                "duration_seconds": e.duration_seconds,
                "notes": e.notes,
                "created_at": e.created_at.isoformat() if e.created_at else None,
                "has_snapshot": has_snap,
                "has_annotated": has_ann,
                "has_clip": has_clip,
                "snapshot_url": f"/api/evidence/{e.id}/snapshot" if has_snap else None,
                "annotated_url": f"/api/evidence/{e.id}/annotated" if has_ann else (f"/api/evidence/{e.id}/snapshot" if has_snap else None),
                "clip_url": f"/api/evidence/{e.id}/clip" if has_clip else None,
                "playback_url": f"/api/evidence/{e.id}/playback" if has_clip else None,
            })

        return {
            "total": total,
            "evidence": results,
            "limit": limit,
            "offset": offset,
        }
    finally:
        db.close()


@router.get("/{evidence_id}")
def get_evidence_metadata(evidence_id: str) -> Dict[str, Any]:
    """Retrieve metadata and provenance for an individual evidence item."""
    _validate_evidence_id(evidence_id)
    service = EvidenceService()
    item = service.get_evidence_by_id(evidence_id)
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence with ID '{evidence_id}' not found.",
        )
    if item.get("clip_path"):
        item["playback_url"] = f"/api/evidence/{evidence_id}/playback"
    return item


@router.get("/{evidence_id}/snapshot")
def get_evidence_snapshot(evidence_id: str):
    """Securely stream original evidence snapshot image (JPEG)."""
    import os
    _validate_evidence_id(evidence_id)
    service = EvidenceService()
    item = service.get_evidence_by_id(evidence_id)
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence with ID '{evidence_id}' not found.",
        )

    snap_path = item.get("snapshot_path")
    if not snap_path or not os.path.exists(snap_path):
        cand = service.evidence_dir / f"{evidence_id}_snapshot.jpg"
        if cand.exists():
            snap_path = str(cand)
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Snapshot file for evidence '{evidence_id}' not found.",
            )

    return _safe_serve_file(str(snap_path), "image/jpeg", f"{evidence_id}_snapshot.jpg")


@router.get("/{evidence_id}/annotated")
def get_evidence_annotated(evidence_id: str):
    """Securely stream annotated evidence snapshot image with bounding box (JPEG)."""
    import os
    _validate_evidence_id(evidence_id)
    service = EvidenceService()
    item = service.get_evidence_by_id(evidence_id)
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence with ID '{evidence_id}' not found.",
        )

    ann_path = item.get("annotated_snapshot_path")
    if not ann_path or not os.path.exists(ann_path):
        cand = service.evidence_dir / f"{evidence_id}_annotated.jpg"
        if cand.exists():
            ann_path = str(cand)
        else:
            # Fall back to base snapshot if annotated file not generated
            return get_evidence_snapshot(evidence_id)

    return _safe_serve_file(str(ann_path), "image/jpeg", f"{evidence_id}_annotated.jpg")


@router.get("/{evidence_id}/clip")
def get_evidence_clip(evidence_id: str, request: Request):
    """Securely stream evidence video sub-clip with browser-compatible H.264 transcoding and RFC 7233 range support."""
    _validate_evidence_id(evidence_id)
    service = EvidenceService()
    item = service.get_evidence_by_id(evidence_id)
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence with ID '{evidence_id}' not found.",
        )

    db_item = service.evidence_dir / f"{evidence_id}_clip.mp4"
    if not db_item.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence clip file for '{evidence_id}' not found.",
        )

    # Path traversal check
    try:
        db_item.resolve().relative_to(settings.STORAGE_EVIDENCE_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    # Ensure browser-compatible playback file (H.264 + MP4) so standard HTML5 video elements can decode it
    try:
        playback_clip = ensure_evidence_clip_playback(evidence_id, db_item)
        return stream_video_file_with_ranges(playback_clip, request.headers.get("range"))
    except Exception as exc:
        logger.warning(f"Fallback to direct clip stream for {evidence_id}: {exc}")
        return stream_video_file_with_ranges(db_item, request.headers.get("range"))


@router.get("/{evidence_id}/playback-status")
def get_evidence_playback_status_endpoint(evidence_id: str) -> Dict[str, Any]:
    """
    Check the browser playback status for an evidence clip.
    Returns status: ready, preparing, needs_conversion, or failed.
    """
    _validate_evidence_id(evidence_id)
    service = EvidenceService()
    item = service.get_evidence_by_id(evidence_id)
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence with ID '{evidence_id}' not found.",
        )

    original_clip = service.evidence_dir / f"{evidence_id}_clip.mp4"
    return get_evidence_playback_status(evidence_id, original_clip)


@router.api_route("/{evidence_id}/playback", methods=["GET", "HEAD"])
def playback_evidence_clip(evidence_id: str, request: Request):
    """
    Securely stream a browser-compatible H.264 MP4 evidence clip with RFC 7233 HTTP Range support.
    
    If the original evidence clip is encoded in an unsupported codec (e.g. mp4v, hevc),
    it is transcoded once to H.264 (yuv420p + AAC) in storage/evidence_playback/ and cached.
    """
    _validate_evidence_id(evidence_id)
    service = EvidenceService()
    item = service.get_evidence_by_id(evidence_id)
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence with ID '{evidence_id}' not found in database.",
        )

    original_clip = service.evidence_dir / f"{evidence_id}_clip.mp4"
    if not original_clip.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence clip file for '{evidence_id}' not found.",
        )

    # Ensure clip belongs to storage/evidence
    try:
        original_clip.resolve().relative_to(settings.STORAGE_EVIDENCE_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    try:
        playback_clip = ensure_evidence_clip_playback(evidence_id, original_clip)
    except PlaybackNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence clip for '{evidence_id}' not found.",
        )
    except PlaybackError as err:
        logger.warning(f"Evidence clip playback conversion deferred/failed for {evidence_id}: {err}; falling back to direct clip stream.")
        playback_clip = original_clip

    range_header = request.headers.get("range")
    return stream_video_file_with_ranges(playback_clip, range_header, is_head=(request.method == "HEAD"))
