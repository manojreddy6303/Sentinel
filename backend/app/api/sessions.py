"""
backend/app/api/sessions.py — Phase 18

REST API router for Surveillance Session and Cross-Camera Intelligence.

All endpoints enforce:
  - session_id existence validation
  - video-in-session ownership checks
  - pagination limits (max 200 per page)
  - input sanitization (delegated to service layer)
  - session isolation (no cross-session data exposure)

Privacy: all endpoints may reference tracks by ID only.
No identity claims, no biometric data in any response.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

try:
    from database.session import SessionLocal
    from app.services.session_service import (
        SurveillanceSessionService,
        SessionNotFoundError,
        VideoNotFoundError,
        CameraNotFoundError,
        CameraAlreadyInSessionError,
    )
    from app.services.cross_camera_service import (
        CrossCameraAnalysisService,
        AssociationNotFoundError,
    )
except ImportError:
    from database.session import SessionLocal            # type: ignore
    from backend.app.services.session_service import (  # type: ignore
        SurveillanceSessionService,
        SessionNotFoundError,
        VideoNotFoundError,
        CameraNotFoundError,
        CameraAlreadyInSessionError,
    )
    from backend.app.services.cross_camera_service import (  # type: ignore
        CrossCameraAnalysisService,
        AssociationNotFoundError,
    )

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["Multi-Camera Sessions"])

_session_svc = SurveillanceSessionService()
_cross_cam_svc = CrossCameraAnalysisService()

# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------
_ID_RE = re.compile(r"^[a-zA-Z0-9\-]{1,64}$")


def _validate_id(value: str, label: str) -> str:
    if not _ID_RE.match(value):
        raise HTTPException(status_code=400, detail=f"Invalid {label}: '{value}'")
    return value


def _get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    site_name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)


class AddCameraRequest(BaseModel):
    video_id: str = Field(..., min_length=1, max_length=64)
    camera_label: str = Field(..., min_length=1, max_length=100)
    position_hint: Optional[str] = Field(None, max_length=255)
    field_of_view_hint: Optional[str] = Field(None, max_length=255)
    adjacency_hints: Optional[List[str]] = Field(None, max_length=20)


class UpdateVerdictRequest(BaseModel):
    verdict: str = Field(..., pattern="^(PENDING|CONFIRMED|REJECTED)$")
    notes: Optional[str] = Field(None, max_length=2000)


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

@router.post("", status_code=status.HTTP_201_CREATED, summary="Create surveillance session")
def create_session(body: CreateSessionRequest) -> Dict[str, Any]:
    """Create a new surveillance session grouping multiple camera sources."""
    db = SessionLocal()
    try:
        session = _session_svc.create_session(
            db,
            name=body.name,
            site_name=body.site_name,
            description=body.description,
        )
        return _session_svc.to_dict(session, include_cameras=True)
    except Exception as exc:
        logger.error("create_session error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        db.close()


@router.get("", summary="List all surveillance sessions")
def list_sessions(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """List all surveillance sessions ordered by creation date (newest first)."""
    db = SessionLocal()
    try:
        sessions = _session_svc.list_sessions(db, limit=limit, offset=offset)
        return {
            "total": len(sessions),
            "sessions": [_session_svc.to_dict(s) for s in sessions],
        }
    finally:
        db.close()


@router.get("/{session_id}", summary="Get session detail")
def get_session(session_id: str) -> Dict[str, Any]:
    """Retrieve full session detail including cameras and association summary."""
    _validate_id(session_id, "session_id")
    db = SessionLocal()
    try:
        try:
            session = _session_svc.get_session(db, session_id)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        return _session_svc.to_dict(session, include_cameras=True)
    finally:
        db.close()


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete session")
def delete_session(session_id: str):
    """Delete a session and cascade to all cameras and associations."""
    _validate_id(session_id, "session_id")
    db = SessionLocal()
    try:
        try:
            _session_svc.delete_session(db, session_id)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Camera management
# ---------------------------------------------------------------------------

@router.post("/{session_id}/cameras", status_code=status.HTTP_201_CREATED, summary="Add camera to session")
def add_camera(session_id: str, body: AddCameraRequest) -> Dict[str, Any]:
    """Add a processed video as a camera source within a session."""
    _validate_id(session_id, "session_id")
    _validate_id(body.video_id, "video_id")
    db = SessionLocal()
    try:
        try:
            camera = _session_svc.add_camera(
                db,
                session_id=session_id,
                video_id=body.video_id,
                camera_label=body.camera_label,
                position_hint=body.position_hint,
                field_of_view_hint=body.field_of_view_hint,
                adjacency_hints=body.adjacency_hints,
            )
            return _session_svc.camera_to_dict(camera)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        except VideoNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except CameraAlreadyInSessionError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
    finally:
        db.close()


@router.get("/{session_id}/cameras", summary="List cameras in session")
def list_cameras(session_id: str) -> Dict[str, Any]:
    """List all camera sources registered in a session."""
    _validate_id(session_id, "session_id")
    db = SessionLocal()
    try:
        try:
            cameras = _session_svc.get_cameras_for_session(db, session_id)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        return {
            "session_id": session_id,
            "camera_count": len(cameras),
            "cameras": [_session_svc.camera_to_dict(c) for c in cameras],
        }
    finally:
        db.close()


@router.delete(
    "/{session_id}/cameras/{camera_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove camera from session",
)
def remove_camera(session_id: str, camera_id: str):
    """Remove a camera source from a session. Cascades associated hypotheses."""
    _validate_id(session_id, "session_id")
    _validate_id(camera_id, "camera_id")
    db = SessionLocal()
    try:
        try:
            _session_svc.remove_camera(db, session_id, camera_id)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        except CameraNotFoundError:
            raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found in session '{session_id}'")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Cross-camera analysis
# ---------------------------------------------------------------------------

@router.post("/{session_id}/analyze", summary="Run cross-camera association analysis")
def run_analysis(session_id: str) -> Dict[str, Any]:
    """
    Trigger cross-camera association analysis for the session.

    Analyses track telemetry across all cameras to identify anonymous subject
    continuity hypotheses. All results require analyst review before use.

    Privacy: Results use observable-attribute matching only.
    No identity claims or biometric data are produced.
    """
    _validate_id(session_id, "session_id")
    db = SessionLocal()
    try:
        try:
            _session_svc.get_session(db, session_id)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        try:
            result = _cross_cam_svc.run_analysis(db, session_id)
            return result
        except Exception as exc:
            logger.error("run_analysis session=%s error: %s", session_id, exc)
            raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Association queries
# ---------------------------------------------------------------------------

@router.get("/{session_id}/associations", summary="List cross-camera associations")
def list_associations(
    session_id: str,
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    association_type: Optional[str] = Query(None),
    analyst_verdict: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """
    List cross-camera associations for a session with optional filters.

    All associations are anonymous observable-attribute hypotheses.
    analyst_verdict options: PENDING, CONFIRMED, REJECTED
    association_type options: SAME_OBJECT, PROBABLE_SAME, POSSIBLE_SAME
    """
    _validate_id(session_id, "session_id")
    db = SessionLocal()
    try:
        try:
            _session_svc.get_session(db, session_id)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        associations = _cross_cam_svc.get_associations(
            db,
            session_id=session_id,
            min_confidence=min_confidence,
            association_type=association_type,
            analyst_verdict=analyst_verdict,
            limit=limit,
            offset=offset,
        )
        return {
            "session_id": session_id,
            "total": len(associations),
            "associations": associations,
        }
    finally:
        db.close()


@router.get("/{session_id}/associations/{association_id}", summary="Get association detail")
def get_association(session_id: str, association_id: str) -> Dict[str, Any]:
    """Retrieve full detail of a cross-camera association including evidence_basis."""
    _validate_id(session_id, "session_id")
    _validate_id(association_id, "association_id")
    db = SessionLocal()
    try:
        try:
            return _cross_cam_svc.get_association(db, session_id, association_id)
        except AssociationNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    finally:
        db.close()


@router.patch(
    "/{session_id}/associations/{association_id}/verdict",
    summary="Update analyst verdict",
)
def update_verdict(
    session_id: str,
    association_id: str,
    body: UpdateVerdictRequest,
) -> Dict[str, Any]:
    """
    Update the analyst verdict for a cross-camera association.

    Permitted verdicts: PENDING, CONFIRMED, REJECTED

    Confirmed associations are included in movement narratives.
    Rejected associations are excluded from further analysis.
    """
    _validate_id(session_id, "session_id")
    _validate_id(association_id, "association_id")
    db = SessionLocal()
    try:
        try:
            return _cross_cam_svc.update_analyst_verdict(
                db,
                session_id=session_id,
                association_id=association_id,
                verdict=body.verdict,
                notes=body.notes,
            )
        except AssociationNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    finally:
        db.close()


@router.delete(
    "/{session_id}/associations/{association_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete association",
)
def delete_association(session_id: str, association_id: str):
    """Analyst purges a false or unwanted cross-camera association."""
    _validate_id(session_id, "session_id")
    _validate_id(association_id, "association_id")
    db = SessionLocal()
    try:
        try:
            _cross_cam_svc.delete_association(db, session_id, association_id)
        except AssociationNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Cross-camera timeline
# ---------------------------------------------------------------------------

@router.get("/{session_id}/timeline", summary="Cross-camera chronological timeline")
def get_timeline(session_id: str) -> Dict[str, Any]:
    """
    Merged chronological timeline across all cameras in the session.

    Returns security events and detection events from all videos,
    annotated with camera labels, sorted by timestamp.
    Video isolation is maintained — only videos in this session are included.
    """
    _validate_id(session_id, "session_id")
    db = SessionLocal()
    try:
        try:
            _session_svc.get_session(db, session_id)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        items = _cross_cam_svc.get_cross_camera_timeline(db, session_id)
        return {
            "session_id": session_id,
            "total_events": len(items),
            "timeline": items,
        }
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Subject trace (cross-camera movement narrative for a specific track cluster)
# ---------------------------------------------------------------------------

@router.get("/{session_id}/tracks/{video_id}/{track_id}/associations", summary="Track association lookup")
def get_track_associations(session_id: str, video_id: str, track_id: str) -> Dict[str, Any]:
    """
    Retrieve all cross-camera associations involving a specific track.

    The video must belong to the session (isolation enforced).
    """
    _validate_id(session_id, "session_id")
    _validate_id(video_id, "video_id")
    db = SessionLocal()
    try:
        try:
            _session_svc.get_session(db, session_id)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

        # Enforce video belongs to session
        if not _session_svc.validate_video_in_session(db, session_id, video_id):
            raise HTTPException(
                status_code=403,
                detail=f"Video '{video_id}' is not part of session '{session_id}'",
            )

        associations = _cross_cam_svc.get_associations_for_track(
            db, session_id, video_id, track_id
        )
        return {
            "session_id": session_id,
            "video_id": video_id,
            "track_id": track_id,
            "associations": associations,
        }
    finally:
        db.close()
