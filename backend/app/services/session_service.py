"""
backend/app/services/session_service.py — Phase 18

SurveillanceSessionService: CRUD for sessions and camera sources.

Enforces:
  - Session isolation: all camera sources are scoped to their session
  - Video existence validation before adding a camera
  - No cross-session data exposure
  - Input sanitization on all string fields
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session

try:
    from database.models import (
        SurveillanceSessionModel,
        CameraSourceModel,
        VideoModel,
    )
except ImportError:
    from database.models import (          # type: ignore
        SurveillanceSessionModel,
        CameraSourceModel,
        VideoModel,
    )

logger = logging.getLogger(__name__)

# Input validation constants
_MAX_NAME_LEN = 255
_MAX_LABEL_LEN = 100
_MAX_HINT_LEN = 255
_SAFE_TEXT_RE = re.compile(r"[^a-zA-Z0-9 _\-\.,:()/\[\]'\"]+")  # strip suspicious chars
_MAX_ADJACENCY_LABELS = 20


def _sanitize(text: Optional[str], max_len: int = _MAX_NAME_LEN) -> Optional[str]:
    """Strip dangerous characters and enforce max length."""
    if text is None:
        return None
    cleaned = _SAFE_TEXT_RE.sub("", text.strip())
    return cleaned[:max_len] or None


class SessionNotFoundError(Exception):
    pass


class VideoNotFoundError(Exception):
    pass


class CameraNotFoundError(Exception):
    pass


class CameraAlreadyInSessionError(Exception):
    pass


class SurveillanceSessionService:
    """
    Service for managing surveillance sessions and their camera sources.

    All methods accept and return plain dicts or ORM model objects;
    no raw SQL is used — only SQLAlchemy ORM queries.
    """

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    def create_session(
        self,
        db: Session,
        name: str,
        site_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> SurveillanceSessionModel:
        """Create a new surveillance session."""
        session = SurveillanceSessionModel(
            name=_sanitize(name) or "Unnamed Session",
            site_name=_sanitize(site_name),
            description=_sanitize(description, max_len=1000),
            status="active",
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        logger.info("Created surveillance session id=%s name=%r", session.id, session.name)
        return session

    def get_session(self, db: Session, session_id: str) -> SurveillanceSessionModel:
        """Retrieve a session by ID. Raises SessionNotFoundError if absent."""
        obj = db.query(SurveillanceSessionModel).filter(
            SurveillanceSessionModel.id == session_id
        ).first()
        if obj is None:
            raise SessionNotFoundError(f"Session '{session_id}' not found")
        return obj

    def list_sessions(self, db: Session, limit: int = 100, offset: int = 0) -> List[SurveillanceSessionModel]:
        """List all sessions ordered by created_at descending."""
        return (
            db.query(SurveillanceSessionModel)
            .order_by(SurveillanceSessionModel.created_at.desc())
            .offset(offset)
            .limit(min(limit, 200))
            .all()
        )

    def delete_session(self, db: Session, session_id: str) -> None:
        """
        Delete a session. Cascades to camera_sources and cross_camera_associations
        via DB foreign-key ON DELETE CASCADE.
        """
        session = self.get_session(db, session_id)
        db.delete(session)
        db.commit()
        logger.info("Deleted surveillance session id=%s", session_id)

    def update_session_status(
        self,
        db: Session,
        session_id: str,
        status: str,
    ) -> SurveillanceSessionModel:
        """Set session status (active / archived)."""
        if status not in ("active", "archived"):
            raise ValueError(f"Invalid status '{status}' — must be 'active' or 'archived'")
        session = self.get_session(db, session_id)
        session.status = status
        session.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(session)
        return session

    # ------------------------------------------------------------------
    # Camera source management
    # ------------------------------------------------------------------

    def add_camera(
        self,
        db: Session,
        session_id: str,
        video_id: str,
        camera_label: str,
        position_hint: Optional[str] = None,
        field_of_view_hint: Optional[str] = None,
        adjacency_hints: Optional[List[str]] = None,
    ) -> CameraSourceModel:
        """
        Add a camera (video) to a session.

        Raises:
            SessionNotFoundError: if session_id unknown
            VideoNotFoundError: if video_id does not exist
            CameraAlreadyInSessionError: if this video is already in the session
        """
        # Validate session exists
        self.get_session(db, session_id)

        # Validate video exists
        video = db.query(VideoModel).filter(VideoModel.id == video_id).first()
        if video is None:
            raise VideoNotFoundError(f"Video '{video_id}' not found")

        # Prevent duplicate: same video in same session
        existing = db.query(CameraSourceModel).filter(
            CameraSourceModel.session_id == session_id,
            CameraSourceModel.video_id == video_id,
        ).first()
        if existing is not None:
            raise CameraAlreadyInSessionError(
                f"Video '{video_id}' is already a camera in session '{session_id}'"
            )

        # Sanitize adjacency_hints
        safe_adjacency: Optional[List[str]] = None
        if adjacency_hints:
            safe_adjacency = [
                _sanitize(lbl, _MAX_LABEL_LEN)
                for lbl in adjacency_hints[:_MAX_ADJACENCY_LABELS]
                if lbl
            ]
            safe_adjacency = [l for l in safe_adjacency if l]  # drop empties

        # Check if physical camera with this label already exists (canonical camera identity)
        clean_label = _sanitize(camera_label, _MAX_LABEL_LEN) or "Camera"
        from sqlalchemy import func
        existing_cam = db.query(CameraSourceModel).filter(
            func.lower(CameraSourceModel.camera_label) == clean_label.lower()
        ).first()

        if existing_cam is not None:
            existing_cam.session_id = session_id
            existing_cam.video_id = video_id
            video.camera_id = existing_cam.id
            if position_hint:
                existing_cam.position_hint = _sanitize(position_hint, _MAX_HINT_LEN)
            if field_of_view_hint:
                existing_cam.field_of_view_hint = _sanitize(field_of_view_hint, _MAX_HINT_LEN)
            if safe_adjacency:
                existing_cam.adjacency_hints = safe_adjacency
            db.commit()
            db.refresh(existing_cam)
            logger.info(
                "Associated existing physical camera id=%s label=%r to session=%s (video=%s)",
                existing_cam.id, existing_cam.camera_label, session_id, video_id,
            )
            return existing_cam

        camera = CameraSourceModel(
            session_id=session_id,
            video_id=video_id,
            camera_label=clean_label,
            position_hint=_sanitize(position_hint, _MAX_HINT_LEN),
            field_of_view_hint=_sanitize(field_of_view_hint, _MAX_HINT_LEN),
            adjacency_hints=safe_adjacency,
            status="ACTIVE",
        )
        db.add(camera)
        db.flush()
        video.camera_id = camera.id
        db.commit()
        db.refresh(camera)
        logger.info(
            "Added camera id=%s label=%r to session=%s (video=%s)",
            camera.id, camera.camera_label, session_id, video_id,
        )
        return camera

    def get_camera(self, db: Session, camera_id: str) -> CameraSourceModel:
        """Retrieve a camera source by ID."""
        cam = db.query(CameraSourceModel).filter(CameraSourceModel.id == camera_id).first()
        if cam is None:
            raise CameraNotFoundError(f"Camera '{camera_id}' not found")
        return cam

    def get_cameras_for_session(
        self,
        db: Session,
        session_id: str,
    ) -> List[CameraSourceModel]:
        """List all cameras in a session, ordered by created_at."""
        self.get_session(db, session_id)
        return (
            db.query(CameraSourceModel)
            .filter(CameraSourceModel.session_id == session_id)
            .order_by(CameraSourceModel.created_at)
            .all()
        )

    def remove_camera(
        self,
        db: Session,
        session_id: str,
        camera_id: str,
    ) -> None:
        """Remove a camera from a session. Cascades cross-camera associations."""
        # Verify camera belongs to this session
        cam = db.query(CameraSourceModel).filter(
            CameraSourceModel.id == camera_id,
            CameraSourceModel.session_id == session_id,
        ).first()
        if cam is None:
            raise CameraNotFoundError(
                f"Camera '{camera_id}' not found in session '{session_id}'"
            )
        db.delete(cam)
        db.commit()
        logger.info("Removed camera id=%s from session=%s", camera_id, session_id)

    def validate_video_in_session(
        self,
        db: Session,
        session_id: str,
        video_id: str,
    ) -> bool:
        """
        Return True if video_id is registered as a camera in session_id.
        Used to enforce cross-session isolation in downstream services.
        """
        cam = db.query(CameraSourceModel).filter(
            CameraSourceModel.session_id == session_id,
            CameraSourceModel.video_id == video_id,
        ).first()
        return cam is not None

    def to_dict(self, session: SurveillanceSessionModel, include_cameras: bool = False) -> Dict[str, Any]:
        """Serialize a session model to a dict."""
        d: Dict[str, Any] = {
            "id": session.id,
            "name": session.name,
            "site_name": session.site_name,
            "description": session.description,
            "status": session.status,
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "updated_at": session.updated_at.isoformat() if session.updated_at else None,
        }
        if include_cameras:
            d["cameras"] = [self.camera_to_dict(c) for c in session.camera_sources]
            d["camera_count"] = len(session.camera_sources)
        return d

    def camera_to_dict(self, camera: CameraSourceModel) -> Dict[str, Any]:
        """Serialize a camera source model to a dict."""
        return {
            "id": camera.id,
            "session_id": camera.session_id,
            "video_id": camera.video_id,
            "camera_label": camera.camera_label,
            "position_hint": camera.position_hint,
            "field_of_view_hint": camera.field_of_view_hint,
            "adjacency_hints": camera.adjacency_hints or [],
            "created_at": camera.created_at.isoformat() if camera.created_at else None,
        }
