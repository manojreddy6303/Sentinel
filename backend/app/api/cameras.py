"""
Cameras API Router
Provides endpoints to list, register, and query CCTV camera sources across Sentinel.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from database.session import SessionLocal
from database.models import CameraSourceModel, VideoModel, SurveillanceSessionModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cameras", tags=["Cameras"])


class RegisterCameraRequest(BaseModel):
    camera_label: str = Field(..., min_length=1, max_length=100)
    video_id: Optional[str] = Field(None, max_length=64)
    session_id: Optional[str] = Field(None, max_length=64)
    position_hint: Optional[str] = Field(None, max_length=255)
    field_of_view_hint: Optional[str] = Field(None, max_length=255)
    location: Optional[str] = Field(None, max_length=255)
    coverage_description: Optional[str] = Field(None, max_length=255)
    adjacency_hints: Optional[List[str]] = Field(default_factory=list)


class AssignVideoRequest(BaseModel):
    video_id: str = Field(..., min_length=1, max_length=64)


@router.get("", summary="List all registered camera sources")
def list_cameras(
    search: Optional[str] = Query(None, description="Search by camera label or location"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """
    List all physical camera sources and their associated surveillance feeds.
    """
    db = SessionLocal()
    try:
        query = db.query(CameraSourceModel)
        if search:
            query = query.filter(
                (CameraSourceModel.camera_label.ilike(f"%{search.strip()}%")) |
                (CameraSourceModel.position_hint.ilike(f"%{search.strip()}%")) |
                (CameraSourceModel.field_of_view_hint.ilike(f"%{search.strip()}%"))
            )

        total = query.count()
        cameras = query.order_by(CameraSourceModel.created_at.desc()).offset(offset).limit(limit).all()

        results = []
        for cam in cameras:
            vid = db.query(VideoModel).filter(VideoModel.id == cam.video_id).first() if cam.video_id else None
            sess = db.query(SurveillanceSessionModel).filter(SurveillanceSessionModel.id == cam.session_id).first() if cam.session_id else None

            # Count videos associated via camera_id foreign key or direct video_id
            assoc_count = db.query(VideoModel).filter(
                (VideoModel.camera_id == cam.id) | (VideoModel.id == cam.video_id)
            ).count()

            results.append({
                "id": cam.id,
                "camera_id": cam.id,
                "camera_label": cam.camera_label,
                "status": getattr(cam, "status", "ACTIVE") or "ACTIVE",
                "video_id": cam.video_id,
                "session_id": cam.session_id,
                "session_name": sess.name if sess else "Main Site",
                "site_name": sess.site_name if sess else "Main Facility",
                "position_hint": cam.position_hint,
                "field_of_view_hint": cam.field_of_view_hint,
                "adjacency_hints": cam.adjacency_hints or [],
                "associated_videos_count": assoc_count,
                "created_at": cam.created_at.isoformat() if cam.created_at else None,
                "video_filename": vid.original_filename if vid else None,
                "video_duration": vid.duration_seconds if vid else None,
                "video_fps": vid.fps if vid else None,
                "video_status": vid.status if vid else None,
                "playback_url": f"/api/videos/{cam.video_id}/playback" if cam.video_id else None,
            })

        return {
            "total": total,
            "cameras": results,
            "limit": limit,
            "offset": offset,
        }
    finally:
        db.close()


@router.post("", status_code=status.HTTP_201_CREATED, summary="Register a new CCTV camera source")
def register_camera(payload: RegisterCameraRequest) -> Dict[str, Any]:
    """
    Register a physical CCTV camera source with spatial and adjacency hints.
    Enforces canonical camera identity: duplicate registrations of the same label
    update metadata and return the existing physical camera rather than creating duplicates.
    """
    from sqlalchemy import func

    db = SessionLocal()
    try:
        norm_label = payload.camera_label.strip()

        # Check for existing camera by canonical label (case-insensitive)
        existing = db.query(CameraSourceModel).filter(
            func.lower(CameraSourceModel.camera_label) == norm_label.lower()
        ).first()

        vid = None
        if payload.video_id:
            vid = db.query(VideoModel).filter(VideoModel.id == payload.video_id).first()
            if not vid:
                raise HTTPException(status_code=404, detail=f"Video '{payload.video_id}' not found.")

        pos_hint = payload.position_hint or payload.location
        fov_hint = payload.field_of_view_hint or payload.coverage_description

        if existing:
            # Update hints on existing physical camera without duplication
            if pos_hint is not None:
                existing.position_hint = pos_hint.strip() or None
            if fov_hint is not None:
                existing.field_of_view_hint = fov_hint.strip() or None
            if payload.adjacency_hints:
                existing.adjacency_hints = payload.adjacency_hints
            if payload.video_id:
                existing.video_id = payload.video_id
                vid.camera_id = existing.id
            db.commit()
            db.refresh(existing)
            return {
                "id": existing.id,
                "camera_id": existing.id,
                "camera_label": existing.camera_label,
                "status": getattr(existing, "status", "ACTIVE") or "ACTIVE",
                "video_id": existing.video_id,
                "session_id": existing.session_id,
                "position_hint": existing.position_hint,
                "field_of_view_hint": existing.field_of_view_hint,
                "location": existing.position_hint,
                "coverage_description": existing.field_of_view_hint,
                "adjacency_hints": existing.adjacency_hints or [],
                "created_at": existing.created_at.isoformat() if existing.created_at else None,
            }

        # Resolve optional session
        session_id = payload.session_id
        if session_id:
            sess = db.query(SurveillanceSessionModel).filter(SurveillanceSessionModel.id == session_id).first()
            if not sess:
                raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

        cam = CameraSourceModel(
            session_id=session_id,
            video_id=payload.video_id,
            camera_label=norm_label,
            position_hint=pos_hint.strip() if pos_hint else None,
            field_of_view_hint=fov_hint.strip() if fov_hint else None,
            adjacency_hints=payload.adjacency_hints or [],
            status="ACTIVE",
            created_at=datetime.now(timezone.utc),
        )
        db.add(cam)
        db.flush()

        if vid:
            vid.camera_id = cam.id

        db.commit()
        db.refresh(cam)

        return {
            "id": cam.id,
            "camera_id": cam.id,
            "camera_label": cam.camera_label,
            "status": cam.status,
            "video_id": cam.video_id,
            "session_id": cam.session_id,
            "position_hint": cam.position_hint,
            "field_of_view_hint": cam.field_of_view_hint,
            "location": cam.position_hint,
            "coverage_description": cam.field_of_view_hint,
            "adjacency_hints": cam.adjacency_hints or [],
            "created_at": cam.created_at.isoformat() if cam.created_at else None,
        }
    finally:
        db.close()


@router.post("/{camera_id}/assign-video", summary="Assign an ingested video to a physical camera source")
def assign_video_to_camera(camera_id: str, payload: AssignVideoRequest) -> Dict[str, Any]:
    """
    Associate an ingested video recording with an existing physical camera.
    """
    db = SessionLocal()
    try:
        cam = db.query(CameraSourceModel).filter(CameraSourceModel.id == camera_id).first()
        if not cam:
            raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")

        vid = db.query(VideoModel).filter(VideoModel.id == payload.video_id).first()
        if not vid:
            raise HTTPException(status_code=404, detail=f"Video '{payload.video_id}' not found.")

        cam.video_id = vid.id
        vid.camera_id = cam.id
        db.commit()
        return {
            "status": "success",
            "camera_id": cam.id,
            "camera_label": cam.camera_label,
            "video_id": vid.id,
            "video_filename": vid.original_filename,
        }
    finally:
        db.close()
