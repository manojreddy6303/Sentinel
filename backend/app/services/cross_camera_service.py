"""
backend/app/services/cross_camera_service.py — Phase 18

CrossCameraAnalysisService: orchestrates cross-camera analysis and persists results.

Responsibilities:
  1. Load tracks from all cameras in a session as CameraObservation objects
  2. Run CrossCameraAssociationEngine → List[CrossCameraHypothesis]
  3. Persist hypotheses as CrossCameraAssociationModel records
  4. Provide query methods for associations, timelines, subject traces
  5. Analyst verdict management (CONFIRMED / REJECTED)

PRIVACY: All track data used is observable CV telemetry only.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session

try:
    from database.models import (
        CrossCameraAssociationModel,
        CameraSourceModel,
        TrackModel,
        VehicleAttributeModel,
        SecurityEventModel,
        EventModel,
    )
    from database.session import SessionLocal
except ImportError:
    from database.models import (          # type: ignore
        CrossCameraAssociationModel,
        CameraSourceModel,
        TrackModel,
        VehicleAttributeModel,
        SecurityEventModel,
        EventModel,
    )
    from database.session import SessionLocal

from ai.multicamera.schemas import (
    CameraObservation,
    CrossCameraHypothesis,
    AssociationType,
    AnalystVerdict,
)
from ai.multicamera.association_engine import CrossCameraAssociationEngine
from ai.multicamera.scene_intelligence import MultiCameraSceneIntelligence

try:
    from app.services.session_service import (
        SurveillanceSessionService,
        SessionNotFoundError,
        CameraNotFoundError,
    )
except ImportError:
    from backend.app.services.session_service import (  # type: ignore
        SurveillanceSessionService,
        SessionNotFoundError,
        CameraNotFoundError,
    )

logger = logging.getLogger(__name__)


class AssociationNotFoundError(Exception):
    pass


class CrossSessionAccessError(Exception):
    pass


class CrossCameraAnalysisService:
    """
    Orchestrates cross-camera track association analysis and result persistence.
    """

    def __init__(self):
        self.session_svc = SurveillanceSessionService()
        self.engine = CrossCameraAssociationEngine()
        self.intelligence = MultiCameraSceneIntelligence()

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def run_analysis(
        self,
        db: Session,
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Run cross-camera association analysis for a session.

        Returns:
            {
                "associations_created": int,
                "associations": [list of association dicts],
                "summary": {...},
            }
        """
        # Load cameras
        cameras = self.session_svc.get_cameras_for_session(db, session_id)
        if len(cameras) < 2:
            return {
                "associations_created": 0,
                "associations": [],
                "summary": {
                    "session_id": session_id,
                    "camera_count": len(cameras),
                    "note": "Cross-camera analysis requires at least 2 cameras in the session.",
                },
            }

        # Build CameraObservation objects from TrackModel records
        camera_observations: Dict[str, List[CameraObservation]] = {}
        for cam in cameras:
            observations = self._build_observations(db, cam)
            camera_observations[cam.id] = observations

        # Run engine
        hypotheses = self.engine.run(session_id, camera_observations)

        # Persist (delete prior results for this session first for idempotency)
        db.query(CrossCameraAssociationModel).filter(
            CrossCameraAssociationModel.session_id == session_id
        ).delete(synchronize_session=False)
        db.commit()

        created_records = []
        for hyp in hypotheses:
            record = self._persist_hypothesis(db, hyp)
            created_records.append(record)

        db.commit()

        # Summary
        summary = self.intelligence.summarize_session(session_id, hypotheses, camera_observations)

        logger.info(
            "CrossCamera analysis session=%s: %d hypotheses created from %d cameras",
            session_id, len(created_records), len(cameras),
        )
        return {
            "associations_created": len(created_records),
            "associations": [self._assoc_to_dict(r) for r in created_records],
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_associations(
        self,
        db: Session,
        session_id: str,
        min_confidence: Optional[float] = None,
        association_type: Optional[str] = None,
        analyst_verdict: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List associations for a session with optional filters."""
        q = db.query(CrossCameraAssociationModel).filter(
            CrossCameraAssociationModel.session_id == session_id
        )
        if min_confidence is not None:
            q = q.filter(CrossCameraAssociationModel.confidence >= min_confidence)
        if association_type is not None:
            q = q.filter(CrossCameraAssociationModel.association_type == association_type)
        if analyst_verdict is not None:
            q = q.filter(CrossCameraAssociationModel.analyst_verdict == analyst_verdict)

        records = (
            q.order_by(CrossCameraAssociationModel.confidence.desc())
            .offset(offset)
            .limit(min(limit, 200))
            .all()
        )
        return [self._assoc_to_dict(r) for r in records]

    def get_association(
        self,
        db: Session,
        session_id: str,
        association_id: str,
    ) -> Dict[str, Any]:
        """Get a single association, verifying it belongs to session_id."""
        record = db.query(CrossCameraAssociationModel).filter(
            CrossCameraAssociationModel.id == association_id,
            CrossCameraAssociationModel.session_id == session_id,
        ).first()
        if record is None:
            raise AssociationNotFoundError(
                f"Association '{association_id}' not found in session '{session_id}'"
            )
        return self._assoc_to_dict(record)

    def get_associations_for_track(
        self,
        db: Session,
        session_id: str,
        video_id: str,
        track_id: str,
    ) -> List[Dict[str, Any]]:
        """Associations where the given track appears as source or target."""
        records = db.query(CrossCameraAssociationModel).filter(
            CrossCameraAssociationModel.session_id == session_id,
            (
                (
                    (CrossCameraAssociationModel.source_video_id == video_id) &
                    (CrossCameraAssociationModel.source_track_id == track_id)
                ) |
                (
                    (CrossCameraAssociationModel.target_video_id == video_id) &
                    (CrossCameraAssociationModel.target_track_id == track_id)
                )
            ),
        ).order_by(CrossCameraAssociationModel.confidence.desc()).all()
        return [self._assoc_to_dict(r) for r in records]

    def get_cross_camera_timeline(
        self,
        db: Session,
        session_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Merged cross-camera chronological timeline.

        Collects SecurityEventModel and EventModel records from all videos
        in the session, annotates with camera_label, and sorts by timestamp.
        """
        cameras = self.session_svc.get_cameras_for_session(db, session_id)
        if not cameras:
            return []

        # Build video_id → camera_label map
        cam_label: Dict[str, str] = {c.video_id: c.camera_label for c in cameras}
        video_ids = list(cam_label.keys())

        items: List[Dict[str, Any]] = []

        # Security events
        sec_events = (
            db.query(SecurityEventModel)
            .filter(SecurityEventModel.video_id.in_(video_ids))
            .order_by(SecurityEventModel.timestamp_seconds)
            .limit(1000)
            .all()
        )
        for ev in sec_events:
            items.append({
                "type": "security_event",
                "camera_label": cam_label.get(ev.video_id, "unknown"),
                "video_id": ev.video_id,
                "timestamp": ev.timestamp_seconds,
                "event_type": ev.event_type,
                "severity": ev.severity,
                "description": ev.description,
                "track_id": ev.track_id,
                "object_class": ev.object_class,
                "confidence": ev.confidence,
                "id": ev.id,
            })

        # Raw detection events (grouped events)
        from database.models import GroupedEventModel
        grp_events = (
            db.query(GroupedEventModel)
            .filter(GroupedEventModel.video_id.in_(video_ids))
            .order_by(GroupedEventModel.start_time)
            .limit(500)
            .all()
        )
        for ev in grp_events:
            items.append({
                "type": "detection_event",
                "camera_label": cam_label.get(ev.video_id, "unknown"),
                "video_id": ev.video_id,
                "timestamp": ev.start_time,
                "event_type": ev.event_type,
                "duration_seconds": ev.duration_seconds,
                "objects_summary": ev.objects_summary,
                "max_confidence": ev.max_confidence,
                "priority": ev.priority,
                "id": ev.id,
            })

        # Sort unified list by timestamp
        items.sort(key=lambda x: x.get("timestamp", 0.0))
        return items

    # ------------------------------------------------------------------
    # Analyst verdict
    # ------------------------------------------------------------------

    def update_analyst_verdict(
        self,
        db: Session,
        session_id: str,
        association_id: str,
        verdict: str,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update analyst verdict on a cross-camera association.

        Permitted verdicts: PENDING, CONFIRMED, REJECTED
        """
        if verdict not in ("PENDING", "CONFIRMED", "REJECTED"):
            raise ValueError(f"Invalid verdict '{verdict}'")

        record = db.query(CrossCameraAssociationModel).filter(
            CrossCameraAssociationModel.id == association_id,
            CrossCameraAssociationModel.session_id == session_id,
        ).first()
        if record is None:
            raise AssociationNotFoundError(
                f"Association '{association_id}' not found in session '{session_id}'"
            )

        record.analyst_verdict = verdict
        record.analyst_notes = notes
        record.analyst_verdict_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(record)
        logger.info(
            "Analyst verdict updated: assoc=%s session=%s verdict=%s",
            association_id, session_id, verdict,
        )
        return self._assoc_to_dict(record)

    def delete_association(
        self,
        db: Session,
        session_id: str,
        association_id: str,
    ) -> None:
        """Analyst purges a false association."""
        record = db.query(CrossCameraAssociationModel).filter(
            CrossCameraAssociationModel.id == association_id,
            CrossCameraAssociationModel.session_id == session_id,
        ).first()
        if record is None:
            raise AssociationNotFoundError(
                f"Association '{association_id}' not found in session '{session_id}'"
            )
        db.delete(record)
        db.commit()
        logger.info("Deleted association id=%s from session=%s", association_id, session_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observations(
        self,
        db: Session,
        camera: CameraSourceModel,
    ) -> List[CameraObservation]:
        """
        Build CameraObservation list for one camera from its TrackModel records.
        Enriches with color data from VehicleAttributeModel if available.
        """
        tracks = (
            db.query(TrackModel)
            .filter(TrackModel.video_id == camera.video_id)
            .all()
        )

        # Color lookup by track_id (best confidence per track)
        color_map: Dict[str, tuple] = {}
        va_records = (
            db.query(VehicleAttributeModel)
            .filter(VehicleAttributeModel.video_id == camera.video_id)
            .all()
        )
        for va in va_records:
            if va.track_id:
                existing = color_map.get(va.track_id)
                if existing is None or va.confidence > existing[1]:
                    color_map[va.track_id] = (va.color, va.confidence)

        observations: List[CameraObservation] = []
        adjacency_labels: List[str] = camera.adjacency_hints or []

        for track in tracks:
            # Extract trajectory telemetry
            exit_direction: Optional[float] = None
            entry_direction: Optional[float] = None
            exit_pos: Optional[tuple] = None
            entry_pos: Optional[tuple] = None
            height_ratio: Optional[float] = None
            width_ratio: Optional[float] = None

            traj = track.trajectory  # [(timestamp, cx, cy), ...]
            if traj and len(traj) >= 2:
                import math
                # Entry direction from first two points
                try:
                    _, x0, y0 = traj[0]
                    _, x1, y1 = traj[1]
                    entry_pos = (x0, y0)
                    dx, dy = x1 - x0, y1 - y0
                    if dx != 0 or dy != 0:
                        entry_direction = math.degrees(math.atan2(dy, dx)) % 360.0
                except Exception:
                    pass

                # Exit direction from last two points
                try:
                    _, xn1, yn1 = traj[-2]
                    _, xn, yn = traj[-1]
                    exit_pos = (xn, yn)
                    dx, dy = xn - xn1, yn - yn1
                    if dx != 0 or dy != 0:
                        exit_direction = math.degrees(math.atan2(dy, dx)) % 360.0
                except Exception:
                    pass

            # Estimate height ratio from current_bbox
            bbox = track.current_bbox
            if bbox and isinstance(bbox, dict):
                try:
                    h = bbox.get("y2", 0) - bbox.get("y1", 0)
                    w = bbox.get("x2", 0) - bbox.get("x1", 0)
                    # Normalize assuming 1080p — approximate; values may be pixel or normalized
                    if h > 1.0:  # pixel space
                        height_ratio = h / 1080.0
                        width_ratio = w / 1920.0
                    else:        # already normalized
                        height_ratio = h
                        width_ratio = w
                except Exception:
                    pass

            color_info = color_map.get(track.track_id)
            color_val = color_info[0] if color_info else track.color
            color_conf = color_info[1] if color_info else track.color_confidence

            obs = CameraObservation(
                camera_id=camera.id,
                camera_label=camera.camera_label,
                video_id=camera.video_id,
                track_id=track.track_id,
                object_class=track.object_class,
                first_seen=track.first_seen,
                last_seen=track.last_seen,
                duration_seconds=track.duration_seconds,
                max_confidence=track.max_confidence,
                color=color_val,
                color_confidence=color_conf,
                estimated_height_ratio=height_ratio,
                estimated_width_ratio=width_ratio,
                exit_direction_degrees=exit_direction,
                entry_direction_degrees=entry_direction,
                exit_position=exit_pos,
                entry_position=entry_pos,
                adjacent_camera_labels=adjacency_labels,
            )
            observations.append(obs)

        return observations

    def _persist_hypothesis(
        self,
        db: Session,
        hyp: CrossCameraHypothesis,
    ) -> CrossCameraAssociationModel:
        """Persist a CrossCameraHypothesis as a DB record."""
        record = CrossCameraAssociationModel(
            session_id=hyp.session_id,
            source_camera_id=hyp.source_camera_id,
            source_video_id=hyp.source_video_id,
            source_track_id=hyp.source_track_id,
            source_last_seen=hyp.source_last_seen,
            target_camera_id=hyp.target_camera_id,
            target_video_id=hyp.target_video_id,
            target_track_id=hyp.target_track_id,
            target_first_seen=hyp.target_first_seen,
            confidence=hyp.confidence,
            association_type=hyp.association_type.value,
            attribute_match_score=hyp.attribute_match_score,
            trajectory_compatibility_score=hyp.trajectory_compatibility_score,
            temporal_gap_seconds=hyp.temporal_gap_seconds,
            temporal_plausibility_score=hyp.temporal_plausibility_score,
            evidence_basis=[e.to_dict() for e in hyp.evidence_basis],
            analyst_review_required=1,
            analyst_verdict="PENDING",
        )
        db.add(record)
        return record

    def _assoc_to_dict(self, record: CrossCameraAssociationModel) -> Dict[str, Any]:
        """Serialize a CrossCameraAssociationModel to API-safe dict."""
        verdict_at = None
        if record.analyst_verdict_at:
            verdict_at = record.analyst_verdict_at.isoformat()
        return {
            "id": record.id,
            "session_id": record.session_id,
            "source_camera_id": record.source_camera_id,
            "source_video_id": record.source_video_id,
            "source_track_id": record.source_track_id,
            "source_last_seen": record.source_last_seen,
            "target_camera_id": record.target_camera_id,
            "target_video_id": record.target_video_id,
            "target_track_id": record.target_track_id,
            "target_first_seen": record.target_first_seen,
            "confidence": record.confidence,
            "association_type": record.association_type,
            "attribute_match_score": record.attribute_match_score,
            "trajectory_compatibility_score": record.trajectory_compatibility_score,
            "temporal_gap_seconds": record.temporal_gap_seconds,
            "temporal_plausibility_score": record.temporal_plausibility_score,
            "evidence_basis": record.evidence_basis or [],
            "analyst_review_required": bool(record.analyst_review_required),
            "analyst_verdict": record.analyst_verdict,
            "analyst_notes": record.analyst_notes,
            "analyst_verdict_at": verdict_at,
            "created_at": record.created_at.isoformat() if record.created_at else None,
            "privacy_note": (
                "Association is an anonymous observable-attribute hypothesis only. "
                "No identity claims are made."
            ),
        }
