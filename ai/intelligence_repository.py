"""
Phase 8: Security Intelligence Repository

Provides database persistence and querying for:
- TrackModel (tracks)
- VehicleAttributeModel (vehicle_attributes)
- FaceDetectionModel (face_detections)
- SecurityZoneModel (security_zones)
- SecurityEventModel (security_events)
"""
import json
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from database.session import SessionLocal, init_db
from database.models import (
    VideoModel,
    TrackModel,
    VehicleAttributeModel,
    FaceDetectionModel,
    SecurityZoneModel,
    SecurityEventModel,
    SpecializedObservationModel,
    CorrelatedIncidentModel,
)
from ai.schemas import (
    TrackedObject,
    VehicleAttribute,
    FaceDetection,
    ZoneDefinition,
    SecurityEvent,
)

logger = logging.getLogger(__name__)


class SecurityIntelligenceRepository:
    """
    SQLAlchemy-backed repository for Phase 8 and Phase 16 advanced surveillance records.
    """

    def __init__(self):
        init_db()

    def save_intelligence_results(
        self,
        video_id: str,
        tracks: List[TrackedObject],
        vehicle_attributes: List[VehicleAttribute],
        face_detections: List[FaceDetection],
        security_events: List[SecurityEvent],
        specialized_observations: Optional[List[Any]] = None,
        correlated_incidents: Optional[List[Any]] = None,
    ) -> None:
        """
        Idempotently persist intelligence outputs for a video.
        """
        db = SessionLocal()
        try:
            # Ensure parent video record exists to satisfy foreign key constraints
            if not db.query(VideoModel.id).filter(VideoModel.id == video_id).first():
                meta_p = Path(f"storage/uploads/{video_id}.json")
                orig_fn = f"{video_id}.mp4"
                stor_p = f"storage/uploads/{video_id}.mp4"
                sz = 0
                if meta_p.exists():
                    try:
                        with open(meta_p, "r", encoding="utf-8") as mf:
                            m = json.load(mf)
                            orig_fn = m.get("filename") or orig_fn
                            stor_p = m.get("storage_path") or stor_p
                            sz = m.get("file_size_bytes", 0)
                    except Exception:
                        pass
                db.add(
                    VideoModel(
                        id=video_id,
                        original_filename=orig_fn,
                        storage_path=stor_p,
                        file_size_bytes=sz,
                        status="processed",
                    )
                )
                db.flush()

            # Clear previous advanced records for this video to ensure idempotency
            db.query(TrackModel).filter(TrackModel.video_id == video_id).delete()
            db.query(VehicleAttributeModel).filter(VehicleAttributeModel.video_id == video_id).delete()
            db.query(FaceDetectionModel).filter(FaceDetectionModel.video_id == video_id).delete()
            db.query(SecurityEventModel).filter(SecurityEventModel.video_id == video_id).delete()
            db.query(SpecializedObservationModel).filter(SpecializedObservationModel.video_id == video_id).delete()
            db.query(CorrelatedIncidentModel).filter(CorrelatedIncidentModel.video_id == video_id).delete()

            # 1. Tracks
            track_models = []
            for t in tracks:
                track_models.append(
                    TrackModel(
                        id=str(uuid.uuid4()),
                        video_id=video_id,
                        track_id=t.track_id,
                        object_class=t.object_class,
                        first_seen=t.first_seen,
                        last_seen=t.last_seen,
                        duration_seconds=t.duration_seconds,
                        detection_count=len(t.history_bboxes),
                        max_confidence=t.confidence,
                        current_bbox=t.current_bbox.to_dict(),
                        trajectory=t.trajectory,
                        color=t.color,
                        color_confidence=t.color_confidence,
                        active=1 if t.active else 0,
                    )
                )
            if track_models:
                db.bulk_save_objects(track_models)

            # 2. Vehicle Attributes
            attr_models = []
            for a in vehicle_attributes:
                attr_models.append(
                    VehicleAttributeModel(
                        id=str(uuid.uuid4()),
                        video_id=video_id,
                        track_id=a.track_id,
                        object_class=a.object_class,
                        color=a.color,
                        confidence=a.confidence,
                        timestamp_seconds=a.timestamp,
                        bounding_box=a.bounding_box.to_dict(),
                        color_space_metrics=a.color_space_metrics or {},
                    )
                )
            if attr_models:
                db.bulk_save_objects(attr_models)

            # 3. Face Detections (Strictly anonymous bounding box regions)
            face_models = []
            for f in face_detections:
                face_models.append(
                    FaceDetectionModel(
                        id=str(uuid.uuid4()),
                        video_id=video_id,
                        track_id=f.track_id,
                        timestamp_seconds=f.timestamp,
                        confidence=f.confidence,
                        bbox_x1=f.bounding_box.x1,
                        bbox_y1=f.bounding_box.y1,
                        bbox_x2=f.bounding_box.x2,
                        bbox_y2=f.bounding_box.y2,
                    )
                )
            if face_models:
                db.bulk_save_objects(face_models)

            # 4. Security Events
            event_models = []
            for e in security_events:
                ev_id = e.event_id or str(uuid.uuid4())
                # If explicit test ID already exists in the database, assign unique ID to prevent UNIQUE constraint violation
                if db.query(SecurityEventModel.id).filter(SecurityEventModel.id == ev_id).first():
                    ev_id = str(uuid.uuid4())
                e.event_id = ev_id

                bbox_dict = None
                if e.bounding_box:
                    bbox_dict = e.bounding_box.to_dict() if hasattr(e.bounding_box, "to_dict") else dict(e.bounding_box)
                    if getattr(e, "person_track_id", None):
                        bbox_dict["person_track_id"] = e.person_track_id
                    if getattr(e, "object_track_id", None):
                        bbox_dict["object_track_id"] = e.object_track_id
                    if getattr(e, "object_class", None) and e.event_type == "POTENTIAL_THEFT":
                        bbox_dict["object_class"] = e.object_class
                    if getattr(e, "object_bounding_box", None):
                        bbox_dict["object_bbox"] = (
                            e.object_bounding_box.to_dict()
                            if hasattr(e.object_bounding_box, "to_dict")
                            else dict(e.object_bounding_box)
                        )
                    if getattr(e, "object_observation_timestamp", None) is not None:
                        bbox_dict["object_observation_timestamp"] = e.object_observation_timestamp
                    if getattr(e, "is_prior_object_observation", False):
                        bbox_dict["is_prior_object_observation"] = True

                e_meta = getattr(e, "incident_metadata", None) or {}
                if not isinstance(e_meta, dict):
                    e_meta = {}
                else:
                    e_meta = dict(e_meta)
                if getattr(e, "pattern_evidence_strength", None) is not None:
                    e_meta.setdefault("pattern_evidence_strength", e.pattern_evidence_strength)
                if getattr(e, "assessment_score", None) is not None:
                    e_meta.setdefault("assessment_score", e.assessment_score)
                if getattr(e, "validation_decision", None) is not None:
                    e_meta.setdefault("validation_decision", e.validation_decision)

                event_models.append(
                    SecurityEventModel(
                        id=ev_id,
                        video_id=video_id,
                        event_type=e.event_type,
                        severity=e.severity,
                        timestamp_seconds=e.timestamp,
                        duration_seconds=e.duration_seconds,
                        track_id=e.track_id,
                        object_class=e.object_class,
                        zone_name=e.zone_name,
                        confidence=e.confidence,
                        bounding_box=bbox_dict,
                        description=e.description,
                        observable_signals=e.observable_signals,
                        evidence_id=e.evidence_id,
                        detector_name=getattr(e, "detector_name", None),
                        detector_version=getattr(e, "detector_version", None),
                        category=getattr(e, "category", None),
                        human_verification_required=int(getattr(e, "human_verification_required", True)),
                        incident_metadata=e_meta,
                    )
                )
            if event_models:
                db.bulk_save_objects(event_models)

            # 5. Specialized Visual Observations (Phase 15)
            spec_models = []
            if specialized_observations:
                for so in specialized_observations:
                    bbox_dict = None
                    if hasattr(so, "bounding_box") and so.bounding_box:
                        bbox_dict = so.bounding_box.to_dict() if hasattr(so.bounding_box, "to_dict") else dict(so.bounding_box)
                    elif isinstance(so, dict) and so.get("bounding_box"):
                        bbox_dict = so["bounding_box"]

                    spec_id = getattr(so, "observation_id", None) or (so.get("observation_id") if isinstance(so, dict) else str(uuid.uuid4()))
                    det_name = getattr(so, "detector_name", None) or (so.get("detector_name") if isinstance(so, dict) else "specialized_detector")
                    det_ver = getattr(so, "detector_version", None) or (so.get("detector_version") if isinstance(so, dict) else "1.0.0")
                    cls_name = getattr(so, "class_name", None) or (so.get("class_name") if isinstance(so, dict) else "unknown")
                    ts = getattr(so, "timestamp", None) or (so.get("timestamp") if isinstance(so, dict) else 0.0)
                    conf = getattr(so, "confidence", None) or (so.get("confidence") if isinstance(so, dict) else 0.5)
                    ev_str = getattr(so, "evidence_strength", None) or (so.get("evidence_strength") if isinstance(so, dict) else conf)
                    v_stat = getattr(so, "validation_status", "RAW")
                    if hasattr(v_stat, "value"):
                        v_stat = v_stat.value
                    elif isinstance(so, dict):
                        v_stat = so.get("validation_status", "RAW")

                    metrics = getattr(so, "visual_metrics", None) or (so.get("visual_metrics") if isinstance(so, dict) else {})
                    ep_id = getattr(so, "episode_id", None) or (so.get("episode_id") if isinstance(so, dict) else None)

                    from ai.common.numeric import ensure_finite

                    spec_models.append(
                        SpecializedObservationModel(
                            id=spec_id,
                            video_id=video_id,
                            detector_name=det_name,
                            detector_version=det_ver,
                            class_name=cls_name,
                            timestamp_seconds=ensure_finite(ts, 0.0),
                            confidence=ensure_finite(conf, 0.0),
                            evidence_strength=ensure_finite(ev_str, 0.0),
                            validation_status=str(v_stat),
                            bounding_box=bbox_dict,
                            metrics=metrics,
                            episode_id=ep_id,
                        )
                    )
            if spec_models:
                db.bulk_save_objects(spec_models)

            # 6. Correlated Incidents (Phase 16)
            corr_models = []
            if correlated_incidents:
                from ai.common.numeric import ensure_finite
                for ci in correlated_incidents:
                    cid = getattr(ci, "incident_id", None) or (ci.get("incident_id") if isinstance(ci, dict) else str(uuid.uuid4()))
                    cat = getattr(ci, "incident_category", None) or (ci.get("incident_category") if isinstance(ci, dict) else "GENERAL")
                    if hasattr(cat, "value"):
                        cat = cat.value
                    cat = str(cat)
                    subcat = getattr(ci, "incident_subcategory", None) or (ci.get("incident_subcategory") if isinstance(ci, dict) else None)
                    if hasattr(subcat, "value"):
                        subcat = subcat.value
                    t_start = getattr(ci, "start_time", 0.0) if not isinstance(ci, dict) else ci.get("start_time", 0.0)
                    t_end = getattr(ci, "end_time", 0.0) if not isinstance(ci, dict) else ci.get("end_time", 0.0)
                    dur = getattr(ci, "duration", 0.0) if not isinstance(ci, dict) else ci.get("duration", 0.0)
                    primary_tracks = getattr(ci, "primary_track_ids", []) if not isinstance(ci, dict) else ci.get("primary_track_ids", [])
                    supp_tracks = getattr(ci, "supporting_track_ids", []) if not isinstance(ci, dict) else ci.get("supporting_track_ids", [])
                    obj_classes = getattr(ci, "involved_object_classes", []) if not isinstance(ci, dict) else ci.get("involved_object_classes", [])
                    src_candidates = getattr(ci, "source_candidate_ids", []) if not isinstance(ci, dict) else ci.get("source_candidate_ids", [])
                    src_detectors = getattr(ci, "source_detector_ids", []) if not isinstance(ci, dict) else ci.get("source_detector_ids", [])
                    supp_signals = getattr(ci, "supporting_signal_ids", []) if not isinstance(ci, dict) else ci.get("supporting_signal_ids", [])
                    evidence_ids = getattr(ci, "evidence_ids", []) if not isinstance(ci, dict) else ci.get("evidence_ids", [])
                    zone_ids = getattr(ci, "zone_ids", []) if not isinstance(ci, dict) else ci.get("zone_ids", [])
                    score = getattr(ci, "assessment_score", 0.0) if not isinstance(ci, dict) else ci.get("assessment_score", 0.0)
                    ev_strength = getattr(ci, "evidence_strength", 0.0) if not isinstance(ci, dict) else ci.get("evidence_strength", 0.0)
                    rel_rating = getattr(ci, "reliability_rating", "LOW") if not isinstance(ci, dict) else ci.get("reliability_rating", "LOW")
                    if hasattr(rel_rating, "value"):
                        rel_rating = rel_rating.value
                    val_decision = getattr(ci, "validation_decision", "REVIEW_REQUIRED") if not isinstance(ci, dict) else ci.get("validation_decision", "REVIEW_REQUIRED")
                    if hasattr(val_decision, "value"):
                        val_decision = val_decision.value
                    val_decision = str(val_decision)
                    neg_ev = getattr(ci, "negative_evidence", []) if not isinstance(ci, dict) else ci.get("negative_evidence", [])
                    ctx_factors = getattr(ci, "contextual_factors", {}) if not isinstance(ci, dict) else ci.get("contextual_factors", {})
                    storyline = getattr(ci, "storyline", "") if not isinstance(ci, dict) else ci.get("storyline", "")
                    provenance = getattr(ci, "provenance", {}) if not isinstance(ci, dict) else ci.get("provenance", {})

                    # Clear any existing record with this id to prevent unique constraint conflicts across runs
                    db.query(CorrelatedIncidentModel).filter(CorrelatedIncidentModel.id == cid).delete()

                    corr_models.append(
                        CorrelatedIncidentModel(
                            id=cid,
                            video_id=video_id,
                            incident_category=str(cat),
                            incident_subcategory=str(subcat) if subcat else None,
                            start_time=ensure_finite(t_start, 0.0),
                            end_time=ensure_finite(t_end, 0.0),
                            duration=ensure_finite(dur, 0.0),
                            primary_track_ids=list(primary_tracks),
                            supporting_track_ids=list(supp_tracks),
                            involved_object_classes=list(obj_classes),
                            source_candidate_ids=list(src_candidates),
                            source_detector_ids=list(src_detectors),
                            supporting_signal_ids=list(supp_signals),
                            evidence_ids=list(evidence_ids),
                            zone_ids=list(zone_ids),
                            assessment_score=ensure_finite(score, 0.0),
                            evidence_strength=ensure_finite(ev_strength, 0.0),
                            reliability_rating=str(rel_rating),
                            validation_decision=str(val_decision),
                            negative_evidence=list(neg_ev),
                            contextual_factors=dict(ctx_factors),
                            storyline=str(storyline),
                            provenance=dict(provenance),
                        )
                    )
            if corr_models:
                db.bulk_save_objects(corr_models)

            db.commit()
            logger.info(
                f"Saved intelligence records for {video_id}: "
                f"{len(track_models)} tracks, {len(attr_models)} attributes, "
                f"{len(face_models)} faces, {len(event_models)} security events, "
                f"{len(spec_models)} specialized observations, "
                f"{len(corr_models)} correlated incidents."
            )
        except Exception as exc:
            db.rollback()
            logger.error(f"Failed to persist intelligence results for {video_id}: {exc}")
            raise
        finally:
            db.close()

    def get_specialized_observations(
        self,
        video_id: str,
        class_name: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve specialized visual detection observations (fire, smoke, weapons, pose)."""
        session = db or SessionLocal()
        should_close = db is None
        try:
            q = session.query(SpecializedObservationModel).filter(SpecializedObservationModel.video_id == video_id)
            if class_name and class_name.lower() != "all":
                q = q.filter(SpecializedObservationModel.class_name.ilike(f"%{class_name}%"))
            obs = q.order_by(SpecializedObservationModel.timestamp_seconds.asc()).all()
            return [
                {
                    "id": o.id,
                    "video_id": o.video_id,
                    "event_id": o.event_id,
                    "episode_id": getattr(o, "episode_id", None),
                    "detector_name": o.detector_name,
                    "detector_version": o.detector_version,
                    "class_name": o.class_name,
                    "timestamp": round(float(o.timestamp_seconds), 4),
                    "timestamp_seconds": round(float(o.timestamp_seconds), 4),
                    "confidence": round(float(o.confidence), 4),
                    "evidence_strength": round(float(o.evidence_strength), 4),
                    "validation_status": o.validation_status,
                    "bounding_box": o.bounding_box,
                    "metrics": o.metrics,
                    "created_at": o.created_at.isoformat() if o.created_at else None,
                }
                for o in obs
            ]
        finally:
            if should_close:
                session.close()

    def get_tracks(self, video_id: str, object_class: Optional[str] = None) -> List[Dict[str, Any]]:
        db = SessionLocal()
        try:
            q = db.query(TrackModel).filter(TrackModel.video_id == video_id)
            if object_class:
                q = q.filter(TrackModel.object_class.ilike(object_class))
            rows = q.order_by(TrackModel.first_seen.asc()).all()
            return [
                {
                    "track_id": r.track_id,
                    "object_class": r.object_class,
                    "first_seen": r.first_seen,
                    "last_seen": r.last_seen,
                    "duration_seconds": r.duration_seconds,
                    "detection_count": r.detection_count,
                    "max_confidence": r.max_confidence,
                    "current_bbox": r.current_bbox,
                    "trajectory": r.trajectory,
                    "color": r.color,
                    "color_confidence": r.color_confidence,
                    "active": bool(r.active),
                }
                for r in rows
            ]
        finally:
            db.close()

    def get_vehicle_attributes(self, video_id: str, color: Optional[str] = None) -> List[Dict[str, Any]]:
        db = SessionLocal()
        try:
            q = db.query(VehicleAttributeModel).filter(VehicleAttributeModel.video_id == video_id)
            if color:
                q = q.filter(VehicleAttributeModel.color.ilike(color))
            rows = q.order_by(VehicleAttributeModel.timestamp_seconds.asc()).all()
            return [
                {
                    "id": r.id,
                    "track_id": r.track_id,
                    "object_class": r.object_class,
                    "color": r.color,
                    "confidence": r.confidence,
                    "timestamp": r.timestamp_seconds,
                    "bounding_box": r.bounding_box,
                    "color_space_metrics": r.color_space_metrics,
                }
                for r in rows
            ]
        finally:
            db.close()

    def get_face_detections(self, video_id: str) -> List[Dict[str, Any]]:
        db = SessionLocal()
        try:
            rows = (
                db.query(FaceDetectionModel)
                .filter(FaceDetectionModel.video_id == video_id)
                .order_by(FaceDetectionModel.timestamp_seconds.asc())
                .all()
            )
            return [
                {
                    "id": r.id,
                    "track_id": r.track_id,
                    "timestamp": r.timestamp_seconds,
                    "confidence": r.confidence,
                    "bounding_box": {
                        "x1": r.bbox_x1,
                        "y1": r.bbox_y1,
                        "x2": r.bbox_x2,
                        "y2": r.bbox_y2,
                    },
                }
                for r in rows
            ]
        finally:
            db.close()

    def get_security_events(
        self,
        video_id: str,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        db = SessionLocal()
        try:
            q = db.query(SecurityEventModel).filter(SecurityEventModel.video_id == video_id)
            if event_type:
                q = q.filter(SecurityEventModel.event_type == event_type)
            if severity:
                q = q.filter(SecurityEventModel.severity == severity)
            rows = q.order_by(SecurityEventModel.timestamp_seconds.asc()).all()
            results = []
            for r in rows:
                meta = r.incident_metadata if isinstance(getattr(r, "incident_metadata", None), dict) else {}
                val_dec = meta.get("validation_decision", "ACCEPTED")
                p_strength = meta.get("pattern_evidence_strength", r.confidence)
                a_score = meta.get("assessment_score", r.confidence)
                results.append({
                    "id": r.id,
                    "event_id": r.id,
                    "event_type": r.event_type,
                    "severity": r.severity,
                    "timestamp": r.timestamp_seconds,
                    "duration_seconds": r.duration_seconds,
                    "duration": r.duration_seconds,
                    "track_id": r.track_id,
                    "object_class": r.object_class,
                    "zone_name": r.zone_name,
                    "confidence": r.confidence,
                    "pattern_evidence_strength": round(float(p_strength), 4),
                    "assessment_score": round(float(a_score), 4),
                    "bounding_box": r.bounding_box,
                    "description": r.description,
                    "observable_signals": r.observable_signals,
                    "evidence_id": r.evidence_id,
                    "detector_name": getattr(r, "detector_name", None),
                    "detector_version": getattr(r, "detector_version", None),
                    "category": getattr(r, "category", None),
                    "human_verification_required": bool(getattr(r, "human_verification_required", 1)),
                    "validation_decision": val_dec,
                    "incident_metadata": r.incident_metadata,
                })
            return results
        finally:
            db.close()

    def get_correlated_incidents(
        self,
        video_id: str,
        category: Optional[str] = None,
        severity: Optional[str] = None,
        validation_decision: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query correlated incidents for a video with optional category/validation filtering.
        """
        db = SessionLocal()
        try:
            from sqlalchemy import func
            q = db.query(CorrelatedIncidentModel).filter(CorrelatedIncidentModel.video_id == video_id)
            if category:
                q = q.filter(func.lower(CorrelatedIncidentModel.incident_category) == category.lower())
            if validation_decision:
                q = q.filter(func.lower(CorrelatedIncidentModel.validation_decision) == validation_decision.lower())
            if severity:
                q = q.filter(func.lower(CorrelatedIncidentModel.reliability_rating) == severity.lower())
            rows = q.order_by(CorrelatedIncidentModel.start_time.asc()).all()
            return [
                {
                    "id": r.id,
                    "incident_id": r.id,
                    "video_id": r.video_id,
                    "incident_category": r.incident_category,
                    "incident_subcategory": r.incident_subcategory,
                    "start_time": r.start_time,
                    "end_time": r.end_time,
                    "duration": r.duration,
                    "primary_track_ids": r.primary_track_ids or [],
                    "supporting_track_ids": r.supporting_track_ids or [],
                    "involved_object_classes": r.involved_object_classes or [],
                    "source_candidate_ids": r.source_candidate_ids or [],
                    "source_detector_ids": r.source_detector_ids or [],
                    "supporting_signal_ids": r.supporting_signal_ids or [],
                    "evidence_ids": r.evidence_ids or [],
                    "zone_ids": r.zone_ids or [],
                    "assessment_score": r.assessment_score,
                    "evidence_strength": r.evidence_strength,
                    "reliability_rating": r.reliability_rating,
                    "validation_decision": r.validation_decision,
                    "negative_evidence": r.negative_evidence or [],
                    "contextual_factors": r.contextual_factors or {},
                    "storyline": r.storyline or "",
                    "provenance": r.provenance or {},
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                }
                for r in rows
            ]
        finally:
            db.close()

    def get_zones(self, video_id: Optional[str] = None) -> List[Dict[str, Any]]:
        db = SessionLocal()
        try:
            q = db.query(SecurityZoneModel)
            if video_id:
                q = q.filter((SecurityZoneModel.video_id == video_id) | (SecurityZoneModel.video_id == None))
            rows = q.order_by(SecurityZoneModel.created_at.asc()).all()
            return [
                {
                    "zone_id": r.id,
                    "name": r.name,
                    "polygon": r.polygon_coordinates,
                    "target_classes": r.target_classes,
                    "enabled": bool(r.enabled),
                }
                for r in rows
            ]
        finally:
            db.close()

    def save_zone(
        self,
        name: str,
        polygon: List[List[float]],
        target_classes: Optional[List[str]] = None,
        video_id: Optional[str] = None,
        zone_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        db = SessionLocal()
        try:
            if video_id and not db.query(VideoModel.id).filter(VideoModel.id == video_id).first():
                db.add(
                    VideoModel(
                        id=video_id,
                        original_filename=f"{video_id}.mp4",
                        storage_path=f"storage/uploads/{video_id}.mp4",
                        status="processed",
                    )
                )
                db.flush()

            zid = zone_id or f"ZONE-{uuid.uuid4().hex[:8]}"
            existing = db.query(SecurityZoneModel).filter(SecurityZoneModel.id == zid).first()
            if not existing:
                new_zone = SecurityZoneModel(
                    id=zid,
                    video_id=video_id,
                    name=name,
                    polygon_coordinates=polygon,
                    target_classes=target_classes or ["person"],
                    enabled=1,
                )
                db.add(new_zone)
            else:
                existing.name = name
                existing.polygon_coordinates = polygon
                existing.target_classes = target_classes or ["person"]
            db.commit()
            return {
                "zone_id": zid,
                "name": name,
                "polygon": polygon,
                "target_classes": target_classes or ["person"],
                "enabled": True,
            }
        except Exception as exc:
            db.rollback()
            logger.error(f"Failed to save security zone: {exc}")
            raise
        finally:
            db.close()

    def delete_zone(self, zone_id: str, video_id: Optional[str] = None) -> bool:
        db = SessionLocal()
        try:
            q = db.query(SecurityZoneModel).filter(SecurityZoneModel.id == zone_id)
            if video_id:
                q = q.filter(SecurityZoneModel.video_id == video_id)
            row = q.first()
            if row:
                db.delete(row)
                db.commit()
                return True
            return False
        except Exception as exc:
            db.rollback()
            logger.error(f"Failed to delete security zone {zone_id}: {exc}")
            raise
        finally:
            db.close()


_intelligence_repo: Optional[SecurityIntelligenceRepository] = None


def get_intelligence_repository() -> SecurityIntelligenceRepository:
    """Return singleton SecurityIntelligenceRepository."""
    global _intelligence_repo
    if _intelligence_repo is None:
        _intelligence_repo = SecurityIntelligenceRepository()
    return _intelligence_repo
