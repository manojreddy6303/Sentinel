"""
backend/app/services/case_service.py — Phase 19

Forensic Case Management & Investigation Workspace Service.

Enforces:
  - Case isolation: cross-case access prevention
  - Non-mutation of raw CCTV media (annotations are overlay metadata only)
  - Clear distinction between ANALYST NOTE and machine observations
  - Forensic reliability: REVIEW_REQUIRED <= 0.65 ceiling preserved
  - Zero biometric / identity inference: anonymous track IDs only
  - Provenance preservation across all timeline items, replay contexts, and exports
"""
from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import or_, and_, desc, asc
from sqlalchemy.orm import Session

try:
    from database.models import (
        CaseModel,
        CaseVideoModel,
        CaseCameraModel,
        CaseIncidentModel,
        CaseEvidenceModel,
        CaseBookmarkModel,
        CaseNoteModel,
        CaseAnnotationModel,
        CaseActivityModel,
        VideoModel,
        CameraSourceModel,
        CrossCameraAssociationModel,
        CorrelatedIncidentModel,
        SecurityEventModel,
        EventModel,
        GroupedEventModel,
        EvidenceModel,
        TrackModel,
    )
except ImportError:
    from database.models import (  # type: ignore
        CaseModel,
        CaseVideoModel,
        CaseCameraModel,
        CaseIncidentModel,
        CaseEvidenceModel,
        CaseBookmarkModel,
        CaseNoteModel,
        CaseAnnotationModel,
        CaseActivityModel,
        VideoModel,
        CameraSourceModel,
        CrossCameraAssociationModel,
        CorrelatedIncidentModel,
        SecurityEventModel,
        EventModel,
        GroupedEventModel,
        EvidenceModel,
        TrackModel,
    )

logger = logging.getLogger(__name__)

# Controlled vocabularies
VALID_STATUSES = {"OPEN", "INVESTIGATING", "REVIEW", "CLOSED"}
VALID_PRIORITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
VALID_ANNOTATION_TYPES = {"POINT", "REGION", "TIMESTAMP_MARKER", "TEXT_NOTE", "INCIDENT_MARKER"}
VALID_ASSOCIATED_TYPES = {"CASE", "INCIDENT", "EVIDENCE", "TRACK", "TIMESTAMP", "CAMERA"}

_SAFE_TEXT_RE = re.compile(r"[^a-zA-Z0-9 _\-\.,:()/\[\]'\"?!\n\r]+")


def _sanitize(text: Optional[str], max_len: int = 1000) -> Optional[str]:
    if text is None:
        return None
    cleaned = _SAFE_TEXT_RE.sub("", text.strip())
    return cleaned[:max_len] or None


class CaseNotFoundError(Exception):
    pass


class CaseEntityNotFoundError(Exception):
    pass


class CaseIsolationError(Exception):
    pass


class CaseValidationError(Exception):
    pass


class CaseService:
    """Service layer managing forensic cases and investigation workflows."""

    # ------------------------------------------------------------------
    # 1. Case Lifecycle & CRUD
    # ------------------------------------------------------------------

    def generate_case_number(self, db: Session) -> str:
        """Generate human-readable, unique case number: CASE-YYYYMMDD-XXXX."""
        today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        short_hex = uuid.uuid4().hex[:4].upper()
        case_num = f"CASE-{today_str}-{short_hex}"
        # Ensure uniqueness
        while db.query(CaseModel).filter(CaseModel.case_number == case_num).first():
            short_hex = uuid.uuid4().hex[:4].upper()
            case_num = f"CASE-{today_str}-{short_hex}"
        return case_num

    def create_case(
        self,
        db: Session,
        title: str,
        description: Optional[str] = None,
        priority: str = "MEDIUM",
        assigned_investigator: Optional[str] = None,
        tags: Optional[List[str]] = None,
        summary: Optional[str] = None,
        initial_video_ids: Optional[List[str]] = None,
        initial_camera_ids: Optional[List[str]] = None,
    ) -> CaseModel:
        """Create a new forensic case and optionally associate initial videos/cameras."""
        p_upper = (priority or "MEDIUM").upper()
        if p_upper not in VALID_PRIORITIES:
            raise CaseValidationError(f"Invalid priority '{priority}'. Must be one of: {sorted(VALID_PRIORITIES)}")

        cleaned_title = _sanitize(title, max_len=255)
        if not cleaned_title:
            raise CaseValidationError("Case title cannot be empty.")

        case_num = self.generate_case_number(db)
        case = CaseModel(
            case_number=case_num,
            title=cleaned_title,
            description=_sanitize(description, max_len=2000),
            status="OPEN",
            priority=p_upper,
            assigned_investigator=_sanitize(assigned_investigator, max_len=100) or "Investigator",
            tags=[_sanitize(t, max_len=50) for t in (tags or []) if _sanitize(t, max_len=50)],
            summary=_sanitize(summary, max_len=5000),
        )
        db.add(case)
        db.commit()
        db.refresh(case)

        # Log activity
        self.log_activity(
            db,
            case_id=case.id,
            action_type="CASE_CREATED",
            description=f"Case {case_num} '{case.title}' created with priority {p_upper}.",
            details={"case_number": case_num, "priority": p_upper},
            actor=case.assigned_investigator,
        )

        # Link initial videos if provided
        if initial_video_ids:
            for vid in initial_video_ids:
                try:
                    self.link_video(db, case.id, vid, actor=case.assigned_investigator)
                except Exception as exc:
                    logger.warning("Failed linking initial video %s to case %s: %s", vid, case.id, exc)

        # Link initial cameras if provided
        if initial_camera_ids:
            for cid in initial_camera_ids:
                try:
                    self.link_camera(db, case.id, cid, actor=case.assigned_investigator)
                except Exception as exc:
                    logger.warning("Failed linking initial camera %s to case %s: %s", cid, case.id, exc)

        db.refresh(case)
        return case

    def get_case(self, db: Session, case_id: str) -> CaseModel:
        """Retrieve a case by ID."""
        case = db.query(CaseModel).filter(CaseModel.id == case_id).first()
        if not case:
            raise CaseNotFoundError(f"Case '{case_id}' not found.")
        return case

    def list_cases(
        self,
        db: Session,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        tag: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[CaseModel]:
        """List cases with optional filters, ordered by updated_at descending."""
        q = db.query(CaseModel)
        if status:
            s_upper = status.upper()
            if s_upper in VALID_STATUSES:
                q = q.filter(CaseModel.status == s_upper)
        if priority:
            p_upper = priority.upper()
            if p_upper in VALID_PRIORITIES:
                q = q.filter(CaseModel.priority == p_upper)
        if search:
            s_clean = f"%{search.strip()}%"
            q = q.filter(
                or_(
                    CaseModel.title.ilike(s_clean),
                    CaseModel.case_number.ilike(s_clean),
                    CaseModel.description.ilike(s_clean),
                )
            )
        q = q.order_by(desc(CaseModel.updated_at))
        cases = q.offset(offset).limit(min(limit, 200)).all()

        if tag:
            t_clean = tag.strip().lower()
            cases = [c for c in cases if c.tags and any(t_clean in str(tg).lower() for tg in c.tags)]

        return cases

    def count_cases(
        self,
        db: Session,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        tag: Optional[str] = None,
        search: Optional[str] = None,
    ) -> int:
        """Count total matching cases across Sentinel."""
        q = db.query(CaseModel)
        if status:
            s_upper = status.upper()
            if s_upper in VALID_STATUSES:
                q = q.filter(CaseModel.status == s_upper)
        if priority:
            p_upper = priority.upper()
            if p_upper in VALID_PRIORITIES:
                q = q.filter(CaseModel.priority == p_upper)
        if search:
            s_clean = f"%{search.strip()}%"
            q = q.filter(
                or_(
                    CaseModel.title.ilike(s_clean),
                    CaseModel.case_number.ilike(s_clean),
                    CaseModel.description.ilike(s_clean),
                )
            )
        if tag:
            t_clean = tag.strip().lower()
            cases = q.all()
            return len([c for c in cases if c.tags and any(t_clean in str(tg).lower() for tg in c.tags)])
        return q.count()

    def update_case(
        self,
        db: Session,
        case_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        assigned_investigator: Optional[str] = None,
        tags: Optional[List[str]] = None,
        summary: Optional[str] = None,
        actor: str = "Investigator",
    ) -> CaseModel:
        """Update case attributes and log activity."""
        case = self.get_case(db, case_id)
        changes: Dict[str, Any] = {}

        if title is not None:
            c_title = _sanitize(title, max_len=255)
            if c_title:
                changes["title"] = (case.title, c_title)
                case.title = c_title

        if description is not None:
            case.description = _sanitize(description, max_len=2000)
            changes["description"] = "updated"

        if status is not None:
            s_upper = status.upper()
            if s_upper not in VALID_STATUSES:
                raise CaseValidationError(f"Invalid status '{status}'. Must be one of: {sorted(VALID_STATUSES)}")
            if case.status != s_upper:
                changes["status"] = (case.status, s_upper)
                case.status = s_upper

        if priority is not None:
            p_upper = priority.upper()
            if p_upper not in VALID_PRIORITIES:
                raise CaseValidationError(f"Invalid priority '{priority}'. Must be one of: {sorted(VALID_PRIORITIES)}")
            if case.priority != p_upper:
                changes["priority"] = (case.priority, p_upper)
                case.priority = p_upper

        if assigned_investigator is not None:
            case.assigned_investigator = _sanitize(assigned_investigator, max_len=100) or case.assigned_investigator
            changes["assigned_investigator"] = case.assigned_investigator

        if tags is not None:
            case.tags = [_sanitize(t, max_len=50) for t in tags if _sanitize(t, max_len=50)]
            changes["tags"] = case.tags

        if summary is not None:
            case.summary = _sanitize(summary, max_len=5000)
            changes["summary"] = "updated"

        case.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(case)

        if changes:
            desc_text = f"Case updated: {', '.join(changes.keys())} modified."
            self.log_activity(
                db,
                case_id=case.id,
                action_type="CASE_UPDATED",
                description=desc_text,
                details=changes,
                actor=actor,
            )

        return case

    def delete_case(self, db: Session, case_id: str) -> None:
        """Delete a case and cascade delete all its associations, bookmarks, notes, annotations."""
        case = self.get_case(db, case_id)
        db.delete(case)
        db.commit()
        logger.info("Deleted case id=%s case_number=%s", case_id, case.case_number)

    # ------------------------------------------------------------------
    # 2. Entity Associations (Videos, Cameras, Incidents)
    # ------------------------------------------------------------------

    def link_video(
        self,
        db: Session,
        case_id: str,
        video_id: str,
        notes: Optional[str] = None,
        actor: str = "Investigator",
    ) -> CaseVideoModel:
        """Link a processed video to a case."""
        case = self.get_case(db, case_id)
        video = db.query(VideoModel).filter(VideoModel.id == video_id).first()
        if not video:
            raise CaseEntityNotFoundError(f"Video '{video_id}' not found.")

        existing = (
            db.query(CaseVideoModel)
            .filter(CaseVideoModel.case_id == case_id, CaseVideoModel.video_id == video_id)
            .first()
        )
        if existing:
            return existing

        case_vid = CaseVideoModel(
            case_id=case_id,
            video_id=video_id,
            notes=_sanitize(notes, max_len=500),
        )
        db.add(case_vid)
        case.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(case_vid)

        self.log_activity(
            db,
            case_id=case_id,
            action_type="VIDEO_LINKED",
            description=f"Video '{video.original_filename}' ({video_id}) linked to case.",
            details={"video_id": video_id, "filename": video.original_filename},
            actor=actor,
        )
        return case_vid

    def unlink_video(
        self,
        db: Session,
        case_id: str,
        video_id: str,
        actor: str = "Investigator",
    ) -> None:
        """Unlink a video from a case."""
        case = self.get_case(db, case_id)
        case_vid = (
            db.query(CaseVideoModel)
            .filter(CaseVideoModel.case_id == case_id, CaseVideoModel.video_id == video_id)
            .first()
        )
        if not case_vid:
            raise CaseEntityNotFoundError(f"Video '{video_id}' is not linked to case '{case_id}'.")

        db.delete(case_vid)
        case.updated_at = datetime.now(timezone.utc)
        db.commit()

        self.log_activity(
            db,
            case_id=case_id,
            action_type="VIDEO_UNLINKED",
            description=f"Video '{video_id}' unlinked from case.",
            details={"video_id": video_id},
            actor=actor,
        )

    def get_linked_videos(self, db: Session, case_id: str) -> List[Dict[str, Any]]:
        """List all videos linked to a case with summary metadata."""
        self.get_case(db, case_id)
        links = db.query(CaseVideoModel).filter(CaseVideoModel.case_id == case_id).all()
        results = []
        for l in links:
            v = l.video
            if v:
                results.append({
                    "link_id": l.id,
                    "video_id": v.id,
                    "filename": v.original_filename,
                    "duration_seconds": round(v.duration_seconds or 0.0, 2),
                    "fps": round(v.fps or 0.0, 2),
                    "status": v.status,
                    "uploaded_at": v.uploaded_at.isoformat() if v.uploaded_at else None,
                    "link_notes": l.notes,
                    "added_at": l.added_at.isoformat() if l.added_at else None,
                })
        return results

    def link_camera(
        self,
        db: Session,
        case_id: str,
        camera_id: str,
        clock_offset_seconds: float = 0.0,
        notes: Optional[str] = None,
        actor: str = "Investigator",
    ) -> CaseCameraModel:
        """Link a camera source to a case."""
        case = self.get_case(db, case_id)
        cam = db.query(CameraSourceModel).filter(CameraSourceModel.id == camera_id).first()
        if not cam:
            raise CaseEntityNotFoundError(f"Camera source '{camera_id}' not found.")

        # Ensure camera's video is linked to the case
        if not db.query(CaseVideoModel).filter(CaseVideoModel.case_id == case_id, CaseVideoModel.video_id == cam.video_id).first():
            self.link_video(db, case_id, cam.video_id, notes=f"Auto-linked for camera {cam.camera_label}", actor=actor)

        existing = (
            db.query(CaseCameraModel)
            .filter(CaseCameraModel.case_id == case_id, CaseCameraModel.camera_id == camera_id)
            .first()
        )
        if existing:
            existing.clock_offset_seconds = clock_offset_seconds
            if notes:
                existing.notes = _sanitize(notes, max_len=500)
            db.commit()
            return existing

        case_cam = CaseCameraModel(
            case_id=case_id,
            camera_id=camera_id,
            clock_offset_seconds=float(clock_offset_seconds),
            notes=_sanitize(notes, max_len=500),
        )
        db.add(case_cam)
        case.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(case_cam)

        self.log_activity(
            db,
            case_id=case_id,
            action_type="CAMERA_LINKED",
            description=f"Camera '{cam.camera_label}' ({camera_id}) linked to case with clock offset {clock_offset_seconds}s.",
            details={"camera_id": camera_id, "camera_label": cam.camera_label, "clock_offset": clock_offset_seconds},
            actor=actor,
        )
        return case_cam

    def unlink_camera(
        self,
        db: Session,
        case_id: str,
        camera_id: str,
        actor: str = "Investigator",
    ) -> None:
        """Unlink a camera from a case."""
        case = self.get_case(db, case_id)
        case_cam = (
            db.query(CaseCameraModel)
            .filter(CaseCameraModel.case_id == case_id, CaseCameraModel.camera_id == camera_id)
            .first()
        )
        if not case_cam:
            raise CaseEntityNotFoundError(f"Camera '{camera_id}' is not linked to case '{case_id}'.")

        db.delete(case_cam)
        case.updated_at = datetime.now(timezone.utc)
        db.commit()

        self.log_activity(
            db,
            case_id=case_id,
            action_type="CAMERA_UNLINKED",
            description=f"Camera '{camera_id}' unlinked from case.",
            details={"camera_id": camera_id},
            actor=actor,
        )

    def get_linked_cameras(self, db: Session, case_id: str) -> List[Dict[str, Any]]:
        """List all cameras linked to a case."""
        self.get_case(db, case_id)
        links = db.query(CaseCameraModel).filter(CaseCameraModel.case_id == case_id).all()
        results = []
        for l in links:
            c = l.camera
            if c:
                results.append({
                    "link_id": l.id,
                    "camera_id": c.id,
                    "session_id": c.session_id,
                    "camera_label": c.camera_label,
                    "position_hint": c.position_hint,
                    "field_of_view_hint": c.field_of_view_hint,
                    "adjacency_hints": c.adjacency_hints or [],
                    "video_id": c.video_id,
                    "clock_offset_seconds": l.clock_offset_seconds,
                    "notes": l.notes,
                    "added_at": l.added_at.isoformat() if l.added_at else None,
                })
        return results

    def link_incident(
        self,
        db: Session,
        case_id: str,
        incident_id: str,
        incident_type: str = "CORRELATED",
        notes: Optional[str] = None,
        actor: str = "Investigator",
    ) -> CaseIncidentModel:
        """Link an incident to a case after verifying video ownership."""
        case = self.get_case(db, case_id)

        # Verify incident exists and get its video_id
        video_id = None
        if incident_type == "CORRELATED":
            corr = db.query(CorrelatedIncidentModel).filter(CorrelatedIncidentModel.id == incident_id).first()
            if not corr:
                raise CaseEntityNotFoundError(f"Correlated incident '{incident_id}' not found.")
            video_id = corr.video_id
        else:
            sec = db.query(SecurityEventModel).filter(SecurityEventModel.id == incident_id).first()
            if not sec:
                raise CaseEntityNotFoundError(f"Security event '{incident_id}' not found.")
            video_id = sec.video_id

        # Ensure video is linked to case
        if not db.query(CaseVideoModel).filter(CaseVideoModel.case_id == case_id, CaseVideoModel.video_id == video_id).first():
            self.link_video(db, case_id, video_id, notes="Auto-linked for incident", actor=actor)

        existing = (
            db.query(CaseIncidentModel)
            .filter(CaseIncidentModel.case_id == case_id, CaseIncidentModel.incident_id == incident_id)
            .first()
        )
        if existing:
            return existing

        case_inc = CaseIncidentModel(
            case_id=case_id,
            incident_id=incident_id,
            incident_type=incident_type,
            notes=_sanitize(notes, max_len=500),
        )
        db.add(case_inc)
        case.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(case_inc)

        self.log_activity(
            db,
            case_id=case_id,
            action_type="INCIDENT_LINKED",
            description=f"Incident '{incident_id}' ({incident_type}) linked to case.",
            details={"incident_id": incident_id, "incident_type": incident_type},
            actor=actor,
        )
        return case_inc

    def unlink_incident(
        self,
        db: Session,
        case_id: str,
        incident_id: str,
        actor: str = "Investigator",
    ) -> None:
        """Unlink an incident from a case."""
        case = self.get_case(db, case_id)
        case_inc = (
            db.query(CaseIncidentModel)
            .filter(CaseIncidentModel.case_id == case_id, CaseIncidentModel.incident_id == incident_id)
            .first()
        )
        if not case_inc:
            raise CaseEntityNotFoundError(f"Incident '{incident_id}' is not explicitly linked to case '{case_id}'.")

        db.delete(case_inc)
        case.updated_at = datetime.now(timezone.utc)
        db.commit()

        self.log_activity(
            db,
            case_id=case_id,
            action_type="INCIDENT_UNLINKED",
            description=f"Incident '{incident_id}' unlinked from case.",
            details={"incident_id": incident_id},
            actor=actor,
        )

    def get_linked_incidents(self, db: Session, case_id: str) -> List[Dict[str, Any]]:
        """List explicitly linked incidents for a case with summary metadata."""
        self.get_case(db, case_id)
        links = db.query(CaseIncidentModel).filter(CaseIncidentModel.case_id == case_id).all()
        results = []
        for l in links:
            inc_meta = {}
            if l.incident_type == "CORRELATED":
                corr = db.query(CorrelatedIncidentModel).filter(CorrelatedIncidentModel.id == l.incident_id).first()
                if corr:
                    v = corr.video
                    inc_meta = {
                        "category": corr.incident_category,
                        "subcategory": corr.incident_subcategory,
                        "title": f"{corr.incident_category.upper()}: {corr.incident_subcategory or 'Anomaly'}",
                        "start_time": corr.start_time,
                        "end_time": corr.end_time,
                        "assessment_score": corr.assessment_score,
                        "validation_decision": corr.validation_decision,
                        "storyline": corr.storyline,
                        "video_id": corr.video_id,
                        "video_filename": v.original_filename if v else None,
                    }
            else:
                sec = db.query(SecurityEventModel).filter(SecurityEventModel.id == l.incident_id).first()
                if sec:
                    v = sec.video
                    inc_meta = {
                        "category": sec.category or sec.event_type,
                        "subcategory": sec.event_type,
                        "title": f"SECURITY EVENT: {sec.event_type}",
                        "start_time": sec.timestamp_seconds,
                        "end_time": sec.timestamp_seconds,
                        "assessment_score": sec.confidence,
                        "validation_decision": "VALIDATED",
                        "storyline": sec.detector_name or "Event",
                        "video_id": sec.video_id,
                        "video_filename": v.original_filename if v else None,
                    }
            results.append({
                "link_id": l.id,
                "incident_id": l.incident_id,
                "incident_type": l.incident_type,
                "notes": l.notes,
                "added_at": l.added_at.isoformat() if l.added_at else None,
                **inc_meta
            })
        return results

    def link_evidence(
        self,
        db: Session,
        case_id: str,
        evidence_id: str,
        notes: Optional[str] = None,
        actor: str = "Investigator",
    ) -> CaseEvidenceModel:
        """Link a preserved forensic evidence artifact to a case."""
        case = self.get_case(db, case_id)
        ev = db.query(EvidenceModel).filter(EvidenceModel.id == evidence_id).first()
        if not ev:
            raise CaseEntityNotFoundError(f"Evidence '{evidence_id}' not found.")

        # Ensure evidence's video is linked to the case
        if not db.query(CaseVideoModel).filter(CaseVideoModel.case_id == case_id, CaseVideoModel.video_id == ev.video_id).first():
            self.link_video(db, case_id, ev.video_id, notes=f"Auto-linked for evidence {ev.object_class}", actor=actor)

        existing = (
            db.query(CaseEvidenceModel)
            .filter(CaseEvidenceModel.case_id == case_id, CaseEvidenceModel.evidence_id == evidence_id)
            .first()
        )
        if existing:
            return existing

        case_ev = CaseEvidenceModel(
            case_id=case_id,
            evidence_id=evidence_id,
            notes=_sanitize(notes, max_len=500),
        )
        db.add(case_ev)
        case.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(case_ev)

        self.log_activity(
            db,
            case_id=case_id,
            action_type="EVIDENCE_LINKED",
            description=f"Evidence '{evidence_id}' ({ev.object_class} @ {ev.timestamp_seconds:.1f}s) linked to case.",
            details={"evidence_id": evidence_id, "object_class": ev.object_class, "timestamp": ev.timestamp_seconds},
            actor=actor,
        )
        return case_ev

    def unlink_evidence(
        self,
        db: Session,
        case_id: str,
        evidence_id: str,
        actor: str = "Investigator",
    ) -> None:
        """Unlink an evidence item from a case."""
        case = self.get_case(db, case_id)
        case_ev = (
            db.query(CaseEvidenceModel)
            .filter(CaseEvidenceModel.case_id == case_id, CaseEvidenceModel.evidence_id == evidence_id)
            .first()
        )
        if not case_ev:
            raise CaseEntityNotFoundError(f"Evidence '{evidence_id}' is not linked to case '{case_id}'.")

        db.delete(case_ev)
        case.updated_at = datetime.now(timezone.utc)
        db.commit()

        self.log_activity(
            db,
            case_id=case_id,
            action_type="EVIDENCE_UNLINKED",
            description=f"Evidence '{evidence_id}' unlinked from case.",
            details={"evidence_id": evidence_id},
            actor=actor,
        )

    def get_linked_evidence(self, db: Session, case_id: str) -> List[Dict[str, Any]]:
        """List all evidence explicitly linked to a case."""
        self.get_case(db, case_id)
        links = db.query(CaseEvidenceModel).filter(CaseEvidenceModel.case_id == case_id).all()
        results = []
        for l in links:
            ev = l.evidence
            if ev:
                results.append({
                    "link_id": l.id,
                    "evidence_id": ev.id,
                    "video_id": ev.video_id,
                    "source_video_name": ev.source_video_name,
                    "object_class": ev.object_class,
                    "confidence": ev.confidence,
                    "timestamp_seconds": ev.timestamp_seconds,
                    "evidence_type": ev.evidence_type,
                    "validation_status": getattr(ev, "validation_status", "VALID") or "VALID",
                    "snapshot_url": f"/api/evidence/{ev.id}/snapshot" if ev.snapshot_path else None,
                    "annotated_url": f"/api/evidence/{ev.id}/annotated" if ev.annotated_snapshot_path else None,
                    "clip_url": f"/api/evidence/{ev.id}/clip" if ev.clip_path else None,
                    "notes": l.notes,
                    "added_at": l.added_at.isoformat() if l.added_at else None,
                })
        return results

    def get_case_incidents(self, db: Session, case_id: str) -> List[Dict[str, Any]]:
        """
        List all incidents available in the case (both explicitly linked and
        those belonging to linked videos).
        """
        self.get_case(db, case_id)
        video_ids = [
            cv.video_id for cv in db.query(CaseVideoModel).filter(CaseVideoModel.case_id == case_id).all()
        ]
        if not video_ids:
            return []

        corrs = (
            db.query(CorrelatedIncidentModel)
            .filter(CorrelatedIncidentModel.video_id.in_(video_ids))
            .order_by(CorrelatedIncidentModel.start_time.asc())
            .all()
        )
        results = []
        for c in corrs:
            results.append({
                "incident_id": c.id,
                "video_id": c.video_id,
                "type": "CORRELATED",
                "category": c.incident_category,
                "subcategory": c.incident_subcategory,
                "start_time": round(c.start_time, 2),
                "end_time": round(c.end_time, 2),
                "duration": round(c.duration, 2),
                "assessment_score": round(c.assessment_score, 4),
                "evidence_strength": round(c.evidence_strength, 4),
                "reliability_rating": c.reliability_rating,
                "validation_decision": c.validation_decision,
                "storyline": c.storyline or "",
                "primary_track_ids": c.primary_track_ids or [],
            })
        return results

    # ------------------------------------------------------------------
    # 3. Bookmarks
    # ------------------------------------------------------------------

    def create_bookmark(
        self,
        db: Session,
        case_id: str,
        video_id: str,
        timestamp_seconds: float,
        title: str,
        description: Optional[str] = None,
        camera_id: Optional[str] = None,
        linked_incident_id: Optional[str] = None,
        linked_track_id: Optional[str] = None,
        linked_evidence_id: Optional[str] = None,
        author: str = "Investigator",
    ) -> CaseBookmarkModel:
        """Create an investigator bookmark pinning a key timestamp."""
        case = self.get_case(db, case_id)
        video = db.query(VideoModel).filter(VideoModel.id == video_id).first()
        if not video:
            raise CaseEntityNotFoundError(f"Video '{video_id}' not found.")

        # Auto-link video to case if not already linked
        if not db.query(CaseVideoModel).filter(CaseVideoModel.case_id == case_id, CaseVideoModel.video_id == video_id).first():
            self.link_video(db, case_id, video_id, notes="Auto-linked for bookmark", actor=author)

        cleaned_title = _sanitize(title, max_len=255)
        if not cleaned_title:
            raise CaseValidationError("Bookmark title cannot be empty.")

        bm = CaseBookmarkModel(
            case_id=case_id,
            video_id=video_id,
            camera_id=camera_id,
            timestamp_seconds=round(max(0.0, float(timestamp_seconds)), 2),
            title=cleaned_title,
            description=_sanitize(description, max_len=1000),
            linked_incident_id=linked_incident_id,
            linked_track_id=linked_track_id,
            linked_evidence_id=linked_evidence_id,
            author=_sanitize(author, max_len=100) or "Investigator",
        )
        db.add(bm)
        case.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(bm)

        self.log_activity(
            db,
            case_id=case_id,
            action_type="BOOKMARK_CREATED",
            description=f"Bookmark '{cleaned_title}' created at {bm.timestamp_seconds}s in video '{video.original_filename}'.",
            details={"bookmark_id": bm.id, "timestamp": bm.timestamp_seconds, "video_id": video_id},
            actor=author,
        )
        return bm

    def list_bookmarks(self, db: Session, case_id: str, video_id: Optional[str] = None) -> List[CaseBookmarkModel]:
        """List bookmarks in a case, optionally filtered by video."""
        self.get_case(db, case_id)
        q = db.query(CaseBookmarkModel).filter(CaseBookmarkModel.case_id == case_id)
        if video_id:
            q = q.filter(CaseBookmarkModel.video_id == video_id)
        return q.order_by(CaseBookmarkModel.timestamp_seconds.asc()).all()

    def update_bookmark(
        self,
        db: Session,
        case_id: str,
        bookmark_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        timestamp_seconds: Optional[float] = None,
    ) -> CaseBookmarkModel:
        """Update a bookmark belonging to the case (enforcing isolation)."""
        self.get_case(db, case_id)
        bm = (
            db.query(CaseBookmarkModel)
            .filter(CaseBookmarkModel.id == bookmark_id, CaseBookmarkModel.case_id == case_id)
            .first()
        )
        if not bm:
            raise CaseIsolationError(f"Bookmark '{bookmark_id}' not found in case '{case_id}'.")

        if title is not None:
            c_title = _sanitize(title, max_len=255)
            if c_title:
                bm.title = c_title
        if description is not None:
            bm.description = _sanitize(description, max_len=1000)
        if timestamp_seconds is not None:
            bm.timestamp_seconds = round(max(0.0, float(timestamp_seconds)), 2)

        bm.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(bm)
        return bm

    def delete_bookmark(
        self,
        db: Session,
        case_id: str,
        bookmark_id: str,
        actor: str = "Investigator",
    ) -> None:
        """Delete a bookmark within the case."""
        self.get_case(db, case_id)
        bm = (
            db.query(CaseBookmarkModel)
            .filter(CaseBookmarkModel.id == bookmark_id, CaseBookmarkModel.case_id == case_id)
            .first()
        )
        if not bm:
            raise CaseIsolationError(f"Bookmark '{bookmark_id}' not found in case '{case_id}'.")

        title = bm.title
        db.delete(bm)
        db.commit()

        self.log_activity(
            db,
            case_id=case_id,
            action_type="BOOKMARK_DELETED",
            description=f"Bookmark '{title}' ({bookmark_id}) deleted.",
            details={"bookmark_id": bookmark_id},
            actor=actor,
        )

    # ------------------------------------------------------------------
    # 4. Investigator Notes
    # ------------------------------------------------------------------

    def create_note(
        self,
        db: Session,
        case_id: str,
        content: str,
        author: str = "Investigator",
        associated_type: str = "CASE",
        associated_id: Optional[str] = None,
        timestamp_seconds: Optional[float] = None,
    ) -> CaseNoteModel:
        """
        Add an investigator note to a case or an entity within it.
        SAFETY: Always classified as ANALYST_NOTE.
        """
        case = self.get_case(db, case_id)
        c_content = _sanitize(content, max_len=4000)
        if not c_content:
            raise CaseValidationError("Note content cannot be empty.")

        a_type = (associated_type or "CASE").upper()
        if a_type not in VALID_ASSOCIATED_TYPES:
            raise CaseValidationError(f"Invalid associated_type '{associated_type}'. Must be one of: {sorted(VALID_ASSOCIATED_TYPES)}")

        note = CaseNoteModel(
            case_id=case_id,
            content=c_content,
            author=_sanitize(author, max_len=100) or "Investigator",
            associated_type=a_type,
            associated_id=associated_id,
            timestamp_seconds=round(timestamp_seconds, 2) if timestamp_seconds is not None else None,
            note_classification="ANALYST_NOTE",
        )
        db.add(note)
        case.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(note)

        self.log_activity(
            db,
            case_id=case_id,
            action_type="NOTE_ADDED",
            description=f"Analyst note added to {a_type} by {note.author}.",
            details={"note_id": note.id, "associated_type": a_type, "associated_id": associated_id},
            actor=note.author,
        )
        return note

    def list_notes(
        self,
        db: Session,
        case_id: str,
        associated_type: Optional[str] = None,
        associated_id: Optional[str] = None,
    ) -> List[CaseNoteModel]:
        """List investigator notes in a case, optionally filtered by association."""
        self.get_case(db, case_id)
        q = db.query(CaseNoteModel).filter(CaseNoteModel.case_id == case_id)
        if associated_type:
            q = q.filter(CaseNoteModel.associated_type == associated_type.upper())
        if associated_id:
            q = q.filter(CaseNoteModel.associated_id == associated_id)
        return q.order_by(CaseNoteModel.created_at.desc()).all()

    def update_note(
        self,
        db: Session,
        case_id: str,
        note_id: str,
        content: str,
    ) -> CaseNoteModel:
        """Update an investigator note content."""
        self.get_case(db, case_id)
        note = (
            db.query(CaseNoteModel)
            .filter(CaseNoteModel.id == note_id, CaseNoteModel.case_id == case_id)
            .first()
        )
        if not note:
            raise CaseIsolationError(f"Note '{note_id}' not found in case '{case_id}'.")

        c_content = _sanitize(content, max_len=4000)
        if not c_content:
            raise CaseValidationError("Note content cannot be empty.")

        note.content = c_content
        note.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(note)
        return note

    def delete_note(
        self,
        db: Session,
        case_id: str,
        note_id: str,
        actor: str = "Investigator",
    ) -> None:
        """Delete an investigator note within the case."""
        self.get_case(db, case_id)
        note = (
            db.query(CaseNoteModel)
            .filter(CaseNoteModel.id == note_id, CaseNoteModel.case_id == case_id)
            .first()
        )
        if not note:
            raise CaseIsolationError(f"Note '{note_id}' not found in case '{case_id}'.")

        db.delete(note)
        db.commit()

        self.log_activity(
            db,
            case_id=case_id,
            action_type="NOTE_DELETED",
            description=f"Analyst note ({note_id}) deleted.",
            details={"note_id": note_id},
            actor=actor,
        )

    # ------------------------------------------------------------------
    # 5. Visual Overlay Annotations
    # ------------------------------------------------------------------

    def create_annotation(
        self,
        db: Session,
        case_id: str,
        video_id: str,
        timestamp_seconds: float,
        annotation_type: str = "REGION",
        data: Optional[Dict[str, Any]] = None,
        author: str = "Investigator",
        camera_id: Optional[str] = None,
        end_timestamp_seconds: Optional[float] = None,
    ) -> CaseAnnotationModel:
        """
        Create a visual overlay annotation.
        SAFETY: Saved strictly as metadata — NEVER modifies raw CCTV media.
        """
        case = self.get_case(db, case_id)
        video = db.query(VideoModel).filter(VideoModel.id == video_id).first()
        if not video:
            raise CaseEntityNotFoundError(f"Video '{video_id}' not found.")

        # Auto-link video if needed
        if not db.query(CaseVideoModel).filter(CaseVideoModel.case_id == case_id, CaseVideoModel.video_id == video_id).first():
            self.link_video(db, case_id, video_id, notes="Auto-linked for annotation", actor=author)

        a_type = (annotation_type or "REGION").upper()
        if a_type not in VALID_ANNOTATION_TYPES:
            raise CaseValidationError(f"Invalid annotation_type '{annotation_type}'. Must be one of: {sorted(VALID_ANNOTATION_TYPES)}")

        ann = CaseAnnotationModel(
            case_id=case_id,
            video_id=video_id,
            camera_id=camera_id,
            timestamp_seconds=round(max(0.0, float(timestamp_seconds)), 2),
            end_timestamp_seconds=round(float(end_timestamp_seconds), 2) if end_timestamp_seconds is not None else None,
            annotation_type=a_type,
            data=data or {},
            author=_sanitize(author, max_len=100) or "Investigator",
        )
        db.add(ann)
        case.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(ann)

        self.log_activity(
            db,
            case_id=case_id,
            action_type="ANNOTATION_ADDED",
            description=f"Overlay annotation ({a_type}) added at {ann.timestamp_seconds}s in video '{video.original_filename}'.",
            details={"annotation_id": ann.id, "type": a_type, "timestamp": ann.timestamp_seconds},
            actor=author,
        )
        return ann

    def list_annotations(
        self,
        db: Session,
        case_id: str,
        video_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[CaseAnnotationModel]:
        """List visual overlay annotations for a case, with optional time range and video filters."""
        self.get_case(db, case_id)
        q = db.query(CaseAnnotationModel).filter(CaseAnnotationModel.case_id == case_id)
        if video_id:
            q = q.filter(CaseAnnotationModel.video_id == video_id)
        if start_time is not None:
            q = q.filter(CaseAnnotationModel.timestamp_seconds >= start_time)
        if end_time is not None:
            q = q.filter(CaseAnnotationModel.timestamp_seconds <= end_time)
        return q.order_by(CaseAnnotationModel.timestamp_seconds.asc()).all()

    def delete_annotation(
        self,
        db: Session,
        case_id: str,
        annotation_id: str,
        actor: str = "Investigator",
    ) -> None:
        """Delete an overlay annotation within the case."""
        self.get_case(db, case_id)
        ann = (
            db.query(CaseAnnotationModel)
            .filter(CaseAnnotationModel.id == annotation_id, CaseAnnotationModel.case_id == case_id)
            .first()
        )
        if not ann:
            raise CaseIsolationError(f"Annotation '{annotation_id}' not found in case '{case_id}'.")

        db.delete(ann)
        db.commit()

        self.log_activity(
            db,
            case_id=case_id,
            action_type="ANNOTATION_DELETED",
            description=f"Annotation ({annotation_id}) deleted.",
            details={"annotation_id": annotation_id},
            actor=actor,
        )

    # ------------------------------------------------------------------
    # 6. Case-Level Unified Timeline
    # ------------------------------------------------------------------

    def get_case_timeline(
        self,
        db: Session,
        case_id: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        layers: Optional[List[str]] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Merged, chronological multi-layer timeline across all linked videos/cameras in a case.
        Layers include:
          - detection_event (grouped detections)
          - security_event (security intelligence events)
          - correlated_incident (correlated incident storylines)
          - evidence (physical snapshots & clips)
          - bookmark (investigator bookmarks)
          - analyst_note (notes with timestamps)
          - annotation (visual overlay markers)

        Strict provenance is preserved. Rejected detections remain labeled REJECTED and are never promoted.
        """
        self.get_case(db, case_id)
        video_links = db.query(CaseVideoModel).filter(CaseVideoModel.case_id == case_id).all()
        video_ids = [vl.video_id for vl in video_links]

        if not video_ids:
            return {
                "case_id": case_id,
                "total_entries": 0,
                "timeline": [],
                "layer_counts": {},
            }

        # Map video_id to filename and camera label if available
        videos_map = {
            v.id: v.original_filename
            for v in db.query(VideoModel).filter(VideoModel.id.in_(video_ids)).all()
        }
        cam_links = db.query(CaseCameraModel).filter(CaseCameraModel.case_id == case_id).all()
        cam_map = {}
        cam_offset_map = {}
        for cl in cam_links:
            cam = cl.camera
            if cam:
                cam_map[cam.video_id] = cam.camera_label
                cam_offset_map[cam.video_id] = cl.clock_offset_seconds

        timeline_entries: List[Dict[str, Any]] = []
        layer_filter = set(layers) if layers else None

        # 1. Detection Events (GroupedEventModel)
        if not layer_filter or "detection_event" in layer_filter:
            ge_q = db.query(GroupedEventModel).filter(GroupedEventModel.video_id.in_(video_ids))
            if start_time is not None:
                ge_q = ge_q.filter(GroupedEventModel.end_time >= start_time)
            if end_time is not None:
                ge_q = ge_q.filter(GroupedEventModel.start_time <= end_time)
            for ge in ge_q.all():
                timeline_entries.append({
                    "layer": "detection_event",
                    "id": ge.id,
                    "video_id": ge.video_id,
                    "source_name": videos_map.get(ge.video_id, "Video"),
                    "camera_label": cam_map.get(ge.video_id),
                    "timestamp": round(ge.start_time, 2),
                    "end_timestamp": round(ge.end_time, 2),
                    "label": ge.event_type,
                    "detail": f"{ge.total_detections} detection(s) (Priority: {ge.priority})",
                    "provenance": {"table": "grouped_events", "record_id": ge.id},
                })

        # 2. Security Events (SecurityEventModel)
        if not layer_filter or "security_event" in layer_filter:
            se_q = db.query(SecurityEventModel).filter(SecurityEventModel.video_id.in_(video_ids))
            if start_time is not None:
                se_q = se_q.filter(SecurityEventModel.timestamp_seconds >= start_time)
            if end_time is not None:
                se_q = se_q.filter(SecurityEventModel.timestamp_seconds <= end_time)
            for se in se_q.all():
                timeline_entries.append({
                    "layer": "security_event",
                    "id": se.id,
                    "video_id": se.video_id,
                    "source_name": videos_map.get(se.video_id, "Video"),
                    "camera_label": cam_map.get(se.video_id),
                    "timestamp": round(se.timestamp_seconds, 2),
                    "end_timestamp": round(se.timestamp_seconds + (se.duration_seconds or 0.0), 2),
                    "label": se.event_type.replace("_", " "),
                    "detail": se.description,
                    "severity": se.severity,
                    "track_id": se.track_id,
                    "human_verification_required": bool(se.human_verification_required),
                    "validation_decision": "REVIEW_REQUIRED" if se.human_verification_required else "ACCEPTED",
                    "provenance": {"table": "security_events", "record_id": se.id, "detector": se.detector_name},
                })

        # 3. Correlated Incidents (CorrelatedIncidentModel)
        if not layer_filter or "correlated_incident" in layer_filter:
            ci_q = db.query(CorrelatedIncidentModel).filter(CorrelatedIncidentModel.video_id.in_(video_ids))
            if start_time is not None:
                ci_q = ci_q.filter(CorrelatedIncidentModel.end_time >= start_time)
            if end_time is not None:
                ci_q = ci_q.filter(CorrelatedIncidentModel.start_time <= end_time)
            for ci in ci_q.all():
                timeline_entries.append({
                    "layer": "correlated_incident",
                    "id": ci.id,
                    "video_id": ci.video_id,
                    "source_name": videos_map.get(ci.video_id, "Video"),
                    "camera_label": cam_map.get(ci.video_id),
                    "timestamp": round(ci.start_time, 2),
                    "end_timestamp": round(ci.end_time, 2),
                    "label": f"{ci.incident_category} — {ci.incident_subcategory or 'Incident'}",
                    "detail": ci.storyline or "",
                    "assessment_score": round(ci.assessment_score, 4),
                    "evidence_strength": round(ci.evidence_strength, 4),
                    "reliability_rating": ci.reliability_rating,
                    "validation_decision": ci.validation_decision,
                    "primary_track_ids": ci.primary_track_ids or [],
                    "provenance": {"table": "correlated_incidents", "record_id": ci.id},
                })

        # 4. Evidence (EvidenceModel)
        if not layer_filter or "evidence" in layer_filter:
            ev_q = db.query(EvidenceModel).filter(EvidenceModel.video_id.in_(video_ids))
            if start_time is not None:
                ev_q = ev_q.filter(EvidenceModel.timestamp_seconds >= start_time)
            if end_time is not None:
                ev_q = ev_q.filter(EvidenceModel.timestamp_seconds <= end_time)
            for ev in ev_q.all():
                timeline_entries.append({
                    "layer": "evidence",
                    "id": ev.id,
                    "video_id": ev.video_id,
                    "source_name": videos_map.get(ev.video_id, "Video"),
                    "camera_label": cam_map.get(ev.video_id),
                    "timestamp": round(ev.timestamp_seconds, 2),
                    "end_timestamp": round(ev.end_time or ev.timestamp_seconds, 2),
                    "label": f"Evidence ({ev.evidence_type})",
                    "detail": ev.notes or f"Evidence preserved for class '{ev.object_class}'",
                    "has_snapshot": bool(ev.snapshot_path),
                    "has_clip": bool(ev.clip_path),
                    "validation_status": ev.validation_status,
                    "provenance": {"table": "evidence", "record_id": ev.id},
                })

        # 5. Bookmarks (CaseBookmarkModel)
        if not layer_filter or "bookmark" in layer_filter:
            bm_q = db.query(CaseBookmarkModel).filter(CaseBookmarkModel.case_id == case_id)
            if start_time is not None:
                bm_q = bm_q.filter(CaseBookmarkModel.timestamp_seconds >= start_time)
            if end_time is not None:
                bm_q = bm_q.filter(CaseBookmarkModel.timestamp_seconds <= end_time)
            for bm in bm_q.all():
                timeline_entries.append({
                    "layer": "bookmark",
                    "id": bm.id,
                    "video_id": bm.video_id,
                    "source_name": videos_map.get(bm.video_id, "Video"),
                    "camera_label": cam_map.get(bm.video_id),
                    "timestamp": round(bm.timestamp_seconds, 2),
                    "end_timestamp": round(bm.timestamp_seconds, 2),
                    "label": f"Bookmark: {bm.title}",
                    "detail": bm.description or "Pinned timestamp",
                    "author": bm.author,
                    "linked_incident_id": bm.linked_incident_id,
                    "provenance": {"table": "case_bookmarks", "record_id": bm.id},
                })

        # 6. Analyst Notes with Timestamps (CaseNoteModel)
        if not layer_filter or "analyst_note" in layer_filter:
            note_q = (
                db.query(CaseNoteModel)
                .filter(CaseNoteModel.case_id == case_id, CaseNoteModel.timestamp_seconds.isnot(None))
            )
            if start_time is not None:
                note_q = note_q.filter(CaseNoteModel.timestamp_seconds >= start_time)
            if end_time is not None:
                note_q = note_q.filter(CaseNoteModel.timestamp_seconds <= end_time)
            for nt in note_q.all():
                timeline_entries.append({
                    "layer": "analyst_note",
                    "id": nt.id,
                    "timestamp": round(nt.timestamp_seconds or 0.0, 2),
                    "end_timestamp": round(nt.timestamp_seconds or 0.0, 2),
                    "label": f"Analyst Note ({nt.author})",
                    "detail": nt.content,
                    "author": nt.author,
                    "classification": "ANALYST_NOTE",
                    "provenance": {"table": "case_notes", "record_id": nt.id},
                })

        # 7. Annotations (CaseAnnotationModel)
        if not layer_filter or "annotation" in layer_filter:
            ann_q = db.query(CaseAnnotationModel).filter(CaseAnnotationModel.case_id == case_id)
            if start_time is not None:
                ann_q = ann_q.filter(CaseAnnotationModel.timestamp_seconds >= start_time)
            if end_time is not None:
                ann_q = ann_q.filter(CaseAnnotationModel.timestamp_seconds <= end_time)
            for ann in ann_q.all():
                timeline_entries.append({
                    "layer": "annotation",
                    "id": ann.id,
                    "video_id": ann.video_id,
                    "source_name": videos_map.get(ann.video_id, "Video"),
                    "camera_label": cam_map.get(ann.video_id),
                    "timestamp": round(ann.timestamp_seconds, 2),
                    "end_timestamp": round(ann.end_timestamp_seconds or ann.timestamp_seconds, 2),
                    "label": f"Annotation ({ann.annotation_type})",
                    "detail": ann.data.get("label") or ann.data.get("text") or "Visual annotation",
                    "annotation_type": ann.annotation_type,
                    "data": ann.data,
                    "author": ann.author,
                    "provenance": {"table": "case_annotations", "record_id": ann.id},
                })

        # Sort chronologically by timestamp
        timeline_entries.sort(key=lambda x: x["timestamp"])

        layer_counts = {}
        for e in timeline_entries:
            layer_counts[e["layer"]] = layer_counts.get(e["layer"], 0) + 1

        total_count = len(timeline_entries)
        paged_entries = timeline_entries[offset : offset + limit]

        return {
            "case_id": case_id,
            "total_entries": total_count,
            "offset": offset,
            "limit": limit,
            "layer_counts": layer_counts,
            "timeline": paged_entries,
        }

    # ------------------------------------------------------------------
    # 7. Incident Replay Context
    # ------------------------------------------------------------------

    def get_incident_replay_context(
        self,
        db: Session,
        case_id: str,
        incident_id: str,
        pre_roll_seconds: float = 5.0,
        post_roll_seconds: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Construct dynamic playback context for an incident:
        BEFORE -> CONTEXT -> INCIDENT -> AFTER.
        Derives time window from actual incident metadata and links clips/stream.
        """
        self.get_case(db, case_id)

        # Check CorrelatedIncidentModel first
        corr = db.query(CorrelatedIncidentModel).filter(CorrelatedIncidentModel.id == incident_id).first()
        sec = None
        if not corr:
            sec = db.query(SecurityEventModel).filter(SecurityEventModel.id == incident_id).first()
            if not sec:
                raise CaseEntityNotFoundError(f"Incident '{incident_id}' not found.")

        if corr:
            video_id = corr.video_id
            incident_start = corr.start_time
            incident_end = corr.end_time
            category = corr.incident_category
            subcategory = corr.incident_subcategory
            assessment_score = corr.assessment_score
            decision = corr.validation_decision
            storyline = corr.storyline
            primary_tracks = corr.primary_track_ids or []
        else:
            assert sec is not None
            video_id = sec.video_id
            incident_start = sec.timestamp_seconds
            incident_end = sec.timestamp_seconds + (sec.duration_seconds or 0.0)
            category = sec.category or "security"
            subcategory = sec.event_type
            assessment_score = sec.confidence
            decision = "REVIEW_REQUIRED" if sec.human_verification_required else "ACCEPTED"
            storyline = sec.description
            primary_tracks = [sec.track_id] if sec.track_id else []

        video = db.query(VideoModel).filter(VideoModel.id == video_id).first()
        max_duration = video.duration_seconds if video and video.duration_seconds else incident_end + 30.0

        # Construct BEFORE -> CONTEXT -> INCIDENT -> AFTER time window
        replay_start = round(max(0.0, incident_start - float(pre_roll_seconds)), 2)
        replay_end = round(min(max_duration, incident_end + float(post_roll_seconds)), 2)

        # Check for existing evidence clips covering this incident
        evidence_clips = (
            db.query(EvidenceModel)
            .filter(
                EvidenceModel.video_id == video_id,
                EvidenceModel.clip_path.isnot(None),
                EvidenceModel.timestamp_seconds >= replay_start - 2.0,
                EvidenceModel.timestamp_seconds <= replay_end + 2.0,
            )
            .all()
        )
        clip_references = [
            {
                "evidence_id": ev.id,
                "timestamp": round(ev.timestamp_seconds, 2),
                "clip_url": f"/api/evidence/{ev.id}/clip",
                "snapshot_url": f"/api/evidence/{ev.id}/snapshot" if ev.snapshot_path else None,
            }
            for ev in evidence_clips
        ]

        # Check multi-camera context if cameras are linked to this case
        cam_links = db.query(CaseCameraModel).filter(CaseCameraModel.case_id == case_id).all()
        camera_contexts = []
        for cl in cam_links:
            cam = cl.camera
            if cam:
                # Aligned time in this camera's video
                aligned_start = max(0.0, replay_start + cl.clock_offset_seconds)
                aligned_end = max(0.0, replay_end + cl.clock_offset_seconds)
                camera_contexts.append({
                    "camera_id": cam.id,
                    "camera_label": cam.camera_label,
                    "video_id": cam.video_id,
                    "clock_offset_seconds": cl.clock_offset_seconds,
                    "aligned_replay_window": [round(aligned_start, 2), round(aligned_end, 2)],
                    "stream_url": f"/api/videos/{cam.video_id}/stream",
                })

        return {
            "incident_id": incident_id,
            "video_id": video_id,
            "video_filename": video.original_filename if video else "Video",
            "category": category,
            "subcategory": subcategory,
            "assessment_score": round(assessment_score, 4),
            "validation_decision": decision,
            "storyline": storyline,
            "primary_tracks": primary_tracks,
            "incident_window": [round(incident_start, 2), round(incident_end, 2)],
            "replay_context": {
                "pre_roll_seconds": float(pre_roll_seconds),
                "post_roll_seconds": float(post_roll_seconds),
                "replay_start": replay_start,
                "replay_end": replay_end,
                "duration": round(replay_end - replay_start, 2),
            },
            "stream_url": f"/api/videos/{video_id}/stream",
            "evidence_clips": clip_references,
            "camera_contexts": camera_contexts,
        }

    # ------------------------------------------------------------------
    # 8. Investigation Focus Mode
    # ------------------------------------------------------------------

    def get_incident_focus_data(
        self,
        db: Session,
        case_id: str,
        incident_id: str,
    ) -> Dict[str, Any]:
        """
        Assemble comprehensive focus view data for a selected incident:
        Incident details, provenance, temporal context, relevant tracks,
        relevant cameras, evidence, analyst notes, and negative/contradictory evidence.
        """
        self.get_case(db, case_id)
        replay = self.get_incident_replay_context(db, case_id, incident_id)

        video_id = replay["video_id"]
        start_time = replay["replay_context"]["replay_start"]
        end_time = replay["replay_context"]["replay_end"]

        # Explainability data
        explanation = self.get_incident_explanation(db, case_id, incident_id)

        # Relevant tracks
        tracks = []
        track_ids = replay.get("primary_tracks", [])
        if track_ids:
            t_records = (
                db.query(TrackModel)
                .filter(TrackModel.video_id == video_id, TrackModel.track_id.in_(track_ids))
                .all()
            )
            for t in t_records:
                tracks.append({
                    "track_id": t.track_id,
                    "object_class": t.object_class,
                    "first_seen": round(t.first_seen, 2),
                    "last_seen": round(t.last_seen, 2),
                    "duration_seconds": round(t.duration_seconds, 2),
                    "detection_count": t.detection_count,
                    "color": t.color,
                })

        # Related incidents in overlapping timeframe
        related_corrs = (
            db.query(CorrelatedIncidentModel)
            .filter(
                CorrelatedIncidentModel.video_id == video_id,
                CorrelatedIncidentModel.id != incident_id,
                CorrelatedIncidentModel.start_time <= end_time + 5.0,
                CorrelatedIncidentModel.end_time >= start_time - 5.0,
            )
            .all()
        )
        related_incidents = [
            {
                "incident_id": rc.id,
                "category": rc.incident_category,
                "subcategory": rc.incident_subcategory,
                "start_time": round(rc.start_time, 2),
                "end_time": round(rc.end_time, 2),
                "assessment_score": round(rc.assessment_score, 4),
                "validation_decision": rc.validation_decision,
            }
            for rc in related_corrs
        ]

        # Analyst notes associated with this incident
        notes = self.list_notes(db, case_id=case_id, associated_type="INCIDENT", associated_id=incident_id)
        notes_serialized = [
            {
                "note_id": n.id,
                "content": n.content,
                "author": n.author,
                "created_at": n.created_at.isoformat(),
            }
            for n in notes
        ]

        return {
            "case_id": case_id,
            "incident": replay,
            "explanation": explanation,
            "tracks": tracks,
            "related_incidents": related_incidents,
            "notes": notes_serialized,
        }

    # ------------------------------------------------------------------
    # 9. "Why Did Sentinel Flag This?" Explainability
    # ------------------------------------------------------------------

    def get_incident_explanation(
        self,
        db: Session,
        case_id: str,
        incident_id: str,
    ) -> Dict[str, Any]:
        """
        Explainability view:
        - Supporting signals (grounded in actual detections)
        - Limiting / Contradicting signals (negative evidence, sensor limits)
        - Pattern Evidence Strength (percentage)
        - Final Assessment (score + REVIEW_REQUIRED ceiling)
        - Human verification requirement
        """
        self.get_case(db, case_id)

        corr = db.query(CorrelatedIncidentModel).filter(CorrelatedIncidentModel.id == incident_id).first()
        sec = None
        if not corr:
            sec = db.query(SecurityEventModel).filter(SecurityEventModel.id == incident_id).first()
            if not sec:
                raise CaseEntityNotFoundError(f"Incident '{incident_id}' not found.")

        supporting_signals: List[str] = []
        limiting_signals: List[str] = []

        if corr:
            # Extract supporting signals from storyline and provenance
            if corr.incident_category:
                supporting_signals.append(f"Incident category '{corr.incident_category}' identified via multi-signal correlation.")
            if corr.incident_subcategory:
                supporting_signals.append(f"Subcategory pattern: '{corr.incident_subcategory}'.")
            if corr.primary_track_ids:
                supporting_signals.append(f"Consistent spatial tracks observed: {', '.join(corr.primary_track_ids)}.")
            if corr.contextual_factors:
                for factor, val in corr.contextual_factors.items() if isinstance(corr.contextual_factors, dict) else []:
                    supporting_signals.append(f"Contextual factor: {factor} = {val}")

            # Extract limiting/contradicting signals from negative_evidence
            if corr.negative_evidence and isinstance(corr.negative_evidence, list):
                for neg in corr.negative_evidence:
                    limiting_signals.append(f"Negative signal: {neg}")
            elif corr.negative_evidence and isinstance(corr.negative_evidence, dict):
                for k, v in corr.negative_evidence.items():
                    limiting_signals.append(f"Negative signal: {k}: {v}")
            else:
                limiting_signals.append("Single-view optical camera perspective; no direct biometric or intent confirmation.")

            pattern_strength = round(corr.evidence_strength * 100, 1)
            final_assessment = round(corr.assessment_score, 4)
            decision = corr.validation_decision
            reliability = corr.reliability_rating
            human_verification = True  # Always required per Sentinel forensic principles
            provenance = corr.provenance or {"source": "ai.correlation.engine"}

        else:
            assert sec is not None
            supporting_signals.append(f"Security event triggered: {sec.event_type.replace('_', ' ')}.")
            if sec.zone_name:
                supporting_signals.append(f"Spatial boundary crossed: Zone '{sec.zone_name}'.")
            if sec.observable_signals and isinstance(sec.observable_signals, list):
                for s in sec.observable_signals:
                    if isinstance(s, dict):
                        supporting_signals.append(f"Observable signal: {s.get('signal', 'telemetry')} (score {s.get('value', '')})")
                    else:
                        supporting_signals.append(f"Observable signal: {s}")
            else:
                supporting_signals.append(f"Observable signal: {sec.description}")

            limiting_signals.append("Visual bounding box heuristics only; intent cannot be inferred.")
            limiting_signals.append("Requires human verification to rule out innocent explanation.")

            pattern_strength = round(sec.confidence * 100, 1)
            final_assessment = round(min(0.65, sec.confidence) if sec.human_verification_required else sec.confidence, 4)
            decision = "REVIEW_REQUIRED" if sec.human_verification_required else "ACCEPTED"
            reliability = "MEDIUM"
            human_verification = bool(sec.human_verification_required)
            provenance = {
                "detector": sec.detector_name or "SentinelSecurityDetector",
                "version": sec.detector_version or "1.0.0",
                "timestamp": sec.timestamp_seconds,
            }

        return {
            "incident_id": incident_id,
            "supporting_signals": supporting_signals,
            "limiting_signals": limiting_signals,
            "pattern_evidence_strength": f"{pattern_strength}%",
            "pattern_evidence_strength_value": pattern_strength / 100.0,
            "final_assessment_score": final_assessment,
            "validation_decision": decision,
            "reliability_rating": reliability,
            "human_verification_required": human_verification,
            "human_verification_notice": (
                "Sentinel provides observational pattern evidence only. "
                "Human analyst verification is strictly required before operational action."
            ),
            "provenance": provenance,
        }

    # ------------------------------------------------------------------
    # 10. Camera Topology & Multi-Camera Workspace
    # ------------------------------------------------------------------

    def get_case_topology(self, db: Session, case_id: str) -> Dict[str, Any]:
        """
        Derive visual camera topology graph for the case:
        Nodes: Cameras linked to the case.
        Edges:
          - ADJACENCY: physically adjacent cameras
          - CANDIDATE_ASSOCIATION: cross-camera track hypotheses
          - CONFIRMED_ASSOCIATION: analyst-confirmed links
        """
        self.get_case(db, case_id)
        cam_links = db.query(CaseCameraModel).filter(CaseCameraModel.case_id == case_id).all()
        if not cam_links:
            return {"case_id": case_id, "total_cameras": 0, "nodes": [], "edges": []}

        nodes = []
        cam_id_set = set()
        session_ids = set()

        for cl in cam_links:
            c = cl.camera
            if c:
                cam_id_set.add(c.id)
                session_ids.add(c.session_id)
                nodes.append({
                    "id": c.id,
                    "label": c.camera_label,
                    "position_hint": c.position_hint or "Site Area",
                    "field_of_view_hint": c.field_of_view_hint,
                    "video_id": c.video_id,
                    "clock_offset_seconds": cl.clock_offset_seconds,
                })

        edges = []
        edge_set = set()

        # 1. Adjacency edges from camera definitions
        for cl in cam_links:
            c = cl.camera
            if c and c.adjacency_hints and isinstance(c.adjacency_hints, list):
                for adj_label in c.adjacency_hints:
                    # Find target camera by label
                    target = next((n for n in nodes if n["label"].lower() == adj_label.lower()), None)
                    if target and target["id"] != c.id:
                        edge_key = tuple(sorted([c.id, target["id"]]))
                        if edge_key not in edge_set:
                            edge_set.add(edge_key)
                            edges.append({
                                "source": c.id,
                                "target": target["id"],
                                "type": "ADJACENCY",
                                "label": "Adjacent FOV",
                                "status": "TOPOLOGY",
                            })

        # 2. Association edges from CrossCameraAssociationModel
        if session_ids:
            assocs = (
                db.query(CrossCameraAssociationModel)
                .filter(
                    CrossCameraAssociationModel.session_id.in_(list(session_ids)),
                    CrossCameraAssociationModel.source_camera_id.in_(list(cam_id_set)),
                    CrossCameraAssociationModel.target_camera_id.in_(list(cam_id_set)),
                )
                .all()
            )
            for a in assocs:
                edges.append({
                    "id": a.id,
                    "source": a.source_camera_id,
                    "target": a.target_camera_id,
                    "type": "CONFIRMED_ASSOCIATION" if a.analyst_verdict == "CONFIRMED" else "CANDIDATE_ASSOCIATION",
                    "confidence": round(a.confidence, 4),
                    "association_type": a.association_type,
                    "verdict": a.analyst_verdict,
                    "temporal_gap": round(a.temporal_gap_seconds or 0.0, 1),
                    "label": f"{a.association_type} ({round(a.confidence*100)}%)",
                    "status": a.analyst_verdict,
                })

        return {
            "case_id": case_id,
            "total_cameras": len(nodes),
            "nodes": nodes,
            "edges": edges,
        }

    # ------------------------------------------------------------------
    # 11. Investigation Storyline / Notebook
    # ------------------------------------------------------------------

    def get_case_storyline(self, db: Session, case_id: str) -> Dict[str, Any]:
        """
        Synthesize chronological, multi-video investigation storyline
        combining correlated incidents, security events, analyst notes, and bookmarks.
        """
        case = self.get_case(db, case_id)
        timeline = self.get_case_timeline(db, case_id=case_id, limit=200)

        storyline_steps = []
        for item in timeline["timeline"]:
            layer = item["layer"]
            # Only include meaningful investigative steps in the high-level storyline
            if layer in {"correlated_incident", "security_event", "bookmark", "analyst_note"}:
                storyline_steps.append({
                    "step_id": item["id"],
                    "layer": layer,
                    "timestamp": item["timestamp"],
                    "timestamp_formatted": f"{int(item['timestamp'] // 60):02d}:{int(item['timestamp'] % 60):02d}",
                    "source_name": item.get("source_name", "Video"),
                    "camera_label": item.get("camera_label") or "Camera",
                    "title": item["label"],
                    "description": item.get("detail", ""),
                    "validation_decision": item.get("validation_decision"),
                    "assessment_score": item.get("assessment_score"),
                    "author": item.get("author"),
                    "provenance": item.get("provenance"),
                })

        return {
            "case_id": case_id,
            "case_number": case.case_number,
            "case_title": case.title,
            "total_steps": len(storyline_steps),
            "storyline": storyline_steps,
        }

    # ------------------------------------------------------------------
    # 12. Activity Audit Trail
    # ------------------------------------------------------------------

    def log_activity(
        self,
        db: Session,
        case_id: str,
        action_type: str,
        description: str,
        details: Optional[Dict[str, Any]] = None,
        actor: str = "Investigator",
    ) -> CaseActivityModel:
        """Record an immutable activity log entry for the case."""
        act = CaseActivityModel(
            case_id=case_id,
            action_type=action_type,
            description=description,
            details=details or {},
            actor=actor or "Investigator",
        )
        db.add(act)
        db.commit()
        db.refresh(act)
        return act

    def get_case_activities(
        self,
        db: Session,
        case_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[CaseActivityModel]:
        """Retrieve activity history for a case."""
        self.get_case(db, case_id)
        return (
            db.query(CaseActivityModel)
            .filter(CaseActivityModel.case_id == case_id)
            .order_by(desc(CaseActivityModel.created_at))
            .offset(offset)
            .limit(min(limit, 200))
            .all()
        )

    # ------------------------------------------------------------------
    # 13. Case Export
    # ------------------------------------------------------------------

    def export_case_data(self, db: Session, case_id: str) -> Dict[str, Any]:
        """
        Assemble a complete, validated forensic case export package.
        Includes case metadata, linked entities, unified timeline,
        bookmarks, notes, annotations, evidence index, storyline, and audit trail.
        """
        case = self.get_case(db, case_id)
        videos = self.get_linked_videos(db, case_id)
        cameras = self.get_linked_cameras(db, case_id)
        incidents = self.get_case_incidents(db, case_id)
        bookmarks = self.list_bookmarks(db, case_id)
        notes = self.list_notes(db, case_id)
        annotations = self.list_annotations(db, case_id)
        timeline = self.get_case_timeline(db, case_id, limit=1000)
        storyline = self.get_case_storyline(db, case_id)
        # Log activity for export first so it is included in the package
        self.log_activity(
            db,
            case_id=case_id,
            action_type="CASE_EXPORTED",
            description=f"Forensic case export package generated for {case.case_number}.",
            actor="Investigator",
        )
        activities = self.get_case_activities(db, case_id, limit=200)

        # Build evidence index
        video_ids = [v["video_id"] for v in videos]
        evidence_items = (
            db.query(EvidenceModel)
            .filter(EvidenceModel.video_id.in_(video_ids))
            .all()
        )
        evidence_index = [
            {
                "evidence_id": ev.id,
                "video_id": ev.video_id,
                "source_video_name": ev.source_video_name,
                "timestamp_seconds": round(ev.timestamp_seconds, 2),
                "evidence_type": ev.evidence_type,
                "validation_status": ev.validation_status,
                "object_class": ev.object_class,
                "confidence": round(ev.confidence, 4) if ev.confidence is not None else None,
                "bounding_box": ev.bounding_box,
                "has_snapshot": bool(ev.snapshot_path),
                "has_clip": bool(ev.clip_path),
                "notes": ev.notes,
            }
            for ev in evidence_items
        ]

        export_payload = {
            "sentinel_platform": {
                "version": "19.0.0",
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "integrity_notice": (
                    "Exported by Sentinel Forensic Case Management Platform. "
                    "Evidence provenance and human verification requirements preserved."
                ),
            },
            "case_metadata": {
                "id": case.id,
                "case_number": case.case_number,
                "title": case.title,
                "description": case.description,
                "status": case.status,
                "priority": case.priority,
                "assigned_investigator": case.assigned_investigator,
                "tags": case.tags or [],
                "summary": case.summary,
                "created_at": case.created_at.isoformat(),
                "updated_at": case.updated_at.isoformat(),
            },
            "linked_videos": videos,
            "linked_cameras": cameras,
            "incidents": incidents,
            "evidence_index": evidence_index,
            "bookmarks": [
                {
                    "bookmark_id": b.id,
                    "video_id": b.video_id,
                    "camera_id": b.camera_id,
                    "timestamp_seconds": b.timestamp_seconds,
                    "title": b.title,
                    "description": b.description,
                    "linked_incident_id": b.linked_incident_id,
                    "author": b.author,
                    "created_at": b.created_at.isoformat(),
                }
                for b in bookmarks
            ],
            "notes": [
                {
                    "note_id": n.id,
                    "content": n.content,
                    "author": n.author,
                    "classification": n.note_classification,
                    "associated_type": n.associated_type,
                    "associated_id": n.associated_id,
                    "timestamp_seconds": n.timestamp_seconds,
                    "created_at": n.created_at.isoformat(),
                }
                for n in notes
            ],
            "annotations": [
                {
                    "annotation_id": a.id,
                    "video_id": a.video_id,
                    "camera_id": a.camera_id,
                    "timestamp_seconds": a.timestamp_seconds,
                    "end_timestamp_seconds": a.end_timestamp_seconds,
                    "annotation_type": a.annotation_type,
                    "data": a.data,
                    "author": a.author,
                    "created_at": a.created_at.isoformat(),
                }
                for a in annotations
            ],
            "timeline": timeline["timeline"],
            "storyline": storyline["storyline"],
            "activity_audit_trail": [
                {
                    "activity_id": act.id,
                    "action_type": act.action_type,
                    "description": act.description,
                    "details": act.details,
                    "actor": act.actor,
                    "timestamp": act.created_at.isoformat(),
                }
                for act in activities
            ],
        }

        return export_payload
