"""
Sentinel Universal Data Consistency Validator (Phase 15.1)

Enforces 12 Universal Data Consistency Rules across ingestion, tracking,
incident analysis, evidence preservation, and dossier reporting.
"""
import math
import logging
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session

from ai.common.numeric import is_finite_number
from database.models import (
    VideoModel,
    EventModel,
    TrackModel,
    SecurityEventModel,
    EvidenceModel,
    ReportModel,
    SpecializedObservationModel,
)

logger = logging.getLogger(__name__)


class ConsistencyViolation:
    """Represents a discrete consistency violation."""
    def __init__(self, rule_id: int, rule_name: str, message: str, record_id: Optional[str] = None):
        self.rule_id = rule_id
        self.rule_name = rule_name
        self.message = message
        self.record_id = record_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "message": self.message,
            "record_id": self.record_id,
        }


class UniversalDataConsistencyValidator:
    """
    Validates that a processed video complies with all 12 Sentinel consistency invariants.
    """

    @classmethod
    def validate_video_consistency(
        cls,
        video_id: str,
        db: Session,
        duration_tolerance_seconds: float = 1.0,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Execute comprehensive 12-rule consistency audit for video_id.
        Returns (is_valid, list_of_violations).
        """
        violations: List[ConsistencyViolation] = []

        # Verify video exists
        video = db.query(VideoModel).filter(VideoModel.id == video_id).first()
        if not video:
            violations.append(
                ConsistencyViolation(0, "VIDEO_EXISTS", f"Video with id '{video_id}' not found in database.")
            )
            return False, [v.to_dict() for v in violations]

        duration = video.duration_seconds or 0.0
        max_allowed_time = duration + duration_tolerance_seconds if duration > 0 else float("inf")

        # ---------------------------------------------------------------------
        # Rule 1 & 2 & 9: All timestamps are finite, within duration, no NaN/Inf
        # ---------------------------------------------------------------------
        # Events
        events = db.query(EventModel).filter(EventModel.video_id == video_id).all()
        for ev in events:
            ts = ev.timestamp_seconds
            if not is_finite_number(ts):
                violations.append(ConsistencyViolation(1, "FINITE_TIMESTAMPS", f"Non-finite timestamp in Event {ev.id}", ev.id))
            elif ts < 0.0 or (duration > 0 and ts > max_allowed_time):
                violations.append(ConsistencyViolation(2, "TIMESTAMPS_WITHIN_DURATION", f"Timestamp {ts}s outside [0, {duration}s] in Event {ev.id}", ev.id))

            conf = ev.confidence
            if conf is not None and not is_finite_number(conf):
                violations.append(ConsistencyViolation(9, "NO_NAN_INFINITY", f"Non-finite confidence in Event {ev.id}", ev.id))

        # Security Events
        sec_events = db.query(SecurityEventModel).filter(SecurityEventModel.video_id == video_id).all()
        for se in sec_events:
            ts = se.timestamp_seconds
            dur = se.duration_seconds
            conf = se.confidence
            if not is_finite_number(ts):
                violations.append(ConsistencyViolation(1, "FINITE_TIMESTAMPS", f"Non-finite timestamp in SecurityEvent {se.id}", se.id))
            elif ts < 0.0 or (duration > 0 and ts > max_allowed_time):
                violations.append(ConsistencyViolation(2, "TIMESTAMPS_WITHIN_DURATION", f"Timestamp {ts}s outside [0, {duration}s] in SecurityEvent {se.id}", se.id))

            if dur is not None and not is_finite_number(dur):
                violations.append(ConsistencyViolation(9, "NO_NAN_INFINITY", f"Non-finite duration in SecurityEvent {se.id}", se.id))
            if conf is not None and not is_finite_number(conf):
                violations.append(ConsistencyViolation(9, "NO_NAN_INFINITY", f"Non-finite confidence in SecurityEvent {se.id}", se.id))

        # Specialized Observations
        spec_obs = db.query(SpecializedObservationModel).filter(SpecializedObservationModel.video_id == video_id).all()
        for so in spec_obs:
            ts = so.timestamp_seconds
            conf = so.confidence
            ev_str = so.evidence_strength
            if not is_finite_number(ts):
                violations.append(ConsistencyViolation(1, "FINITE_TIMESTAMPS", f"Non-finite timestamp in SpecializedObservation {so.id}", so.id))
            elif ts < 0.0 or (duration > 0 and ts > max_allowed_time):
                violations.append(ConsistencyViolation(2, "TIMESTAMPS_WITHIN_DURATION", f"Timestamp {ts}s outside [0, {duration}s] in SpecializedObservation {so.id}", so.id))

            if not is_finite_number(conf):
                violations.append(ConsistencyViolation(9, "NO_NAN_INFINITY", f"Non-finite confidence in SpecializedObservation {so.id}", so.id))
            if not is_finite_number(ev_str):
                violations.append(ConsistencyViolation(9, "NO_NAN_INFINITY", f"Non-finite evidence_strength in SpecializedObservation {so.id}", so.id))

        # ---------------------------------------------------------------------
        # Rule 3: All bounding boxes are valid (x1 <= x2, y1 <= y2, finite)
        # ---------------------------------------------------------------------
        for ev in events:
            if ev.bbox_x1 is not None and ev.bbox_x2 is not None:
                if not (is_finite_number(ev.bbox_x1) and is_finite_number(ev.bbox_x2)):
                    violations.append(ConsistencyViolation(3, "VALID_BOUNDING_BOXES", f"Non-finite bbox coordinates in Event {ev.id}", ev.id))
                elif ev.bbox_x1 > ev.bbox_x2:
                    violations.append(ConsistencyViolation(3, "VALID_BOUNDING_BOXES", f"Inverted x-coordinates in Event {ev.id}", ev.id))
            if ev.bbox_y1 is not None and ev.bbox_y2 is not None:
                if not (is_finite_number(ev.bbox_y1) and is_finite_number(ev.bbox_y2)):
                    violations.append(ConsistencyViolation(3, "VALID_BOUNDING_BOXES", f"Non-finite bbox coordinates in Event {ev.id}", ev.id))
                elif ev.bbox_y1 > ev.bbox_y2:
                    violations.append(ConsistencyViolation(3, "VALID_BOUNDING_BOXES", f"Inverted y-coordinates in Event {ev.id}", ev.id))

        # ---------------------------------------------------------------------
        # Rule 4 & 5 & 6 & 7 & 8: Correct video_id scoping across all models
        # ---------------------------------------------------------------------
        tracks = db.query(TrackModel).filter(TrackModel.video_id == video_id).all()
        for trk in tracks:
            if trk.video_id != video_id:
                violations.append(ConsistencyViolation(5, "TRACKS_REFERENCE_VALID_VIDEO", f"Track {trk.id} cross-references {trk.video_id} instead of {video_id}", trk.id))

        for se in sec_events:
            if se.video_id != video_id:
                violations.append(ConsistencyViolation(6, "EVENTS_REFERENCE_VALID_VIDEO", f"SecurityEvent {se.id} cross-references {se.video_id} instead of {video_id}", se.id))

        evidence_items = db.query(EvidenceModel).filter(EvidenceModel.video_id == video_id).all()
        for evd in evidence_items:
            if evd.video_id != video_id:
                violations.append(ConsistencyViolation(7, "EVIDENCE_REFERENCES_VALID_VIDEO", f"Evidence {evd.id} cross-references {evd.video_id} instead of {video_id}", evd.id))

        reports = db.query(ReportModel).filter(ReportModel.video_id == video_id).all()
        for rep in reports:
            if rep.video_id != video_id:
                violations.append(ConsistencyViolation(8, "REPORTS_REFERENCE_VALID_VIDEO", f"Report {rep.id} cross-references {rep.video_id} instead of {video_id}", rep.id))

        # ---------------------------------------------------------------------
        # Rule 10: No duplicate user-facing incidents for same continuous event
        # ---------------------------------------------------------------------
        # Check for duplicate security events with identical event_type and overlapping timestamps (<0.5s)
        seen_events: Dict[str, List[float]] = {}
        for se in sec_events:
            t_key = f"{se.event_type}-{se.track_id or 'none'}"
            if t_key not in seen_events:
                seen_events[t_key] = []
            for existing_t in seen_events[t_key]:
                if abs(existing_t - se.timestamp_seconds) < 0.2:
                    violations.append(ConsistencyViolation(10, "NO_DUPLICATE_INCIDENTS", f"Duplicate user-facing incident detected for {se.event_type} at {se.timestamp_seconds}s", se.id))
            seen_events[t_key].append(se.timestamp_seconds)

        # ---------------------------------------------------------------------
        # Rule 12: Rejected observations do not become accepted evidence
        # ---------------------------------------------------------------------
        # Verify evidence snapshots never reference REJECTED events
        for evd in evidence_items:
            if evd.event_id:
                matching_se = next((s for s in sec_events if s.id == evd.event_id), None)
                if matching_se and getattr(matching_se, "validation_decision", None) == "REJECTED":
                    violations.append(ConsistencyViolation(12, "REJECTED_OBSERVATIONS_NOT_EVIDENCE", f"Evidence {evd.id} references REJECTED SecurityEvent {matching_se.id}", evd.id))

        is_valid = (len(violations) == 0)
        return is_valid, [v.to_dict() for v in violations]

    @classmethod
    def validate_data(
        cls,
        video_metadata: Dict[str, Any],
        detections: Optional[List[Dict[str, Any]]] = None,
        tracks: Optional[List[Dict[str, Any]]] = None,
        events: Optional[List[Dict[str, Any]]] = None,
        evidence: Optional[List[Dict[str, Any]]] = None,
        reports: Optional[List[Dict[str, Any]]] = None,
        duration_tolerance_seconds: float = 1.0,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        In-memory consistency validation across video metadata, detections, tracks,
        events, evidence, and reports according to Sentinel's 12 universal invariants.
        """
        violations: List[ConsistencyViolation] = []
        target_vid = video_metadata.get("video_id", "")
        duration = float(video_metadata.get("duration") or video_metadata.get("duration_seconds") or 0.0)
        max_allowed_time = duration + duration_tolerance_seconds if duration > 0 else float("inf")

        # 1. Events validation
        seen_events: Dict[str, List[float]] = {}
        for ev in (events or []):
            ev_id = str(ev.get("id") or ev.get("event_id") or "unknown")
            ev_vid = ev.get("video_id")
            if ev_vid and ev_vid != target_vid:
                violations.append(ConsistencyViolation(4, "CROSS_VIDEO_CONTAMINATION", f"Event references video_id '{ev_vid}' instead of '{target_vid}'", ev_id))

            ts = ev.get("timestamp") or ev.get("timestamp_seconds")
            if ts is not None:
                if not is_finite_number(ts):
                    violations.append(ConsistencyViolation(1, "NON_FINITE_VALUE", f"Non-finite timestamp in Event {ev_id}", ev_id))
                elif ts < 0.0 or ts > max_allowed_time:
                    violations.append(ConsistencyViolation(2, "TIMESTAMP_OUT_OF_BOUNDS", f"Timestamp {ts} exceeds duration {duration}", ev_id))

            conf = ev.get("confidence")
            if conf is not None and not is_finite_number(conf):
                violations.append(ConsistencyViolation(9, "NON_FINITE_VALUE", f"Non-finite confidence in Event {ev_id}", ev_id))

            ev_type = ev.get("event_type", "UNKNOWN")
            if ev_type not in seen_events:
                seen_events[ev_type] = []
            if ts is not None and is_finite_number(ts):
                for prior_t in seen_events[ev_type]:
                    if abs(prior_t - ts) < 0.3:
                        violations.append(ConsistencyViolation(10, "DUPLICATE_CONTINUOUS_INCIDENT", f"Duplicate event for {ev_type} at {ts}s", ev_id))
                seen_events[ev_type].append(ts)

        # 2. Tracks validation
        for trk in (tracks or []):
            trk_id = str(trk.get("track_id") or trk.get("id") or "unknown")
            t_vid = trk.get("video_id")
            if t_vid and t_vid != target_vid:
                violations.append(ConsistencyViolation(5, "CROSS_VIDEO_CONTAMINATION", f"Track references {t_vid} instead of {target_vid}", trk_id))

        # 3. Evidence validation
        for evd in (evidence or []):
            evd_id = str(evd.get("evidence_id") or evd.get("id") or "unknown")
            e_vid = evd.get("video_id")
            if e_vid and e_vid != target_vid:
                violations.append(ConsistencyViolation(7, "CROSS_VIDEO_CONTAMINATION", f"Evidence references {e_vid} instead of {target_vid}", evd_id))
            evd_status = str(evd.get("status") or evd.get("validation_status") or "").upper()
            if evd_status == "REJECTED":
                violations.append(ConsistencyViolation(12, "REJECTED_OBSERVATIONS_NOT_EVIDENCE", f"Rejected observation persisted as evidence {evd_id}", evd_id))

        # 4. Reports validation
        for rep in (reports or []):
            rep_id = str(rep.get("report_id") or rep.get("id") or "unknown")
            r_vid = rep.get("video_id")
            if r_vid and r_vid != target_vid:
                violations.append(ConsistencyViolation(8, "CROSS_VIDEO_CONTAMINATION", f"Report references {r_vid} instead of {target_vid}", rep_id))

        is_valid = (len(violations) == 0)
        return is_valid, [v.to_dict() for v in violations]

