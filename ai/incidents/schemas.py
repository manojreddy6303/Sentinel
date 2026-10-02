"""
Universal Incident Intelligence Engine Schemas (Phase 10)

Standardized incident candidates, context, supporting signals, and motion structures.
Adheres strictly to observational safety rules and provides 100% interoperability
with existing SecurityEvent models and timelines.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import math
import uuid

from ai.schemas import BoundingBox, TrackedObject, VehicleAttribute, FaceDetection, ZoneDefinition, SecurityEvent

# Canonical 1080p surveillance reference diagonal: sqrt(1920^2 + 1080^2) ~= 2202.906
CANONICAL_REFERENCE_DIAGONAL: float = 2202.906


class IncidentCategory(str, Enum):
    VEHICLE = "vehicle"
    PERSON = "person"
    PROPERTY = "property"
    CROWD = "crowd"
    ZONE = "zone"
    ENVIRONMENT = "environment"
    OBJECT = "object"
    MULTI_SIGNAL = "multi_signal"
    GENERAL = "general"


class IncidentValidationStatus(str, Enum):
    VALID = "VALID"
    UNCERTAIN = "UNCERTAIN"
    DISMISSED = "DISMISSED"


class ValidationDecision(str, Enum):
    ACCEPTED = "ACCEPTED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    REJECTED = "REJECTED"


@dataclass
class SupportingSignal:
    """
    Observable physical signal supporting an incident candidate.
    Must reflect observable telemetry (e.g. motion, proximity, timing) rather than intent.
    """
    signal_type: str
    description: str
    confidence: float
    timestamp: Optional[float] = None
    track_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "description": self.description,
            "confidence": round(float(self.confidence), 4),
            "timestamp": round(float(self.timestamp), 4) if self.timestamp is not None else None,
            "track_id": self.track_id,
            "metadata": self.metadata,
        }


@dataclass
class EvidenceCandidate:
    """
    Recommended timestamp and bounding box for forensic evidence capture.
    """
    timestamp: float
    pre_seconds: float = 3.0
    post_seconds: float = 4.0
    bounding_box: Optional[BoundingBox] = None
    target_track_id: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(float(self.timestamp), 4),
            "pre_seconds": round(float(self.pre_seconds), 2),
            "post_seconds": round(float(self.post_seconds), 2),
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "target_track_id": self.target_track_id,
            "reason": self.reason,
        }


@dataclass
class SpatialContext:
    """
    Spatial reasoning context for an incident.
    Coordinates are in pixel or normalized coordinate space.
    """
    centroid: Optional[Tuple[float, float]] = None
    bounding_box: Optional[BoundingBox] = None
    zone_name: Optional[str] = None
    relative_distance: Optional[float] = None
    inter_track_distances: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "centroid": [round(float(c), 2) for c in self.centroid] if self.centroid else None,
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "zone_name": self.zone_name,
            "relative_distance": round(float(self.relative_distance), 2) if self.relative_distance is not None else None,
            "inter_track_distances": {k: round(float(v), 2) for k, v in self.inter_track_distances.items()},
            "metadata": self.metadata,
        }


@dataclass
class TemporalContext:
    """
    Temporal reasoning context for an incident candidate.
    """
    start_time: float
    end_time: float
    duration_seconds: float
    persistence_score: float = 1.0
    onset_timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": round(float(self.start_time), 4),
            "end_time": round(float(self.end_time), 4),
            "duration_seconds": round(float(self.duration_seconds), 4),
            "persistence_score": round(float(self.persistence_score), 4),
            "onset_timestamp": round(float(self.onset_timestamp), 4) if self.onset_timestamp is not None else None,
            "metadata": self.metadata,
        }


@dataclass
class TrackMotion:
    """
    Derived motion telemetry for a track at a specific timestamp.
    Calculated in pixel/normalized space without uncalibrated km/h claims.
    """
    track_id: str
    timestamp: float
    position: Tuple[float, float]  # (cx, cy)
    displacement: float            # Net displacement from track start
    distance_traveled: float       # Cumulative path distance
    direction_radians: float       # Angle of instantaneous motion
    direction_degrees: float       # 0 - 360 degrees
    velocity_estimate: float       # pixels/normalized units per second
    acceleration_estimate: float   # velocity delta per second
    speed_change: float            # absolute speed difference
    is_stationary: bool            # whether subject is localized/resting
    stationary_duration: float     # consecutive seconds stationary
    movement_duration: float       # consecutive seconds moving
    path_consistency: float        # straight-line displacement / total path distance (0.0 to 1.0)
    confidence: float              # reliability of track observations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "timestamp": round(float(self.timestamp), 4),
            "position": [round(float(p), 2) for p in self.position],
            "displacement": round(float(self.displacement), 2),
            "distance_traveled": round(float(self.distance_traveled), 2),
            "direction_radians": round(float(self.direction_radians), 4),
            "direction_degrees": round(float(self.direction_degrees), 2),
            "velocity_estimate": round(float(self.velocity_estimate), 2),
            "acceleration_estimate": round(float(self.acceleration_estimate), 2),
            "speed_change": round(float(self.speed_change), 2),
            "is_stationary": bool(self.is_stationary),
            "stationary_duration": round(float(self.stationary_duration), 2),
            "movement_duration": round(float(self.movement_duration), 2),
            "path_consistency": round(float(self.path_consistency), 4),
            "confidence": round(float(self.confidence), 4),
        }


@dataclass
class IncidentCandidate:
    """
    Standardized Universal Incident Candidate Structure.
    Emitted by any BaseIncidentDetector and processed by the IncidentFusionEngine.
    """
    incident_id: str
    video_id: str
    event_type: str
    category: str  # vehicle, person, property, crowd, zone, environment, multi_signal
    start_time: float
    end_time: float
    duration: float
    severity: str  # LOW, NORMAL, HIGH
    confidence: float  # 0.0 to 1.0 (calibrated assessment score)
    pattern_evidence_strength: float = 0.5  # Uncapped multi-signal telemetry evidence strength
    assessment_score: float = 0.5  # Calibrated final assessment score
    track_ids: List[str] = field(default_factory=list)
    object_classes: List[str] = field(default_factory=list)
    source_detection_ids: List[str] = field(default_factory=list)
    supporting_signals: List[SupportingSignal] = field(default_factory=list)
    contradictory_signals: List[SupportingSignal] = field(default_factory=list)
    spatial_context: Optional[SpatialContext] = None
    temporal_context: Optional[TemporalContext] = None
    explanation: str = ""
    evidence_candidates: List[EvidenceCandidate] = field(default_factory=list)
    validation_status: str = "VALID"
    validation_decision: str = "ACCEPTED"  # ACCEPTED, REVIEW_REQUIRED, REJECTED
    validation_reasons: List[str] = field(default_factory=list)
    evidence_eligible: bool = True
    evidence_ineligibility_reason: Optional[str] = None
    human_verification_required: bool = True
    detector_name: str = "universal_engine"
    detector_version: str = "1.0.0"
    incident_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def reasons(self) -> List[str]:
        return self.validation_reasons

    def to_security_event(self) -> SecurityEvent:
        """
        Convert IncidentCandidate to Sentinel's core SecurityEvent for 100% backward
        compatibility with database queries, timeline rendering, evidence vault, and reports.
        """
        # Collect formatted observable signal strings
        signals_text = [f"{s.signal_type}: {s.description}" for s in self.supporting_signals]
        if self.contradictory_signals:
            for cs in self.contradictory_signals:
                signals_text.append(f"{cs.signal_type}: {cs.description}")
        
        # Authoritative score telemetry: do not mix ACCEPTED with REVIEW_REQUIRED
        final_pct = int(round(self.assessment_score * 100))
        if self.validation_decision == "REVIEW_REQUIRED":
            signals_text.append(f"Final Assessment: {final_pct}% (REVIEW_REQUIRED) — Human verification required")
        elif self.validation_decision == "ACCEPTED":
            signals_text.append(f"Final Assessment: {final_pct}% (ACCEPTED)")

        primary_track = self.track_ids[0] if self.track_ids else None
        primary_class = self.object_classes[0] if self.object_classes else None
        if self.event_type == "POTENTIAL_THEFT":
            non_person_classes = [c for c in self.object_classes if c != "person"]
            if non_person_classes:
                primary_class = non_person_classes[0]
            elif len(self.object_classes) > 1:
                primary_class = self.object_classes[1]
        bbox = self.spatial_context.bounding_box if self.spatial_context else None
        zone_name = self.spatial_context.zone_name if self.spatial_context else None

        # Build SecurityEvent
        sec_ev = SecurityEvent(
            event_id=self.incident_id,
            event_type=self.event_type,
            severity=self.severity,
            timestamp=self.start_time,
            duration_seconds=self.duration,
            confidence=self.confidence,
            description=self.explanation,
            observable_signals=signals_text,
            track_id=primary_track,
            object_class=primary_class,
            zone_name=zone_name,
            bounding_box=bbox,
            detector_name=self.detector_name,
            detector_version=self.detector_version,
            category=self.category,
            human_verification_required=self.human_verification_required,
            incident_metadata={
                "incident_id": self.incident_id,
                "category": self.category,
                "pattern_evidence_strength": self.pattern_evidence_strength,
                "assessment_score": self.assessment_score,
                "track_ids": self.track_ids,
                "object_classes": self.object_classes,
                "evidence_candidates": [e.to_dict() for e in self.evidence_candidates],
                "spatial_context": self.spatial_context.to_dict() if self.spatial_context else None,
                "validation_decision": self.validation_decision,
                "validation_reasons": self.validation_reasons,
                "contradictory_signals": [s.to_dict() for s in self.contradictory_signals],
                "evidence_eligible": self.evidence_eligible,
                "evidence_ineligibility_reason": self.evidence_ineligibility_reason,
                **(self.incident_metadata or {}),
            },
        )
        return sec_ev

    def to_dict(self) -> Dict[str, Any]:
        from ai.common.numeric import ensure_finite
        return {
            "incident_id": self.incident_id,
            "video_id": self.video_id,
            "event_type": self.event_type,
            "category": self.category,
            "start_time": round(ensure_finite(self.start_time, 0.0), 4),
            "end_time": round(ensure_finite(self.end_time, 0.0), 4),
            "duration": round(ensure_finite(self.duration, 0.0), 4),
            "severity": self.severity,
            "confidence": round(ensure_finite(self.confidence, 0.0), 4),
            "track_ids": self.track_ids,
            "object_classes": self.object_classes,
            "source_detection_ids": self.source_detection_ids,
            "supporting_signals": [s.to_dict() for s in self.supporting_signals],
            "contradictory_signals": [s.to_dict() for s in self.contradictory_signals],
            "spatial_context": self.spatial_context.to_dict() if self.spatial_context else None,
            "temporal_context": self.temporal_context.to_dict() if self.temporal_context else None,
            "explanation": self.explanation,
            "evidence_candidates": [e.to_dict() for e in self.evidence_candidates],
            "validation_status": self.validation_status,
            "validation_decision": self.validation_decision,
            "validation_reasons": self.validation_reasons,
            "evidence_eligible": self.evidence_eligible,
            "evidence_ineligibility_reason": self.evidence_ineligibility_reason,
            "human_verification_required": self.human_verification_required,
            "detector_name": self.detector_name,
            "detector_version": self.detector_version,
            "created_at": self.created_at,
        }


@dataclass
class IncidentContext:
    """
    Complete structured context passed to detectors.
    Precludes individual detectors from directly querying the database or filesystem.
    """
    video_id: str
    fps: float = 30.0
    duration_seconds: float = 0.0
    sample_rate_fps: float = 1.0
    validated_detections: List[Dict[str, Any]] = field(default_factory=list)
    tracks: List[TrackedObject] = field(default_factory=list)
    vehicle_attributes: List[VehicleAttribute] = field(default_factory=list)
    face_detections: List[FaceDetection] = field(default_factory=list)
    zones: List[ZoneDefinition] = field(default_factory=list)
    track_motions: Dict[str, List[TrackMotion]] = field(default_factory=dict)  # track_id -> List[TrackMotion]
    motion_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # track_id -> summary stats
    scene_density: Dict[str, Any] = field(default_factory=dict)
    video_metadata: Dict[str, Any] = field(default_factory=dict)
    scene_context: Optional[Any] = None  # SceneContextData
    specialized_observations: List[Any] = field(default_factory=list)
    specialized_tracks: List[Any] = field(default_factory=list)
    specialized_episodes: Optional[List[Any]] = None


    def get_track_by_id(self, track_id: str) -> Optional[TrackedObject]:
        for t in self.tracks:
            if t.track_id == track_id:
                return t
        return None

    def get_tracks_by_class(self, object_class: str) -> List[TrackedObject]:
        return [t for t in self.tracks if t.object_class == object_class]

    def get_motion_summary(self, track_id: str) -> Optional[Dict[str, Any]]:
        return self.motion_summaries.get(track_id)

    # Canonical 1080p surveillance reference diagonal: sqrt(1920^2 + 1080^2) ~= 2202.906
    CANONICAL_REFERENCE_DIAGONAL: float = 2202.906

    @property
    def frame_dimensions(self) -> Tuple[float, float]:
        """Return (width, height) from video metadata, detections, or canonical default."""
        if self.video_metadata:
            w = self.video_metadata.get("width")
            h = self.video_metadata.get("height")
            if w and h and float(w) > 0 and float(h) > 0:
                return float(w), float(h)
        for d in self.validated_detections:
            w = d.get("image_width")
            h = d.get("image_height")
            if w and h and float(w) > 0 and float(h) > 0:
                return float(w), float(h)
        for t in self.tracks:
            if t.current_bbox:
                if t.current_bbox.x2 > 1920 or t.current_bbox.y2 > 1080:
                    return 3840.0, 2160.0
        return 1920.0, 1080.0

    @property
    def frame_diagonal(self) -> float:
        w, h = self.frame_dimensions
        return math.hypot(w, h)

    @property
    def resolution_scale_factor(self) -> float:
        """Scale factor relative to canonical 1080p reference frame."""
        diag = self.frame_diagonal
        return (diag / self.CANONICAL_REFERENCE_DIAGONAL) if diag > 0 else 1.0

    def normalize_distance(self, distance_pixels: float) -> float:
        """Convert a pixel distance to normalized fraction of frame diagonal."""
        diag = self.frame_diagonal
        return float(distance_pixels) / diag if diag > 0 else 0.0

    def denormalize_distance(self, normalized_fraction: float) -> float:
        """Convert a normalized fraction of frame diagonal to current frame pixels."""
        return float(normalized_fraction) * self.frame_diagonal
