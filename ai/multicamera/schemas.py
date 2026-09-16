"""
ai/multicamera/schemas.py — Phase 18 Typed Dataclasses

Defines the pure-Python data structures used throughout the cross-camera
association pipeline. These types carry NO biometric data, NO identity
attributes, and NO facial features.

PRIVACY: All schemas represent observable object telemetry only.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any
import uuid


class AssociationType(str, Enum):
    """
    Classification of cross-camera association confidence.

    SAME_OBJECT   : confidence >= 0.80 — strong multi-signal corroboration
    PROBABLE_SAME : confidence 0.60–0.79 — multiple weak signals agree
    POSSIBLE_SAME : confidence 0.40–0.59 — limited corroborating signals
    (below 0.40)  : no association created — correct abstention

    NOTE: These labels describe observable-attribute similarity, NOT identity.
    """
    SAME_OBJECT = "SAME_OBJECT"
    PROBABLE_SAME = "PROBABLE_SAME"
    POSSIBLE_SAME = "POSSIBLE_SAME"


class AnalystVerdict(str, Enum):
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"


@dataclass
class CameraObservation:
    """
    A summary of one tracked object as seen from one camera.

    All fields are derived from observable CV telemetry.
    No biometric fields are present.
    """
    camera_id: str                          # CameraSourceModel.id
    camera_label: str                       # Human-readable label
    video_id: str                           # VideoModel.id
    track_id: str                           # TrackModel.track_id
    object_class: str                       # e.g. "person", "car"
    first_seen: float                       # seconds into video
    last_seen: float                        # seconds into video
    duration_seconds: float
    max_confidence: float

    # Observable attributes (NO biometrics)
    color: Optional[str] = None             # Color vocabulary: "red", "blue", etc.
    color_confidence: Optional[float] = None
    estimated_height_ratio: Optional[float] = None  # bbox height / frame height
    estimated_width_ratio: Optional[float] = None   # bbox width / frame width

    # Trajectory telemetry at track exit/entry
    exit_direction_degrees: Optional[float] = None   # direction track left camera FOV
    entry_direction_degrees: Optional[float] = None  # direction track entered camera FOV
    exit_position: Optional[tuple] = None            # (cx_norm, cy_norm) at last detection
    entry_position: Optional[tuple] = None           # (cx_norm, cy_norm) at first detection

    # Adjacency context
    adjacent_camera_labels: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "camera_label": self.camera_label,
            "video_id": self.video_id,
            "track_id": self.track_id,
            "object_class": self.object_class,
            "first_seen": round(self.first_seen, 4),
            "last_seen": round(self.last_seen, 4),
            "duration_seconds": round(self.duration_seconds, 4),
            "max_confidence": round(self.max_confidence, 4),
            "color": self.color,
            "color_confidence": round(self.color_confidence, 4) if self.color_confidence is not None else None,
            "estimated_height_ratio": round(self.estimated_height_ratio, 4) if self.estimated_height_ratio is not None else None,
            "estimated_width_ratio": round(self.estimated_width_ratio, 4) if self.estimated_width_ratio is not None else None,
            "exit_direction_degrees": round(self.exit_direction_degrees, 2) if self.exit_direction_degrees is not None else None,
            "entry_direction_degrees": round(self.entry_direction_degrees, 2) if self.entry_direction_degrees is not None else None,
            "exit_position": list(self.exit_position) if self.exit_position else None,
            "entry_position": list(self.entry_position) if self.entry_position else None,
            "adjacent_camera_labels": self.adjacent_camera_labels,
        }


@dataclass
class AssociationEvidence:
    """
    One observable signal contributing to a cross-camera association.

    PRIVACY: signal_type must NEVER be a biometric type.
    Permitted signal types:
      - "object_class_match"
      - "color_match"
      - "size_ratio_match"
      - "trajectory_direction_compatibility"
      - "temporal_gap_plausibility"
      - "adjacency_hint"
    """
    signal_type: str
    description: str
    score: float                  # 0.0–1.0 contribution to association confidence
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Safeguard: reject biometric signal types at construction time
    _FORBIDDEN_TYPES = frozenset({
        "facial_similarity", "face_match", "biometric", "identity",
        "voice_match", "gait_recognition", "iris_match",
    })

    def __post_init__(self):
        if self.signal_type.lower() in self._FORBIDDEN_TYPES:
            raise ValueError(
                f"AssociationEvidence: forbidden signal_type '{self.signal_type}'. "
                "Cross-camera association must not use biometric signals."
            )
        self.score = max(0.0, min(1.0, float(self.score)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "description": self.description,
            "score": round(self.score, 4),
            "metadata": self.metadata,
        }


@dataclass
class CrossCameraHypothesis:
    """
    The hypothesis that a track observed in source_camera is the same
    physical object as a track observed in target_camera.

    This is an evidence-supported hypothesis, NOT an identity assertion.
    analyst_review_required is always True on creation.
    """
    hypothesis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""

    # Source observation
    source_camera_id: str = ""
    source_video_id: str = ""
    source_track_id: str = ""
    source_last_seen: Optional[float] = None

    # Target observation
    target_camera_id: str = ""
    target_video_id: str = ""
    target_track_id: str = ""
    target_first_seen: Optional[float] = None

    # Association scoring
    confidence: float = 0.0
    association_type: AssociationType = AssociationType.POSSIBLE_SAME
    attribute_match_score: float = 0.0
    trajectory_compatibility_score: float = 0.0
    temporal_gap_seconds: Optional[float] = None
    temporal_plausibility_score: float = 0.0

    # Evidence provenance (observable attributes only)
    evidence_basis: List[AssociationEvidence] = field(default_factory=list)

    # Analyst workflow — always pending on creation
    analyst_review_required: bool = True
    analyst_verdict: AnalystVerdict = AnalystVerdict.PENDING
    analyst_notes: Optional[str] = None

    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def classify_type(self) -> AssociationType:
        """Derive association_type from confidence score."""
        if self.confidence >= 0.80:
            return AssociationType.SAME_OBJECT
        elif self.confidence >= 0.60:
            return AssociationType.PROBABLE_SAME
        elif self.confidence >= 0.40:
            return AssociationType.POSSIBLE_SAME
        else:
            raise ValueError(f"Confidence {self.confidence:.3f} is below abstention threshold 0.40")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "session_id": self.session_id,
            "source_camera_id": self.source_camera_id,
            "source_video_id": self.source_video_id,
            "source_track_id": self.source_track_id,
            "source_last_seen": round(self.source_last_seen, 4) if self.source_last_seen is not None else None,
            "target_camera_id": self.target_camera_id,
            "target_video_id": self.target_video_id,
            "target_track_id": self.target_track_id,
            "target_first_seen": round(self.target_first_seen, 4) if self.target_first_seen is not None else None,
            "confidence": round(self.confidence, 4),
            "association_type": self.association_type.value,
            "attribute_match_score": round(self.attribute_match_score, 4),
            "trajectory_compatibility_score": round(self.trajectory_compatibility_score, 4),
            "temporal_gap_seconds": round(self.temporal_gap_seconds, 4) if self.temporal_gap_seconds is not None else None,
            "temporal_plausibility_score": round(self.temporal_plausibility_score, 4),
            "evidence_basis": [e.to_dict() for e in self.evidence_basis],
            "analyst_review_required": self.analyst_review_required,
            "analyst_verdict": self.analyst_verdict.value,
            "analyst_notes": self.analyst_notes,
            "created_at": self.created_at,
        }
