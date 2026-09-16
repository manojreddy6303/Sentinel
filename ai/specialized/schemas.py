"""
Specialized Visual Detection Schemas (Phase 15)

Defines data models for specialized visual observations, model statuses,
and temporal telemetry.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple

from ai.schemas import BoundingBox


class SpecializedDetectorStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    FAILED = "FAILED"


class SpecializedValidationStatus(str, Enum):
    RAW = "RAW"
    VALID = "VALID"
    UNCERTAIN = "UNCERTAIN"
    REJECTED = "REJECTED"


@dataclass
class SpecializedModelInfo:
    """Metadata describing a specialized visual model."""
    name: str
    version: str = "1.0.0"
    source: str = "internal_heuristic"  # e.g., "yolov8_custom", "internal_heuristic"
    license: str = "Proprietary / Sentinel Core"
    input_format: str = "BGR Frame"
    output_format: str = "BoundingBox + Confidence + Visual Metrics"
    expected_resources: str = "Lightweight CPU / GPU optional"
    status: SpecializedDetectorStatus = SpecializedDetectorStatus.AVAILABLE
    status_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "source": self.source,
            "license": self.license,
            "input_format": self.input_format,
            "output_format": self.output_format,
            "expected_resources": self.expected_resources,
            "status": self.status.value if isinstance(self.status, SpecializedDetectorStatus) else str(self.status),
            "status_reason": self.status_reason,
        }


@dataclass
class SpecializedObservation:
    """
    Single-frame raw or validated visual observation from a specialized detector.
    Does NOT equate to a definitive security incident on its own.
    """
    observation_id: str
    detector_name: str
    detector_version: str
    class_name: str  # e.g., "fire", "smoke", "weapon_handgun", "weapon_knife", "pose_lying"
    timestamp: float
    confidence: float
    evidence_strength: float  # Calibrated non-probabilistic semantic score [0.0, 1.0]
    bounding_box: Optional[BoundingBox] = None
    validation_status: SpecializedValidationStatus = SpecializedValidationStatus.RAW
    validation_reasons: List[str] = field(default_factory=list)
    visual_metrics: Dict[str, Any] = field(default_factory=dict)
    frame_number: Optional[int] = None
    episode_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "detector_name": self.detector_name,
            "detector_version": self.detector_version,
            "class_name": self.class_name,
            "timestamp": round(float(self.timestamp), 4),
            "confidence": round(float(self.confidence), 4),
            "evidence_strength": round(float(self.evidence_strength), 4),
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "validation_status": self.validation_status.value if isinstance(self.validation_status, SpecializedValidationStatus) else str(self.validation_status),
            "validation_reasons": self.validation_reasons,
            "visual_metrics": self.visual_metrics,
            "frame_number": self.frame_number,
            "episode_id": self.episode_id,
            "created_at": self.created_at,
        }


@dataclass
class SpecializedTemporalTrack:
    """
    Tracks a sustained specialized visual phenomenon across contiguous frames.
    Prevents single-frame transient anomalies from triggering false alarms.
    """
    track_id: str
    class_name: str
    detector_name: str
    first_seen: float
    last_seen: float
    observation_count: int = 1
    max_confidence: float = 0.0
    mean_confidence: float = 0.0
    confidence_history: List[float] = field(default_factory=list)
    bounding_boxes: List[Dict[str, Any]] = field(default_factory=list)
    visual_metrics_history: List[Dict[str, Any]] = field(default_factory=list)
    spatial_dispersion: float = 0.0  # Centroid movement over time
    episode_id: Optional[str] = None

    @property
    def persistence_duration(self) -> float:
        from ai.common.numeric import ensure_finite
        first = ensure_finite(self.first_seen, default=0.0) or 0.0
        last = ensure_finite(self.last_seen, default=first) or first
        return max(0.0, last - first)

    def add_observation(self, obs: SpecializedObservation) -> None:
        from ai.common.numeric import ensure_finite
        obs_ts = ensure_finite(obs.timestamp, default=self.last_seen)
        if obs_ts is not None:
            self.last_seen = obs_ts
        self.observation_count += 1
        conf = ensure_finite(obs.confidence, default=0.0) or 0.0
        self.confidence_history.append(conf)
        self.max_confidence = max(self.max_confidence, conf)
        if self.confidence_history:
            self.mean_confidence = sum(self.confidence_history) / len(self.confidence_history)
        else:
            self.mean_confidence = conf
        if obs.bounding_box:
            self.bounding_boxes.append(obs.bounding_box.to_dict())
        if obs.visual_metrics:
            self.visual_metrics_history.append(obs.visual_metrics)

    def to_dict(self) -> Dict[str, Any]:
        from ai.common.numeric import ensure_finite
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "detector_name": self.detector_name,
            "first_seen": round(ensure_finite(self.first_seen, 0.0), 4),
            "last_seen": round(ensure_finite(self.last_seen, 0.0), 4),
            "persistence_duration": round(ensure_finite(self.persistence_duration, 0.0), 4),
            "observation_count": self.observation_count,
            "max_confidence": round(ensure_finite(self.max_confidence, 0.0), 4),
            "mean_confidence": round(ensure_finite(self.mean_confidence, 0.0), 4),
            "bounding_boxes_count": len(self.bounding_boxes),
            "episode_id": self.episode_id,
        }


