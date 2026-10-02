"""
Phase 8: Advanced Security Intelligence Core Schemas
"""
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from ai.common.numeric import is_finite_number, ensure_finite, clamp_finite


class DetectionValidationStatus(str, Enum):
    RAW = "RAW"
    VALID = "VALID"
    UNCERTAIN = "UNCERTAIN"
    REJECTED = "REJECTED"


class TrackLifecycleState(str, Enum):
    TENTATIVE = "TENTATIVE"    # 1 detection, awaiting confirmation
    CONFIRMED = "CONFIRMED"    # >= 2 detections with sufficient confidence
    OCCLUDED = "OCCLUDED"      # Temporarily missed due to known overlap/occlusion
    COASTING = "COASTING"      # Missed 1-2 frames within tolerance window
    LOST = "LOST"              # Exceeded coasting window
    ENDED = "ENDED"            # Stream ended or track cleanly exited


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self):
        if not math.isfinite(self.x1) or not math.isfinite(self.y1) or not math.isfinite(self.x2) or not math.isfinite(self.y2):
            raise ValueError(f"BoundingBox coordinates must be finite numbers: ({self.x1}, {self.y1}, {self.x2}, {self.y2})")
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"BoundingBox inverted coordinates x2 < x1 or y2 < y1: ({self.x1}, {self.y1}, {self.x2}, {self.y2})")
        self.x1 = float(self.x1)
        self.y1 = float(self.y1)
        self.x2 = float(self.x2)
        self.y2 = float(self.y2)

    @property
    def is_valid(self) -> bool:
        return (
            math.isfinite(self.x1) and math.isfinite(self.y1) and
            math.isfinite(self.x2) and math.isfinite(self.y2) and
            self.x2 > self.x1 and self.y2 > self.y1
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "x1": round(ensure_finite(self.x1, 0.0), 4),
            "y1": round(ensure_finite(self.y1, 0.0), 4),
            "x2": round(ensure_finite(self.x2, 0.0), 4),
            "y2": round(ensure_finite(self.y2, 0.0), 4),
        }

    @property
    def centroid(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def is_edge_clipped(self, frame_width: Optional[float] = None, frame_height: Optional[float] = None, margin: float = 4.0) -> bool:
        """Check if bounding box touches or is truncated by any camera frame edge."""
        if self.x1 <= margin or self.y1 <= margin:
            return True
        if frame_width is not None and frame_width > 0 and self.x2 >= (frame_width - margin):
            return True
        if frame_height is not None and frame_height > 0 and self.y2 >= (frame_height - margin):
            return True
        return False

    def edge_clip_boundaries(self, frame_width: Optional[float] = None, frame_height: Optional[float] = None, margin: float = 4.0) -> List[str]:
        """Return list of touched frame boundaries ('left', 'top', 'right', 'bottom')."""
        boundaries = []
        if self.x1 <= margin:
            boundaries.append("left")
        if self.y1 <= margin:
            boundaries.append("top")
        if frame_width is not None and frame_width > 0 and self.x2 >= (frame_width - margin):
            boundaries.append("right")
        if frame_height is not None and frame_height > 0 and self.y2 >= (frame_height - margin):
            boundaries.append("bottom")
        return boundaries

    def iou(self, other: "BoundingBox") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union_area = self.area + other.area - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area


@dataclass
class CanonicalDetection:
    """
    Universal Standardized Detection Contract (Phase 15.1 Hardened).
    Standardizes all object detector outputs across YOLO and specialized detectors.
    Provides complete backward compatibility with dict-based consumers via mapping methods.
    """
    video_id: str
    frame_index: int
    timestamp_seconds: float
    class_name: str
    confidence: float
    bounding_box: BoundingBox
    detector_name: str = "yolo_detector"
    detector_version: str = "1.0.0"
    validation_status: DetectionValidationStatus = DetectionValidationStatus.RAW
    observation_source: str = "yolo_detector"
    model_or_heuristic: str = "trained_model"
    class_id: Optional[int] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    is_edge_clipped: bool = False
    edge_clip_boundaries: List[str] = field(default_factory=list)
    validation_score: Optional[float] = None
    validation_reason: Optional[str] = None
    track_id: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.timestamp_seconds = ensure_finite(self.timestamp_seconds, 0.0)
        self.confidence = clamp_finite(self.confidence, 0.0, 1.0, default=0.0)
        if self.validation_score is not None:
            self.validation_score = clamp_finite(self.validation_score, 0.0, 1.0, default=0.0)

        # Enforce bounding box validity invariant
        if not self.bounding_box.is_valid:
            self.validation_status = DetectionValidationStatus.REJECTED
            if not self.validation_reason:
                self.validation_reason = "Invalid bounding box geometry (zero area or non-positive dimensions)"

        # Check edge clipping if dimensions provided
        if not self.is_edge_clipped and (self.image_width or self.image_height):
            self.is_edge_clipped = self.bounding_box.is_edge_clipped(self.image_width, self.image_height)
            self.edge_clip_boundaries = self.bounding_box.edge_clip_boundaries(self.image_width, self.image_height)

    def to_dict(self) -> Dict[str, Any]:
        val_status = self.validation_status.value if isinstance(self.validation_status, DetectionValidationStatus) else str(self.validation_status)
        return {
            "video_id": self.video_id,
            "frame_index": self.frame_index,
            "timestamp_seconds": round(ensure_finite(self.timestamp_seconds, 0.0), 4),
            "timestamp": round(ensure_finite(self.timestamp_seconds, 0.0), 4),  # Backward compatibility
            "class_name": self.class_name,
            "object_class": self.class_name,  # Backward compatibility
            "class_id": self.class_id,
            "confidence": round(ensure_finite(self.confidence, 0.0), 4),
            "bounding_box": self.bounding_box.to_dict(),
            "detector_name": self.detector_name,
            "detector_version": self.detector_version,
            "validation_status": val_status,
            "observation_source": self.observation_source,
            "model_or_heuristic": self.model_or_heuristic,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "is_edge_clipped": self.is_edge_clipped,
            "edge_clip_boundaries": self.edge_clip_boundaries,
            "validation_score": round(self.validation_score, 4) if self.validation_score is not None else None,
            "validation_reason": self.validation_reason,
            "track_id": self.track_id,
            "metrics": self.metrics,
        }

    # Dictionary emulation interface for 100% backward compatibility
    def __getitem__(self, item: str) -> Any:
        return self.to_dict()[item]

    def get(self, item: str, default: Any = None) -> Any:
        return self.to_dict().get(item, default)

    def __contains__(self, item: str) -> bool:
        return item in self.to_dict()


@dataclass
class TrackedObject:
    track_id: str
    object_class: str
    first_seen: float
    last_seen: float
    confidence: float
    current_bbox: BoundingBox
    trajectory: List[Tuple[float, float, float]] = field(default_factory=list)  # [(timestamp, cx, cy), ...]
    history_bboxes: List[Dict[str, Any]] = field(default_factory=list)  # [{"timestamp": t, "bbox": {...}}]
    active: bool = True
    color: Optional[str] = None
    color_confidence: Optional[float] = None
    state: TrackLifecycleState = TrackLifecycleState.TENTATIVE
    video_id: str = ""
    visual_attributes: Optional[Dict[str, Any]] = None
    attribute_history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.first_seen = ensure_finite(self.first_seen, 0.0)
        self.last_seen = ensure_finite(self.last_seen, 0.0)
        self.confidence = clamp_finite(self.confidence, 0.0, 1.0, default=0.0)

    @property
    def duration_seconds(self) -> float:
        return max(0.0, round(ensure_finite(self.last_seen - self.first_seen, 0.0), 4))

    @property
    def detection_count(self) -> int:
        return len(self.history_bboxes)

    @property
    def is_validated(self) -> bool:
        """
        Distinguish persistent, reliable visual tracks from weak isolated single-frame detections.
        - Multi-frame tracks (detection_count >= 2) are verified.
        - For vehicle classes ('bus', 'truck', 'car'), require >= 2 detections OR confidence >= 0.65.
          Weak single-frame vehicle predictions (like isolated indoor noise) are rejected.
        - Non-vehicle objects (people, belongings, portable items) are valid tracks.
        """
        count = len(self.history_bboxes)
        if count >= 2:
            return True
        if self.object_class in {"bus", "truck", "car"}:
            return self.confidence >= 0.65
        return True

    def to_dict(self) -> Dict[str, Any]:
        st = self.state.value if isinstance(self.state, TrackLifecycleState) else str(self.state)
        return {
            "track_id": self.track_id,
            "video_id": self.video_id,
            "object_class": self.object_class,
            "first_seen": round(ensure_finite(self.first_seen, 0.0), 4),
            "last_seen": round(ensure_finite(self.last_seen, 0.0), 4),
            "duration_seconds": self.duration_seconds,
            "confidence": round(ensure_finite(self.confidence, 0.0), 4),
            "current_bbox": self.current_bbox.to_dict(),
            "trajectory": self.trajectory,
            "detection_count": len(self.history_bboxes),
            "active": self.active,
            "state": st,
            "is_validated": self.is_validated,
            "color": self.color,
            "color_confidence": round(self.color_confidence, 4) if self.color_confidence is not None else None,
            "visual_attributes": self.visual_attributes,
            "attribute_history": self.attribute_history,
        }


@dataclass
class ClothingColor:
    """Non-biometric clothing color attribute with illumination and temporal confidence."""
    color_name: str
    confidence: float
    observation_count: int = 1
    is_illumination_uncertain: bool = False
    region: str = "upper"  # upper, lower, outerwear, carried_object
    color_space_metrics: Optional[Dict[str, Any]] = None

    @property
    def color(self) -> str:
        return self.color_name

    @property
    def is_confirmed(self) -> bool:
        return self.observation_count >= 3 and not self.is_illumination_uncertain and self.confidence >= 0.60

    def to_dict(self) -> Dict[str, Any]:
        return {
            "color_name": self.color_name,
            "color": self.color_name,
            "confidence": round(ensure_finite(self.confidence, 0.0), 4),
            "observation_count": self.observation_count,
            "is_confirmed": self.is_confirmed,
            "is_illumination_uncertain": self.is_illumination_uncertain,
            "region": self.region,
            "color_space_metrics": self.color_space_metrics or {},
        }


@dataclass
class FaceRegionTelemetry:
    """
    CRITICAL NON-BIOMETRIC PRIVACY BOUNDARY:
    Localizes face region coordinates and optical properties only.
    Contains ABSOLUTELY ZERO identity tokens, facial recognition embeddings,
    database lookup references, or biometric profiles.
    """
    face_present: bool
    visibility_score: float = 0.0
    quality_score: float = 0.0
    approximate_orientation: str = "unknown"  # frontal, profile_left, profile_right, downward, unknown
    is_occluded: bool = False
    sharpness_score: float = 0.0
    resolution: Tuple[int, int] = (0, 0)
    face_bbox: Optional[BoundingBox] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "face_present": self.face_present,
            "visibility_score": round(ensure_finite(self.visibility_score, 0.0), 4),
            "quality_score": round(ensure_finite(self.quality_score, 0.0), 4),
            "approximate_orientation": self.approximate_orientation,
            "is_occluded": self.is_occluded,
            "sharpness_score": round(ensure_finite(self.sharpness_score, 0.0), 2),
            "resolution": list(self.resolution),
            "face_bbox": self.face_bbox.to_dict() if self.face_bbox else None,
        }


@dataclass
class PersonVisualAttributes:
    """Universal non-biometric visual attributes for a tracked person."""
    track_id: str
    timestamp: float
    bounding_box: BoundingBox
    upper_clothing_color: ClothingColor
    lower_clothing_color: Optional[ClothingColor] = None
    outerwear_color: Optional[ClothingColor] = None
    carried_object_category: Optional[str] = None
    carried_object_color: Optional[str] = None
    headwear: Optional[str] = None
    face_telemetry: Optional[FaceRegionTelemetry] = None
    occlusion_level: float = 0.0
    posture: str = "standing"  # standing, walking, sitting, bending, unknown

    @property
    def upper_clothing(self) -> ClothingColor:
        return self.upper_clothing_color

    @property
    def lower_clothing(self) -> Optional[ClothingColor]:
        return self.lower_clothing_color

    @property
    def outerwear(self) -> Optional[ClothingColor]:
        return self.outerwear_color

    @property
    def headwear_present(self) -> bool:
        return self.headwear is not None and self.headwear != "none"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "timestamp": round(ensure_finite(self.timestamp, 0.0), 4),
            "bounding_box": self.bounding_box.to_dict(),
            "upper_clothing_color": self.upper_clothing_color.to_dict() if self.upper_clothing_color else None,
            "lower_clothing_color": self.lower_clothing_color.to_dict() if self.lower_clothing_color else None,
            "outerwear_color": self.outerwear_color.to_dict() if self.outerwear_color else None,
            "carried_object_category": self.carried_object_category,
            "carried_object_color": self.carried_object_color,
            "headwear": self.headwear,
            "face_telemetry": self.face_telemetry.to_dict() if self.face_telemetry else None,
            "occlusion_level": round(ensure_finite(self.occlusion_level, 0.0), 4),
            "posture": self.posture,
        }


@dataclass
class GeneralObjectAttributes:
    """Universal physical attributes for detected surveillance items."""
    object_class: str
    color: str = "unknown"
    confidence: float = 0.0
    size: Tuple[float, float] = (0.0, 0.0)
    position: Tuple[float, float] = (0.0, 0.0)
    persistence_seconds: float = 0.0
    interaction_state: str = "stationary"  # stationary, carried, in_motion
    track_id: Optional[str] = None
    timestamp: float = 0.0
    bounding_box: Optional[BoundingBox] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_class": self.object_class,
            "color": self.color,
            "confidence": round(ensure_finite(self.confidence, 0.0), 4),
            "size": list(self.size),
            "position": list(self.position),
            "persistence_seconds": round(ensure_finite(self.persistence_seconds, 0.0), 4),
            "interaction_state": self.interaction_state,
            "track_id": self.track_id,
            "timestamp": round(ensure_finite(self.timestamp, 0.0), 4),
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
        }


@dataclass
class VehicleAttribute:
    color: str
    confidence: float
    timestamp: float
    bounding_box: BoundingBox
    object_class: str = "car"
    track_id: Optional[str] = None
    color_space_metrics: Optional[Dict[str, Any]] = None
    secondary_color: Optional[str] = None
    secondary_confidence: Optional[float] = None
    approximate_orientation: str = "unknown"
    is_occluded: bool = False
    is_illumination_uncertain: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "color": self.color,
            "confidence": round(self.confidence, 4),
            "secondary_color": self.secondary_color,
            "secondary_confidence": round(self.secondary_confidence, 4) if self.secondary_confidence is not None else None,
            "approximate_orientation": self.approximate_orientation,
            "is_occluded": self.is_occluded,
            "is_illumination_uncertain": self.is_illumination_uncertain,
            "timestamp": round(self.timestamp, 4),
            "bounding_box": self.bounding_box.to_dict(),
            "object_class": self.object_class,
            "track_id": self.track_id,
            "color_space_metrics": self.color_space_metrics or {},
        }


@dataclass
class FaceDetection:
    """
    CRITICAL SAFETY CONSTRAINT:
    Contains visual bounding box and confidence ONLY.
    Never contains identity, name, facial recognition embeddings, or biometric tokens.
    """
    timestamp: float
    bounding_box: BoundingBox
    confidence: float
    track_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 4),
            "bounding_box": self.bounding_box.to_dict(),
            "confidence": round(self.confidence, 4),
            "track_id": self.track_id,
        }


@dataclass
class ZoneDefinition:
    zone_id: str
    name: str
    polygon: List[Tuple[float, float]]  # Normalized or pixel coordinates [(x, y), ...]
    target_classes: List[str] = field(default_factory=lambda: ["person"])
    enabled: bool = True
    alert_on_entry: bool = True
    loitering_threshold_seconds: float = 30.0

    @property
    def zone_name(self) -> str:
        return self.name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "zone_name": self.name,
            "polygon": self.polygon,
            "target_classes": self.target_classes,
            "enabled": self.enabled,
            "alert_on_entry": self.alert_on_entry,
            "loitering_threshold_seconds": self.loitering_threshold_seconds,
        }


@dataclass
class SecurityEvent:
    event_type: str  # POTENTIAL_INTRUSION, PROLONGED_PRESENCE, POTENTIAL_ABANDONED_OBJECT, HIGH_ACTIVITY_PERIOD, OBSERVATIONAL_ANOMALY, POTENTIAL_THEFT
    severity: str  # LOW, NORMAL, HIGH
    timestamp: float
    duration_seconds: float
    confidence: float
    description: str
    observable_signals: List[str] = field(default_factory=list)
    track_id: Optional[str] = None
    object_class: Optional[str] = None
    zone_name: Optional[str] = None
    bounding_box: Optional[BoundingBox] = None
    event_id: Optional[str] = None
    evidence_id: Optional[str] = None
    # Spatial grounding & supporting evidence fields
    person_track_id: Optional[str] = None
    object_track_id: Optional[str] = None
    object_bounding_box: Optional[BoundingBox] = None
    object_observation_timestamp: Optional[float] = None
    is_prior_object_observation: bool = False
    detector_name: Optional[str] = None
    detector_version: Optional[str] = None
    category: Optional[str] = None
    human_verification_required: bool = True
    validation_decision: str = "ACCEPTED"
    pattern_evidence_strength: Optional[float] = None
    assessment_score: Optional[float] = None
    incident_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        p_str = self.pattern_evidence_strength if self.pattern_evidence_strength is not None else self.confidence
        a_score = self.assessment_score if self.assessment_score is not None else self.confidence
        res = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "severity": self.severity,
            "timestamp": round(self.timestamp, 4),
            "duration_seconds": round(self.duration_seconds, 4),
            "confidence": round(self.confidence, 4),
            "pattern_evidence_strength": round(float(p_str), 4),
            "assessment_score": round(float(a_score), 4),
            "description": self.description,
            "observable_signals": self.observable_signals,
            "track_id": self.track_id,
            "object_class": self.object_class,
            "zone_name": self.zone_name,
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "evidence_id": self.evidence_id,
            "person_track_id": self.person_track_id,
            "object_track_id": self.object_track_id,
            "object_bounding_box": self.object_bounding_box.to_dict() if self.object_bounding_box else None,
            "object_observation_timestamp": round(self.object_observation_timestamp, 4) if self.object_observation_timestamp is not None else None,
            "is_prior_object_observation": self.is_prior_object_observation,
            "detector_name": self.detector_name,
            "detector_version": self.detector_version,
            "category": self.category,
            "human_verification_required": self.human_verification_required,
            "validation_decision": self.validation_decision,
            "incident_metadata": self.incident_metadata,
        }
        return res
