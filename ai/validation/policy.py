"""
Sentinel Object Detection Validation Policy

Centralized, class-aware rules and geometric/temporal criteria for validating raw YOLO detections.
Prevents probabilistic noise, transient false positives, and degenerate bounding boxes from
propagating into downstream intelligence pipelines, while preserving legitimate distant and small objects.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, Tuple, Set


class ValidationStatus(str, Enum):
    VALID = "VALID"
    UNCERTAIN = "UNCERTAIN"
    REJECTED = "REJECTED"


@dataclass
class ClassValidationRule:
    """Class-specific validation thresholds and constraints."""
    min_confidence_standalone: float
    min_confidence_with_temporal: float
    min_area_fraction: float = 0.0001
    min_pixel_dimension: int = 10
    min_pixel_area: int = 100
    min_aspect_ratio: float = 0.05       # width / height min
    max_aspect_ratio: float = 15.0       # width / height max
    requires_temporal_support: bool = False
    context_boost_classes: Set[str] = field(default_factory=set)


# Standard COCO Class Profiles tailored for Surveillance CCTV
DEFAULT_CLASS_RULES: Dict[str, ClassValidationRule] = {
    # Heavy / Large Commercial Vehicles
    "bus": ClassValidationRule(
        min_confidence_standalone=0.45,
        min_confidence_with_temporal=0.32,
        min_area_fraction=0.0003,
        min_pixel_dimension=14,
        min_pixel_area=200,
        min_aspect_ratio=0.20,
        max_aspect_ratio=5.0,
        requires_temporal_support=True,  # Large vehicle in CCTV should have temporal support unless high conf
    ),
    "truck": ClassValidationRule(
        min_confidence_standalone=0.45,
        min_confidence_with_temporal=0.30,
        min_area_fraction=0.0003,
        min_pixel_dimension=14,
        min_pixel_area=200,
        min_aspect_ratio=0.20,
        max_aspect_ratio=5.0,
        requires_temporal_support=True,
    ),
    # Standard Motor & Human-Powered Vehicles
    "car": ClassValidationRule(
        min_confidence_standalone=0.38,
        min_confidence_with_temporal=0.28,
        min_area_fraction=0.00015,
        min_pixel_dimension=12,
        min_pixel_area=140,
        min_aspect_ratio=0.20,
        max_aspect_ratio=6.0,
        requires_temporal_support=False,
    ),
    "motorcycle": ClassValidationRule(
        min_confidence_standalone=0.38,
        min_confidence_with_temporal=0.28,
        min_area_fraction=0.00008,
        min_pixel_dimension=10,
        min_pixel_area=100,
        min_aspect_ratio=0.15,
        max_aspect_ratio=5.0,
        requires_temporal_support=False,
    ),
    "bicycle": ClassValidationRule(
        min_confidence_standalone=0.35,
        min_confidence_with_temporal=0.26,
        min_area_fraction=0.00008,
        min_pixel_dimension=10,
        min_pixel_area=100,
        min_aspect_ratio=0.15,
        max_aspect_ratio=5.0,
        requires_temporal_support=False,
    ),
    # Infrastructure & Roadside Elements
    "traffic light": ClassValidationRule(
        min_confidence_standalone=0.30,
        min_confidence_with_temporal=0.22,
        min_area_fraction=0.00003,
        min_pixel_dimension=8,
        min_pixel_area=60,
        min_aspect_ratio=0.10,
        max_aspect_ratio=6.0,
        requires_temporal_support=False,
    ),
    # Pedestrians / Persons (Crucial: allow distant legitimate people)
    "person": ClassValidationRule(
        min_confidence_standalone=0.35,
        min_confidence_with_temporal=0.25,
        min_area_fraction=0.00005,  # Distant people can be small
        min_pixel_dimension=10,
        min_pixel_area=100,
        min_aspect_ratio=0.10,     # Upright person: width/height ~ 0.25 to 0.5
        max_aspect_ratio=4.0,      # Crouching/crawling can be wider
        requires_temporal_support=False,
    ),
    # Luggage / Personal Items (May legitimately appear briefly during theft / handling)
    "suitcase": ClassValidationRule(
        min_confidence_standalone=0.30,
        min_confidence_with_temporal=0.24,
        min_area_fraction=0.00005,
        min_pixel_dimension=10,
        min_pixel_area=100,
        min_aspect_ratio=0.15,
        max_aspect_ratio=6.0,
        requires_temporal_support=False,
        context_boost_classes={"person"},
    ),
    "backpack": ClassValidationRule(
        min_confidence_standalone=0.30,
        min_confidence_with_temporal=0.24,
        min_area_fraction=0.00005,
        min_pixel_dimension=10,
        min_pixel_area=100,
        min_aspect_ratio=0.15,
        max_aspect_ratio=5.0,
        requires_temporal_support=False,
        context_boost_classes={"person"},
    ),
    "handbag": ClassValidationRule(
        min_confidence_standalone=0.30,
        min_confidence_with_temporal=0.24,
        min_area_fraction=0.00005,
        min_pixel_dimension=8,
        min_pixel_area=80,
        min_aspect_ratio=0.15,
        max_aspect_ratio=5.0,
        requires_temporal_support=False,
        context_boost_classes={"person"},
    ),
    "umbrella": ClassValidationRule(
        min_confidence_standalone=0.32,
        min_confidence_with_temporal=0.25,
        min_area_fraction=0.00005,
        min_pixel_dimension=10,
        min_pixel_area=100,
        min_aspect_ratio=0.15,
        max_aspect_ratio=6.0,
        requires_temporal_support=False,
        context_boost_classes={"person"},
    ),
    # Small Portable Belongings & Merchandise
    "cell phone": ClassValidationRule(
        min_confidence_standalone=0.28,
        min_confidence_with_temporal=0.20,
        min_area_fraction=0.00001,
        min_pixel_dimension=6,
        min_pixel_area=36,
        min_aspect_ratio=0.15,
        max_aspect_ratio=6.0,
        requires_temporal_support=False,
        context_boost_classes={"person"},
    ),
    "bottle": ClassValidationRule(
        min_confidence_standalone=0.28,
        min_confidence_with_temporal=0.20,
        min_area_fraction=0.000015,
        min_pixel_dimension=6,
        min_pixel_area=36,
        min_aspect_ratio=0.15,
        max_aspect_ratio=6.0,
        requires_temporal_support=False,
        context_boost_classes={"person"},
    ),
    "laptop": ClassValidationRule(
        min_confidence_standalone=0.30,
        min_confidence_with_temporal=0.22,
        min_area_fraction=0.00003,
        min_pixel_dimension=8,
        min_pixel_area=64,
        min_aspect_ratio=0.25,
        max_aspect_ratio=5.0,
        requires_temporal_support=False,
        context_boost_classes={"person"},
    ),
    "book": ClassValidationRule(
        min_confidence_standalone=0.30,
        min_confidence_with_temporal=0.22,
        min_area_fraction=0.00002,
        min_pixel_dimension=6,
        min_pixel_area=36,
        min_aspect_ratio=0.20,
        max_aspect_ratio=5.0,
        requires_temporal_support=False,
        context_boost_classes={"person"},
    ),
    "box": ClassValidationRule(
        min_confidence_standalone=0.30,
        min_confidence_with_temporal=0.22,
        min_area_fraction=0.00003,
        min_pixel_dimension=8,
        min_pixel_area=64,
        min_aspect_ratio=0.20,
        max_aspect_ratio=5.0,
        requires_temporal_support=False,
        context_boost_classes={"person"},
    ),
    "package": ClassValidationRule(
        min_confidence_standalone=0.30,
        min_confidence_with_temporal=0.22,
        min_area_fraction=0.00003,
        min_pixel_dimension=8,
        min_pixel_area=64,
        min_aspect_ratio=0.20,
        max_aspect_ratio=5.0,
        requires_temporal_support=False,
        context_boost_classes={"person"},
    ),
    "merchandise": ClassValidationRule(
        min_confidence_standalone=0.28,
        min_confidence_with_temporal=0.20,
        min_area_fraction=0.000015,
        min_pixel_dimension=6,
        min_pixel_area=36,
        min_aspect_ratio=0.15,
        max_aspect_ratio=6.0,
        requires_temporal_support=False,
        context_boost_classes={"person"},
    ),
    "general_object": ClassValidationRule(
        min_confidence_standalone=0.28,
        min_confidence_with_temporal=0.20,
        min_area_fraction=0.000015,
        min_pixel_dimension=6,
        min_pixel_area=36,
        min_aspect_ratio=0.10,
        max_aspect_ratio=8.0,
        requires_temporal_support=False,
        context_boost_classes={"person"},
    ),
}

# Fallback profile for unseen / unknown COCO classes
DEFAULT_FALLBACK_RULE = ClassValidationRule(
    min_confidence_standalone=0.45,
    min_confidence_with_temporal=0.32,
    min_area_fraction=0.0001,
    min_pixel_dimension=10,
    min_pixel_area=100,
    min_aspect_ratio=0.08,
    max_aspect_ratio=12.0,
    requires_temporal_support=False,
)


@dataclass
class DetectionValidationPolicy:
    """Central configuration for Sentinel detection validation."""
    temporal_window_seconds: float = 1.5
    temporal_match_iou: float = 0.10
    temporal_max_centroid_disp: float = 0.35  # Relative to frame dimensions
    context_proximity_distance: float = 0.40  # Normalized distance for person/luggage context
    high_confidence_override: float = 0.55    # Detections above this pass standalone if geometrically sound
    canonical_reference_width: float = 1920.0
    canonical_reference_height: float = 1080.0
    rules: Dict[str, ClassValidationRule] = field(default_factory=lambda: dict(DEFAULT_CLASS_RULES))
    fallback_rule: ClassValidationRule = field(default_factory=lambda: DEFAULT_FALLBACK_RULE)

    def get_rule(self, object_class: Optional[str]) -> ClassValidationRule:
        if not object_class:
            return self.fallback_rule
        return self.rules.get(object_class.lower().strip(), self.fallback_rule)
