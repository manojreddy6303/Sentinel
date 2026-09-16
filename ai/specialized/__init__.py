"""
Sentinel Specialized Visual Detection Core (Phase 15)

Provides specialized, failure-isolated visual models for:
- Fire detection (POTENTIAL_FIRE)
- Smoke detection (POTENTIAL_SMOKE)
- Weapon / suspicious object detection foundation (POTENTIAL_WEAPON_VISUAL)
- Pose / action telemetry foundation
- Multi-frame temporal validation and negative evidence
"""
from ai.specialized.schemas import (
    SpecializedDetectorStatus,
    SpecializedValidationStatus,
    SpecializedObservation,
    SpecializedModelInfo,
    SpecializedTemporalTrack,
)
from ai.specialized.base import BaseSpecializedDetector
from ai.specialized.registry import (
    SpecializedDetectorRegistry,
    get_specialized_registry,
)
from ai.specialized.temporal import SpecializedTemporalTracker
from ai.specialized.negative_evidence import SpecializedNegativeEvidenceEngine
from ai.specialized.fire_smoke import FireVisualDetector, SmokeVisualDetector
from ai.specialized.weapon import WeaponVisualDetector
from ai.specialized.pose import PoseActionDetector
from ai.specialized.validator import SpecializedValidationEngine

__all__ = [
    "SpecializedDetectorStatus",
    "SpecializedValidationStatus",
    "SpecializedObservation",
    "SpecializedModelInfo",
    "SpecializedTemporalTrack",
    "BaseSpecializedDetector",
    "SpecializedDetectorRegistry",
    "get_specialized_registry",
    "SpecializedTemporalTracker",
    "SpecializedNegativeEvidenceEngine",
    "FireVisualDetector",
    "SmokeVisualDetector",
    "WeaponVisualDetector",
    "PoseActionDetector",
    "SpecializedValidationEngine",
]
