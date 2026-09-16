"""
Person Incident Intelligence Core Package (Sentinel Phase 12)
Exposes core person feature engines and detectors.
"""
from ai.incidents.detectors.person import (
    PersonMotionFeatureEngine,
    PoseFeatureEngine,
    PersonFallDetector,
    PersonDownDetector,
    PanicRunningDetector,
    UnusualRapidPersonMovementDetector,
    PhysicalAltercationDetector,
    ForcedMovementDetector,
    PersonFollowingDetector,
    CoordinatedPersonMovementDetector,
)

__all__ = [
    "PersonMotionFeatureEngine",
    "PoseFeatureEngine",
    "PersonFallDetector",
    "PersonDownDetector",
    "PanicRunningDetector",
    "UnusualRapidPersonMovementDetector",
    "PhysicalAltercationDetector",
    "ForcedMovementDetector",
    "PersonFollowingDetector",
    "CoordinatedPersonMovementDetector",
]
