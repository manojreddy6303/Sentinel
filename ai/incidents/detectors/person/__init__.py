"""
Person Incident Intelligence Detectors Package (Sentinel Phase 12)

Exports modular detectors for observable person incidents:
- PersonFallDetector (POTENTIAL_PERSON_FALL)
- PersonDownDetector (POTENTIAL_PERSON_DOWN)
- PanicRunningDetector (POTENTIAL_PANIC_RUNNING)
- UnusualRapidPersonMovementDetector (UNUSUAL_RAPID_PERSON_MOVEMENT)
- PhysicalAltercationDetector (POTENTIAL_PHYSICAL_ALTERCATION)
- ForcedMovementDetector (POTENTIAL_FORCED_MOVEMENT)
- PersonFollowingDetector (PERSON_FOLLOWING)
- CoordinatedPersonMovementDetector (COORDINATED_PERSON_MOVEMENT)
"""
from ai.incidents.detectors.person.motion_features import PersonMotionFeatureEngine
from ai.incidents.detectors.person.pose_features import PoseFeatureEngine
from ai.incidents.detectors.person.fall import PersonFallDetector
from ai.incidents.detectors.person.person_down import PersonDownDetector
from ai.incidents.detectors.person.panic_running import PanicRunningDetector
from ai.incidents.detectors.person.rapid_movement import UnusualRapidPersonMovementDetector
from ai.incidents.detectors.person.altercation import PhysicalAltercationDetector
from ai.incidents.detectors.person.forced_movement import ForcedMovementDetector
from ai.incidents.detectors.person.following import PersonFollowingDetector
from ai.incidents.detectors.person.coordinated_movement import CoordinatedPersonMovementDetector

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
