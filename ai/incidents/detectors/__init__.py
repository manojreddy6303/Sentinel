"""
Detectors Package (Phase 10)

Exports standard incident detectors and registration helper.
"""
from ai.incidents.detectors.zone_intrusion import ZoneIntrusionDetector
from ai.incidents.detectors.prolonged_presence import ProlongedPresenceDetector
from ai.incidents.detectors.abandoned_object import AbandonedObjectDetector
from ai.incidents.detectors.theft_and_takeaway import TheftAndTakeawayDetector
from ai.incidents.detectors.activity_analysis import ActivityAnalysisDetector
from ai.incidents.detectors.observational_anomaly import ObservationalAnomalyDetector
from ai.incidents.detectors.vehicle import (
    VehicleCollisionDetector,
    NearCollisionDetector,
    SuddenStopDetector,
    WrongWayVehicleDetector,
    UnusualTrajectoryDetector,
    StationaryVehicleDetector,
)
from ai.incidents.detectors.person import (
    PersonFallDetector,
    PersonDownDetector,
    PanicRunningDetector,
    UnusualRapidPersonMovementDetector,
    PhysicalAltercationDetector,
    ForcedMovementDetector,
    PersonFollowingDetector,
    CoordinatedPersonMovementDetector,
)

from ai.incidents.detectors.crowd import (
    CrowdDensityDetector,
    CrowdSurgeDetector,
    CrowdDispersalDetector,
    UnusualCrowdMovementDetector,
    ZoneOccupancyAndActivityDetector,
)

__all__ = [
    "ZoneIntrusionDetector",
    "ProlongedPresenceDetector",
    "AbandonedObjectDetector",
    "TheftAndTakeawayDetector",
    "ActivityAnalysisDetector",
    "ObservationalAnomalyDetector",
    "VehicleCollisionDetector",
    "NearCollisionDetector",
    "SuddenStopDetector",
    "WrongWayVehicleDetector",
    "UnusualTrajectoryDetector",
    "StationaryVehicleDetector",
    "PersonFallDetector",
    "PersonDownDetector",
    "PanicRunningDetector",
    "UnusualRapidPersonMovementDetector",
    "PhysicalAltercationDetector",
    "ForcedMovementDetector",
    "PersonFollowingDetector",
    "CoordinatedPersonMovementDetector",
    "ObjectLeftBehindDetector",
    "ObjectPickupDetector",
    "ObjectDisplacementDetector",
    "ObjectRemovalDetector",
    "PropertyTamperingDetector",
    "RestrictedObjectMovementDetector",
    "CrowdDensityDetector",
    "CrowdSurgeDetector",
    "CrowdDispersalDetector",
    "UnusualCrowdMovementDetector",
    "ZoneOccupancyAndActivityDetector",
    "SpecializedFireIncidentDetector",
    "SpecializedSmokeIncidentDetector",
    "SpecializedWeaponIncidentDetector",
    "register_standard_detectors",
]

from ai.incidents.detectors.property import (
    ObjectLeftBehindDetector,
    ObjectPickupDetector,
    ObjectDisplacementDetector,
    ObjectRemovalDetector,
    PropertyTamperingDetector,
    RestrictedObjectMovementDetector,
)
from ai.incidents.detectors.specialized import (
    SpecializedFireIncidentDetector,
    SpecializedSmokeIncidentDetector,
    SpecializedWeaponIncidentDetector,
)


def register_standard_detectors(registry) -> None:
    """Register all core detectors into the provided IncidentDetectorRegistry."""
    registry.register(ZoneIntrusionDetector())
    registry.register(ProlongedPresenceDetector())
    registry.register(AbandonedObjectDetector())
    registry.register(TheftAndTakeawayDetector())
    registry.register(ActivityAnalysisDetector())
    registry.register(ObservationalAnomalyDetector())
    registry.register(VehicleCollisionDetector())
    registry.register(NearCollisionDetector())
    registry.register(SuddenStopDetector())
    registry.register(WrongWayVehicleDetector())
    registry.register(UnusualTrajectoryDetector())
    registry.register(StationaryVehicleDetector())
    registry.register(PersonFallDetector())
    registry.register(PersonDownDetector())
    registry.register(PanicRunningDetector())
    registry.register(UnusualRapidPersonMovementDetector())
    registry.register(PhysicalAltercationDetector())
    registry.register(ForcedMovementDetector())
    registry.register(PersonFollowingDetector())
    registry.register(CoordinatedPersonMovementDetector())
    registry.register(ObjectLeftBehindDetector())
    registry.register(ObjectPickupDetector())
    registry.register(ObjectDisplacementDetector())
    registry.register(ObjectRemovalDetector())
    registry.register(PropertyTamperingDetector())
    registry.register(RestrictedObjectMovementDetector())
    registry.register(CrowdDensityDetector())
    registry.register(CrowdSurgeDetector())
    registry.register(CrowdDispersalDetector())
    registry.register(UnusualCrowdMovementDetector())
    registry.register(ZoneOccupancyAndActivityDetector())
    # Phase 15 Specialized Detectors
    registry.register(SpecializedFireIncidentDetector())
    registry.register(SpecializedSmokeIncidentDetector())
    registry.register(SpecializedWeaponIncidentDetector())

