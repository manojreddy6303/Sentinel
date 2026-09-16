"""
Vehicle Incident Detectors Package (Sentinel Phase 11)

Exports specialized modular vehicle detectors:
- VehicleCollisionDetector (POTENTIAL_VEHICLE_COLLISION)
- NearCollisionDetector (POTENTIAL_NEAR_COLLISION)
- SuddenStopDetector (POTENTIAL_SUDDEN_VEHICLE_STOP)
- WrongWayVehicleDetector (POTENTIAL_WRONG_WAY_VEHICLE)
- UnusualTrajectoryDetector (POTENTIAL_UNUSUAL_VEHICLE_TRAJECTORY)
- StationaryVehicleDetector (POTENTIAL_STATIONARY_VEHICLE)

Also exports:
- VehicleInteractionModel
- AdaptiveTemporalSamplingEngine
"""
from ai.incidents.detectors.vehicle.collision import VehicleCollisionDetector
from ai.incidents.detectors.vehicle.near_collision import NearCollisionDetector
from ai.incidents.detectors.vehicle.sudden_stop import SuddenStopDetector
from ai.incidents.detectors.vehicle.wrong_way import WrongWayVehicleDetector
from ai.incidents.detectors.vehicle.unusual_trajectory import UnusualTrajectoryDetector
from ai.incidents.detectors.vehicle.stationary_vehicle import StationaryVehicleDetector
from ai.incidents.detectors.vehicle.interaction import VehicleInteractionModel
from ai.incidents.detectors.vehicle.sampling import AdaptiveTemporalSamplingEngine

__all__ = [
    "VehicleCollisionDetector",
    "NearCollisionDetector",
    "SuddenStopDetector",
    "WrongWayVehicleDetector",
    "UnusualTrajectoryDetector",
    "StationaryVehicleDetector",
    "VehicleInteractionModel",
    "AdaptiveTemporalSamplingEngine",
]
