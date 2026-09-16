"""
Future Detector Plug-In Support & Extensibility Blueprints (Phase 10 & 11)

Maintains backward-compatible exports for modular detectors.
"""
from ai.incidents.detectors.vehicle.collision import VehicleCollisionDetector

__all__ = ["VehicleCollisionDetector"]
