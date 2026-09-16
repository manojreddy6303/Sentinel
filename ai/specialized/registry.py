"""
Specialized Detector Registry (Phase 15)

Coordinates discovery, lifecycle, and failure-isolated execution of specialized visual detectors.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from ai.specialized.base import BaseSpecializedDetector
from ai.specialized.schemas import SpecializedObservation, SpecializedDetectorStatus

logger = logging.getLogger(__name__)


class SpecializedDetectorRegistry:
    """
    Failure-isolated registry for specialized visual models.
    """

    def __init__(self):
        self._detectors: Dict[str, BaseSpecializedDetector] = {}

    def register(self, detector: BaseSpecializedDetector) -> None:
        """Register a detector instance by its unique name."""
        if not isinstance(detector, BaseSpecializedDetector):
            raise TypeError(f"Detector must inherit from BaseSpecializedDetector, got {type(detector)}")
        self._detectors[detector.detector_name] = detector
        logger.debug(f"Registered specialized detector: {detector.detector_name} (v{detector.detector_version})")

    def unregister(self, detector_name: str) -> Optional[BaseSpecializedDetector]:
        """Remove a detector from the registry."""
        return self._detectors.pop(detector_name, None)

    def clear(self) -> None:
        """Clear all registered detectors."""
        self._detectors.clear()

    def get(self, detector_name: str) -> Optional[BaseSpecializedDetector]:
        """Retrieve a registered detector by name."""
        return self._detectors.get(detector_name)

    def list_detectors(self) -> List[Dict[str, Any]]:
        """List all registered detectors with current status and model metadata."""
        results = []
        for name, det in self._detectors.items():
            info = det.get_model_info()
            results.append({
                "detector_name": name,
                "version": det.detector_version,
                "enabled": det.enabled,
                "supported_classes": det.supported_classes,
                "status": det.get_status().value,
                "model_info": info.to_dict(),
            })
        return results

    def execute_all(
        self,
        frame: Optional[np.ndarray],
        timestamp: float,
        frame_idx: int = 0,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[SpecializedObservation], Dict[str, Any]]:
        """
        Execute all registered and enabled specialized detectors on a frame.
        Complete failure isolation: failure of one detector does not impact others.
        """
        all_observations: List[SpecializedObservation] = []
        diagnostics: Dict[str, Any] = {
            "executed": 0,
            "succeeded": 0,
            "failed": 0,
            "skipped": 0,
            "errors": {},
        }

        for name, det in self._detectors.items():
            if not det.enabled:
                diagnostics["skipped"] += 1
                continue

            diagnostics["executed"] += 1
            try:
                obs = det.safe_detect(frame=frame, timestamp=timestamp, frame_idx=frame_idx, context=context)
                all_observations.extend(obs)
                diagnostics["succeeded"] += 1
            except Exception as exc:
                diagnostics["failed"] += 1
                diagnostics["errors"][name] = str(exc)
                logger.error(f"Detector {name} crashed unhandled in registry: {exc}")

        return all_observations, diagnostics


# Global singleton registry instance
_specialized_registry = SpecializedDetectorRegistry()


def get_specialized_registry() -> SpecializedDetectorRegistry:
    return _specialized_registry
