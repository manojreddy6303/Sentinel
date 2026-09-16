"""
Base Specialized Detector Interface (Phase 15)

Defines the contract for all specialized visual models in Sentinel.
Mandates strict failure isolation, non-crashing execution, and graceful degradation.
"""
from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, List, Optional
import numpy as np

from ai.specialized.schemas import (
    SpecializedDetectorStatus,
    SpecializedObservation,
    SpecializedModelInfo,
)

logger = logging.getLogger(__name__)


class BaseSpecializedDetector(ABC):
    """
    Abstract contract for specialized visual detectors (Fire, Smoke, Weapon, Pose, etc.).
    Guarantees:
    - Never crashes the Sentinel pipeline if model or hardware is unavailable.
    - Explicitly reports availability and status reasons.
    - Zero fabrication: never synthesizes false observations.
    """

    detector_name: str = "base_specialized_detector"
    detector_version: str = "1.0.0"
    supported_classes: List[str] = []

    def __init__(self, enabled: bool = True, config: Optional[Dict[str, Any]] = None):
        self.enabled = enabled
        self.config = config or {}
        self._is_initialized = False
        self._status = SpecializedDetectorStatus.NOT_CONFIGURED
        self._status_reason = "Uninitialized"

    @abstractmethod
    def initialize(self) -> bool:
        """
        Attempt to initialize detector and load weights/models if available.
        Must NOT download large weights silently over the network at runtime.
        Returns True if operational, False otherwise.
        """
        raise NotImplementedError("Specialized detectors must implement initialize().")

    @abstractmethod
    def detect_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_idx: int = 0,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[SpecializedObservation]:
        """
        Process a single BGR frame and return detected specialized visual observations.
        Must handle corrupted/empty frames gracefully.
        """
        raise NotImplementedError("Specialized detectors must implement detect_frame().")

    def safe_detect(
        self,
        frame: Optional[np.ndarray],
        timestamp: float,
        frame_idx: int = 0,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[SpecializedObservation]:
        """
        Failure-isolated wrapper around detect_frame.
        Catches all runtime exceptions to prevent pipeline failure.
        """
        if not self.enabled:
            return []

        if not self._is_initialized:
            try:
                self.initialize()
            except Exception as e:
                logger.warning(f"Specialized detector {self.detector_name} failed during lazy initialization: {e}")
                self._status = SpecializedDetectorStatus.FAILED
                self._status_reason = str(e)
                return []

        if self._status != SpecializedDetectorStatus.AVAILABLE:
            return []

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        try:
            return self.detect_frame(frame, timestamp, frame_idx, context)
        except Exception as exc:
            logger.error(f"Specialized detector {self.detector_name} error on frame at {timestamp:.2f}s: {exc}", exc_info=True)
            return []

    def get_status(self) -> SpecializedDetectorStatus:
        """Return the current operational status of the detector."""
        return self._status

    def get_model_info(self) -> SpecializedModelInfo:
        """Return metadata regarding the model, version, license, and resources."""
        return SpecializedModelInfo(
            name=self.detector_name,
            version=self.detector_version,
            status=self._status,
            status_reason=self._status_reason,
        )
