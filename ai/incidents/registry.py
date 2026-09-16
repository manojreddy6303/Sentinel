"""
Incident Detector Registry (Phase 10)

Manages discovery, registration, configuration, and execution of incident detectors.
Provides strict failure isolation: an exception in one detector will NEVER crash the
analysis pipeline or prevent other detectors from completing.
"""
import logging
import time
import traceback
from typing import Dict, List, Any, Optional, Tuple

from ai.incidents.base_detector import BaseIncidentDetector
from ai.incidents.schemas import IncidentCandidate, IncidentContext

logger = logging.getLogger(__name__)


class IncidentDetectorRegistry:
    """
    Central registry for modular incident detectors.
    """

    def __init__(self):
        self._detectors: Dict[str, BaseIncidentDetector] = {}

    def register(self, detector: BaseIncidentDetector) -> None:
        """Register a detector instance."""
        if not isinstance(detector, BaseIncidentDetector):
            raise TypeError(f"Detector must inherit from BaseIncidentDetector, got {type(detector)}")
        self._detectors[detector.detector_name] = detector
        logger.info(f"Registered incident detector: {detector.detector_name} (v{detector.detector_version}, {detector.category})")

    def unregister(self, detector_name: str) -> bool:
        """Remove a detector by name."""
        if detector_name in self._detectors:
            del self._detectors[detector_name]
            return True
        return False

    def get(self, detector_name: str) -> Optional[BaseIncidentDetector]:
        return self._detectors.get(detector_name)

    def set_enabled(self, detector_name: str, enabled: bool) -> bool:
        """Enable or disable a specific detector."""
        det = self._detectors.get(detector_name)
        if det:
            det.enabled = enabled
            return True
        return False

    def list_detectors(self) -> List[Dict[str, Any]]:
        """List metadata for all registered detectors."""
        return [
            {
                "name": d.detector_name,
                "version": d.detector_version,
                "category": d.category.value if hasattr(d.category, "value") else str(d.category),
                "enabled": d.enabled,
            }
            for d in self._detectors.values()
        ]

    def execute_all(self, context: IncidentContext) -> Tuple[List[IncidentCandidate], Dict[str, Any]]:
        """
        Execute all enabled detectors with complete failure isolation.
        Returns:
            (all_candidates, diagnostics_dict)
        """
        all_candidates: List[IncidentCandidate] = []
        diagnostics: Dict[str, Any] = {
            "total_detectors": len(self._detectors),
            "executed_detectors": 0,
            "failed_detectors": 0,
            "execution_details": {},
            "errors": {},
        }

        for name, detector in self._detectors.items():
            if not detector.enabled:
                diagnostics["execution_details"][name] = {"status": "disabled", "candidates_count": 0}
                continue

            t_start = time.perf_counter()
            try:
                candidates = detector.analyze(context)
                elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

                # Validate candidates contract
                valid_candidates = []
                for cand in candidates:
                    if isinstance(cand, IncidentCandidate):
                        valid_candidates.append(cand)
                    else:
                        logger.warning(f"Detector '{name}' returned non-IncidentCandidate object: {type(cand)}")

                all_candidates.extend(valid_candidates)
                diagnostics["executed_detectors"] += 1
                diagnostics["execution_details"][name] = {
                    "status": "success",
                    "elapsed_ms": elapsed_ms,
                    "candidates_count": len(valid_candidates),
                }

            except Exception as exc:
                elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
                error_msg = str(exc)
                stack_trace = traceback.format_exc()

                logger.error(
                    f"Detector failure isolation: '{name}' raised an error during analyze(): {error_msg}\n{stack_trace}"
                )

                diagnostics["failed_detectors"] += 1
                diagnostics["execution_details"][name] = {
                    "status": "failed",
                    "elapsed_ms": elapsed_ms,
                    "error": error_msg,
                }
                diagnostics["errors"][name] = {
                    "error": error_msg,
                    "traceback_summary": stack_trace.splitlines()[-3:],
                }
                # Failure is isolated! Loop proceeds to next detector.

        return all_candidates, diagnostics


# Global instance
_global_registry = IncidentDetectorRegistry()


def get_detector_registry() -> IncidentDetectorRegistry:
    return _global_registry
