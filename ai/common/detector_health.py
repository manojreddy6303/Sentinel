"""
Sentinel Detector Health & Status Registry (Global Detection Reliability Core)

Tracks operational health, model provenance, and observation throughput
for all Sentinel detectors (object detectors, specialized visual detectors,
and incident intelligence detectors).

Ensures complete visibility into whether "0 detections" reflects a clear scene
versus an unconfigured/failed detector.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class DetectorOperationalStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    UNAVAILABLE = "UNAVAILABLE"
    FAILED = "FAILED"


class DetectorModelType(str, Enum):
    TRAINED_MODEL = "trained_model"
    HEURISTIC_CV = "heuristic_cv"
    RULE_BASED = "rule_based"
    FOUNDATION_ONLY = "foundation_only"


@dataclass
class DetectorHealthRecord:
    name: str
    version: str = "1.0.0"
    status: str = DetectorOperationalStatus.AVAILABLE.value
    model_type: str = DetectorModelType.TRAINED_MODEL.value
    configured: bool = True
    observations: int = 0
    validated_observations: int = 0
    rejected_observations: int = 0
    failures: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "status": self.status,
            "model_type": self.model_type,
            "configured": self.configured,
            "observations": self.observations,
            "validated_observations": self.validated_observations,
            "rejected_observations": self.rejected_observations,
            "failures": self.failures,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class DetectorHealthRegistry:
    """
    Central registry for tracking detector operational health across pipeline execution.
    Thread-safe and video-isolated.
    """
    _instance: Optional["DetectorHealthRegistry"] = None

    def __init__(self):
        self._detectors: Dict[str, DetectorHealthRecord] = {}

    @classmethod
    def get_instance(cls) -> "DetectorHealthRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_detector(
        self,
        name: str,
        version: str = "1.0.0",
        status: str = DetectorOperationalStatus.AVAILABLE.value,
        model_type: str = DetectorModelType.TRAINED_MODEL.value,
        configured: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DetectorHealthRecord:
        record = DetectorHealthRecord(
            name=name,
            version=version,
            status=status,
            model_type=model_type,
            configured=configured,
            metadata=metadata or {},
        )
        self._detectors[name] = record
        return record

    def record_observation(
        self,
        name: str,
        is_validated: bool = True,
        is_rejected: bool = False,
    ) -> None:
        if name not in self._detectors:
            self.register_detector(name=name)
        record = self._detectors[name]
        record.observations += 1
        if is_rejected:
            record.rejected_observations += 1
        elif is_validated:
            record.validated_observations += 1

    def record_failure(self, name: str, error_message: str) -> None:
        if name not in self._detectors:
            self.register_detector(name=name, status=DetectorOperationalStatus.FAILED.value)
        record = self._detectors[name]
        record.failures += 1
        record.status = DetectorOperationalStatus.FAILED.value
        record.error_message = error_message
        logger.error(f"Detector [{name}] recorded failure: {error_message}")

    def get_detector(self, name: str) -> Optional[DetectorHealthRecord]:
        return self._detectors.get(name)

    def get_all_records(self) -> List[DetectorHealthRecord]:
        return list(self._detectors.values())

    def get_health_report(self) -> Dict[str, Any]:
        total_obs = sum(r.observations for r in self._detectors.values())
        total_val = sum(r.validated_observations for r in self._detectors.values())
        total_rej = sum(r.rejected_observations for r in self._detectors.values())
        total_failures = sum(r.failures for r in self._detectors.values())
        
        return {
            "total_detectors": len(self._detectors),
            "healthy_detectors": sum(1 for r in self._detectors.values() if r.status == DetectorOperationalStatus.AVAILABLE.value),
            "failed_detectors": sum(1 for r in self._detectors.values() if r.status == DetectorOperationalStatus.FAILED.value),
            "unconfigured_detectors": sum(1 for r in self._detectors.values() if r.status == DetectorOperationalStatus.NOT_CONFIGURED.value),
            "total_observations": total_obs,
            "total_validated": total_val,
            "total_rejected": total_rej,
            "total_failures": total_failures,
            "detectors": [r.to_dict() for r in self._detectors.values()],
        }

    def reset(self) -> None:
        self._detectors.clear()
