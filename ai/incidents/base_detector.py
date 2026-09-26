"""
Base Incident Detector Interface (Phase 10)

Defines the contract that every incident detector must implement.
Keeps detectors strictly decoupled from the database and UI layers.
"""
from abc import ABC, abstractmethod
import logging
from typing import List, Optional, Dict, Any
import uuid

from ai.incidents.schemas import (
    IncidentCandidate,
    IncidentContext,
    IncidentCategory,
    SupportingSignal,
    EvidenceCandidate,
    SpatialContext,
    TemporalContext,
)

logger = logging.getLogger(__name__)


class BaseIncidentDetector(ABC):
    """
    Abstract contract for all current and future Sentinel incident detectors.
    """

    detector_name: str = "base_detector"
    detector_version: str = "1.0.0"
    category: IncidentCategory = IncidentCategory.GENERAL
    enabled: bool = True

    # Phase Z: Future-Proof Detector Contract Declarations
    required_signals: List[str] = []
    supporting_signals_declared: List[str] = []
    contradictory_signals_declared: List[str] = []
    context_requirements: Dict[str, Any] = {}
    evidence_requirements: Dict[str, Any] = {}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(self, enabled: bool = True, config: Optional[Dict[str, Any]] = None):
        self.enabled = enabled
        self.config = config or {}

    @abstractmethod
    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        """
        Execute analytical logic across structured IncidentContext.
        Must return a list of IncidentCandidate objects.
        Must NOT perform database writes or network queries directly.
        """
        raise NotImplementedError("Every incident detector must implement analyze(context).")

    def build_candidate(
        self,
        video_id: str,
        event_type: str,
        start_time: float,
        end_time: float,
        severity: str,
        confidence: float,
        explanation: str,
        track_ids: Optional[List[str]] = None,
        object_classes: Optional[List[str]] = None,
        source_detection_ids: Optional[List[str]] = None,
        supporting_signals: Optional[List[SupportingSignal]] = None,
        contradictory_signals: Optional[List[SupportingSignal]] = None,
        spatial_context: Optional[SpatialContext] = None,
        temporal_context: Optional[TemporalContext] = None,
        evidence_candidates: Optional[List[EvidenceCandidate]] = None,
        human_verification_required: bool = True,
        prefix: str = "INC",
        incident_metadata: Optional[Dict[str, Any]] = None,
        pattern_evidence_strength: Optional[float] = None,
        assessment_score: Optional[float] = None,
    ) -> IncidentCandidate:
        """
        Utility to construct a fully formed IncidentCandidate with safety guarantees.
        """
        from ai.common.numeric import ensure_finite, clamp_finite

        start_t = max(0.0, round(ensure_finite(start_time, 0.0), 4))
        end_t = max(start_t, round(ensure_finite(end_time, start_t), 4))
        duration = max(0.0, round(end_t - start_t, 4))
        inc_id = f"{prefix}-{uuid.uuid4().hex[:8]}"

        # Guarantee temporal context exists
        if temporal_context is None:
            temporal_context = TemporalContext(
                start_time=start_t,
                end_time=end_t,
                duration_seconds=duration,
                onset_timestamp=start_t,
            )

        p_strength = pattern_evidence_strength if pattern_evidence_strength is not None else confidence
        a_score = assessment_score if assessment_score is not None else confidence

        return IncidentCandidate(
            incident_id=inc_id,
            video_id=video_id,
            event_type=event_type,
            category=self.category.value if isinstance(self.category, IncidentCategory) else str(self.category),
            start_time=start_t,
            end_time=end_t,
            duration=duration,
            severity=severity,
            confidence=round(clamp_finite(confidence, 0.0, 1.0, default=0.0), 4),
            pattern_evidence_strength=round(clamp_finite(p_strength, 0.0, 1.0, default=0.0), 4),
            assessment_score=round(clamp_finite(a_score, 0.0, 1.0, default=0.0), 4),
            track_ids=track_ids or [],
            object_classes=object_classes or [],
            source_detection_ids=source_detection_ids or [],
            supporting_signals=supporting_signals or [],
            contradictory_signals=contradictory_signals or [],
            spatial_context=spatial_context,
            temporal_context=temporal_context,
            explanation=explanation,
            evidence_candidates=evidence_candidates or [],
            human_verification_required=human_verification_required,
            detector_name=self.detector_name,
            detector_version=self.detector_version,
            incident_metadata=incident_metadata or {},
        )


