"""
Universal Incident Intelligence Engine (Sentinel Phase 10)

Public package exports.
"""
from ai.incidents.schemas import (
    IncidentCandidate,
    IncidentContext,
    IncidentCategory,
    IncidentValidationStatus,
    ValidationDecision,
    SupportingSignal,
    EvidenceCandidate,
    SpatialContext,
    TemporalContext,
    TrackMotion,
)
from ai.incidents.base_detector import BaseIncidentDetector
from ai.incidents.registry import IncidentDetectorRegistry, get_detector_registry
from ai.incidents.motion import UniversalMotionEngine
from ai.incidents.spatial import SpatialRelationshipEngine
from ai.incidents.temporal import TemporalAnalysisEngine
from ai.incidents.context import IncidentContextBuilder
from ai.incidents.scoring import IncidentScorer
from ai.incidents.fusion import IncidentFusionEngine
from ai.incidents.negative_evidence import NegativeEvidenceEngine
from ai.incidents.scene_context import SceneContextEngine, SceneContextData, NormalBehaviorBaseline
from ai.incidents.validator import IncidentCandidateValidator
from ai.incidents.evidence_gate import EvidenceEligibilityGate
from ai.incidents.engine import IncidentIntelligenceEngine

__all__ = [
    "IncidentCandidate",
    "IncidentContext",
    "IncidentCategory",
    "IncidentValidationStatus",
    "ValidationDecision",
    "SupportingSignal",
    "EvidenceCandidate",
    "SpatialContext",
    "TemporalContext",
    "TrackMotion",
    "BaseIncidentDetector",
    "IncidentDetectorRegistry",
    "get_detector_registry",
    "UniversalMotionEngine",
    "SpatialRelationshipEngine",
    "TemporalAnalysisEngine",
    "IncidentContextBuilder",
    "IncidentScorer",
    "IncidentFusionEngine",
    "NegativeEvidenceEngine",
    "SceneContextEngine",
    "SceneContextData",
    "NormalBehaviorBaseline",
    "IncidentCandidateValidator",
    "EvidenceEligibilityGate",
    "IncidentIntelligenceEngine",
]

