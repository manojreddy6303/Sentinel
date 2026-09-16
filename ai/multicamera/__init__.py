"""
ai/multicamera — Phase 18: Multi-Camera Forensic Intelligence

Provides cross-camera track association, temporal compatibility reasoning,
and scene-level synthesis across multiple CCTV cameras within a
SurveillanceSession.

PRIVACY INVARIANTS (enforced throughout this package):
- No facial recognition. No biometric data. No identity claims.
- Cross-camera continuity is a hypothesis, never an assertion.
- Correct abstention (confidence < 0.40) is preferred over a false match.
- All association evidence_basis contains only observable attributes.
"""

from ai.multicamera.schemas import (
    CameraObservation,
    AssociationEvidence,
    CrossCameraHypothesis,
    AssociationType,
    AnalystVerdict,
)
from ai.multicamera.attribute_matcher import AttributeMatcher
from ai.multicamera.temporal_reasoner import TemporalCompatibilityReasoner
from ai.multicamera.association_engine import CrossCameraAssociationEngine
from ai.multicamera.scene_intelligence import MultiCameraSceneIntelligence

__all__ = [
    "CameraObservation",
    "AssociationEvidence",
    "CrossCameraHypothesis",
    "AssociationType",
    "AnalystVerdict",
    "AttributeMatcher",
    "TemporalCompatibilityReasoner",
    "CrossCameraAssociationEngine",
    "MultiCameraSceneIntelligence",
]
