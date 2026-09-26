"""
ai/correlation/models.py
========================
Canonical domain models and typed schemas for Phase 16:
Advanced Incident Correlation, Contextual Fusion, and Incident Storylines.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple


class CorrelationRelationshipType(str, Enum):
    """Observational relationships between tracks, objects, zones, and incidents."""
    APPROACHES = "approaches"
    FOLLOWS = "follows"
    INTERACTS_WITH = "interacts_with"
    CARRIES = "carries"
    DEPARTS_WITH = "departs_with"
    ENTERS = "enters"
    EXITS = "exits"
    COLLIDES_WITH = "collides_with"
    MOVES_WITH = "moves_with"
    REMAINS_NEAR = "remains_near"
    SEPARATES_FROM = "separates_from"
    DISPLACES = "displaces"
    DISAPPEARS_AFTER_INTERACTION = "disappears_after_interaction"


class TemporalRelationType(str, Enum):
    """Temporal interval relationship types."""
    OVERLAPPING = "OVERLAPPING"
    IMMEDIATELY_PRECEDING = "IMMEDIATELY_PRECEDING"
    FOLLOWING = "FOLLOWING"
    CONTINUATION = "CONTINUATION"
    CONCURRENT = "CONCURRENT"
    SEPARATED = "SEPARATED"


class HypothesisOutcome(str, Enum):
    """Arbitration outcomes for competing behavioral hypotheses."""
    ACCEPTED = "ACCEPTED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    ABSTAINED = "ABSTAINED"
    SUPERSEDED = "SUPERSEDED_BY_STRONGER_HYPOTHESIS"


@dataclass
class ObservationalRelationship:
    """
    Physical observable relationship between two tracks or entities.
    Strictly grounded in geometry and timing; does not infer intent.
    """
    subject_track_id: str
    target_track_id: str
    relationship_type: str  # CorrelationRelationshipType value
    confidence: float
    start_time: float
    end_time: float
    spatial_proximity: float = 0.0  # normalized or pixel distance
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_track_id": self.subject_track_id,
            "target_track_id": self.target_track_id,
            "relationship_type": self.relationship_type,
            "confidence": round(float(self.confidence), 4),
            "start_time": round(float(self.start_time), 4),
            "end_time": round(float(self.end_time), 4),
            "spatial_proximity": round(float(self.spatial_proximity), 2),
            "metadata": self.metadata,
        }


@dataclass
class CorrelatedIncident:
    """
    Unified Correlated Incident domain model (Phase 16).
    Combines multi-detector signals, temporal-spatial sequences,
    and physical relationships into a coherent situation narrative.
    """
    incident_id: str
    video_id: str
    incident_category: str  # property, vehicle, person, crowd, specialized, multi_signal
    incident_subcategory: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    severity: str = "NORMAL"  # LOW, NORMAL, HIGH
    confidence: float = 0.5  # [0.0, 1.0]
    assessment_score: float = 0.5  # Calibrated non-probabilistic semantic score [0.0, 1.0]
    evidence_strength: float = 0.5  # Empirical signal strength [0.0, 1.0]
    pattern_evidence_strength: float = 0.5  # Multi-signal pattern evidence strength [0.0, 1.0]
    reliability_rating: str = "MEDIUM"  # "HIGH", "MEDIUM", "LOW"
    validation_decision: str = "REVIEW_REQUIRED"  # "ACCEPTED", "REVIEW_REQUIRED", "ABSTAINED"

    primary_track_ids: List[str] = field(default_factory=list)
    supporting_track_ids: List[str] = field(default_factory=list)
    involved_object_classes: List[str] = field(default_factory=list)

    source_candidate_ids: List[str] = field(default_factory=list)
    source_detector_ids: List[str] = field(default_factory=list)
    supporting_signals: List[Dict[str, Any]] = field(default_factory=list)
    contradictory_signals: List[Dict[str, Any]] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    zone_ids: List[str] = field(default_factory=list)

    relationships: List[ObservationalRelationship] = field(default_factory=list)
    contextual_factors: Dict[str, Any] = field(default_factory=dict)
    alternate_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    negative_evidence: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    storyline: str = ""
    provenance_graph: Dict[str, Any] = field(default_factory=dict)
    human_verification_required: bool = True
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "video_id": self.video_id,
            "incident_category": self.incident_category,
            "incident_subcategory": self.incident_subcategory,
            "start_time": round(float(self.start_time), 4),
            "end_time": round(float(self.end_time), 4),
            "duration": round(float(self.duration), 4),
            "severity": self.severity,
            "confidence": round(float(self.confidence), 4),
            "assessment_score": round(float(self.assessment_score), 4),
            "evidence_strength": round(float(self.evidence_strength), 4),
            "pattern_evidence_strength": round(float(self.pattern_evidence_strength), 4),
            "reliability_rating": self.reliability_rating,
            "validation_decision": self.validation_decision,
            "primary_track_ids": self.primary_track_ids,
            "supporting_track_ids": self.supporting_track_ids,
            "involved_object_classes": self.involved_object_classes,
            "source_candidate_ids": self.source_candidate_ids,
            "source_detector_ids": self.source_detector_ids,
            "supporting_signals": self.supporting_signals,
            "contradictory_signals": self.contradictory_signals,
            "evidence_ids": self.evidence_ids,
            "zone_ids": self.zone_ids,
            "negative_evidence": self.negative_evidence,
            "provenance": self.provenance,
            "relationships": [r.to_dict() for r in self.relationships],
            "contextual_factors": self.contextual_factors,
            "alternate_hypotheses": self.alternate_hypotheses,
            "storyline": self.storyline,
            "provenance_graph": self.provenance_graph,
            "human_verification_required": self.human_verification_required,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
