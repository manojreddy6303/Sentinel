"""
ai/correlation/engine.py
========================
Master Advanced Incident Correlation Engine for Phase 16.
Orchestrates relationship graph derivation, competing hypothesis arbitration,
domain fusion policies, and deterministic storyline synthesis.
"""

import logging
from typing import List, Dict, Any, Optional

from ai.schemas import TrackedObject
from ai.incidents.schemas import IncidentCandidate
from ai.correlation.models import CorrelatedIncident, ObservationalRelationship, HypothesisOutcome
from ai.correlation.relationship_graph import IncidentRelationshipGraph
from ai.correlation.arbitrator import CompetingHypothesisArbitrator
from ai.correlation.fusion_policies import (
    VehicleCorrelationPolicy,
    PropertyCorrelationPolicy,
    PersonCorrelationPolicy,
    CrowdZoneCorrelationPolicy,
    SpecializedVisualPolicy,
)
from ai.correlation.storyline_generator import IncidentStorylineGenerator

logger = logging.getLogger(__name__)


class AdvancedIncidentCorrelationEngine:
    """
    Coordinates multi-signal correlation, contextual fusion, duplicate suppression,
    and incident storyline generation across Sentinel surveillance data.
    """

    def __init__(self, proximity_threshold: float = 120.0):
        self.relationship_graph = IncidentRelationshipGraph(proximity_threshold=proximity_threshold)
        self.arbitrator = CompetingHypothesisArbitrator()

    def correlate_incidents(
        self,
        video_id: str,
        candidates: List[IncidentCandidate],
        tracks: List[TrackedObject],
        fps: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Execute end-to-end incident correlation and storyline synthesis.
        Returns:
            {
                "correlated_incidents": List[CorrelatedIncident],
                "relationships": List[ObservationalRelationship],
                "diagnostics": Dict[str, Any],
            }
        """
        if not candidates:
            return {
                "correlated_incidents": [],
                "relationships": [],
                "diagnostics": {
                    "raw_candidates_count": 0,
                    "validated_candidates_count": 0,
                    "correlated_groups_count": 0,
                    "independent_incidents_count": 0,
                    "superseded_candidates_count": 0,
                    "rejected_candidates_count": 0,
                    "review_required_candidates_count": 0,
                    "abstained_candidates_count": 0,
                    "correlated_incidents_count": 0,
                    "fused_incidents_count": 0,
                    "standalone_incidents_count": 0,
                    "relationships_derived": 0,
                    "candidate_dispositions": {},
                },
            }

        # 1. Derive pairwise observational relationships
        relationships = self.relationship_graph.build_track_relationships(tracks, fps=fps)

        # Filter out rejected or invalid candidates
        valid_candidates = [
            c for c in candidates
            if str(getattr(c, "validation_decision", "REVIEW_REQUIRED")).upper() not in ("REJECTED", "VALIDATIONDECISION.REJECTED")
            and getattr(c, "confidence", 1.0) > 0.0
        ]
        rejected_candidates = [c for c in candidates if c not in valid_candidates]

        # 2. Arbitrate competing hypotheses
        arbitrated_tuples = self.arbitrator.arbitrate_candidates(valid_candidates)
        accepted_candidates = [cand for cand, outcome, _ in arbitrated_tuples if outcome != HypothesisOutcome.SUPERSEDED]
        superseded_candidates = [cand for cand, outcome, _ in arbitrated_tuples if outcome == HypothesisOutcome.SUPERSEDED]

        # 3. Apply domain fusion policies
        correlated_veh = VehicleCorrelationPolicy.correlate(accepted_candidates, video_id, relationships)
        correlated_prop = PropertyCorrelationPolicy.correlate(accepted_candidates, video_id, relationships)
        correlated_pers = PersonCorrelationPolicy.correlate(accepted_candidates, video_id, relationships)
        correlated_crwd = CrowdZoneCorrelationPolicy.correlate(accepted_candidates, video_id, relationships)
        correlated_spec = SpecializedVisualPolicy.correlate(accepted_candidates, video_id, relationships)

        fused_correlated = correlated_veh + correlated_prop + correlated_pers + correlated_crwd + correlated_spec

        # Track which candidate IDs were absorbed into fused incidents
        absorbed_candidate_ids = set()
        for ci in fused_correlated:
            for cid in ci.source_candidate_ids:
                absorbed_candidate_ids.add(cid)

        # 4. Wrap any remaining unabsorbed accepted candidates into CorrelatedIncident format
        standalone_correlated: List[CorrelatedIncident] = []
        for cand in accepted_candidates:
            if cand.incident_id not in absorbed_candidate_ids:
                matched_rels = [
                    r for r in relationships
                    if any(t in (r.subject_track_id, r.target_track_id) for t in cand.track_ids)
                ]
                is_review = str(cand.validation_decision).upper() in ("REVIEW_REQUIRED", "VALIDATIONDECISION.REVIEW_REQUIRED", "HYPOTHESISOUTCOME.REVIEW_REQUIRED")
                val_dec = "REVIEW_REQUIRED" if is_review else "ACCEPTED"
                score = cand.confidence
                if val_dec == "REVIEW_REQUIRED":
                    score = min(score, 0.65)
                    rel_rating = "MODERATE"
                    ev_str = min(score * 0.9, 0.65)
                else:
                    rel_rating = "HIGH" if score >= 0.80 else "MODERATE"
                    ev_str = score * 0.9

                storyline = IncidentStorylineGenerator.generate_storyline(
                    category=cand.category,
                    subcategory=cand.event_type.lower(),
                    start_time=cand.start_time,
                    end_time=cand.end_time,
                    duration=cand.duration,
                    primary_tracks=cand.track_ids,
                    supporting_tracks=[],
                    object_classes=cand.object_classes,
                    relationships=matched_rels,
                    supporting_signals=[s.to_dict() for s in cand.supporting_signals],
                    confidence=score,
                    validation_decision=val_dec,
                )
                ci = CorrelatedIncident(
                    incident_id=f"CORR-{cand.incident_id[:8]}",
                    video_id=video_id,
                    incident_category=cand.category,
                    incident_subcategory=cand.event_type.lower(),
                    start_time=cand.start_time,
                    end_time=cand.end_time,
                    duration=cand.duration,
                    severity=cand.severity,
                    confidence=cand.confidence,
                    assessment_score=score,
                    evidence_strength=ev_str,
                    reliability_rating=rel_rating,
                    validation_decision=val_dec,
                    primary_track_ids=cand.track_ids,
                    involved_object_classes=cand.object_classes,
                    source_candidate_ids=[cand.incident_id],
                    source_detector_ids=[cand.detector_name],
                    supporting_signals=[s.to_dict() for s in cand.supporting_signals],
                    relationships=matched_rels,
                    storyline=storyline,
                    provenance_graph={"source_event_type": cand.event_type},
                )
                standalone_correlated.append(ci)

        all_correlated = fused_correlated + standalone_correlated
        all_correlated.sort(key=lambda c: c.start_time)

        # Build candidate disposition accounting
        candidate_dispositions = {}
        for c in rejected_candidates:
            candidate_dispositions[c.incident_id] = "rejected"
        for c in superseded_candidates:
            candidate_dispositions[c.incident_id] = "superseded"
        for cid in absorbed_candidate_ids:
            candidate_dispositions[cid] = "correlated"
        for ci in standalone_correlated:
            for cid in ci.source_candidate_ids:
                candidate_dispositions[cid] = "independent"

        review_req_cands = [c for c in candidates if "REVIEW_REQUIRED" in str(getattr(c, "validation_decision", "")).upper()]
        abstained_cands = [c for c in candidates if "ABSTAIN" in str(getattr(c, "validation_decision", "")).upper()]

        diagnostics = {
            "raw_candidates_count": len(candidates),
            "validated_candidates_count": len(valid_candidates),
            "correlated_groups_count": len(fused_correlated),
            "independent_incidents_count": len(standalone_correlated),
            "superseded_candidates_count": len(superseded_candidates),
            "rejected_candidates_count": len(rejected_candidates),
            "review_required_candidates_count": len(review_req_cands),
            "abstained_candidates_count": len(abstained_cands),
            "correlated_incidents_count": len(all_correlated),
            "fused_incidents_count": len(fused_correlated),
            "standalone_incidents_count": len(standalone_correlated),
            "relationships_derived": len(relationships),
            "candidate_dispositions": candidate_dispositions,
        }

        logger.info(
            f"Advanced Incident Correlation Engine completed for video {video_id}: "
            f"{len(candidates)} raw candidates -> {len(all_correlated)} correlated incident storylines "
            f"({len(fused_correlated)} fused groups, {len(standalone_correlated)} independent, "
            f"{len(relationships)} relationships, {len(superseded_candidates)} superseded, {len(rejected_candidates)} rejected)."
        )

        return {
            "correlated_incidents": all_correlated,
            "relationships": relationships,
            "diagnostics": diagnostics,
        }
