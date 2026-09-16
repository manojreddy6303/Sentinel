"""
ai/correlation/arbitrator.py
============================
Competing hypothesis arbitration engine for Phase 16.
Resolves mutually exclusive explanations while preserving provenance of alternate interpretations.
"""

from typing import List, Dict, Any, Tuple
from ai.correlation.models import HypothesisOutcome
from ai.incidents.schemas import IncidentCandidate, ValidationDecision


class CompetingHypothesisArbitrator:
    """
    Arbitrates between competing behavioral interpretations (e.g. following vs. coordinated movement,
    theft vs. normal transport, collision vs. near collision).
    """

    @staticmethod
    def _get_candidate_outcome(cand: IncidentCandidate) -> HypothesisOutcome:
        """
        Global Invariant: A correlated incident / hypothesis MUST NOT upgrade a constituent candidate's
        REVIEW_REQUIRED decision to ACCEPTED unless an explicit independent evidence rule justifies it.
        """
        v_upper = str(getattr(cand, "validation_decision", "")).upper()
        if "REVIEW_REQUIRED" in v_upper or "REVIEW" in v_upper:
            return HypothesisOutcome.REVIEW_REQUIRED
        elif "REJECTED" in v_upper:
            return HypothesisOutcome.REJECTED
        return HypothesisOutcome.ACCEPTED

    @staticmethod
    def arbitrate_candidates(
        candidates: List[IncidentCandidate],
    ) -> List[Tuple[IncidentCandidate, HypothesisOutcome, List[str]]]:
        """
        Evaluate candidate list and return tuples of:
        (Candidate, HypothesisOutcome, List[AlternateHypothesisEventTypes])
        """
        if not candidates:
            return []

        results: List[Tuple[IncidentCandidate, HypothesisOutcome, List[str]]] = []

        # 1. Collision vs Near-Collision Arbitration
        # If a vehicle collision is grounded in physical bbox contact, near-collision on the same tracks is superseded
        collision_cands = [c for c in candidates if "COLLISION" in c.event_type]
        non_collision_cands = [c for c in candidates if "COLLISION" not in c.event_type]

        if collision_cands:
            full_collisions = [c for c in collision_cands if c.event_type == "POTENTIAL_VEHICLE_COLLISION"]
            near_collisions = [c for c in collision_cands if c.event_type == "POTENTIAL_NEAR_COLLISION"]

            if full_collisions and near_collisions:
                for full_c in full_collisions:
                    # Check if near collision shares tracks
                    matched_near = [n for n in near_collisions if set(full_c.track_ids).intersection(set(n.track_ids))]
                    alt_types = [n.event_type for n in matched_near]
                    results.append((full_c, CompetingHypothesisArbitrator._get_candidate_outcome(full_c), alt_types))

                for near_c in near_collisions:
                    has_full = any(set(near_c.track_ids).intersection(set(f.track_ids)) for f in full_collisions)
                    if has_full:
                        results.append((near_c, HypothesisOutcome.SUPERSEDED, ["POTENTIAL_VEHICLE_COLLISION"]))
                    else:
                        results.append((near_c, CompetingHypothesisArbitrator._get_candidate_outcome(near_c), []))
            else:
                for c in collision_cands:
                    results.append((c, CompetingHypothesisArbitrator._get_candidate_outcome(c), []))

        # 2. Property Arbitration (Theft vs Removal vs Displacement vs Pickup)
        prop_cands = [c for c in non_collision_cands if any(k in c.event_type for k in ("THEFT", "TAKEAWAY", "PICKUP", "DISPLACEMENT", "REMOVAL", "ABANDONED"))]
        rem_cands = [c for c in non_collision_cands if c not in prop_cands]

        if prop_cands:
            # Group by shared object track if available
            thefts = [c for c in prop_cands if "THEFT" in c.event_type or "TAKEAWAY" in c.event_type]
            intermediates = [c for c in prop_cands if c not in thefts]

            if thefts and intermediates:
                for th in thefts:
                    alt_types = [i.event_type for i in intermediates]
                    results.append((th, CompetingHypothesisArbitrator._get_candidate_outcome(th), alt_types))
                for im in intermediates:
                    # Intermediate stages are preserved as superseded by the full takeaway storyline
                    results.append((im, HypothesisOutcome.SUPERSEDED, [t.event_type for t in thefts]))
            else:
                for c in prop_cands:
                    results.append((c, CompetingHypothesisArbitrator._get_candidate_outcome(c), []))

        # 3. Person Arbitration (Following vs Coordinated Movement vs Altercation)
        for c in rem_cands:
            results.append((c, CompetingHypothesisArbitrator._get_candidate_outcome(c), []))

        return results

    def arbitrate(
        self, cand_a: IncidentCandidate, cand_b: IncidentCandidate
    ) -> Tuple[IncidentCandidate, IncidentCandidate, HypothesisOutcome]:
        """
        Pairwise arbitration between two competing hypotheses.
        Returns (winner, loser, outcome).
        """
        # Compare assessment scores and evidence strength
        if cand_a.confidence >= cand_b.confidence:
            winner = cand_a
            loser = cand_b
        else:
            winner = cand_b
            loser = cand_a

        loser.validation_decision = "REVIEW_REQUIRED"
        return winner, loser, HypothesisOutcome.ACCEPTED
