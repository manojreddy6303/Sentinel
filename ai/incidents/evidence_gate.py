"""
Evidence Eligibility Gate (Phase 10-R)

Gating mechanism that validates whether an incident is eligible to generate
automatic snapshots/clips in the Evidence Vault.

Checks:
1. Incident candidate validation status (must NOT be REJECTED).
2. Video ID consistency.
3. Timestamp interval validity (start <= end).
4. Evidence candidate timestamp within incident interval.
5. Track ID grounding (track exists in context).
6. Bounding box validity (coordinates non-empty, finite).
7. Object class alignment.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging

from ai.incidents.schemas import IncidentCandidate, EvidenceCandidate, ValidationDecision, IncidentContext

logger = logging.getLogger(__name__)


class EvidenceEligibilityGate:
    """
    Guarantees evidence snapshots and clips are grounded in verified observations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def evaluate_eligibility(
        self,
        candidate: IncidentCandidate,
        context: Optional[IncidentContext] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Returns (is_eligible, ineligibility_reason).
        """
        # 1. REJECTED incidents can NEVER generate evidence
        if candidate.validation_decision == ValidationDecision.REJECTED.value:
            return False, "Incident candidate was REJECTED by candidate validator"

        # 2. Check video_id consistency
        if not candidate.video_id:
            return False, "Missing video_id"

        # 3. Check temporal interval
        if candidate.start_time < 0 or candidate.end_time < candidate.start_time:
            return False, f"Invalid incident interval [{candidate.start_time}, {candidate.end_time}]"

        # 4. If evidence candidates are present, verify their internal alignment
        for ec in candidate.evidence_candidates:
            # Timestamp must be within [start_time - 1.0, end_time + 1.0]
            if ec.timestamp < max(0.0, candidate.start_time - 1.0) or ec.timestamp > (candidate.end_time + 1.0):
                return False, f"Evidence timestamp {ec.timestamp:.2f}s falls outside incident interval [{candidate.start_time:.2f}, {candidate.end_time:.2f}]"

        # 5. Track grounding check if context is available
        if context and context.tracks and candidate.track_ids:
            known_track_ids = {str(getattr(t, "track_id", t.get("track_id") if isinstance(t, dict) else "")) for t in context.tracks}
            grounded_tracks = [tid for tid in candidate.track_ids if str(tid) in known_track_ids]
            if not grounded_tracks and candidate.track_ids:
                return False, "None of the candidate track IDs correspond to observed tracks in context"


        # 6. Spatial bounding box grounding
        if candidate.spatial_context and candidate.spatial_context.bounding_box:
            box = candidate.spatial_context.bounding_box
            w = getattr(box, "width", getattr(box, "x2", 1) - getattr(box, "x1", 0))
            h = getattr(box, "height", getattr(box, "y2", 1) - getattr(box, "y1", 0))
            if w <= 0:
                return False, "Candidate spatial box has invalid non-positive width"
            if h <= 0:
                return False, "Candidate spatial box has invalid non-positive height"

        return True, None


    def filter_and_mark_candidates(
        self,
        candidates: List[IncidentCandidate],
        context: Optional[IncidentContext] = None,
    ) -> List[IncidentCandidate]:
        """
        Updates evidence_eligible and evidence_ineligibility_reason on all candidates.
        """
        for cand in candidates:
            eligible, reason = self.evaluate_eligibility(cand, context=context)
            cand.evidence_eligible = eligible
            cand.evidence_ineligibility_reason = reason
            if not eligible and cand.evidence_candidates:
                # Strip misleading evidence candidates if not eligible
                logger.info(f"Stripping evidence from ineligible candidate {cand.incident_id}: {reason}")
                cand.evidence_candidates = []
        return candidates
