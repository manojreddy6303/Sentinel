"""
Potential Physical Altercation Detector (Sentinel Phase 12)

Detects observable physical agitation or altercation patterns between individuals:
- Sustained close proximity (<= 1.2 person body widths)
- Repeated abrupt reciprocal motion and opposing movement vectors
- Multi-signal validation strictly suppressing normal conversation, queues, and parallel walking
- Strictly conservative observational review labeling (NEVER definitive 'fight')
"""
from typing import List, Dict, Any, Optional

from ai.incidents.base_detector import BaseIncidentDetector
from ai.incidents.schemas import (
    IncidentCandidate,
    IncidentContext,
    IncidentCategory,
    SupportingSignal,
    EvidenceCandidate,
    SpatialContext,
)
from ai.incidents.scoring import IncidentScorer
from ai.incidents.negative_evidence import NegativeEvidenceEngine
from ai.incidents.detectors.person.motion_features import PersonMotionFeatureEngine


class PhysicalAltercationDetector(BaseIncidentDetector):
    """
    Evaluates pairwise person interactions for observable physical altercation kinematics.
    """

    detector_name: str = "physical_altercation_detector"
    detector_version: str = "1.0.0"
    category: IncidentCategory = IncidentCategory.PERSON

    required_signals: List[str] = ["Sustained Close Proximity", "Reciprocal Agitation Motion"]
    supporting_signals_declared: List[str] = [
        "Sustained Close Proximity",
        "Rapid Reciprocal Motion Oscillations",
        "Opposing Movement Vectors",
    ]
    contradictory_signals_declared: List[str] = [
        "Negative: Parallel Side-by-Side Walking",
        "Negative: Pedestrian Queue Formation",
        "Negative: Normal Low-Mobility Conversation",
        "Negative: Brief Passing Clearance",
    ]
    context_requirements: Dict[str, Any] = {"pedestrian_context": True}
    evidence_requirements: Dict[str, Any] = {"verified_tracks": True, "bounding_box_grounding": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        min_interaction_seconds: float = 1.5,
        min_reciprocal_score: float = 0.50,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.min_interaction_seconds = min_interaction_seconds
        self.min_reciprocal_score = min_reciprocal_score

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        persons = [t for t in context.tracks if t.object_class == "person" and t.is_validated]
        if len(persons) < 2:
            return candidates

        checked_pairs = set()
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                p1, p2 = persons[i], persons[j]
                pair_key = (min(p1.track_id, p2.track_id), max(p1.track_id, p2.track_id))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                # 1. Evaluate Reciprocal Motion
                recip = PersonMotionFeatureEngine.compute_reciprocal_motion(p1, p2)
                if not recip["is_reciprocal"] or recip["reciprocal_score"] < self.min_reciprocal_score or recip.get("is_passing"):
                    continue

                event_t = (max(p1.first_seen, p2.first_seen) + min(p1.last_seen, p2.last_seen)) / 2.0

                # 2. Negative Evidence Evaluation
                neg_signals = NegativeEvidenceEngine.evaluate_altercation_negative_evidence(p1, p2, context, event_t)
                neg_names = {s.signal_type for s in neg_signals}
                if any(k in neg_names for k in [
                    "Negative: Parallel Side-by-Side Walking",
                    "Negative: Pedestrian Queue Formation",
                    "Negative: Normal Low-Mobility Conversation",
                    "Negative: Transient Passing Pedestrians",
                ]):
                    continue

                supporting_signals = [
                    SupportingSignal(
                        signal_type="Sustained Close Proximity",
                        description=f"Persons [{p1.track_id}] and [{p2.track_id}] interacted at distance of {recip['min_distance_px']:.1f}px",
                        confidence=0.75,
                        timestamp=event_t,
                    ),
                    SupportingSignal(
                        signal_type="Rapid Reciprocal Motion Oscillations",
                        description=(
                            f"Exhibited {recip['oscillations']} distance oscillation cycles and "
                            f"{recip['opposing_count']} opposing heading vectors"
                        ),
                        confidence=0.78,
                        timestamp=event_t,
                    ),
                ]

                scoring = IncidentScorer.calculate_evidence_score(
                    base_confidence=0.60,
                    supporting_signals=supporting_signals,
                    tracks=[p1, p2],
                    duration_seconds=min(p1.duration_seconds, p2.duration_seconds),
                    contradictory_signals=neg_signals,
                    validation_decision="REVIEW_REQUIRED",
                )

                cand = self.build_candidate(
                    video_id=context.video_id,
                    event_type="POTENTIAL_PHYSICAL_ALTERCATION",
                    start_time=max(0.0, event_t - 1.0),
                    end_time=min(context.duration_seconds, event_t + 2.0),
                    severity="NORMAL",
                    confidence=min(0.65, scoring["confidence"]),
                    track_ids=[p1.track_id, p2.track_id],
                    object_classes=["person", "person"],
                    supporting_signals=supporting_signals,
                    contradictory_signals=neg_signals,
                    explanation=(
                        f"Potential physical altercation pattern: [{p1.track_id}] and [{p2.track_id}] "
                        f"exhibited sustained close proximity and reciprocal agitation oscillations. "
                        f"Observational pattern only; visual kinematics cannot establish interpersonal conflict. Strict human verification required."
                    ),
                    evidence_candidates=[
                        EvidenceCandidate(
                            timestamp=event_t,
                            pre_seconds=2.0,
                            post_seconds=3.0,
                            bounding_box=p1.current_bbox,
                            target_track_id=p1.track_id,
                            reason="Forensic capture of potential altercation pattern",
                        )
                    ],
                    human_verification_required=True,
                )
                cand.validation_decision = "REVIEW_REQUIRED"
                candidates.append(cand)

        return candidates
