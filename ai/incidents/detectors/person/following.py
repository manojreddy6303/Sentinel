"""
Person Following Detector (Sentinel Phase 12)

Detects persistent following relationships between anonymous tracked persons:
- Lagged trajectory correlation (Person B traverses Person A's path with temporal lag dt in [0.5s, 3.0s])
- Spatial persistence over time (>= 2.5s)
- Negative evidence filtering queues, pairs walking abreast, and shared pedestrian corridors
- Strictly anonymous observational track relationships (NO identity inference)
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


class PersonFollowingDetector(BaseIncidentDetector):
    """
    Evaluates pairwise trajectories for lagged following patterns.
    """

    detector_name: str = "person_following_detector"
    detector_version: str = "1.0.0"
    category: IncidentCategory = IncidentCategory.PERSON

    required_signals: List[str] = ["Lagged Trajectory Correlation", "Spatial Path Alignment"]
    supporting_signals_declared: List[str] = [
        "Lagged Trajectory Correlation",
        "Consistent Inter-Person Following Lag",
    ]
    contradictory_signals_declared: [
        "Negative: Pedestrian Queue Formation",
        "Negative: Walking Abreast",
    ]
    context_requirements: Dict[str, Any] = {"pedestrian_context": True}
    evidence_requirements: Dict[str, Any] = {"verified_tracks": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        min_duration_seconds: float = 2.5,
        max_lag_seconds: float = 3.0,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.min_duration_seconds = min_duration_seconds
        self.max_lag_seconds = max_lag_seconds

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        persons = [t for t in context.tracks if t.object_class == "person" and t.is_validated]
        if len(persons) < 2:
            return candidates

        resolved_followers: Dict[frozenset, Dict[str, Any]] = {}

        for i in range(len(persons)):
            for j in range(len(persons)):
                if i == j:
                    continue
                p_lead, p_follow = persons[i], persons[j]
                shared_dur = min(p_lead.duration_seconds, p_follow.duration_seconds)
                if shared_dur < self.min_duration_seconds:
                    continue

                lag_res = PersonMotionFeatureEngine.compute_lagged_trajectory_similarity(
                    p_lead, p_follow, max_lag_seconds=self.max_lag_seconds
                )
                if not lag_res["is_following"]:
                    continue

                pair_key = frozenset([p_lead.track_id, p_follow.track_id])
                # Check if inverse relationship already evaluated
                if pair_key in resolved_followers:
                    existing = resolved_followers[pair_key]
                    # If both claim to follow each other with similar error, it's symmetric walking abreast -> reject both!
                    if abs(lag_res["mean_path_deviation_px"] - existing["res"]["mean_path_deviation_px"]) < 20.0:
                        del resolved_followers[pair_key]
                        continue
                    # Otherwise keep only the one with significantly better path deviation
                    if lag_res["mean_path_deviation_px"] < existing["res"]["mean_path_deviation_px"]:
                        resolved_followers[pair_key] = {"lead": p_lead, "follow": p_follow, "res": lag_res, "dur": shared_dur}
                else:
                    resolved_followers[pair_key] = {"lead": p_lead, "follow": p_follow, "res": lag_res, "dur": shared_dur}

        for pair_key, item in resolved_followers.items():
            p_lead = item["lead"]
            p_follow = item["follow"]
            lag_res = item["res"]
            shared_dur = item["dur"]

            event_t = (max(p_lead.first_seen, p_follow.first_seen) + min(p_lead.last_seen, p_follow.last_seen)) / 2.0

            # Negative evidence: check queue or group walking abreast
            neg_signals = NegativeEvidenceEngine.evaluate_following_negative_evidence(
                p_lead, p_follow, context, event_t
            )
            neg_names = {s.signal_type for s in neg_signals}
            if any(k in neg_names for k in ["Negative: Pedestrian Queue Formation", "Negative: Symmetric Trajectory Correlation"]):
                continue

            supporting_signals = [
                SupportingSignal(
                    signal_type="Lagged Trajectory Correlation",
                    description=(
                        f"[{p_follow.track_id}] traversed path of [{p_lead.track_id}] with ~{lag_res['best_lag_seconds']:.1f}s lag "
                        f"(mean path deviation: {lag_res['mean_path_deviation_px']:.1f}px, precedence: {lag_res.get('spatial_precedence_px', 0):.1f}px)"
                    ),
                    confidence=lag_res["confidence"],
                    timestamp=event_t,
                ),
            ]

            scoring = IncidentScorer.calculate_evidence_score(
                base_confidence=0.68,
                supporting_signals=supporting_signals,
                tracks=[p_lead, p_follow],
                duration_seconds=shared_dur,
                contradictory_signals=neg_signals,
            )

            cand = self.build_candidate(
                video_id=context.video_id,
                event_type="PERSON_FOLLOWING",
                start_time=max(0.0, event_t - 1.0),
                end_time=min(context.duration_seconds, event_t + 2.0),
                severity="NORMAL",
                confidence=scoring["confidence"],
                track_ids=[p_lead.track_id, p_follow.track_id],
                object_classes=["person", "person"],
                supporting_signals=supporting_signals,
                contradictory_signals=neg_signals,
                explanation=(
                    f"Person following pattern: [{p_follow.track_id}] traversed path of [{p_lead.track_id}] "
                    f"with consistent {lag_res['best_lag_seconds']:.1f}s lag over {shared_dur:.1f}s. Observational finding."
                ),
                evidence_candidates=[
                    EvidenceCandidate(
                        timestamp=event_t,
                        pre_seconds=2.0,
                        post_seconds=2.0,
                        bounding_box=p_follow.current_bbox,
                        target_track_id=p_follow.track_id,
                        reason="Forensic capture of person following pattern",
                    )
                ],
                human_verification_required=True,
            )
            candidates.append(cand)

        return candidates
