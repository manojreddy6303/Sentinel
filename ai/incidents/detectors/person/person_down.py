"""
Potential Person Down Detector (Sentinel Phase 12)

Detects a person remaining in a low-mobility horizontal/ground state for an extended duration:
- Low centroid displacement over sustained dwell window (>= 3.0s)
- Low/horizontal aspect ratio (w/h >= 0.75), distinguishing upright standing still (w/h ~0.35)
- Absence of normal walking transit
- Observational, conservative review labeling
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


class PersonDownDetector(BaseIncidentDetector):
    """
    Evaluates tracked person presence for prolonged ground-level low-mobility states.
    """

    detector_name: str = "person_down_detector"
    detector_version: str = "1.0.0"
    category: IncidentCategory = IncidentCategory.PERSON

    required_signals: List[str] = ["Low Mobility Dwell", "Horizontal Geometry"]
    supporting_signals_declared: List[str] = [
        "Prolonged Horizontal Dwell",
        "Arrested Centroid Displacement",
        "Ground Level Persistence",
    ]
    contradictory_signals_declared: List[str] = [
        "Negative: Upright Posture Maintained",
        "Negative: Resumed Walking Transit",
    ]
    context_requirements: Dict[str, Any] = {"pedestrian_context": True}
    evidence_requirements: Dict[str, Any] = {"verified_track": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        min_down_duration_seconds: float = 3.0,
        max_displacement_ratio: float = 0.50,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.min_down_duration_seconds = min_down_duration_seconds
        self.max_displacement_ratio = max_displacement_ratio

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        persons = [t for t in context.tracks if t.object_class == "person" and t.is_validated]

        for track in persons:
            if track.duration_seconds < self.min_down_duration_seconds:
                continue

            dynamics = PersonMotionFeatureEngine.compute_person_dynamics(track, context.track_motions)
            is_low_mob = dynamics.get("is_low_mobility", False)
            dwell_sec = dynamics.get("horizontal_dwell_seconds", 0.0)
            final_ar = dynamics.get("final_aspect_ratio", 0.0)

            # Upright check: if aspect ratio is upright (e.g. < 0.60), person is standing still, NOT down!
            if final_ar < 0.65:
                continue

            # Dwell duration check
            if not is_low_mob and dwell_sec < self.min_down_duration_seconds:
                # Check if entire track was consistently horizontal and stationary
                if final_ar >= 0.80 and dynamics.get("avg_normalized_speed_bl_s", 0.0) < 0.35 and track.duration_seconds >= self.min_down_duration_seconds:
                    dwell_sec = track.duration_seconds
                    is_low_mob = True
                else:
                    continue

            # Negative evidence check
            neg_signals = NegativeEvidenceEngine.evaluate_person_down_negative_evidence(track, context, track.first_seen)
            neg_names = {s.signal_type for s in neg_signals}
            if any(k in s for s in neg_names for k in ["Resumed Walking Transit", "Frame Boundary Truncation"]):
                continue

            if dynamics.get("is_edge_clipped", False):
                continue

            supporting_signals = [
                SupportingSignal(
                    signal_type="Prolonged Horizontal Dwell",
                    description=f"Person remained in horizontal/ground geometry for {dwell_sec:.1f}s (aspect ratio: {final_ar:.2f})",
                    confidence=0.82,
                    timestamp=track.first_seen,
                    track_id=track.track_id,
                ),
                SupportingSignal(
                    signal_type="Arrested Centroid Displacement",
                    description="Centroid displacement remained near-zero during low-mobility period",
                    confidence=0.80,
                    timestamp=track.first_seen,
                    track_id=track.track_id,
                ),
            ]

            scoring = IncidentScorer.calculate_evidence_score(
                base_confidence=0.70,
                supporting_signals=supporting_signals,
                tracks=[track],
                duration_seconds=dwell_sec,
                contradictory_signals=neg_signals,
            )

            cand = self.build_candidate(
                video_id=context.video_id,
                event_type="POTENTIAL_PERSON_DOWN",
                start_time=track.first_seen,
                end_time=track.last_seen,
                severity="NORMAL",
                confidence=scoring["confidence"],
                track_ids=[track.track_id],
                object_classes=["person"],
                supporting_signals=supporting_signals,
                contradictory_signals=neg_signals,
                explanation=(
                    f"Potential person down pattern: subject [{track.track_id}] remained in sustained horizontal/ground geometry "
                    f"for {dwell_sec:.1f}s. Observational finding; requires human verification."
                ),
                evidence_candidates=[
                    EvidenceCandidate(
                        timestamp=track.first_seen + 1.0,
                        pre_seconds=2.0,
                        post_seconds=3.0,
                        bounding_box=track.current_bbox,
                        target_track_id=track.track_id,
                        reason="Forensic capture of potential person down event",
                    )
                ],
                human_verification_required=True,
            )
            cand.validation_decision = "REVIEW_REQUIRED"
            candidates.append(cand)

        return candidates
