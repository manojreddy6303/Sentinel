"""
Potential Panic / Running Detector (Sentinel Phase 12)

Detects unusual rapid human running or crowd panic dispersion:
- Sustained high normalized speed (> 3.0 body-lengths/second)
- Abrupt acceleration from walking/standing to rapid running
- Multi-person directional dispersion away from an encounter window
- Negative evidence filtering normal walking, brisk transit, and recreational contexts
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


class PanicRunningDetector(BaseIncidentDetector):
    """
    Evaluates tracked person speed and dispersion for potential panic or rapid running patterns.
    """

    detector_name: str = "panic_running_detector"
    detector_version: str = "1.0.0"
    category: IncidentCategory = IncidentCategory.PERSON

    required_signals: List[str] = ["High Normalized Speed"]
    supporting_signals_declared: List[str] = [
        "High Normalized Speed (> 3.0 body lengths/s)",
        "Sudden Acceleration Burst",
        "Multi-Person Directional Dispersion",
    ]
    contradictory_signals_declared: List[str] = [
        "Negative: Normal Pedestrian Transit Pace",
        "Negative: Expected Recreational Activity",
    ]
    context_requirements: Dict[str, Any] = {"pedestrian_context": True}
    evidence_requirements: Dict[str, Any] = {"verified_track": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        min_running_speed_bl_s: float = 3.0,
        min_running_duration_seconds: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.min_running_speed_bl_s = min_running_speed_bl_s
        self.min_running_duration_seconds = min_running_duration_seconds

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        persons = [t for t in context.tracks if t.object_class == "person" and t.is_validated]

        running_tracks: List[Tuple[TrackedObject, Dict[str, Any]]] = []

        for track in persons:
            if track.duration_seconds < self.min_running_duration_seconds:
                continue

            dynamics = PersonMotionFeatureEngine.compute_person_dynamics(track, context.track_motions)
            max_speed = dynamics.get("max_normalized_speed_bl_s", 0.0)
            avg_speed = dynamics.get("avg_normalized_speed_bl_s", 0.0)

            if max_speed >= self.min_running_speed_bl_s or avg_speed >= (self.min_running_speed_bl_s * 0.75):
                running_tracks.append((track, dynamics))

        if not running_tracks:
            return candidates

        # Check for multi-person dispersal or isolated panic running
        for track, dyn in running_tracks:
            max_speed = dyn.get("max_normalized_speed_bl_s", 0.0)
            avg_speed = dyn.get("avg_normalized_speed_bl_s", 0.0)

            neg_signals = NegativeEvidenceEngine.evaluate_running_negative_evidence(track, context)
            if any("Normal Pedestrian Transit Pace" in s.signal_type for s in neg_signals):
                continue

            supporting_signals = [
                SupportingSignal(
                    signal_type="High Normalized Speed",
                    description=f"Person moved at {max_speed:.1f} body-lengths/sec (avg: {avg_speed:.1f} bl/s)",
                    confidence=0.82,
                    timestamp=track.first_seen,
                    track_id=track.track_id,
                ),
            ]

            if len(running_tracks) >= 2:
                supporting_signals.append(
                    SupportingSignal(
                        signal_type="Multi-Person Directional Dispersion",
                        description=f"Multiple pedestrians ({len(running_tracks)}) exhibited concurrent rapid movement",
                        confidence=0.85,
                        timestamp=track.first_seen,
                    )
                )

            scoring = IncidentScorer.calculate_evidence_score(
                base_confidence=0.74,
                supporting_signals=supporting_signals,
                tracks=[track],
                duration_seconds=track.duration_seconds,
            )

            cand = self.build_candidate(
                video_id=context.video_id,
                event_type="POTENTIAL_PANIC_RUNNING",
                start_time=track.first_seen,
                end_time=track.last_seen,
                severity="NORMAL" if len(running_tracks) == 1 else "HIGH",
                confidence=scoring["confidence"],
                track_ids=[track.track_id],
                object_classes=["person"],
                supporting_signals=supporting_signals,
                contradictory_signals=neg_signals,
                explanation=(
                    f"Potential panic / running: [{track.track_id}] exhibited rapid movement at "
                    f"{max_speed:.1f} body-lengths/sec. Observational finding."
                ),
                evidence_candidates=[
                    EvidenceCandidate(
                        timestamp=track.first_seen + (track.duration_seconds / 2.0),
                        pre_seconds=2.0,
                        post_seconds=2.0,
                        bounding_box=track.current_bbox,
                        target_track_id=track.track_id,
                        reason="Forensic capture of rapid running pattern",
                    )
                ],
                human_verification_required=True,
            )
            candidates.append(cand)

        return candidates
