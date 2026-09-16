"""
Unusual Rapid Person Movement Detector (Sentinel Phase 12)

Detects person movement that is statistically or kinematically anomalous relative
to other individuals in the same scene or local scene baseline:
- Compares subject speed with peer pedestrian distribution in the video
- Disproportionate velocity disparity (> 2.2x local pedestrian mean)
- Avoids brittle universal pixel thresholds across different camera depths
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


class UnusualRapidPersonMovementDetector(BaseIncidentDetector):
    """
    Evaluates tracked person speed relative to the local scene pedestrian distribution.
    """

    detector_name: str = "unusual_rapid_person_movement_detector"
    detector_version: str = "1.0.0"
    category: IncidentCategory = IncidentCategory.PERSON

    required_signals: List[str] = ["Scene-Disproportionate Velocity"]
    supporting_signals_declared: List[str] = [
        "Scene-Disproportionate Velocity",
        "Kinematic Outlier",
    ]
    contradictory_signals_declared: [
        "Negative: Synchronized Pedestrian Flow Pace",
    ]
    context_requirements: Dict[str, Any] = {"pedestrian_context": True}
    evidence_requirements: Dict[str, Any] = {"verified_track": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        velocity_disparity_ratio: float = 2.2,
        min_absolute_speed_bl_s: float = 2.2,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.velocity_disparity_ratio = velocity_disparity_ratio
        self.min_absolute_speed_bl_s = min_absolute_speed_bl_s

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        persons = [t for t in context.tracks if t.object_class == "person" and t.is_validated]
        if not persons:
            return candidates

        # Compute dynamics for all persons in scene
        person_dynamics: Dict[str, Dict[str, Any]] = {}
        for p in persons:
            person_dynamics[p.track_id] = PersonMotionFeatureEngine.compute_person_dynamics(p, context.track_motions)

        # Baseline scene pedestrian speed (body-lengths/s)
        speeds = [d.get("avg_normalized_speed_bl_s", 1.0) for d in person_dynamics.values() if d.get("avg_normalized_speed_bl_s", 0.0) > 0.0]
        scene_avg_speed = sum(speeds) / len(speeds) if speeds else 1.0

        for track in persons:
            dyn = person_dynamics[track.track_id]
            max_spd = dyn.get("max_normalized_speed_bl_s", 0.0)
            avg_spd = dyn.get("avg_normalized_speed_bl_s", 0.0)

            # Must exceed minimum absolute threshold
            if max_spd < self.min_absolute_speed_bl_s:
                continue

            # Must be significantly faster than other pedestrians in scene (if multi-person scene)
            if len(persons) >= 2:
                other_speeds = [
                    d.get("avg_normalized_speed_bl_s", 1.0)
                    for tid, d in person_dynamics.items()
                    if tid != track.track_id and d.get("avg_normalized_speed_bl_s", 0.0) > 0.0
                ]
                other_avg = sum(other_speeds) / len(other_speeds) if other_speeds else scene_avg_speed
                if avg_spd < (other_avg * self.velocity_disparity_ratio) and max_spd < (other_avg * (self.velocity_disparity_ratio + 0.5)):
                    continue

            supporting_signals = [
                SupportingSignal(
                    signal_type="Scene-Disproportionate Velocity",
                    description=(
                        f"Pedestrian velocity ({max_spd:.1f} bl/s) was {max_spd / max(0.5, scene_avg_speed):.1f}x "
                        f"the average observed scene pedestrian pace ({scene_avg_speed:.1f} bl/s)"
                    ),
                    confidence=0.80,
                    timestamp=track.first_seen,
                    track_id=track.track_id,
                ),
            ]

            scoring = IncidentScorer.calculate_evidence_score(
                base_confidence=0.72,
                supporting_signals=supporting_signals,
                tracks=[track],
                duration_seconds=track.duration_seconds,
            )

            cand = self.build_candidate(
                video_id=context.video_id,
                event_type="UNUSUAL_RAPID_PERSON_MOVEMENT",
                start_time=track.first_seen,
                end_time=track.last_seen,
                severity="NORMAL",
                confidence=scoring["confidence"],
                track_ids=[track.track_id],
                object_classes=["person"],
                supporting_signals=supporting_signals,
                explanation=(
                    f"Unusual rapid movement: [{track.track_id}] moved at {max_spd:.1f} body-lengths/sec, "
                    f"disproportionate to local pedestrian activity. Observational finding."
                ),
                evidence_candidates=[
                    EvidenceCandidate(
                        timestamp=track.first_seen + (track.duration_seconds / 2.0),
                        pre_seconds=2.0,
                        post_seconds=2.0,
                        bounding_box=track.current_bbox,
                        target_track_id=track.track_id,
                        reason="Forensic capture of unusual rapid movement",
                    )
                ],
                human_verification_required=True,
            )
            candidates.append(cand)

        return candidates
