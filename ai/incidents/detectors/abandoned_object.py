"""
Abandoned Object Detector (Phase 10)

Detects stationary, unattended belongings (backpacks, suitcases, handbags)
where nearby person tracks have departed the immediate perimeter.
"""
import math
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


class AbandonedObjectDetector(BaseIncidentDetector):
    """
    Evaluates portable belonging tracks for unattended stationary patterns.
    """

    detector_name: str = "abandoned_object_detector"
    detector_version: str = "1.1.0"
    category: IncidentCategory = IncidentCategory.PROPERTY

    # Phase Z Contract Declarations
    required_signals: List[str] = ["Stationary Belonging", "Unattended State"]
    supporting_signals_declared: List[str] = ["Stationary Belonging", "Unattended State"]
    contradictory_signals_declared: List[str] = ["Negative: Person Nearby", "Negative: Moving Belonging"]
    context_requirements: Dict[str, Any] = {"belonging_classes": ["backpack", "suitcase", "handbag"]}
    evidence_requirements: Dict[str, Any] = {"stationary_track": True, "bounding_box_grounding": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        stationary_threshold_seconds: float = 4.0,
        abandoned_separation_distance: float = 120.0,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.stationary_threshold_seconds = stationary_threshold_seconds
        self.abandoned_separation_distance = abandoned_separation_distance
        self.belonging_classes = {"backpack", "suitcase", "handbag"}


    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        person_tracks = [t for t in context.tracks if t.object_class == "person"]
        belonging_tracks = [t for t in context.tracks if t.object_class in self.belonging_classes]

        for b_track in belonging_tracks:
            if b_track.duration_seconds < self.stationary_threshold_seconds or len(b_track.trajectory) < 2:
                continue

            summary = context.get_motion_summary(b_track.track_id) or {}
            net_disp = summary.get("net_displacement", 0.0)
            if net_disp > 40.0:  # Moving object is being carried, not abandoned
                continue

            b_start_x, b_start_y = b_track.trajectory[0][1], b_track.trajectory[0][2]
            b_end_x, b_end_y = b_track.trajectory[-1][1], b_track.trajectory[-1][2]

            # Was there a person nearby at the start?
            associated_person_id = None
            for p_track in person_tracks:
                if not p_track.trajectory:
                    continue
                p_early = [pt for pt in p_track.trajectory if pt[0] <= b_track.first_seen + 2.0]
                for pt in p_early:
                    if math.hypot(pt[1] - b_start_x, pt[2] - b_start_y) <= self.abandoned_separation_distance:
                        associated_person_id = p_track.track_id
                        break
                if associated_person_id:
                    break

            # Is any person currently nearby at the end?
            person_nearby_at_end = False
            for p_track in person_tracks:
                if not p_track.trajectory:
                    continue
                p_late = [pt for pt in p_track.trajectory if pt[0] >= b_track.last_seen - 2.0]
                for pt in p_late:
                    if math.hypot(pt[1] - b_end_x, pt[2] - b_end_y) <= self.abandoned_separation_distance:
                        person_nearby_at_end = True
                        break
                if person_nearby_at_end:
                    break

            if not person_nearby_at_end:
                signals = [
                    SupportingSignal(
                        signal_type="Stationary Belonging",
                        description=f"{b_track.object_class} [{b_track.track_id}] remained stationary for {b_track.duration_seconds:.1f}s",
                        confidence=b_track.confidence,
                        timestamp=b_track.first_seen,
                        track_id=b_track.track_id,
                    ),
                    SupportingSignal(
                        signal_type="Unattended State",
                        description=f"No tracked person in proximity at {b_track.last_seen:.1f}s (initial associate: {associated_person_id or 'none'})",
                        confidence=0.88,
                        timestamp=b_track.last_seen,
                        track_id=b_track.track_id,
                    ),
                ]

                scoring = IncidentScorer.calculate_evidence_score(
                    base_confidence=0.80,
                    supporting_signals=signals,
                    tracks=[b_track],
                    duration_seconds=b_track.duration_seconds,
                    expected_duration_threshold=self.stationary_threshold_seconds,
                )

                spatial_ctx = SpatialContext(
                    centroid=b_track.current_bbox.centroid if b_track.current_bbox else None,
                    bounding_box=b_track.current_bbox,
                )

                cand = self.build_candidate(
                    video_id=context.video_id,
                    event_type="POTENTIAL_ABANDONED_OBJECT",
                    start_time=b_track.last_seen,
                    end_time=b_track.last_seen + 1.0,
                    severity="HIGH",
                    confidence=scoring["score"],
                    explanation=(
                        f"Potential abandoned object candidate: stationary {b_track.object_class} "
                        f"[{b_track.track_id}] observed unattended for {b_track.duration_seconds:.1f}s."
                    ),
                    track_ids=[b_track.track_id],
                    object_classes=[b_track.object_class],
                    supporting_signals=signals,
                    spatial_context=spatial_ctx,
                    evidence_candidates=[
                        EvidenceCandidate(
                            timestamp=b_track.last_seen,
                            bounding_box=b_track.current_bbox,
                            target_track_id=b_track.track_id,
                            reason="Unattended stationary object candidate",
                        )
                    ],
                    prefix="ABAN",
                )
                candidates.append(cand)

        return candidates
