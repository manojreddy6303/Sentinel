"""
Potential Object Left Behind Detector (Phase 13)

Detects the specific sequence where a person carries or interacts with an object,
departs from the object's vicinity, and the object subsequently remains stationary
and unattended over a meaningful temporal duration.
"""
import math
from typing import List, Dict, Any, Optional

from ai.schemas import BoundingBox, TrackedObject
from ai.incidents.base_detector import BaseIncidentDetector
from ai.incidents.schemas import (
    IncidentCandidate,
    IncidentContext,
    IncidentCategory,
    SupportingSignal,
    EvidenceCandidate,
    SpatialContext,
    ValidationDecision,
)
from ai.incidents.scoring import IncidentScorer
from ai.incidents.detectors.property.state_machine import ObjectStateMachine, ObjectTemporalState
from ai.incidents.detectors.property.association import ObjectPersonAssociationEngine
from ai.incidents.detectors.property.camera_stability import CameraStabilityEngine


class ObjectLeftBehindDetector(BaseIncidentDetector):
    """
    Evaluates portable belongings for object-left-behind patterns with grounded person departure.
    """

    detector_name: str = "object_left_behind_detector"
    detector_version: str = "1.0.0"
    category: IncidentCategory = IncidentCategory.PROPERTY

    required_signals: List[str] = ["Object-Person Interaction", "Person Departure Vector", "Stationary Residual State"]
    supporting_signals_declared: List[str] = ["Spatial Interaction Proximity", "Person Departure Vector", "Stationary Residual State"]
    contradictory_signals_declared: List[str] = ["Negative: Person Returned", "Negative: Object Moved With Person"]
    context_requirements: Dict[str, Any] = {"belonging_classes": True}
    evidence_requirements: Dict[str, Any] = {"object_track": True, "person_track": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        min_residual_seconds: float = 3.0,
        departure_min_distance: float = 65.0,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.min_residual_seconds = min_residual_seconds
        self.departure_min_distance = departure_min_distance
        self.target_classes = {
            "backpack", "handbag", "suitcase", "laptop", "cell phone",
            "bottle", "umbrella", "box", "package", "merchandise", "book", "general_object", "bicycle",
        }
        self.state_machine = ObjectStateMachine()
        self.association_engine = ObjectPersonAssociationEngine(
            min_departure_distance=departure_min_distance
        )
        self.stability_engine = CameraStabilityEngine()

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        person_tracks = [t for t in context.tracks if t.object_class == "person"]
        belonging_tracks = [t for t in context.tracks if t.object_class in self.target_classes]

        stability = self.stability_engine.assess_stability(context)
        if not stability.is_camera_stable:
            # Skip or downgrade if global camera movement compromises tracking
            return candidates

        for o_track in belonging_tracks:
            if o_track.duration_seconds < self.min_residual_seconds or not o_track.trajectory:
                continue

            state_record = self.state_machine.analyze_track_state(o_track, person_tracks, context)
            if state_record.is_edge_clipped:
                continue

            # Need at least one candidate person who departed
            for p_track in person_tracks:
                assoc = self.association_engine.evaluate_association(p_track, o_track)
                if not assoc.is_associated or not assoc.departure_detected:
                    continue

                # Verify object remained at original coordinates after departure
                orig_x, orig_y = o_track.trajectory[0][1], o_track.trajectory[0][2]
                residual_points = [pt for pt in o_track.trajectory if pt[0] >= assoc.interaction_end]
                if not residual_points:
                    continue

                residual_duration = residual_points[-1][0] - residual_points[0][0]
                if residual_duration < self.min_residual_seconds:
                    continue

                max_o_disp = max(math.hypot(pt[1] - orig_x, pt[2] - orig_y) for pt in residual_points)
                if max_o_disp > 30.0:
                    # Object moved, not left behind
                    continue

                # Check if person returned
                person_returned = False
                post_dep_p = [pt for pt in p_track.trajectory if pt[0] > assoc.interaction_end + 2.0]
                for ppt in post_dep_p:
                    if math.hypot(ppt[1] - orig_x, ppt[2] - orig_y) <= 40.0:
                        person_returned = True
                        break

                contradictory_signals = []
                if person_returned:
                    contradictory_signals.append(
                        SupportingSignal(
                            signal_type="Negative: Person Returned",
                            description=f"Person [{p_track.track_id}] returned to {o_track.object_class} vicinity",
                            confidence=0.90,
                            timestamp=assoc.interaction_end + 2.0,
                        )
                    )
                    continue

                signals = [
                    SupportingSignal(
                        signal_type="Object-Person Interaction",
                        description=f"Person [{p_track.track_id}] interacted with {o_track.object_class} [{o_track.track_id}] (dwell: {assoc.interaction_duration:.1f}s)",
                        confidence=assoc.association_confidence,
                        timestamp=assoc.interaction_start,
                        track_id=p_track.track_id,
                    ),
                    SupportingSignal(
                        signal_type="Person Departure Vector",
                        description=f"Person [{p_track.track_id}] departed ({assoc.departure_displacement:.1f}px displacement)",
                        confidence=0.86,
                        timestamp=assoc.interaction_end,
                        track_id=p_track.track_id,
                    ),
                    SupportingSignal(
                        signal_type="Stationary Residual State",
                        description=f"{o_track.object_class} [{o_track.track_id}] persisted unattended at original coordinates for {residual_duration:.1f}s",
                        confidence=0.88,
                        timestamp=assoc.interaction_end,
                        track_id=o_track.track_id,
                    ),
                ]

                scoring = IncidentScorer.calculate_evidence_score(
                    base_confidence=0.72,
                    supporting_signals=signals,
                    tracks=[o_track, p_track],
                    duration_seconds=residual_duration,
                    expected_duration_threshold=self.min_residual_seconds,
                    contradictory_signals=contradictory_signals,
                )

                spatial_ctx = SpatialContext(
                    centroid=o_track.current_bbox.centroid if o_track.current_bbox else None,
                    bounding_box=o_track.current_bbox,
                    metadata={
                        "associated_person_track_id": p_track.track_id,
                        "object_track_id": o_track.track_id,
                        "object_class": o_track.object_class,
                        "residual_duration": round(residual_duration, 2),
                    }
                )

                cand = self.build_candidate(
                    video_id=context.video_id,
                    event_type="POTENTIAL_OBJECT_LEFT_BEHIND",
                    start_time=assoc.interaction_start,
                    end_time=o_track.last_seen,
                    severity="NORMAL",
                    confidence=min(0.70, scoring["score"]),
                    explanation=(
                        f"Potential Object Left Behind ({scoring['verification_note']}): "
                        f"Person [{p_track.track_id}] interacted with {o_track.object_class} [{o_track.track_id}] "
                        f"and departed ({assoc.departure_displacement:.1f}px), after which the object remained stationary for {residual_duration:.1f}s."
                    ),
                    track_ids=[o_track.track_id, p_track.track_id],
                    object_classes=[o_track.object_class, "person"],
                    supporting_signals=signals,
                    contradictory_signals=contradictory_signals,
                    spatial_context=spatial_ctx,
                    evidence_candidates=[
                        EvidenceCandidate(
                            timestamp=assoc.interaction_end,
                            bounding_box=o_track.current_bbox,
                            target_track_id=o_track.track_id,
                            reason="Object left behind following person departure",
                        )
                    ],
                    prefix="OBJLB",
                )
                cand.validation_decision = ValidationDecision.REVIEW_REQUIRED
                candidates.append(cand)

        return candidates
