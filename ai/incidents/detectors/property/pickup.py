"""
Potential Object Pickup Detector (Phase 13)

Detects the physical sequence where a stationary object is approached by a person,
entered into contact/proximity, and subsequently transitioned into motion concurrently
with the person.
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
from ai.incidents.detectors.property.association import ObjectPersonAssociationEngine
from ai.incidents.detectors.property.camera_stability import CameraStabilityEngine


class ObjectPickupDetector(BaseIncidentDetector):
    """
    Evaluates portable object tracks for stationary-to-movement transitions initiated by person contact.
    """

    detector_name: str = "object_pickup_detector"
    detector_version: str = "1.0.0"
    category: IncidentCategory = IncidentCategory.PROPERTY

    required_signals: List[str] = ["Prior Stationary State", "Person Approach & Contact", "Post-Interaction Motion"]
    supporting_signals_declared: List[str] = ["Prior Stationary State", "Person Approach & Contact", "Post-Interaction Motion"]
    contradictory_signals_declared: List[str] = ["Negative: Object Never Moved", "Negative: Transient Passerby"]
    context_requirements: Dict[str, Any] = {"belonging_classes": True}
    evidence_requirements: Dict[str, Any] = {"object_track": True, "person_track": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        min_prior_stationary_seconds: float = 2.0,
        min_pickup_displacement: float = 35.0,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.min_prior_stationary_seconds = min_prior_stationary_seconds
        self.min_pickup_displacement = min_pickup_displacement
        self.target_classes = {
            "backpack", "handbag", "suitcase", "laptop", "cell phone",
            "bottle", "umbrella", "box", "package", "merchandise", "book", "general_object", "unknown_portable_object", "bicycle",
        }
        self.association_engine = ObjectPersonAssociationEngine()
        self.stability_engine = CameraStabilityEngine()

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        person_tracks = [t for t in context.tracks if t.object_class == "person"]
        belonging_tracks = [t for t in context.tracks if t.object_class in self.target_classes]

        stability = self.stability_engine.assess_stability(context)
        if not stability.is_camera_stable:
            return candidates

        for o_track in belonging_tracks:
            if not o_track.trajectory or len(o_track.trajectory) < 3:
                continue

            orig_ox, orig_oy = o_track.trajectory[0][1], o_track.trajectory[0][2]
            for p_track in person_tracks:
                assoc = self.association_engine.evaluate_association(p_track, o_track)
                if not assoc.is_associated:
                    continue

                # 1. Verify object was stationary before interaction
                pre_interaction_o = [pt for pt in o_track.trajectory if pt[0] <= assoc.interaction_start]
                if not pre_interaction_o:
                    continue

                pre_disp = max(math.hypot(pt[1] - orig_ox, pt[2] - orig_oy) for pt in pre_interaction_o)
                if pre_disp > 20.0:
                    # Was already moving, not stationary prior to pickup
                    continue

                # 2. Verify post-interaction object movement
                post_interaction_o = [pt for pt in o_track.trajectory if pt[0] >= assoc.interaction_start]
                if not post_interaction_o:
                    continue

                final_o = post_interaction_o[-1]
                post_disp = math.hypot(final_o[1] - orig_ox, final_o[2] - orig_oy)
                if post_disp < self.min_pickup_displacement:
                    # Object did not move substantially
                    continue

                # 3. Verify co-movement with person
                signals = [
                    SupportingSignal(
                        signal_type="Prior Stationary State",
                        description=f"{o_track.object_class} [{o_track.track_id}] was stationary prior to interaction ({pre_disp:.1f}px drift)",
                        confidence=0.88,
                        timestamp=pre_interaction_o[0][0],
                        track_id=o_track.track_id,
                    ),
                    SupportingSignal(
                        signal_type="Person Approach & Contact",
                        description=f"Person [{p_track.track_id}] approached within {assoc.min_distance:.1f}px (dwell: {assoc.interaction_duration:.1f}s)",
                        confidence=assoc.association_confidence,
                        timestamp=assoc.interaction_start,
                        track_id=p_track.track_id,
                    ),
                    SupportingSignal(
                        signal_type="Post-Interaction Motion",
                        description=f"{o_track.object_class} [{o_track.track_id}] displaced {post_disp:.1f}px following interaction",
                        confidence=0.86,
                        timestamp=assoc.interaction_end,
                        track_id=o_track.track_id,
                    ),
                ]

                scoring = IncidentScorer.calculate_evidence_score(
                    base_confidence=0.72,
                    supporting_signals=signals,
                    tracks=[o_track, p_track],
                    duration_seconds=assoc.interaction_duration,
                    expected_duration_threshold=1.0,
                )

                spatial_ctx = SpatialContext(
                    centroid=o_track.current_bbox.centroid if o_track.current_bbox else None,
                    bounding_box=o_track.current_bbox,
                    metadata={
                        "person_track_id": p_track.track_id,
                        "object_track_id": o_track.track_id,
                        "object_class": o_track.object_class,
                        "pickup_displacement": round(post_disp, 2),
                    }
                )

                cand = self.build_candidate(
                    video_id=context.video_id,
                    event_type="POTENTIAL_OBJECT_PICKUP",
                    start_time=assoc.interaction_start,
                    end_time=o_track.last_seen,
                    severity="NORMAL",
                    confidence=min(0.70, scoring["score"]),
                    explanation=(
                        f"Potential Object Pickup ({scoring['verification_note']}): "
                        f"Stationary {o_track.object_class} [{o_track.track_id}] was approached by "
                        f"person [{p_track.track_id}] and subsequently displaced {post_disp:.1f}px."
                    ),
                    track_ids=[o_track.track_id, p_track.track_id],
                    object_classes=[o_track.object_class, "person"],
                    supporting_signals=signals,
                    spatial_context=spatial_ctx,
                    evidence_candidates=[
                        EvidenceCandidate(
                            timestamp=assoc.interaction_start,
                            bounding_box=o_track.current_bbox,
                            target_track_id=o_track.track_id,
                            reason="Object transition from stationary to motion",
                        )
                    ],
                    prefix="OBJPK",
                )
                cand.validation_decision = ValidationDecision.REVIEW_REQUIRED
                candidates.append(cand)

        return candidates
