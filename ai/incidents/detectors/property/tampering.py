"""
Potential Property Tampering Detector (Phase 13)

Detects sustained, repeated, or abnormal physical manipulation of property fixtures,
barriers, vehicles, or infrastructure elements by a person. Requires observable temporal
interaction and physical state or coordinate disturbance.
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


class PropertyTamperingDetector(BaseIncidentDetector):
    """
    Evaluates physical interaction and micro-displacement on property structures and stationary fixtures.
    """

    detector_name: str = "property_tampering_detector"
    detector_version: str = "1.0.0"
    category: IncidentCategory = IncidentCategory.PROPERTY

    required_signals: List[str] = ["Sustained Contact Proximity", "Observable Property Agitation"]
    supporting_signals_declared: List[str] = ["Sustained Contact Proximity", "Observable Property Agitation", "Fixture Coordinate Shift"]
    contradictory_signals_declared: List[str] = ["Negative: Normal Passing Proximity", "Negative: Zero Property Displacement"]
    context_requirements: Dict[str, Any] = {"property_classes": True}
    evidence_requirements: Dict[str, Any] = {"person_track": True, "property_track": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        min_contact_seconds: float = 3.0,
        contact_max_distance: float = 55.0,
        min_fixture_shift_px: float = 12.0,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.min_contact_seconds = min_contact_seconds
        self.contact_max_distance = contact_max_distance
        self.min_fixture_shift_px = min_fixture_shift_px
        self.property_classes = {
            "bicycle", "car", "motorcycle", "truck", "door", "gate",
            "barrier", "box", "suitcase", "bench", "chair",
        }

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        person_tracks = [t for t in context.tracks if t.object_class == "person"]
        property_tracks = [t for t in context.tracks if t.object_class in self.property_classes]

        for prop in property_tracks:
            if not prop.trajectory or len(prop.trajectory) < 3:
                continue

            orig_px, orig_py = prop.trajectory[0][1], prop.trajectory[0][2]

            for person in person_tracks:
                if not person.trajectory:
                    continue

                contact_points = []
                for pt in person.trajectory:
                    t, px, py = pt[0], pt[1], pt[2]
                    # Find closest property point in time
                    closest_p = min(prop.trajectory, key=lambda opt: abs(opt[0] - t))
                    if abs(closest_p[0] - t) <= 1.0:
                        dist = math.hypot(px - closest_p[1], py - closest_p[2])
                        if dist <= self.contact_max_distance:
                            contact_points.append((t, dist))

                if not contact_points:
                    continue

                contact_start = min(cp[0] for cp in contact_points)
                contact_end = max(cp[0] for cp in contact_points)
                contact_dur = contact_end - contact_start

                if contact_dur < self.min_contact_seconds:
                    # Transient passing, not tampering
                    continue

                # Check if property exhibited measurable coordinate shift or agitation
                prop_contact_pts = [pt for pt in prop.trajectory if contact_start <= pt[0] <= contact_end + 2.0]
                if not prop_contact_pts:
                    continue

                prop_shift = max(math.hypot(pt[1] - orig_px, pt[2] - orig_py) for pt in prop_contact_pts)

                contradictory_signals = []
                if prop_shift < self.min_fixture_shift_px:
                    # Contact without any observable state change
                    contradictory_signals.append(
                        SupportingSignal(
                            signal_type="Negative: Zero Property Displacement",
                            description=f"Property [{prop.object_class} • {prop.track_id}] had no physical state change ({prop_shift:.1f}px shift)",
                            confidence=0.88,
                            timestamp=contact_end,
                        )
                    )
                    continue

                signals = [
                    SupportingSignal(
                        signal_type="Sustained Contact Proximity",
                        description=f"Person [{person.track_id}] maintained direct physical contact with {prop.object_class} [{prop.track_id}] for {contact_dur:.1f}s",
                        confidence=0.86,
                        timestamp=contact_start,
                        track_id=person.track_id,
                    ),
                    SupportingSignal(
                        signal_type="Observable Property Agitation",
                        description=f"Property {prop.object_class} [{prop.track_id}] shifted {prop_shift:.1f}px during interaction",
                        confidence=0.84,
                        timestamp=contact_end,
                        track_id=prop.track_id,
                    ),
                ]

                scoring = IncidentScorer.calculate_evidence_score(
                    base_confidence=0.68,
                    supporting_signals=signals,
                    tracks=[prop, person],
                    duration_seconds=contact_dur,
                    expected_duration_threshold=self.min_contact_seconds,
                )

                spatial_ctx = SpatialContext(
                    centroid=prop.current_bbox.centroid if prop.current_bbox else None,
                    bounding_box=prop.current_bbox,
                    metadata={
                        "person_track_id": person.track_id,
                        "property_track_id": prop.track_id,
                        "property_class": prop.object_class,
                        "contact_duration": round(contact_dur, 2),
                        "property_shift_px": round(prop_shift, 2),
                    }
                )

                cand = self.build_candidate(
                    video_id=context.video_id,
                    event_type="POTENTIAL_PROPERTY_TAMPERING",
                    start_time=contact_start,
                    end_time=max(person.last_seen, prop.last_seen),
                    severity="NORMAL",
                    confidence=min(0.65, scoring["score"]),
                    explanation=(
                        f"Potential Property Tampering ({scoring['verification_note']}): "
                        f"Person [{person.track_id}] maintained sustained physical contact with "
                        f"{prop.object_class} [{prop.track_id}] for {contact_dur:.1f}s, inducing {prop_shift:.1f}px coordinate displacement."
                    ),
                    track_ids=[prop.track_id, person.track_id],
                    object_classes=[prop.object_class, "person"],
                    supporting_signals=signals,
                    contradictory_signals=contradictory_signals,
                    spatial_context=spatial_ctx,
                    evidence_candidates=[
                        EvidenceCandidate(
                            timestamp=contact_start,
                            bounding_box=prop.current_bbox,
                            target_track_id=prop.track_id,
                            reason="Physical interaction with property fixture",
                        )
                    ],
                    prefix="TAMPER",
                )
                cand.validation_decision = ValidationDecision.REVIEW_REQUIRED
                candidates.append(cand)

        return candidates
