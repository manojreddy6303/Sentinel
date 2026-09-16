"""
Potential Restricted Object Movement Detector (Phase 13)

Integrates physical object movement tracking with configured SecurityZone boundaries.
Identifies portable property items transiting into or moving within restricted perimeter zones.
"""
from typing import List, Dict, Any, Optional

from ai.schemas import BoundingBox, TrackedObject, ZoneDefinition
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
from ai.incidents.spatial import SpatialRelationshipEngine
from ai.incidents.scoring import IncidentScorer


class RestrictedObjectMovementDetector(BaseIncidentDetector):
    """
    Evaluates object tracks against configured restricted security zones.
    """

    detector_name: str = "restricted_object_movement_detector"
    detector_version: str = "1.0.0"
    category: IncidentCategory = IncidentCategory.PROPERTY

    required_signals: List[str] = ["Restricted Zone Transit", "Object Motion Grounding"]
    supporting_signals_declared: List[str] = ["Restricted Zone Transit", "Object Motion Grounding", "Zone Dwell Duration"]
    contradictory_signals_declared: List[str] = ["Negative: Exterior Trajectory"]
    context_requirements: Dict[str, Any] = {"zones": True}
    evidence_requirements: Dict[str, Any] = {"object_track": True, "zone": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        min_zone_dwell_seconds: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.min_zone_dwell_seconds = min_zone_dwell_seconds
        self.portable_classes = {
            "backpack", "handbag", "suitcase", "laptop", "cell phone",
            "box", "package", "bicycle",
        }

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        if not context.zones:
            return candidates

        for zone in context.zones:
            if not zone.enabled or not zone.polygon:
                continue

            for track in context.tracks:
                if track.object_class not in self.portable_classes or not track.trajectory:
                    continue

                # Evaluate trajectory points inside zone polygon
                inside_points = []
                for pt in track.trajectory:
                    t, x, y = pt[0], pt[1], pt[2]
                    if SpatialRelationshipEngine.point_in_polygon((x, y), zone.polygon):
                        inside_points.append((t, x, y))

                if not inside_points:
                    continue

                dwell_start = inside_points[0][0]
                dwell_end = inside_points[-1][0]
                dwell_duration = max(0.5, dwell_end - dwell_start)

                if dwell_duration < self.min_zone_dwell_seconds and len(inside_points) < 2:
                    continue

                signals = [
                    SupportingSignal(
                        signal_type="Restricted Zone Transit",
                        description=f"{track.object_class} [{track.track_id}] entered restricted zone '{zone.zone_name}'",
                        confidence=0.88,
                        timestamp=dwell_start,
                        track_id=track.track_id,
                    ),
                    SupportingSignal(
                        signal_type="Zone Dwell Duration",
                        description=f"Dwelled inside zone for {dwell_duration:.1f}s ({len(inside_points)} detections)",
                        confidence=0.86,
                        timestamp=dwell_end,
                        track_id=track.track_id,
                    ),
                ]

                scoring = IncidentScorer.calculate_evidence_score(
                    base_confidence=0.74,
                    supporting_signals=signals,
                    tracks=[track],
                    duration_seconds=dwell_duration,
                    expected_duration_threshold=self.min_zone_dwell_seconds,
                )

                spatial_ctx = SpatialContext(
                    centroid=(inside_points[0][1], inside_points[0][2]),
                    bounding_box=track.current_bbox,
                    metadata={
                        "zone_id": zone.zone_id,
                        "zone_name": zone.zone_name,
                        "dwell_duration": round(dwell_duration, 2),
                    }
                )

                cand = self.build_candidate(
                    video_id=context.video_id,
                    event_type="POTENTIAL_RESTRICTED_OBJECT_MOVEMENT",
                    start_time=dwell_start,
                    end_time=dwell_end,
                    severity="NORMAL",
                    confidence=min(0.72, scoring["score"]),
                    explanation=(
                        f"Potential Restricted Object Movement ({scoring['verification_note']}): "
                        f"{track.object_class} [{track.track_id}] was tracked within restricted zone '{zone.zone_name}' "
                        f"for {dwell_duration:.1f}s."
                    ),
                    track_ids=[track.track_id],
                    object_classes=[track.object_class],
                    supporting_signals=signals,
                    spatial_context=spatial_ctx,
                    evidence_candidates=[
                        EvidenceCandidate(
                            timestamp=dwell_start,
                            bounding_box=track.current_bbox,
                            target_track_id=track.track_id,
                            reason=f"Object transit in zone '{zone.zone_name}'",
                        )
                    ],
                    prefix="OBJZN",
                )
                cand.validation_decision = ValidationDecision.REVIEW_REQUIRED
                candidates.append(cand)

        return candidates
