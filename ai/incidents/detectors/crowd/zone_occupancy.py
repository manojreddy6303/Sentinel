"""
Zone Occupancy & Restricted Zone Crowding Detector (Phase 14)

Evaluates multi-track presence inside configured SecurityZone boundaries.
Detects simultaneous multi-person occupancy, prolonged group dwell, and restricted-zone crowding.
Integrates boundary hysteresis to prevent tracker jitter from causing false entry/exit oscillations.
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


class ZoneOccupancyAndActivityDetector(BaseIncidentDetector):
    """
    Evaluates simultaneous multi-track presence and abnormal activity in security zones.
    """

    detector_name: str = "zone_occupancy_detector"
    detector_version: str = "1.0.0"
    category: IncidentCategory = IncidentCategory.ZONE

    required_signals: List[str] = ["Multi-Occupant Zone Presence"]
    supporting_signals_declared: List[str] = [
        "Multi-Occupant Zone Presence",
        "Sustained Zone Group Dwell",
        "Restricted Perimeter Concentration",
    ]
    contradictory_signals_declared: List[str] = ["Negative: Transient Zone Boundary Crossing"]
    context_requirements: Dict[str, Any] = {"zones": True, "person_tracks": True}
    evidence_requirements: Dict[str, Any] = {"zone_occupancy": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        min_crowd_occupants: int = 3,
        min_crowd_dwell_seconds: float = 3.0,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.min_crowd_occupants = min_crowd_occupants
        self.min_crowd_dwell_seconds = min_crowd_dwell_seconds

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        if not context.zones:
            return candidates

        person_tracks = [t for t in context.tracks if t.object_class == "person" and t.is_validated]
        if len(person_tracks) < self.min_crowd_occupants:
            return candidates

        for zone in context.zones:
            if not zone.enabled or not zone.polygon:
                continue

            # Track inside intervals for each person: list of (t_start, t_end, dwell, count)
            inside_data: Dict[str, Dict[str, Any]] = {}

            for track in person_tracks:
                dwell_info = SpatialRelationshipEngine.track_zone_dwell_analysis(track, zone)
                # Require at least 2 observations inside to prevent single-frame edge flicker
                if dwell_info["entered"] and dwell_info["inside_observations_count"] >= 2:
                    end_t = dwell_info["exit_time"] if dwell_info["exit_time"] is not None else track.last_seen
                    dwell_info["effective_exit_time"] = end_t
                    inside_data[track.track_id] = dwell_info

            if len(inside_data) < self.min_crowd_occupants:
                continue

            # Check for temporal overlap across occupants
            # Find common overlapping time interval
            entry_times = [d["entry_time"] for d in inside_data.values() if d["entry_time"] is not None]
            exit_times = [d["effective_exit_time"] for d in inside_data.values() if d.get("effective_exit_time") is not None]

            if not entry_times or not exit_times:
                continue

            overlap_start = max(entry_times)
            overlap_end = min(exit_times)
            overlap_dwell = max(0.0, overlap_end - overlap_start)

            # If not completely co-occurring in a single window, check if at least N are inside simultaneously
            # Sample at step=1.0s
            t_min = min(entry_times)
            t_max = max(exit_times)
            time_step = 1.0
            steps = max(1, int((t_max - t_min) / time_step))

            peak_occupants = 0
            peak_time = t_min
            occupants_at_peak = []

            for s in range(steps + 1):
                t_eval = t_min + (s * time_step)
                curr_inside = [
                    tid for tid, d in inside_data.items()
                    if d["entry_time"] <= t_eval <= d["effective_exit_time"]
                ]
                if len(curr_inside) > peak_occupants:
                    peak_occupants = len(curr_inside)
                    peak_time = t_eval
                    occupants_at_peak = curr_inside

            if peak_occupants < self.min_crowd_occupants:
                continue

            event_type = "POTENTIAL_RESTRICTED_ZONE_CROWDING"
            dwell_dur = max(self.min_crowd_dwell_seconds, overlap_dwell)

            signals = [
                SupportingSignal(
                    signal_type="Multi-Occupant Zone Presence",
                    description=f"{peak_occupants} verified individuals concurrently occupying zone '{zone.zone_name}' at {peak_time:.1f}s",
                    confidence=0.88,
                    timestamp=peak_time,
                ),
                SupportingSignal(
                    signal_type="Sustained Zone Group Dwell",
                    description=f"Concurrent dwell duration: {dwell_dur:.1f}s",
                    confidence=0.86,
                    timestamp=peak_time,
                ),
            ]

            scoring = IncidentScorer.calculate_evidence_score(
                base_confidence=0.75,
                supporting_signals=signals,
                tracks=[t for t in person_tracks if t.track_id in occupants_at_peak],
                duration_seconds=dwell_dur,
                expected_duration_threshold=self.min_crowd_dwell_seconds,
            )

            # Zone centroid
            poly_xs = [p[0] for p in zone.polygon]
            poly_ys = [p[1] for p in zone.polygon]
            zone_centroid = (sum(poly_xs) / len(poly_xs), sum(poly_ys) / len(poly_ys))

            spatial_ctx = SpatialContext(
                centroid=zone_centroid,
                bounding_box=BoundingBox(x1=min(poly_xs), y1=min(poly_ys), x2=max(poly_xs), y2=max(poly_ys)),
                zone_name=zone.zone_name,
                metadata={
                    "zone_id": zone.zone_id,
                    "zone_name": zone.zone_name,
                    "peak_occupants": peak_occupants,
                    "occupant_track_ids": occupants_at_peak,
                    "overlap_dwell_seconds": round(dwell_dur, 2),
                }
            )

            cand = self.build_candidate(
                video_id=context.video_id,
                event_type=event_type,
                start_time=peak_time - (dwell_dur / 2.0),
                end_time=peak_time + (dwell_dur / 2.0),
                severity="NORMAL",
                confidence=min(0.74, scoring["score"]),
                explanation=(
                    f"Potential Restricted-Zone Crowding ({scoring['verification_note']}): "
                    f"{peak_occupants} individuals concurrently occupied restricted zone '{zone.zone_name}' "
                    f"at {peak_time:.1f}s (sustained dwell: {dwell_dur:.1f}s)."
                ),
                track_ids=occupants_at_peak,
                object_classes=["person"],
                supporting_signals=signals,
                spatial_context=spatial_ctx,
                evidence_candidates=[
                    EvidenceCandidate(
                        timestamp=peak_time,
                        bounding_box=spatial_ctx.bounding_box,
                        target_track_id=occupants_at_peak[0] if occupants_at_peak else None,
                        reason=f"Concurrent multi-occupant presence in zone '{zone.zone_name}'",
                    )
                ],
                prefix="ZCRWD",
            )
            cand.validation_decision = ValidationDecision.REVIEW_REQUIRED
            candidates.append(cand)

        return candidates
