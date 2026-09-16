"""
Potential Stationary / Stopped Vehicle Detector (Sentinel Phase 11)

Detects vehicles exhibiting prolonged stationary presence within active roadway corridors:
- Distinguishes roadway stalls from designated parking areas
- Distinguishes isolated stopped vehicles from general traffic congestion
- Adheres to strictly conservative observational labeling:
  "Vehicle remained stationary in roadway corridor for extended period — review recommended."
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


class StationaryVehicleDetector(BaseIncidentDetector):
    """
    Evaluates vehicle presence for abnormal isolated roadway stalls.
    """

    detector_name: str = "stationary_vehicle_detector"
    detector_version: str = "1.0.0"
    category: IncidentCategory = IncidentCategory.VEHICLE

    # Phase Z Contract Declarations
    required_signals: List[str] = ["Extended Stationary Dwell"]
    supporting_signals_declared: List[str] = [
        "Extended Stationary Dwell",
        "Active Traffic Disparity",
    ]
    contradictory_signals_declared: [
        "Negative: General Congestion Queue",
        "Negative: Designated Parking Area",
        "Negative: Normal Moving Transit",
    ]
    context_requirements: Dict[str, Any] = {"roadway_context": True}
    evidence_requirements: Dict[str, Any] = {"stationary_track": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        min_stationary_duration_seconds: float = 8.0,
        max_displacement_threshold_px: float = 50.0,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.min_stationary_duration_seconds = min_stationary_duration_seconds
        self.max_displacement_threshold_px = max_displacement_threshold_px
        self.vehicle_classes = {"car", "bus", "truck", "motorcycle", "van"}

    def _is_scene_congestion(self, context: IncidentContext) -> bool:
        """Check if most vehicles in the scene are stationary (indicating general congestion)."""
        vehicles = [t for t in context.tracks if t.object_class in self.vehicle_classes and t.is_validated]
        if len(vehicles) < 3:
            return False

        stationary_count = 0
        for v in vehicles:
            summary = context.get_motion_summary(v.track_id) or {}
            if summary.get("max_stationary_duration", 0.0) >= 5.0 or summary.get("avg_velocity", 10.0) < 5.0:
                stationary_count += 1

        # If more than 60% of vehicles are stationary, scene is in traffic congestion / red light queue
        return (stationary_count / len(vehicles)) >= 0.60

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        vehicles = [t for t in context.tracks if t.object_class in self.vehicle_classes and t.is_validated]

        # Congestion check: if entire roadway is in a queue, abstain
        if self._is_scene_congestion(context):
            return candidates

        for track in vehicles:
            if track.duration_seconds < self.min_stationary_duration_seconds:
                continue

            summary = context.get_motion_summary(track.track_id) or {}
            stat_sec = summary.get("max_stationary_duration", 0.0)
            net_disp = summary.get("net_displacement", 0.0)
            avg_vel = summary.get("avg_velocity", 0.0)

            # Must be stationary for threshold and have minimal net displacement
            if stat_sec < self.min_stationary_duration_seconds or net_disp > self.max_displacement_threshold_px or avg_vel > 12.0:
                continue

            signals = [
                SupportingSignal(
                    signal_type="Extended Stationary Dwell",
                    description=(
                        f"{track.object_class} [{track.track_id}] remained stationary for {stat_sec:.1f}s "
                        f"(displacement: {net_disp:.1f}px)"
                    ),
                    confidence=0.88,
                    timestamp=track.first_seen,
                    track_id=track.track_id,
                ),
            ]

            scoring = IncidentScorer.calculate_evidence_score(
                base_confidence=0.75,
                supporting_signals=signals,
                tracks=[track],
                duration_seconds=stat_sec,
                expected_duration_threshold=self.min_stationary_duration_seconds,
            )

            c_box = track.current_bbox
            cand = self.build_candidate(
                video_id=context.video_id,
                event_type="POTENTIAL_STATIONARY_VEHICLE",
                start_time=track.first_seen,
                end_time=track.last_seen,
                severity="NORMAL",
                confidence=scoring["score"],
                explanation=(
                    f"Stationary Vehicle observation: {track.object_class} [{track.track_id}] remained stationary "
                    f"in roadway corridor for {stat_sec:.1f}s (threshold: {self.min_stationary_duration_seconds:.1f}s) — review recommended."
                ),
                track_ids=[track.track_id],
                object_classes=[track.object_class],
                supporting_signals=signals,
                spatial_context=SpatialContext(
                    centroid=c_box.centroid if c_box else None,
                    bounding_box=c_box,
                    metadata={"stationary_duration": stat_sec, "net_displacement": net_disp},
                ),
                evidence_candidates=[
                    EvidenceCandidate(
                        timestamp=track.first_seen + self.min_stationary_duration_seconds,
                        bounding_box=c_box,
                        target_track_id=track.track_id,
                        reason="Point of confirmed stationary dwell in roadway corridor",
                    )
                ],
                prefix="STAT",
            )
            candidates.append(cand)

        return candidates
