"""
Potential Sudden Vehicle Stop Detector (Sentinel Phase 11)

Detects abrupt, unexpected vehicle stopping events:
- High initial transit speed
- Significant, rapid velocity drop (hard braking)
- Subsequent stationary or near-stationary state
- Contextual distinction from synchronized queue stopping and traffic lights
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


class SuddenStopDetector(BaseIncidentDetector):
    """
    Evaluates vehicle kinematic profiles for isolated abrupt deceleration and stopping anomalies.
    """

    detector_name: str = "sudden_stop_detector"
    detector_version: str = "1.0.0"
    category: IncidentCategory = IncidentCategory.VEHICLE

    # Phase Z Contract Declarations
    required_signals: List[str] = ["High Pre-Stop Velocity", "Abrupt Velocity Reduction"]
    supporting_signals_declared: List[str] = [
        "High Pre-Stop Velocity",
        "Abrupt Velocity Reduction",
        "Subsequent Stationary Dwell",
    ]
    contradictory_signals_declared: [
        "Negative: Synchronized Traffic Slowdown",
        "Negative: Normal Gradual Deceleration",
    ]
    context_requirements: Dict[str, Any] = {"roadway_context": True}
    evidence_requirements: Dict[str, Any] = {"verified_track": True, "kinematic_motion": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        min_pre_stop_velocity: float = 25.0,
        deceleration_drop_threshold: float = 20.0,
        min_stop_duration_seconds: float = 2.0,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.min_pre_stop_velocity = min_pre_stop_velocity
        self.deceleration_drop_threshold = deceleration_drop_threshold
        self.min_stop_duration_seconds = min_stop_duration_seconds
        self.vehicle_classes = {"car", "bus", "truck", "motorcycle", "van"}

    def _is_synchronized_traffic_slowdown(
        self,
        target_track_id: str,
        stop_time: float,
        context: IncidentContext,
    ) -> bool:
        """
        Check if other vehicles in the scene simultaneously decelerated,
        indicating a traffic light, red signal, or queue rather than an isolated sudden stop.
        """
        other_vehicles = [
            t for t in context.tracks
            if t.object_class in self.vehicle_classes and t.track_id != target_track_id
        ]
        if not other_vehicles:
            return False

        simultaneous_deceleration_count = 0
        for ov in other_vehicles:
            ov_motions = context.track_motions.get(ov.track_id, [])
            near_m = [m for m in ov_motions if abs(m.timestamp - stop_time) <= 2.0]
            if len(near_m) >= 2:
                # Check if this vehicle also slowed down
                v_pre = near_m[0].velocity_estimate
                v_post = near_m[-1].velocity_estimate
                if (v_pre - v_post) >= 15.0 or near_m[-1].is_stationary:
                    simultaneous_deceleration_count += 1

        # If 2 or more other vehicles also decelerated simultaneously, it's synchronized traffic flow
        return simultaneous_deceleration_count >= 2

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        vehicles = [t for t in context.tracks if t.object_class in self.vehicle_classes and t.is_validated]

        for track in vehicles:
            motions = context.track_motions.get(track.track_id, [])
            if len(motions) < 3:
                continue

            # Look for abrupt drop from high speed to stationary
            for i in range(len(motions) - 1):
                m_prev = motions[i]
                m_curr = motions[i + 1]

                # Check if vehicle was moving at good speed and dropped sharply
                vel_drop = m_prev.velocity_estimate - m_curr.velocity_estimate
                if m_prev.velocity_estimate >= self.min_pre_stop_velocity and vel_drop >= self.deceleration_drop_threshold:
                    # Check subsequent stationary state
                    subsequent_motions = motions[i + 1:]
                    has_stationary = any(
                        m.is_stationary and m.stationary_duration >= self.min_stop_duration_seconds
                        for m in subsequent_motions
                    )
                    if not has_stationary:
                        continue

                    stop_t = m_curr.timestamp

                    # Negative evidence check: synchronized slowdown
                    is_synchronized = self._is_synchronized_traffic_slowdown(track.track_id, stop_t, context)
                    if is_synchronized:
                        # Normal traffic light or queue slowdown. ABSTAIN.
                        continue

                    signals = [
                        SupportingSignal(
                            signal_type="High Pre-Stop Velocity",
                            description=f"{track.object_class} [{track.track_id}] was in active transit at {m_prev.velocity_estimate:.1f}px/s",
                            confidence=0.88,
                            timestamp=m_prev.timestamp,
                            track_id=track.track_id,
                        ),
                        SupportingSignal(
                            signal_type="Abrupt Velocity Reduction",
                            description=f"Rapid deceleration of {vel_drop:.1f}px/s observed at {stop_t:.1f}s",
                            confidence=0.90,
                            timestamp=stop_t,
                            track_id=track.track_id,
                        ),
                        SupportingSignal(
                            signal_type="Subsequent Stationary Dwell",
                            description=f"Vehicle remained stationary/stopped post-deceleration",
                            confidence=0.85,
                            timestamp=stop_t + 1.0,
                            track_id=track.track_id,
                        ),
                    ]

                    scoring = IncidentScorer.calculate_evidence_score(
                        base_confidence=0.75,
                        supporting_signals=signals,
                        tracks=[track],
                    )

                    c_bbox = track.current_bbox
                    cand = self.build_candidate(
                        video_id=context.video_id,
                        event_type="POTENTIAL_SUDDEN_VEHICLE_STOP",
                        start_time=max(0.0, m_prev.timestamp - 0.5),
                        end_time=stop_t + self.min_stop_duration_seconds,
                        severity="NORMAL",
                        confidence=scoring["score"],
                        explanation=(
                            f"Potential Sudden Stop: {track.object_class} [{track.track_id}] decelerated rapidly "
                            f"from {m_prev.velocity_estimate:.1f}px/s (drop: {vel_drop:.1f}px/s) to a complete stop at {stop_t:.1f}s."
                        ),
                        track_ids=[track.track_id],
                        object_classes=[track.object_class],
                        supporting_signals=signals,
                        spatial_context=SpatialContext(
                            centroid=c_bbox.centroid if c_bbox else None,
                            bounding_box=c_bbox,
                        ),
                        evidence_candidates=[
                            EvidenceCandidate(
                                timestamp=stop_t,
                                bounding_box=c_bbox,
                                target_track_id=track.track_id,
                                reason="Point of abrupt vehicle deceleration and stop",
                            )
                        ],
                        prefix="STOP",
                    )
                    candidates.append(cand)
                    break  # Avoid duplicate stop events on the same track

        return candidates
