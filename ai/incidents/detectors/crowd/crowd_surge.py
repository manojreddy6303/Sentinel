"""
Potential Crowd Surge Detector (Phase 14)

Conservative multi-signal candidate for sudden directional crowd surges.
Requires simultaneous high pedestrian density, abnormal group velocity, directional coherence,
and spatial compression. Integrates queue detection as counter-evidence to suppress orderly lines.
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
from ai.incidents.detectors.crowd.density_engine import CrowdDensityEngine
from ai.incidents.detectors.crowd.queue_flow_engine import QueueAndFlowEngine


class CrowdSurgeDetector(BaseIncidentDetector):
    """
    Evaluates multi-frame pedestrian telemetry for potential directional crowd surges.
    """

    detector_name: str = "crowd_surge_detector"
    detector_version: str = "1.0.0"
    category: IncidentCategory = IncidentCategory.CROWD

    required_signals: List[str] = [
        "Elevated Crowd Density",
        "Directional Heading Alignment",
        "Abnormal Group Velocity",
    ]
    supporting_signals_declared: List[str] = [
        "Elevated Crowd Density",
        "Directional Heading Alignment",
        "Abnormal Group Velocity",
        "Spatial Cluster Compression",
    ]
    contradictory_signals_declared: List[str] = [
        "Negative: Orderly Queue Flow",
        "Negative: Dispersed Non-Cohesive",
        "Negative: Normal Walking Pace",
    ]
    context_requirements: Dict[str, Any] = {"person_tracks": True}
    evidence_requirements: Dict[str, Any] = {"group_velocity": True, "heading_coherence": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        min_surge_persons: int = 4,
        min_surge_velocity_px_s: float = 24.0,
        max_inter_distance_px: float = 85.0,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.min_surge_persons = min_surge_persons
        self.min_surge_velocity_px_s = min_surge_velocity_px_s
        self.max_inter_distance_px = max_inter_distance_px
        self.density_engine = CrowdDensityEngine()
        self.queue_engine = QueueAndFlowEngine()

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        person_tracks = [t for t in context.tracks if t.object_class == "person"]
        if len(person_tracks) < self.min_surge_persons:
            return candidates

        windows = self.density_engine.evaluate_windows(context)
        if not windows:
            return candidates

        for w in windows:
            if w.active_person_count < self.min_surge_persons:
                continue

            # Check queue counter-evidence first
            queue_assessment = self.queue_engine.evaluate_queue_pattern(person_tracks, w.timestamp)
            contradictory_signals = []
            if queue_assessment.is_queue_detected:
                # Orderly queue explains the linear group flow
                contradictory_signals.append(
                    SupportingSignal(
                        signal_type="Negative: Orderly Queue Flow",
                        description=f"Pedestrians aligned in orderly single/double file ({queue_assessment.queue_member_count} persons)",
                        confidence=0.88,
                        timestamp=w.timestamp,
                    )
                )
                continue

            # Evaluate group velocities and headings across active tracks in this window
            active_tracks = [t for t in person_tracks if t.track_id in w.unique_person_track_ids]
            velocities = []
            headings = []

            for t in active_tracks:
                summary = context.get_motion_summary(t.track_id) or {}
                v = summary.get("avg_velocity", 0.0)
                if v <= 0.0 and t.trajectory and len(t.trajectory) >= 2:
                    dt = t.trajectory[-1][0] - t.trajectory[0][0]
                    if dt > 0:
                        dist = math.hypot(t.trajectory[-1][1] - t.trajectory[0][1], t.trajectory[-1][2] - t.trajectory[0][2])
                        v = dist / dt
                velocities.append(v)
                if t.trajectory and len(t.trajectory) >= 2:
                    dx = t.trajectory[-1][1] - t.trajectory[0][1]
                    dy = t.trajectory[-1][2] - t.trajectory[0][2]
                    headings.append(math.degrees(math.atan2(dy, dx)) % 360.0)

            if not velocities or len(headings) < self.min_surge_persons:
                continue

            mean_vel = sum(velocities) / len(velocities)
            if mean_vel < self.min_surge_velocity_px_s:
                # Normal slow walking pace
                continue

            # Heading coherence (circular variance)
            cos_sum = sum(math.cos(math.radians(h)) for h in headings)
            sin_sum = sum(math.sin(math.radians(h)) for h in headings)
            coherence = math.hypot(cos_sum, sin_sum) / len(headings)

            if coherence < 0.70:
                # Dispersed, divergent movement directions
                continue

            # Check spatial compression
            if w.avg_inter_person_dist > self.max_inter_distance_px:
                # Too spread out for a surge
                continue

            signals = [
                SupportingSignal(
                    signal_type="Elevated Crowd Density",
                    description=f"Group of {w.active_person_count} pedestrians in close proximity ({w.avg_inter_person_dist:.1f}px inter-distance)",
                    confidence=0.86,
                    timestamp=w.timestamp,
                ),
                SupportingSignal(
                    signal_type="Directional Heading Alignment",
                    description=f"High directional coherence: {coherence:.2f} across active individuals",
                    confidence=0.85,
                    timestamp=w.timestamp,
                ),
                SupportingSignal(
                    signal_type="Abnormal Group Velocity",
                    description=f"Mean group velocity: {mean_vel:.1f}px/s exceeds normal transit threshold",
                    confidence=0.84,
                    timestamp=w.timestamp,
                ),
            ]

            scoring = IncidentScorer.calculate_evidence_score(
                base_confidence=0.74,
                supporting_signals=signals,
                tracks=active_tracks,
                duration_seconds=w.window_end - w.window_start,
                expected_duration_threshold=2.0,
                contradictory_signals=contradictory_signals,
            )

            primary_cluster = w.clusters[0] if w.clusters else None
            centroid = primary_cluster.centroid if primary_cluster else (960.0, 540.0)
            bbox = primary_cluster.bounding_box if primary_cluster else None

            spatial_ctx = SpatialContext(
                centroid=centroid,
                bounding_box=bbox,
                metadata={
                    "active_person_count": w.active_person_count,
                    "mean_velocity_px_s": round(mean_vel, 1),
                    "heading_coherence": round(coherence, 2),
                    "average_inter_distance": round(w.avg_inter_person_dist, 1),
                }
            )

            cand = self.build_candidate(
                video_id=context.video_id,
                event_type="POTENTIAL_CROWD_SURGE",
                start_time=w.window_start,
                end_time=w.window_end,
                severity="NORMAL",
                confidence=min(0.72, scoring["score"]),
                explanation=(
                    f"Potential Crowd Surge ({scoring['verification_note']}): "
                    f"Cohesive group of {w.active_person_count} individuals exhibited rapid, "
                    f"unidirectional movement (vel: {mean_vel:.1f}px/s, coherence: {coherence:.2f}) at {w.timestamp:.1f}s."
                ),
                track_ids=w.unique_person_track_ids,
                object_classes=["person"],
                supporting_signals=signals,
                contradictory_signals=contradictory_signals,
                spatial_context=spatial_ctx,
                evidence_candidates=[
                    EvidenceCandidate(
                        timestamp=w.timestamp,
                        bounding_box=bbox,
                        target_track_id=w.unique_person_track_ids[0] if w.unique_person_track_ids else None,
                        reason="Coherent group velocity and density surge",
                    )
                ],
                prefix="SURGE",
            )
            cand.validation_decision = ValidationDecision.REVIEW_REQUIRED
            candidates.append(cand)

        return candidates
