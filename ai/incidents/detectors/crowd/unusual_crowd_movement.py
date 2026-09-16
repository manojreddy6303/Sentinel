"""
Potential Unusual Crowd Movement Detector (Phase 14)

Detects synchronized group transit that deviates substantially in direction,
velocity, or trajectory from the learned scene baseline pedestrian flow.
Observational only: does not infer panic or intent.
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
from ai.incidents.scene_context import SceneContextEngine


class UnusualCrowdMovementDetector(BaseIncidentDetector):
    """
    Evaluates group flow vectors against the baseline directional scene orientation.
    """

    detector_name: str = "unusual_crowd_movement_detector"
    detector_version: str = "1.0.0"
    category: IncidentCategory = IncidentCategory.CROWD

    required_signals: List[str] = ["Group Directional Coherence", "Baseline Flow Deviation"]
    supporting_signals_declared: List[str] = [
        "Group Directional Coherence",
        "Baseline Flow Deviation",
        "Synchronized Trajectory Deviation",
    ]
    contradictory_signals_declared: List[str] = ["Negative: Normal Commute Flow Direction"]
    context_requirements: Dict[str, Any] = {"person_tracks": True, "scene_context": True}
    evidence_requirements: Dict[str, Any] = {"heading_deviation": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        min_group_size: int = 4,
        min_heading_deviation_deg: float = 75.0,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.min_group_size = min_group_size
        self.min_heading_deviation_deg = min_heading_deviation_deg
        self.density_engine = CrowdDensityEngine()

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        person_tracks = [t for t in context.tracks if t.object_class == "person"]
        if len(person_tracks) < self.min_group_size:
            return candidates

        # Infer or retrieve scene baseline dominant heading
        scene_ctx = context.scene_context
        if not scene_ctx:
            scene_ctx = SceneContextEngine.infer_scene_context(context)

        baseline_heading = scene_ctx.baseline.dominant_heading_degrees if scene_ctx else None

        windows = self.density_engine.evaluate_windows(context)
        if not windows:
            return candidates

        for w in windows:
            if w.active_person_count < self.min_group_size:
                continue

            active_tracks = [t for t in person_tracks if t.track_id in w.unique_person_track_ids]
            headings = []

            for t in active_tracks:
                if t.trajectory and len(t.trajectory) >= 2:
                    dx = t.trajectory[-1][1] - t.trajectory[0][1]
                    dy = t.trajectory[-1][2] - t.trajectory[0][2]
                    headings.append(math.degrees(math.atan2(dy, dx)) % 360.0)

            if len(headings) < self.min_group_size:
                continue

            # Check group directional alignment
            cos_sum = sum(math.cos(math.radians(h)) for h in headings)
            sin_sum = sum(math.sin(math.radians(h)) for h in headings)
            coherence = math.hypot(cos_sum, sin_sum) / len(headings)

            if coherence < 0.70:
                continue

            mean_group_heading = math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0

            # Compare against baseline heading if available
            is_unusual_flow = False
            dev_deg = 0.0
            if baseline_heading is not None:
                diff = abs(mean_group_heading - baseline_heading) % 360.0
                if diff > 180.0:
                    diff = 360.0 - diff
                dev_deg = diff
                if diff >= self.min_heading_deviation_deg:
                    is_unusual_flow = True
            else:
                # If no baseline heading, check for rapid directional shift (>60 deg) across trajectory
                is_unusual_flow = False

            if not is_unusual_flow:
                continue

            signals = [
                SupportingSignal(
                    signal_type="Group Directional Coherence",
                    description=f"Group of {w.active_person_count} pedestrians transiting cohesively (coherence: {coherence:.2f})",
                    confidence=0.86,
                    timestamp=w.timestamp,
                ),
                SupportingSignal(
                    signal_type="Baseline Flow Deviation",
                    description=f"Group heading {mean_group_heading:.1f}° deviates {dev_deg:.1f}° from dominant scene flow ({baseline_heading:.1f}°)",
                    confidence=0.85,
                    timestamp=w.timestamp,
                ),
            ]

            scoring = IncidentScorer.calculate_evidence_score(
                base_confidence=0.70,
                supporting_signals=signals,
                tracks=active_tracks,
                duration_seconds=w.window_end - w.window_start,
                expected_duration_threshold=3.0,
            )

            primary_cluster = w.clusters[0] if w.clusters else None
            centroid = primary_cluster.centroid if primary_cluster else (960.0, 540.0)
            bbox = primary_cluster.bounding_box if primary_cluster else None

            spatial_ctx = SpatialContext(
                centroid=centroid,
                bounding_box=bbox,
                metadata={
                    "group_size": w.active_person_count,
                    "mean_heading_deg": round(mean_group_heading, 1),
                    "baseline_heading_deg": round(baseline_heading, 1) if baseline_heading is not None else None,
                    "heading_deviation_deg": round(dev_deg, 1),
                }
            )

            cand = self.build_candidate(
                video_id=context.video_id,
                event_type="POTENTIAL_UNUSUAL_CROWD_MOVEMENT",
                start_time=w.window_start,
                end_time=w.window_end,
                severity="NORMAL",
                confidence=min(0.70, scoring["score"]),
                explanation=(
                    f"Potential Unusual Crowd Movement ({scoring['verification_note']}): "
                    f"Synchronized group of {w.active_person_count} pedestrians transited at {mean_group_heading:.1f}°, "
                    f"deviating {dev_deg:.1f}° from normal baseline flow."
                ),
                track_ids=w.unique_person_track_ids,
                object_classes=["person"],
                supporting_signals=signals,
                spatial_context=spatial_ctx,
                evidence_candidates=[
                    EvidenceCandidate(
                        timestamp=w.timestamp,
                        bounding_box=bbox,
                        target_track_id=w.unique_person_track_ids[0] if w.unique_person_track_ids else None,
                        reason="Unusual crowd directional flow relative to baseline",
                    )
                ],
                prefix="CFLOW",
            )
            cand.validation_decision = ValidationDecision.REVIEW_REQUIRED
            candidates.append(cand)

        return candidates
