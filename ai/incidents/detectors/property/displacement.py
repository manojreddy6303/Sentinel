"""
Potential Object Displacement Detector (Phase 13)

Detects meaningful spatial translation of an object from an initial stable location
to a distinct subsequent stable position. Uses object-scale normalization and camera
motion protection to reject detector jitter and perspective artifacts.
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
from ai.incidents.detectors.property.camera_stability import CameraStabilityEngine


class ObjectDisplacementDetector(BaseIncidentDetector):
    """
    Evaluates portable and property objects for significant physical relocations.
    """

    detector_name: str = "object_displacement_detector"
    detector_version: str = "1.0.0"
    category: IncidentCategory = IncidentCategory.PROPERTY

    required_signals: List[str] = ["Scale-Normalized Displacement", "Multi-Phase Coordinate Stability"]
    supporting_signals_declared: List[str] = ["Scale-Normalized Displacement", "Multi-Phase Coordinate Stability"]
    contradictory_signals_declared: List[str] = ["Negative: Displacement Below Noise Threshold", "Negative: Camera Jitter"]
    context_requirements: Dict[str, Any] = {"portable_objects": True}
    evidence_requirements: Dict[str, Any] = {"object_track": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        min_displacement_px: float = 40.0,
        min_scale_ratio: float = 0.8,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.min_displacement_px = min_displacement_px
        self.min_scale_ratio = min_scale_ratio
        self.target_classes = {
            "backpack", "handbag", "suitcase", "laptop", "cell phone",
            "bottle", "umbrella", "box", "package", "merchandise", "book", "general_object", "unknown_portable_object", "bicycle", "chair",
        }
        self.stability_engine = CameraStabilityEngine()

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        stability = self.stability_engine.assess_stability(context)
        if not stability.is_camera_stable:
            return candidates

        for track in context.tracks:
            if track.object_class not in self.target_classes or not track.trajectory or len(track.trajectory) < 4:
                continue

            # Need initial stable cluster and terminal stable cluster
            first_pts = track.trajectory[:len(track.trajectory) // 3]
            last_pts = track.trajectory[-(len(track.trajectory) // 3):]

            init_x = sum(p[1] for p in first_pts) / len(first_pts)
            init_y = sum(p[2] for p in first_pts) / len(first_pts)
            end_x = sum(p[1] for p in last_pts) / len(last_pts)
            end_y = sum(p[2] for p in last_pts) / len(last_pts)

            disp = math.hypot(end_x - init_x, end_y - init_y)
            if disp < self.min_displacement_px:
                continue

            # Check object scale normalization
            bbox = track.current_bbox
            scale = 50.0
            if bbox:
                w = abs(bbox.x2 - bbox.x1)
                h = abs(bbox.y2 - bbox.y1)
                scale = max(20.0, math.hypot(w, h))

            scale_ratio = disp / scale
            if scale_ratio < self.min_scale_ratio:
                # Displacement is within detector jitter / aspect change range
                continue

            signals = [
                SupportingSignal(
                    signal_type="Scale-Normalized Displacement",
                    description=(
                        f"{track.object_class} [{track.track_id}] translated {disp:.1f}px "
                        f"({scale_ratio:.2f}x body scale) from initial coordinates"
                    ),
                    confidence=0.86,
                    timestamp=track.last_seen,
                    track_id=track.track_id,
                ),
                SupportingSignal(
                    signal_type="Multi-Phase Coordinate Stability",
                    description=(
                        f"Initial stable centroid: ({init_x:.1f}, {init_y:.1f}) -> "
                        f"Final stable centroid: ({end_x:.1f}, {end_y:.1f})"
                    ),
                    confidence=0.84,
                    timestamp=track.first_seen,
                    track_id=track.track_id,
                ),
            ]

            scoring = IncidentScorer.calculate_evidence_score(
                base_confidence=0.70,
                supporting_signals=signals,
                tracks=[track],
                duration_seconds=track.duration_seconds,
                expected_duration_threshold=3.0,
            )

            spatial_ctx = SpatialContext(
                centroid=(end_x, end_y),
                bounding_box=bbox,
                metadata={
                    "initial_centroid": [round(init_x, 1), round(init_y, 1)],
                    "final_centroid": [round(end_x, 1), round(end_y, 1)],
                    "net_displacement": round(disp, 2),
                    "scale_ratio": round(scale_ratio, 2),
                }
            )

            cand = self.build_candidate(
                video_id=context.video_id,
                event_type="POTENTIAL_OBJECT_DISPLACEMENT",
                start_time=track.first_seen,
                end_time=track.last_seen,
                severity="NORMAL",
                confidence=min(0.68, scoring["score"]),
                explanation=(
                    f"Potential Object Displacement ({scoring['verification_note']}): "
                    f"{track.object_class} [{track.track_id}] exhibited physical displacement "
                    f"of {disp:.1f}px ({scale_ratio:.1f}x scale) between stable positions."
                ),
                track_ids=[track.track_id],
                object_classes=[track.object_class],
                supporting_signals=signals,
                spatial_context=spatial_ctx,
                evidence_candidates=[
                    EvidenceCandidate(
                        timestamp=track.last_seen,
                        bounding_box=bbox,
                        target_track_id=track.track_id,
                        reason="Object displacement to new stable coordinates",
                    )
                ],
                prefix="OBJDISP",
            )
            cand.validation_decision = ValidationDecision.REVIEW_REQUIRED
            candidates.append(cand)

        return candidates
