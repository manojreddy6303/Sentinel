"""
Potential Object Removal Detector (Phase 13)

Detects the unpredicted disappearance of a previously stable object from the monitored scene.
Distinguishes genuine removal candidates from track loss, boundary exit, detector jitter,
and temporary occlusion.
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


class ObjectRemovalDetector(BaseIncidentDetector):
    """
    Evaluates persistent object tracks that cease detection inside the camera FOV.
    """

    detector_name: str = "object_removal_detector"
    detector_version: str = "1.0.0"
    category: IncidentCategory = IncidentCategory.PROPERTY

    required_signals: List[str] = ["Persistent Prior Presence", "Mid-Scene Track Cessation"]
    supporting_signals_declared: List[str] = ["Persistent Prior Presence", "Mid-Scene Track Cessation", "Interior FOV Grounding"]
    contradictory_signals_declared: List[str] = [
        "Negative: Frame Boundary Exit",
        "Negative: Temporary Occlusion",
        "Negative: Camera Jitter / Motion",
    ]
    context_requirements: Dict[str, Any] = {"portable_objects": True}
    evidence_requirements: Dict[str, Any] = {"object_track": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        min_stable_seconds: float = 3.0,
        frame_edge_margin_px: float = 30.0,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.min_stable_seconds = min_stable_seconds
        self.frame_edge_margin_px = frame_edge_margin_px
        self.target_classes = {
            "backpack", "handbag", "suitcase", "laptop", "cell phone",
            "bottle", "umbrella", "box", "package", "merchandise", "book", "general_object", "bicycle",
        }
        self.stability_engine = CameraStabilityEngine(frame_margin_px=frame_edge_margin_px)

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        stability = self.stability_engine.assess_stability(context)
        if not stability.is_camera_stable:
            return candidates

        video_duration = context.duration_seconds if context.duration_seconds else 0.0

        for track in context.tracks:
            if track.object_class not in self.target_classes or not track.trajectory:
                continue

            # 1. Must have had persistent stable presence
            if track.duration_seconds < self.min_stable_seconds:
                continue

            # 2. Must have ceased detection BEFORE the end of video (mid-scene disappearance)
            if video_duration > 0 and track.last_seen >= video_duration - 1.5:
                # Video ended while object was still present -> not removed
                continue

            # 3. Last observed bbox must NOT be at camera boundary
            last_bbox = track.current_bbox
            if not last_bbox:
                continue

            frame_w = context.video_metadata.get("width") if context.video_metadata else None
            frame_h = context.video_metadata.get("height") if context.video_metadata else None
            is_edge = last_bbox.is_edge_clipped(frame_width=frame_w, frame_height=frame_h, margin=self.frame_edge_margin_px)
            if is_edge:
                # Exited frame boundary, not mid-scene removal
                continue

            # 4. Check for obvious occlusion by an overlapping track at last seen
            is_occluded = False
            for other in context.tracks:
                if other.track_id == track.track_id or not other.current_bbox:
                    continue
                if abs(other.last_seen - track.last_seen) <= 1.0 or (other.first_seen <= track.last_seen <= other.last_seen):
                    ob = other.current_bbox
                    # Overlap check
                    ix1 = max(last_bbox.x1, ob.x1)
                    iy1 = max(last_bbox.y1, ob.y1)
                    ix2 = min(last_bbox.x2, ob.x2)
                    iy2 = min(last_bbox.y2, ob.y2)
                    if ix2 > ix1 and iy2 > iy1:
                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                        box_area = (last_bbox.x2 - last_bbox.x1) * (last_bbox.y2 - last_bbox.y1)
                        if box_area > 0 and (inter_area / box_area) > 0.40:
                            is_occluded = True
                            break

            contradictory_signals = []
            if is_occluded:
                contradictory_signals.append(
                    SupportingSignal(
                        signal_type="Negative: Temporary Occlusion",
                        description=f"{track.object_class} [{track.track_id}] overlapped by concurrent track at cessation",
                        confidence=0.88,
                        timestamp=track.last_seen,
                    )
                )
                continue

            signals = [
                SupportingSignal(
                    signal_type="Persistent Prior Presence",
                    description=f"{track.object_class} [{track.track_id}] stably observed for {track.duration_seconds:.1f}s",
                    confidence=0.88,
                    timestamp=track.first_seen,
                    track_id=track.track_id,
                ),
                SupportingSignal(
                    signal_type="Mid-Scene Track Cessation",
                    description=f"Detection ceased at {track.last_seen:.1f}s without frame boundary transit",
                    confidence=0.86,
                    timestamp=track.last_seen,
                    track_id=track.track_id,
                ),
                SupportingSignal(
                    signal_type="Interior FOV Grounding",
                    description=f"Last coordinates: ({last_bbox.centroid[0]:.1f}, {last_bbox.centroid[1]:.1f}) well inside FOV margins",
                    confidence=0.85,
                    timestamp=track.last_seen,
                    track_id=track.track_id,
                ),
            ]

            scoring = IncidentScorer.calculate_evidence_score(
                base_confidence=0.70,
                supporting_signals=signals,
                tracks=[track],
                duration_seconds=track.duration_seconds,
                expected_duration_threshold=self.min_stable_seconds,
            )

            spatial_ctx = SpatialContext(
                centroid=last_bbox.centroid,
                bounding_box=last_bbox,
                metadata={
                    "last_seen_timestamp": track.last_seen,
                    "last_bbox": last_bbox.to_dict(),
                }
            )

            cand = self.build_candidate(
                video_id=context.video_id,
                event_type="POTENTIAL_OBJECT_REMOVAL",
                start_time=track.first_seen,
                end_time=track.last_seen,
                severity="NORMAL",
                confidence=min(0.66, scoring["score"]),
                explanation=(
                    f"Potential Object Removal ({scoring['verification_note']}): "
                    f"{track.object_class} [{track.track_id}] was stably tracked for {track.duration_seconds:.1f}s "
                    f"and ceased detection at {track.last_seen:.1f}s within interior camera boundaries."
                ),
                track_ids=[track.track_id],
                object_classes=[track.object_class],
                supporting_signals=signals,
                contradictory_signals=contradictory_signals,
                spatial_context=spatial_ctx,
                evidence_candidates=[
                    EvidenceCandidate(
                        timestamp=track.last_seen,
                        bounding_box=last_bbox,
                        target_track_id=track.track_id,
                        reason="Object cessation inside camera FOV",
                    )
                ],
                prefix="OBJREM",
            )
            cand.validation_decision = ValidationDecision.REVIEW_REQUIRED
            candidates.append(cand)

        return candidates
