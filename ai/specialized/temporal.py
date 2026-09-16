"""
Specialized Temporal Validation Engine (Phase 15)

Tracks and aggregates frame-level specialized visual observations across time.
Guarantees that single-frame artifacts, compression glitches, or transient
sensor noise never trigger security incidents without temporal persistence.
"""
import uuid
import math
from typing import List, Dict, Any, Optional
from collections import defaultdict

from ai.schemas import BoundingBox
from ai.specialized.schemas import SpecializedObservation, SpecializedTemporalTrack


class SpecializedTemporalTracker:
    """
    Associates frame-by-frame specialized visual detections into coherent temporal tracks.
    """

    def __init__(
        self,
        max_time_gap_seconds: float = 1.5,
        min_spatial_iou: float = 0.15,
        max_centroid_distance: float = 150.0,
    ):
        self.max_time_gap = max_time_gap_seconds
        self.min_spatial_iou = min_spatial_iou
        self.max_centroid_distance = max_centroid_distance
        self.active_tracks: List[SpecializedTemporalTrack] = []
        self.completed_tracks: List[SpecializedTemporalTrack] = []

    def update(self, observations: List[SpecializedObservation]) -> None:
        """Update tracker with observations from a new frame timestamp."""
        if not observations:
            return

        unmatched_obs = list(observations)
        for trk in self.active_tracks:
            best_match_idx = -1
            best_score = -1.0

            for idx, obs in enumerate(unmatched_obs):
                if obs.class_name != trk.class_name:
                    continue

                time_diff = obs.timestamp - trk.last_seen
                if time_diff < 0 or time_diff > self.max_time_gap:
                    continue

                # Spatial match score
                score = self._compute_spatial_match(trk, obs)
                if score > best_score and score >= 0.2:
                    best_score = score
                    best_match_idx = idx

            if best_match_idx >= 0:
                matched_obs = unmatched_obs.pop(best_match_idx)
                trk.add_observation(matched_obs)

        # Create new tracks for remaining unmatched observations
        for obs in unmatched_obs:
            trk_id = f"SPEC-{obs.class_name[:4].upper()}-{uuid.uuid4().hex[:6]}"
            new_trk = SpecializedTemporalTrack(
                track_id=trk_id,
                class_name=obs.class_name,
                detector_name=obs.detector_name,
                first_seen=obs.timestamp,
                last_seen=obs.timestamp,
                observation_count=1,
                max_confidence=obs.confidence,
                mean_confidence=obs.confidence,
                confidence_history=[obs.confidence],
                bounding_boxes=[obs.bounding_box.to_dict()] if obs.bounding_box else [],
                visual_metrics_history=[obs.visual_metrics] if obs.visual_metrics else [],
            )
            self.active_tracks.append(new_trk)

    def prune_stale(self, current_time: float) -> None:
        """Move inactive tracks to completed_tracks."""
        still_active = []
        for trk in self.active_tracks:
            if current_time - trk.last_seen > self.max_time_gap:
                self.completed_tracks.append(trk)
            else:
                still_active.append(trk)
        self.active_tracks = still_active

    def finalize(self) -> List[SpecializedTemporalTrack]:
        """Finalize and return all tracks."""
        all_tracks = self.completed_tracks + self.active_tracks
        self.active_tracks = []
        self.completed_tracks = []
        return all_tracks

    def _compute_spatial_match(self, trk: SpecializedTemporalTrack, obs: SpecializedObservation) -> float:
        """Compute spatial affinity between a track's last box and the new observation."""
        if not trk.bounding_boxes or not obs.bounding_box:
            return 0.5

        last_b = trk.bounding_boxes[-1]
        try:
            last_box = BoundingBox(
                x1=float(last_b.get("x1", 0.0)),
                y1=float(last_b.get("y1", 0.0)),
                x2=float(last_b.get("x2", 0.0)),
                y2=float(last_b.get("y2", 0.0)),
            )
        except Exception:
            return 0.0

        try:
            iou = last_box.iou(obs.bounding_box)
            if math.isnan(iou) or math.isinf(iou):
                iou = 0.0
        except Exception:
            iou = 0.0

        if iou >= self.min_spatial_iou:
            return min(1.0, max(0.0, iou))

        # Centroid distance scale-normalized against object scale
        try:
            cx1, cy1 = (last_box.x1 + last_box.x2) / 2.0, (last_box.y1 + last_box.y2) / 2.0
            cx2, cy2 = (obs.bounding_box.x1 + obs.bounding_box.x2) / 2.0, (obs.bounding_box.y1 + obs.bounding_box.y2) / 2.0
            dist = math.hypot(cx2 - cx1, cy2 - cy1)
            if math.isnan(dist) or math.isinf(dist):
                return 0.0
        except Exception:
            return 0.0

        # Scale-adaptive distance threshold: allow movement proportional to object size or fixed baseline
        obj_diag = max(
            math.hypot(last_box.width, last_box.height),
            math.hypot(obs.bounding_box.width, obs.bounding_box.height),
            50.0
        )
        adaptive_threshold = max(self.max_centroid_distance, obj_diag * 1.5)

        if dist <= adaptive_threshold:
            score = 1.0 - (dist / adaptive_threshold)
            return max(0.2, min(1.0, score))

        return 0.0

