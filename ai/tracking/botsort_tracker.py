"""
BoT-SORT Multi-Object Tracker for Sentinel (Phase 21A)

Camera-motion compensated, Kalman-filtered multi-object tracker with:
- Kalman filter (8-dim state: cx, cy, w, h, vx, vy, vw, vh)
- Global Motion Compensation (GMC) via sparse optical flow
- Hungarian (optimal) assignment via scipy or greedy fallback
- Two-stage ByteTrack-style association (high-conf → low-conf)
- Proper occlusion handling using the OCCLUDED lifecycle state
- No ReID / appearance features (Phase 21A scope)

Outputs standard TrackedObject instances, fully compatible with the
existing SecurityIntelligencePipeline, IncidentEngine, CorrelationEngine,
and evidence generation pipeline.

This tracker is NOT the production default. It is a benchmark candidate
selectable via TRACKER_TYPE configuration.

SAFETY CONSTRAINT:
Track ID represents only 'same visual object within this video session'.
It does NOT infer, establish, or imply personal identity.
"""
import logging
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set

import numpy as np

from ai.schemas import BoundingBox, TrackedObject, TrackLifecycleState, DetectionValidationStatus
from ai.tracking.kalman_filter import KalmanFilter
from ai.tracking.gmc import GlobalMotionCompensation
from ai.tracking.matching import (
    iou_batch,
    iou_cost_matrix,
    linear_assignment,
    centroid_distance_matrix,
    centroid_distance_assignment,
)

logger = logging.getLogger(__name__)

# Mahalanobis gating threshold for 4-DOF chi-squared (99.5% confidence)
_CHI2_THRESHOLD = 16.27


@dataclass
class _STrack:
    """Internal track state for BoT-SORT. Not exposed to external consumers."""

    track_id: str
    object_class: str
    first_seen: float
    last_seen: float
    confidence: float
    kalman_mean: np.ndarray
    kalman_cov: np.ndarray
    state: TrackLifecycleState
    trajectory: List[Tuple[float, float, float]] = field(default_factory=list)
    history_bboxes: List[Dict[str, Any]] = field(default_factory=list)
    active: bool = True
    video_id: str = ""
    visual_attributes: Optional[Dict[str, Any]] = None
    attribute_history: List[Dict[str, Any]] = field(default_factory=list)
    color: Optional[str] = None
    color_confidence: Optional[float] = None
    frames_since_update: int = 0

    @property
    def predicted_bbox_tlbr(self) -> Tuple[float, float, float, float]:
        """Get predicted bounding box in [x1, y1, x2, y2] format."""
        return KalmanFilter.state_to_bbox(self.kalman_mean)

    @property
    def detection_count(self) -> int:
        return len(self.history_bboxes)

    @property
    def is_validated(self) -> bool:
        """Same validation logic as TrackedObject."""
        count = len(self.history_bboxes)
        if count >= 2:
            return True
        if self.object_class in {"bus", "truck", "car"}:
            return self.confidence >= 0.65
        return True

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    def to_tracked_object(self) -> TrackedObject:
        """Convert internal track to standard TrackedObject output."""
        x1, y1, x2, y2 = self.predicted_bbox_tlbr
        return TrackedObject(
            track_id=self.track_id,
            video_id=self.video_id,
            object_class=self.object_class,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            confidence=self.confidence,
            current_bbox=BoundingBox(
                x1=max(0.0, x1), y1=max(0.0, y1),
                x2=max(1.0, x2), y2=max(1.0, y2),
            ),
            trajectory=list(self.trajectory),
            history_bboxes=list(self.history_bboxes),
            active=self.active,
            state=self.state,
            visual_attributes=self.visual_attributes,
            attribute_history=list(self.attribute_history),
            color=self.color,
            color_confidence=self.color_confidence,
        )


class BoTSORTTracker:
    """
    BoT-SORT-style multi-object tracker with Kalman filtering,
    Global Motion Compensation, and Hungarian assignment.

    Compatible with SecurityIntelligencePipeline via the same update() interface
    as ObjectTracker, plus optional frame_bgr for GMC.
    """

    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_distance_threshold: float = 0.25,
        max_missing_seconds: float = 2.5,
        max_occluded_seconds: float = 5.0,
        high_conf_threshold: float = 0.40,
        low_conf_threshold: float = 0.20,
        occlusion_iou_threshold: float = 0.3,
        trackable_classes: Optional[List[str]] = None,
        video_id: str = "",
        enable_gmc: bool = True,
        enable_centroid_fallback: bool = True,
        ambiguity_threshold: float = 0.01,
        relative_ambiguity_margin: float = 0.10,
    ):
        self.iou_threshold = iou_threshold
        self.max_distance_threshold = max_distance_threshold
        self.max_missing_seconds = max_missing_seconds
        self.max_occluded_seconds = max_occluded_seconds
        self.high_conf_threshold = high_conf_threshold
        self.low_conf_threshold = low_conf_threshold
        self.occlusion_iou_threshold = occlusion_iou_threshold
        self.video_id = video_id
        self.enable_gmc = enable_gmc
        self.enable_centroid_fallback = enable_centroid_fallback
        self.ambiguity_threshold = ambiguity_threshold
        self.relative_ambiguity_margin = relative_ambiguity_margin

        self.trackable_classes = set(
            trackable_classes
            or [
                "person", "car", "bus", "truck", "motorcycle",
                "bicycle", "backpack", "suitcase", "handbag",
            ]
        )

        self._next_id = 1
        self._tracks: Dict[str, _STrack] = {}
        self._kalman = KalmanFilter()
        self._gmc = GlobalMotionCompensation()
        self._last_timestamp: Optional[float] = None

    def reset(self, video_id: str = "") -> None:
        """Reset tracker state between videos."""
        self._next_id = 1
        self._tracks.clear()
        self._gmc.reset()
        self._last_timestamp = None
        if video_id:
            self.video_id = video_id

    def update(
        self,
        timestamp: float,
        detections: List[Dict[str, Any]],
        frame_width: Optional[float] = None,
        frame_height: Optional[float] = None,
        frame_bgr: Optional[np.ndarray] = None,
    ) -> List[TrackedObject]:
        """
        Update tracks with detections from the current frame.

        Uses BoT-SORT pipeline:
        1. GMC: estimate camera motion from frame_bgr
        2. Kalman predict all active tracks (with GMC warp)
        3. Stage 1: Hungarian match high-conf detections
        4. Stage 2: Hungarian match low-conf detections with remaining tracks
        5. Occlusion handling for unmatched tracks
        6. Track initiation from unmatched high-conf detections
        7. Lifecycle management

        Args:
            timestamp: current frame timestamp in seconds
            detections: list of detection dicts
            frame_width: frame width in pixels (for distance normalization)
            frame_height: frame height in pixels (for distance normalization)
            frame_bgr: raw BGR frame for GMC (optional — falls back to no GMC)

        Returns:
            List of active TrackedObject instances
        """
        # --- Parse detections ---
        current_dets = self._parse_detections(detections)

        # --- Compute dt ---
        dt = 1.0
        if self._last_timestamp is not None and timestamp > self._last_timestamp:
            dt = min(timestamp - self._last_timestamp, 5.0)  # cap dt to prevent runaway
        self._last_timestamp = timestamp

        # --- GMC: estimate camera motion ---
        affine = np.eye(2, 3, dtype=np.float64)
        if self.enable_gmc and frame_bgr is not None:
            try:
                affine = self._gmc.estimate(frame_bgr)
            except Exception as e:
                logger.debug("GMC estimation failed, using identity: %s", e)

        # --- Age out tracks exceeding their max missing window ---
        for strack in self._tracks.values():
            if not strack.active:
                continue
            if strack.state == TrackLifecycleState.OCCLUDED:
                max_missing = self.max_occluded_seconds
            else:
                max_missing = self.max_missing_seconds
            if (timestamp - strack.last_seen) > max_missing:
                strack.active = False
                strack.state = TrackLifecycleState.LOST

        # --- Kalman predict all active tracks + apply GMC ---
        active_stracks = [s for s in self._tracks.values() if s.active]
        for strack in active_stracks:
            # Apply GMC warp to Kalman state before prediction
            if not GlobalMotionCompensation.is_identity(affine):
                cx, cy, w, h = strack.kalman_mean[:4]
                new_cx, new_cy, _, _ = GlobalMotionCompensation.apply_affine_to_cxcywh(
                    affine, cx, cy, w, h
                )
                strack.kalman_mean[0] = new_cx
                strack.kalman_mean[1] = new_cy

            # Kalman predict
            strack.kalman_mean, strack.kalman_cov = self._kalman.predict(
                strack.kalman_mean, strack.kalman_cov, dt=dt
            )

        # --- Separate detections into high-conf and low-conf ---
        high_det_indices: List[int] = []
        low_det_indices: List[int] = []
        for i, (_, conf, _, raw_det) in enumerate(current_dets):
            val_stat = str(raw_det.get("validation_status", "VALID")).upper()
            if conf >= self.high_conf_threshold or val_stat == "VALID":
                high_det_indices.append(i)
            elif conf >= self.low_conf_threshold or val_stat == "UNCERTAIN":
                low_det_indices.append(i)

        matched_track_ids: Set[str] = set()
        matched_det_indices: Set[int] = set()

        # =====================================================================
        # STAGE 1: Hungarian match active tracks ↔ high-confidence detections
        # =====================================================================
        if active_stracks and high_det_indices:
            s1_matches, s1_unmatched_trk, s1_unmatched_det = self._hungarian_match(
                tracks=active_stracks,
                det_indices=high_det_indices,
                all_dets=current_dets,
                threshold=1.0 - self.iou_threshold,  # IoU cost threshold
                frame_width=frame_width,
                frame_height=frame_height,
            )

            for trk_idx, det_idx in s1_matches:
                strack = active_stracks[trk_idx]
                self._update_track(strack, current_dets[det_idx], timestamp)
                matched_track_ids.add(strack.track_id)
                matched_det_indices.add(det_idx)

        # =====================================================================
        # STAGE 2: Hungarian match remaining tracks ↔ low-conf detections
        # =====================================================================
        remaining_stracks = [s for s in active_stracks if s.track_id not in matched_track_ids]
        remaining_low_dets = [i for i in low_det_indices if i not in matched_det_indices]

        if remaining_stracks and remaining_low_dets:
            s2_matches, _, _ = self._hungarian_match(
                tracks=remaining_stracks,
                det_indices=remaining_low_dets,
                all_dets=current_dets,
                threshold=1.0 - self.iou_threshold,
                frame_width=frame_width,
                frame_height=frame_height,
            )

            for trk_idx, det_idx in s2_matches:
                strack = remaining_stracks[trk_idx]
                self._update_track(strack, current_dets[det_idx], timestamp)
                matched_track_ids.add(strack.track_id)
                matched_det_indices.add(det_idx)

        # =====================================================================
        # STAGE 3: Centroid-distance fallback for remaining unmatched tracks ↔ detections
        # =====================================================================
        if self.enable_centroid_fallback:
            candidate_fallback_stracks = [
                s for s in active_stracks if s.track_id not in matched_track_ids
            ]
            candidate_fallback_dets = [
                i for i in range(len(current_dets)) if i not in matched_det_indices
            ]

            if candidate_fallback_stracks and candidate_fallback_dets:
                fb_matches = self._centroid_fallback_match(
                    tracks=candidate_fallback_stracks,
                    det_indices=candidate_fallback_dets,
                    all_dets=current_dets,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    current_ts=timestamp,
                )
                for trk_idx, det_idx in fb_matches:
                    strack = candidate_fallback_stracks[trk_idx]
                    self._update_track(strack, current_dets[det_idx], timestamp)
                    matched_track_ids.add(strack.track_id)
                    matched_det_indices.add(det_idx)

        # =====================================================================
        # OCCLUSION HANDLING: Classify unmatched active tracks
        # =====================================================================
        # Get bboxes of all matched detections (objects that are visible now)
        matched_det_bboxes = []
        for idx in matched_det_indices:
            _, _, d_bbox, _ = current_dets[idx]
            matched_det_bboxes.append([d_bbox.x1, d_bbox.y1, d_bbox.x2, d_bbox.y2])
        matched_det_arr = np.array(matched_det_bboxes) if matched_det_bboxes else np.zeros((0, 4))

        for strack in self._tracks.values():
            if not strack.active or strack.track_id in matched_track_ids:
                continue

            # Check if this track's predicted position overlaps a visible detection
            pred_x1, pred_y1, pred_x2, pred_y2 = strack.predicted_bbox_tlbr
            pred_arr = np.array([[pred_x1, pred_y1, pred_x2, pred_y2]])

            is_occluded = False
            if len(matched_det_arr) > 0:
                ious = iou_batch(pred_arr, matched_det_arr)
                if ious.max() >= self.occlusion_iou_threshold:
                    is_occluded = True

            time_since_seen = timestamp - strack.last_seen
            if is_occluded:
                strack.state = TrackLifecycleState.OCCLUDED
                strack.frames_since_update += 1
            elif time_since_seen > self.max_missing_seconds:
                strack.active = False
                strack.state = TrackLifecycleState.LOST
            else:
                strack.state = TrackLifecycleState.COASTING
                strack.frames_since_update += 1

        # =====================================================================
        # TRACK INITIATION: from unmatched HIGH-conf detections only
        # =====================================================================
        for idx in high_det_indices:
            if idx in matched_det_indices:
                continue
            d_cls, d_conf, d_bbox, raw_det = current_dets[idx]
            track_id = f"TRACK-{self._next_id:03d}"
            self._next_id += 1

            measurement = KalmanFilter.bbox_to_measurement(
                d_bbox.x1, d_bbox.y1, d_bbox.x2, d_bbox.y2
            )
            mean, cov = self._kalman.initiate(measurement)
            cx, cy = d_bbox.centroid

            strack = _STrack(
                track_id=track_id,
                object_class=d_cls,
                first_seen=timestamp,
                last_seen=timestamp,
                confidence=d_conf,
                kalman_mean=mean,
                kalman_cov=cov,
                state=TrackLifecycleState.TENTATIVE,
                trajectory=[(timestamp, round(cx, 2), round(cy, 2))],
                history_bboxes=[{"timestamp": timestamp, "bbox": d_bbox.to_dict()}],
                active=True,
                video_id=self.video_id,
            )
            if isinstance(raw_det, dict):
                raw_det["track_id"] = track_id
            self._tracks[track_id] = strack
            matched_track_ids.add(track_id)

        # --- Return active TrackedObject list ---
        return [s.to_tracked_object() for s in self._tracks.values() if s.active]

    def _parse_detections(
        self, detections: List[Dict[str, Any]]
    ) -> List[Tuple[str, float, BoundingBox, Dict[str, Any]]]:
        """Parse and filter detections for trackable classes."""
        result = []
        for det in detections:
            val_status = det.get("validation_status")
            if val_status in ("REJECTED", "rejected", DetectionValidationStatus.REJECTED):
                continue
            cls = det.get("object_class") or det.get("class_name")
            if not cls or cls not in self.trackable_classes:
                continue
            bb_dict = det.get("bounding_box", {})
            try:
                bbox = BoundingBox(
                    x1=float(bb_dict.get("x1", 0.0)),
                    y1=float(bb_dict.get("y1", 0.0)),
                    x2=float(bb_dict.get("x2", 0.0)),
                    y2=float(bb_dict.get("y2", 0.0)),
                )
                if not bbox.is_valid:
                    continue
            except Exception:
                continue
            conf = float(det.get("confidence", 0.0))
            result.append((cls, conf, bbox, det))
        return result

    def _hungarian_match(
        self,
        tracks: List[_STrack],
        det_indices: List[int],
        all_dets: List[Tuple[str, float, BoundingBox, Dict[str, Any]]],
        threshold: float,
        frame_width: Optional[float],
        frame_height: Optional[float],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Build cost matrix and run Hungarian assignment with class gating.

        Returns:
            (matches, unmatched_track_indices, unmatched_det_indices)
            where matches are (track_list_index, det_global_index) tuples
        """
        n_tracks = len(tracks)
        n_dets = len(det_indices)

        # Build predicted track bboxes
        track_bboxes = np.zeros((n_tracks, 4), dtype=np.float64)
        for i, strack in enumerate(tracks):
            track_bboxes[i] = strack.predicted_bbox_tlbr

        # Build detection bboxes
        det_bboxes = np.zeros((n_dets, 4), dtype=np.float64)
        for j, idx in enumerate(det_indices):
            _, _, d_bbox, _ = all_dets[idx]
            det_bboxes[j] = [d_bbox.x1, d_bbox.y1, d_bbox.x2, d_bbox.y2]

        # IoU cost matrix
        cost = iou_cost_matrix(track_bboxes, det_bboxes)

        # Apply class gating: set cost to infinity for class mismatches
        for i, strack in enumerate(tracks):
            for j, idx in enumerate(det_indices):
                d_cls = all_dets[idx][0]
                if d_cls != strack.object_class:
                    cost[i, j] = 1e6  # effectively infinite

        # Apply Mahalanobis gating
        for i, strack in enumerate(tracks):
            for j, idx in enumerate(det_indices):
                if cost[i, j] >= 1e6:
                    continue
                d_bbox = all_dets[idx][2]
                measurement = KalmanFilter.bbox_to_measurement(
                    d_bbox.x1, d_bbox.y1, d_bbox.x2, d_bbox.y2
                )
                maha_dist = self._kalman.gating_distance(
                    strack.kalman_mean, strack.kalman_cov, measurement
                )
                if maha_dist > _CHI2_THRESHOLD:
                    cost[i, j] = 1e6

        # Solve assignment
        matches_raw, unmatched_trk, unmatched_det_local = linear_assignment(
            cost, threshold
        )

        # Map local det indices back to global indices
        matches = [(trk_idx, det_indices[det_local]) for trk_idx, det_local in matches_raw]
        unmatched_det = [det_indices[d] for d in unmatched_det_local]

        return matches, unmatched_trk, unmatched_det

    def _centroid_fallback_match(
        self,
        tracks: List[_STrack],
        det_indices: List[int],
        all_dets: List[Tuple[str, float, BoundingBox, Dict[str, Any]]],
        frame_width: Optional[float],
        frame_height: Optional[float],
        current_ts: float,
    ) -> List[Tuple[int, int]]:
        """
        Match remaining tracks and detections via normalized centroid distance.
        Includes class gating, temporal gap gating, maximum distance gating,
        and ambiguity abstention.
        """
        n_tracks = len(tracks)
        n_dets = len(det_indices)
        if n_tracks == 0 or n_dets == 0:
            return []

        # Frame diagonal for normalization
        diag = (
            math.hypot(frame_width, frame_height)
            if (frame_width and frame_height and frame_width > 0 and frame_height > 0)
            else 1000.0
        )

        track_bboxes = np.zeros((n_tracks, 4), dtype=np.float64)
        for i, strack in enumerate(tracks):
            track_bboxes[i] = strack.predicted_bbox_tlbr

        det_bboxes = np.zeros((n_dets, 4), dtype=np.float64)
        for j, idx in enumerate(det_indices):
            _, _, d_bbox, _ = all_dets[idx]
            det_bboxes[j] = [d_bbox.x1, d_bbox.y1, d_bbox.x2, d_bbox.y2]

        cost = centroid_distance_matrix(track_bboxes, det_bboxes, frame_diag=diag)

        # Gate by object class and temporal gap
        for i, strack in enumerate(tracks):
            time_since_seen = current_ts - strack.last_seen
            max_allowed_gap = (
                self.max_occluded_seconds
                if strack.state == TrackLifecycleState.OCCLUDED
                else self.max_missing_seconds
            )
            is_expired = time_since_seen > max_allowed_gap

            for j, idx in enumerate(det_indices):
                d_cls = all_dets[idx][0]
                if is_expired or d_cls != strack.object_class:
                    cost[i, j] = 1e6

        matches_local, _, _ = centroid_distance_assignment(
            cost_matrix=cost,
            distance_threshold=self.max_distance_threshold,
            ambiguity_threshold=self.ambiguity_threshold,
            relative_ambiguity_margin=self.relative_ambiguity_margin,
        )
        return [(trk_idx, det_indices[det_local]) for trk_idx, det_local in matches_local]

    def _update_track(
        self,
        strack: _STrack,
        det_tuple: Tuple[str, float, BoundingBox, Dict[str, Any]],
        timestamp: float,
    ) -> None:
        """Update an existing track with a matched detection."""
        d_cls, d_conf, d_bbox, raw_det = det_tuple

        # Kalman update
        measurement = KalmanFilter.bbox_to_measurement(
            d_bbox.x1, d_bbox.y1, d_bbox.x2, d_bbox.y2
        )
        strack.kalman_mean, strack.kalman_cov = self._kalman.update(
            strack.kalman_mean, strack.kalman_cov, measurement
        )

        # Update metadata
        strack.last_seen = timestamp
        strack.confidence = max(strack.confidence, d_conf)
        strack.frames_since_update = 0
        cx, cy = d_bbox.centroid
        strack.trajectory.append((timestamp, round(cx, 2), round(cy, 2)))
        strack.history_bboxes.append({"timestamp": timestamp, "bbox": d_bbox.to_dict()})

        # Lifecycle state update
        if strack.is_validated or len(strack.history_bboxes) >= 2:
            strack.state = TrackLifecycleState.CONFIRMED
        else:
            strack.state = TrackLifecycleState.TENTATIVE

        # Propagate track_id to raw detection dict
        if isinstance(raw_det, dict):
            raw_det["track_id"] = strack.track_id

    def finalize(self, only_validated: bool = False) -> List[TrackedObject]:
        """Mark all remaining tracks inactive and return track catalogue."""
        for strack in self._tracks.values():
            strack.active = False
            if strack.state != TrackLifecycleState.LOST:
                strack.state = TrackLifecycleState.ENDED
        if only_validated:
            return [s.to_tracked_object() for s in self._tracks.values() if s.is_validated]
        return [s.to_tracked_object() for s in self._tracks.values()]

    def get_validated_tracks(self) -> List[TrackedObject]:
        """Return only tracks with multi-frame temporal persistence."""
        return [s.to_tracked_object() for s in self._tracks.values() if s.is_validated]

    def get_tracks(self) -> List[TrackedObject]:
        """Return all tracks recorded so far."""
        return [s.to_tracked_object() for s in self._tracks.values()]
