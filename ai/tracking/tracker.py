"""
Multi-Frame Object Tracker for Sentinel Surveillance Intelligence

Associates detections across consecutive sampled video frames using ByteTrack-style
two-stage association (high-confidence primary matching + low-confidence secondary
recovery) combined with linear velocity motion prediction and Euclidean distance normalization.

SAFETY CONSTRAINT:
Track ID represents only 'same visual object within this video session'.
It does NOT infer, establish, or imply personal identity.
"""
import math
from typing import List, Dict, Any, Optional, Tuple, Set
from ai.schemas import BoundingBox, TrackedObject, TrackLifecycleState, DetectionValidationStatus


class ObjectTracker:
    """
    ByteTrack-style multi-object spatial tracker for surveillance video with
    two-stage candidate association, linear velocity prediction, and coasting recovery.
    """

    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_distance_threshold: float = 0.25,  # Normalized distance relative to frame dimensions
        max_missing_seconds: float = 2.5,     # Coasting window before deactivating a track
        high_conf_threshold: float = 0.40,    # Stage 1 primary association threshold
        low_conf_threshold: float = 0.20,     # Stage 2 secondary recovery threshold
        trackable_classes: Optional[List[str]] = None,
        video_id: str = "",
    ):
        self.iou_threshold = iou_threshold
        self.max_distance_threshold = max_distance_threshold
        self.max_missing_seconds = max_missing_seconds
        self.high_conf_threshold = high_conf_threshold
        self.low_conf_threshold = low_conf_threshold
        self.video_id = video_id
        self.trackable_classes = set(
            trackable_classes
            or [
                "person",
                "car",
                "bus",
                "truck",
                "motorcycle",
                "bicycle",
                "backpack",
                "suitcase",
                "handbag",
            ]
        )
        self._next_id = 1
        self._tracks: Dict[str, TrackedObject] = {}

    def reset(self, video_id: str = "") -> None:
        """Reset tracker state between videos."""
        self._next_id = 1
        self._tracks.clear()
        if video_id:
            self.video_id = video_id

    @staticmethod
    def _predict_bbox(track: TrackedObject, current_ts: float) -> BoundingBox:
        """
        Predict track bounding box at current timestamp using linear velocity vector.
        Clamped to max 2.0s prediction horizon to prevent runaway extrapolation drift.
        """
        if len(track.trajectory) < 2:
            return track.current_bbox

        t1, x1, y1 = track.trajectory[-2]
        t2, x2, y2 = track.trajectory[-1]
        dt = t2 - t1
        if dt <= 0:
            return track.current_bbox

        pred_dt = min(2.0, max(0.0, current_ts - t2))
        if pred_dt <= 0.001:
            return track.current_bbox

        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        dx = vx * pred_dt
        dy = vy * pred_dt

        orig = track.current_bbox
        return BoundingBox(
            x1=orig.x1 + dx,
            y1=orig.y1 + dy,
            x2=orig.x2 + dx,
            y2=orig.y2 + dy,
        )

    def _associate_candidates(
        self,
        tracks: List[TrackedObject],
        detection_indices: List[int],
        all_detections: List[Tuple[str, float, BoundingBox, Dict[str, Any]]],
        current_ts: float,
        frame_width: Optional[float],
        frame_height: Optional[float],
    ) -> Tuple[List[Tuple[TrackedObject, int]], Set[str], Set[int]]:
        """
        Score and match candidate detections with candidate tracks.
        Returns (matched_pairs, matched_track_ids, matched_det_indices).
        """
        candidate_matches: List[Tuple[float, TrackedObject, int]] = []
        diag = math.hypot(frame_width, frame_height) if (frame_width and frame_height and frame_width > 0 and frame_height > 0) else None

        for track in tracks:
            pred_bbox = self._predict_bbox(track, current_ts)
            ptcx, ptcy = pred_bbox.centroid

            for idx in detection_indices:
                d_cls, d_conf, d_bbox, _ = all_detections[idx]
                if d_cls != track.object_class:
                    continue

                # Compute IoU between predicted bbox and detection bbox
                overlap = pred_bbox.iou(d_bbox)

                # Compute normalized centroid distance
                dcx, dcy = d_bbox.centroid
                dist = math.hypot(dcx - ptcx, dcy - ptcy)
                if diag:
                    norm_dist = dist / diag
                else:
                    norm_dist = dist if dist <= 1.0 else (dist / 1000.0)

                # Matching score: prioritize positive IoU overlap, fallback to spatial proximity
                if overlap >= self.iou_threshold:
                    score = overlap + 1.0  # Tier 1: IoU overlap
                elif norm_dist <= self.max_distance_threshold:
                    score = 1.0 - norm_dist  # Tier 2: normalized centroid proximity
                else:
                    score = 0.0

                if score > 0.0:
                    candidate_matches.append((score, track, idx))

        # Greedy assignment in descending score order
        candidate_matches.sort(key=lambda item: item[0], reverse=True)
        matched_pairs: List[Tuple[TrackedObject, int]] = []
        matched_track_ids: Set[str] = set()
        matched_det_indices: Set[int] = set()

        for score, track, idx in candidate_matches:
            if track.track_id in matched_track_ids or idx in matched_det_indices:
                continue
            matched_pairs.append((track, idx))
            matched_track_ids.add(track.track_id)
            matched_det_indices.add(idx)

        return matched_pairs, matched_track_ids, matched_det_indices

    def update(
        self,
        timestamp: float,
        detections: List[Dict[str, Any]],
        frame_width: Optional[float] = None,
        frame_height: Optional[float] = None,
    ) -> List[TrackedObject]:
        """
        Update tracks with detections from the current frame using ByteTrack two-stage matching.
        """
        # Filter detections for trackable surveillance classes
        current_dets: List[Tuple[str, float, BoundingBox, Dict[str, Any]]] = []
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
            current_dets.append((cls, conf, bbox, det))

        # Age out coasting tracks exceeding max_missing_seconds BEFORE matching current frame
        for track in self._tracks.values():
            if track.active and (timestamp - track.last_seen) > self.max_missing_seconds:
                track.active = False
                track.state = TrackLifecycleState.LOST

        # Separate candidates into Stage 1 (high confidence / VALID) and Stage 2 (low confidence / UNCERTAIN)
        high_det_indices: List[int] = []
        low_det_indices: List[int] = []
        for i, (_, conf, _, raw_det) in enumerate(current_dets):
            val_stat = str(raw_det.get("validation_status", "VALID")).upper()
            if conf >= self.high_conf_threshold or val_stat == "VALID":
                high_det_indices.append(i)
            elif conf >= self.low_conf_threshold or val_stat == "UNCERTAIN":
                low_det_indices.append(i)

        # Active tracks available for association
        active_tracks = [t for t in self._tracks.values() if t.active]
        matched_track_ids: Set[str] = set()
        matched_det_indices: Set[int] = set()

        # =====================================================================
        # STAGE 1: Primary association with high-confidence detections
        # =====================================================================
        stage1_matches, s1_tracks, s1_dets = self._associate_candidates(
            tracks=active_tracks,
            detection_indices=high_det_indices,
            all_detections=current_dets,
            current_ts=timestamp,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        matched_track_ids.update(s1_tracks)
        matched_det_indices.update(s1_dets)

        for track, idx in stage1_matches:
            d_cls, d_conf, d_bbox, raw_det = current_dets[idx]
            track.last_seen = timestamp
            track.confidence = max(track.confidence, d_conf)
            track.current_bbox = d_bbox
            cx, cy = d_bbox.centroid
            track.trajectory.append((timestamp, round(cx, 2), round(cy, 2)))
            track.history_bboxes.append({"timestamp": timestamp, "bbox": d_bbox.to_dict()})
            if track.is_validated or len(track.history_bboxes) >= 2:
                track.state = TrackLifecycleState.CONFIRMED
            else:
                track.state = TrackLifecycleState.TENTATIVE
            if isinstance(raw_det, dict):
                raw_det["track_id"] = track.track_id

        # =====================================================================
        # STAGE 2: Secondary recovery association with low-confidence detections
        # Matches remaining unmatched active/coasting tracks to preserve continuity
        # =====================================================================
        unmatched_tracks_stage1 = [t for t in active_tracks if t.track_id not in matched_track_ids]
        if unmatched_tracks_stage1 and low_det_indices:
            stage2_matches, s2_tracks, s2_dets = self._associate_candidates(
                tracks=unmatched_tracks_stage1,
                detection_indices=low_det_indices,
                all_detections=current_dets,
                current_ts=timestamp,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            matched_track_ids.update(s2_tracks)
            matched_det_indices.update(s2_dets)

            for track, idx in stage2_matches:
                d_cls, d_conf, d_bbox, raw_det = current_dets[idx]
                track.last_seen = timestamp
                track.confidence = max(track.confidence, d_conf)
                track.current_bbox = d_bbox
                cx, cy = d_bbox.centroid
                track.trajectory.append((timestamp, round(cx, 2), round(cy, 2)))
                track.history_bboxes.append({"timestamp": timestamp, "bbox": d_bbox.to_dict()})
                if track.is_validated or len(track.history_bboxes) >= 2:
                    track.state = TrackLifecycleState.CONFIRMED
                if isinstance(raw_det, dict):
                    raw_det["track_id"] = track.track_id

        # =====================================================================
        # STAGE 3: Track Initiation ONLY from unmatched HIGH-CONFIDENCE detections
        # Low-confidence detections never spawn new tracks (prevents noise tracks)
        # =====================================================================
        for idx in high_det_indices:
            if idx in matched_det_indices:
                continue
            d_cls, d_conf, d_bbox, raw_det = current_dets[idx]
            track_id = f"TRACK-{self._next_id:03d}"
            self._next_id += 1
            cx, cy = d_bbox.centroid
            new_track = TrackedObject(
                track_id=track_id,
                video_id=self.video_id,
                object_class=d_cls,
                first_seen=timestamp,
                last_seen=timestamp,
                confidence=d_conf,
                current_bbox=d_bbox,
                trajectory=[(timestamp, round(cx, 2), round(cy, 2))],
                history_bboxes=[{"timestamp": timestamp, "bbox": d_bbox.to_dict()}],
                active=True,
                state=TrackLifecycleState.TENTATIVE,
            )
            if isinstance(raw_det, dict):
                raw_det["track_id"] = track_id
            self._tracks[track_id] = new_track
            matched_track_ids.add(track_id)

        # Update lifecycle state for tracks that were unmatched in both stages
        for track in self._tracks.values():
            if track.active and track.track_id not in matched_track_ids:
                if (timestamp - track.last_seen) > self.max_missing_seconds:
                    track.active = False
                    track.state = TrackLifecycleState.LOST
                else:
                    track.state = TrackLifecycleState.COASTING

        return [t for t in self._tracks.values() if t.active]

    def finalize(self, only_validated: bool = False) -> List[TrackedObject]:
        """Mark all remaining tracks inactive and return track catalogue."""
        for track in self._tracks.values():
            track.active = False
            if track.state != TrackLifecycleState.LOST:
                track.state = TrackLifecycleState.ENDED
        if only_validated:
            return [t for t in self._tracks.values() if t.is_validated]
        return list(self._tracks.values())

    def get_validated_tracks(self) -> List[TrackedObject]:
        """Return only tracks with multi-frame temporal persistence or verified confidence."""
        return [t for t in self._tracks.values() if t.is_validated]

    def get_tracks(self) -> List[TrackedObject]:
        """Return all tracks recorded so far."""
        return list(self._tracks.values())
