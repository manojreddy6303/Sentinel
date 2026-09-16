"""
Multi-Frame Object Tracker for Sentinel Surveillance Intelligence

Associates detections across consecutive sampled video frames using spatial IoU
and Euclidean centroid distance matching. Maintains active tracks with motion history.

SAFETY CONSTRAINT:
Track ID represents only 'same visual object within this video session'.
It does NOT infer, establish, or imply personal identity.
"""
import math
from typing import List, Dict, Any, Optional, Tuple
from ai.schemas import BoundingBox, TrackedObject, TrackLifecycleState, DetectionValidationStatus


class ObjectTracker:
    """
    Lightweight, reliable multi-object spatial tracker for surveillance video.
    """

    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_distance_threshold: float = 0.25,  # Normalized distance relative to frame dimensions
        max_missing_seconds: float = 2.5,     # Coasting window before deactivating a track
        trackable_classes: Optional[List[str]] = None,
        video_id: str = "",
    ):
        self.iou_threshold = iou_threshold
        self.max_distance_threshold = max_distance_threshold
        self.max_missing_seconds = max_missing_seconds
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

    def update(
        self,
        timestamp: float,
        detections: List[Dict[str, Any]],
        frame_width: Optional[float] = None,
        frame_height: Optional[float] = None,
    ) -> List[TrackedObject]:
        """
        Update tracks with detections from the current frame.
        
        Args:
            timestamp: Video timestamp in seconds.
            detections: List of detection dicts with 'object_class', 'confidence', 'bounding_box'.
            frame_width: Optional pixel width for distance normalization.
            frame_height: Optional pixel height for distance normalization.

        Returns:
            List of currently active TrackedObject instances.
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

        # Separate currently active tracks
        active_tracks = [t for t in self._tracks.values() if t.active]
        unmatched_dets = list(range(len(current_dets)))
        matched_tracks = set()

        # Match tracks of the same class by IoU and distance
        for track in active_tracks:
            best_det_idx = -1
            best_score = -1.0

            for idx in unmatched_dets:
                d_cls, d_conf, d_bbox, _ = current_dets[idx]
                if d_cls != track.object_class:
                    continue

                # Compute IoU
                overlap = track.current_bbox.iou(d_bbox)
                
                # Compute centroid distance
                tcx, tcy = track.current_bbox.centroid
                dcx, dcy = d_bbox.centroid
                dist = math.hypot(dcx - tcx, dcy - tcy)
                if frame_width and frame_height and frame_width > 0 and frame_height > 0:
                    norm_dist = dist / math.hypot(frame_width, frame_height)
                else:
                    norm_dist = dist if dist <= 1.0 else (dist / 1000.0)

                # Matching score: prioritize IoU, fallback to proximity
                if overlap >= self.iou_threshold:
                    score = overlap + 1.0  # Priority tier for positive overlap
                elif norm_dist <= self.max_distance_threshold:
                    score = 1.0 - norm_dist  # Secondary tier for close proximity
                else:
                    score = 0.0

                if score > best_score and score > 0.0:
                    best_score = score
                    best_det_idx = idx

            if best_det_idx >= 0:
                # Update existing track
                d_cls, d_conf, d_bbox, raw_det = current_dets[best_det_idx]
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
                matched_tracks.add(track.track_id)
                unmatched_dets.remove(best_det_idx)

        # Create new tracks for unmatched detections
        for idx in unmatched_dets:
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
            matched_tracks.add(track_id)

        # Mark coasting tracks as inactive if exceeding max_missing_seconds
        for track in self._tracks.values():
            if track.active and track.track_id not in matched_tracks:
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
