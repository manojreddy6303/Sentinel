"""
Sentinel Micro-Object and Interaction Recovery Engine (Phase 0.2)

Discovers unmodeled small portable objects (<32px or outside standard COCO vocabulary)
via selective spatio-temporal ROI analysis around dwelling person tracks and interaction zones.
Preserves Railway 1GB memory bounds by operating only on localized interaction crops.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2

from ai.schemas import BoundingBox, CanonicalDetection, DetectionValidationStatus

logger = logging.getLogger(__name__)


class MicroObjectRecoveryEngine:
    """
    Spatio-temporal recovery engine for micro-objects and unmodeled portable property.
    """

    def __init__(
        self,
        min_dwell_seconds: float = 3.0,
        max_normalized_speed: float = 0.12,  # body heights / second
        min_contour_area: float = 35.0,
        max_contour_area: float = 2500.0,
        min_displacement_px: float = 25.0,
        confidence: float = 0.65,
    ):
        self.min_dwell_seconds = float(min_dwell_seconds)
        self.max_normalized_speed = float(max_normalized_speed)
        self.min_contour_area = float(min_contour_area)
        self.max_contour_area = float(max_contour_area)
        self.min_displacement_px = float(min_displacement_px)
        self.confidence = float(confidence)

    def discover_dwelling_person_intervals(
        self,
        tracks: List[Any],
        fps: float,
    ) -> List[Dict[str, Any]]:
        """
        Identify dwelling person tracks with localized stationary presence near potential surfaces.
        Uses normalized body-height velocity to avoid false triggers on distant walking pedestrians.
        """
        dwelling_intervals: List[Dict[str, Any]] = []

        for track in tracks:
            cls = getattr(track, "object_class", None) or getattr(track, "class_name", "")
            if cls != "person":
                continue

            dur = getattr(track, "duration_seconds", 0.0)
            if dur < self.min_dwell_seconds:
                continue

            traj = getattr(track, "trajectory", [])
            hist = getattr(track, "history_bboxes", [])
            if len(traj) < 2 or not hist:
                continue

            # Calculate spatial span and normalized velocity in body heights / s
            pts = np.array([(p[1], p[2]) for p in traj])
            span = float(np.hypot(pts.max(axis=0)[0] - pts.min(axis=0)[0], pts.max(axis=0)[1] - pts.min(axis=0)[1]))
            avg_height = float(np.mean([max(10.0, float(h["bbox"]["y2"]) - float(h["bbox"]["y1"])) for h in hist]))
            norm_speed = span / max(1.0, avg_height * dur)

            if norm_speed <= self.max_normalized_speed:
                min_x = min(float(h["bbox"]["x1"]) for h in hist)
                min_y = min(float(h["bbox"]["y1"]) for h in hist)
                max_x = max(float(h["bbox"]["x2"]) for h in hist)
                max_y = max(float(h["bbox"]["y2"]) for h in hist)

                t_first = getattr(track, "first_seen", traj[0][0])
                t_last = getattr(track, "last_seen", traj[-1][0])

                dwelling_intervals.append({
                    "track_id": getattr(track, "track_id", "PERSON"),
                    "start_time": t_first,
                    "end_time": t_last,
                    "duration": dur,
                    "person_bbox_union": (min_x, min_y, max_x, max_y),
                    "trajectory": traj,
                    "history_bboxes": hist,
                })

        return dwelling_intervals

    def recover_micro_objects_from_video(
        self,
        video_path: str,
        dwelling_intervals: List[Dict[str, Any]],
        fps: float,
        frame_width: int,
        frame_height: int,
        sampled_frames_cache: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Selectively analyze high-resolution candidate reach ROIs to recover unmodeled micro-objects.
        Enforces coherent trajectory tracking and displacement verification.
        """
        if not dwelling_intervals:
            return []

        recovered: List[Dict[str, Any]] = []
        cap = None
        if not sampled_frames_cache and video_path:
            cap = cv2.VideoCapture(video_path)

        try:
            for interval in dwelling_intervals:
                bx1, by1, bx2, by2 = interval["person_bbox_union"]
                pw = max(10.0, bx2 - bx1)
                ph = max(10.0, by2 - by1)

                # Reach ROI: focus around the hands/reach perimeter
                roi_x1 = max(0, int(bx1 - 0.40 * pw))
                roi_x2 = min(frame_width, int(bx2 + 0.40 * pw))
                roi_y1 = max(0, int(by1 + 0.20 * ph))
                roi_y2 = min(frame_height, int(by1 + 0.85 * ph))

                roi_w = roi_x2 - roi_x1
                roi_h = roi_y2 - roi_y1
                if roi_w < 20 or roi_h < 20:
                    continue

                t_start = interval["start_time"]
                t_end = interval["end_time"]
                step_s = 0.5
                sample_times = np.arange(t_start, t_end + 0.1, step_s)

                window_frames: List[Tuple[float, np.ndarray]] = []
                for st in sample_times:
                    frame = None
                    if sampled_frames_cache:
                        best_k = None
                        min_d = 0.4
                        for k in sampled_frames_cache:
                            diff = abs(k - st)
                            if diff < min_d:
                                min_d = diff
                                best_k = k
                        if best_k is not None:
                            frame = sampled_frames_cache[best_k]

                    if frame is None and cap and cap.isOpened():
                        f_idx = int(round(st * fps))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                        ret, f = cap.read()
                        if ret and f is not None:
                            frame = f

                    if frame is not None and frame.size > 0:
                        window_frames.append((float(st), frame))

                if len(window_frames) < 3:
                    continue

                ref_ts, ref_img = window_frames[0]
                ref_crop = ref_img[roi_y1:roi_y2, roi_x1:roi_x2]
                ref_gray = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)
                ref_gray = cv2.GaussianBlur(ref_gray, (5, 5), 0)

                raw_candidate_dets: List[Dict[str, Any]] = []

                for ts, cur_img in window_frames[1:]:
                    cur_crop = cur_img[roi_y1:roi_y2, roi_x1:roi_x2]
                    cur_gray = cv2.cvtColor(cur_crop, cv2.COLOR_BGR2GRAY)
                    cur_gray = cv2.GaussianBlur(cur_gray, (5, 5), 0)

                    diff = cv2.absdiff(ref_gray, cur_gray)
                    _, thresh = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for c in contours:
                        area = cv2.contourArea(c)
                        if self.min_contour_area <= area <= self.max_contour_area:
                            cx, cy, cw, ch = cv2.boundingRect(c)
                            aspect = max(cw, ch) / max(1, min(cw, ch))
                            if aspect <= 4.0 and 8 <= cw <= 90 and 8 <= ch <= 90:
                                gx1 = roi_x1 + cx
                                gy1 = roi_y1 + cy
                                gx2 = gx1 + cw
                                gy2 = gy1 + ch
                                raw_candidate_dets.append({
                                    "timestamp": ts,
                                    "bbox": (gx1, gy1, gx2, gy2),
                                    "centroid": ((gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0),
                                    "area": area,
                                })

                if len(raw_candidate_dets) < 2:
                    continue

                # Form coherent spatio-temporal trajectories
                raw_candidate_dets.sort(key=lambda o: o["timestamp"])
                trajectories: List[List[Dict[str, Any]]] = []

                for det in raw_candidate_dets:
                    matched = False
                    for traj in trajectories:
                        prev = traj[-1]
                        dt = det["timestamp"] - prev["timestamp"]
                        if 0.1 <= dt <= 1.5:
                            dist = float(np.hypot(det["centroid"][0] - prev["centroid"][0], det["centroid"][1] - prev["centroid"][1]))
                            if dist <= 40.0:  # smooth physical continuity
                                traj.append(det)
                                matched = True
                                break
                    if not matched:
                        trajectories.append([det])

                # Select validated trajectories exhibiting genuine physical displacement (> 25px)
                valid_trajs = []
                for traj in trajectories:
                    if len(traj) >= 2:
                        p_start = traj[0]["centroid"]
                        p_end = traj[-1]["centroid"]
                        disp = float(np.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1]))
                        if disp >= self.min_displacement_px:
                            valid_trajs.append((disp, traj))

                valid_trajs.sort(key=lambda item: item[0], reverse=True)

                # Cap to top coherent micro-object track per interaction window
                if valid_trajs:
                    best_disp, best_traj = valid_trajs[0]
                    seen_sec = set()
                    for item in best_traj:
                        sec_key = round(item["timestamp"], 1)
                        if sec_key in seen_sec:
                            continue
                        seen_sec.add(sec_key)

                        ox1, oy1, ox2, oy2 = item["bbox"]
                        recovered.append({
                            "video_id": "",
                            "frame_index": int(round(item["timestamp"] * fps)),
                            "timestamp_seconds": item["timestamp"],
                            "timestamp": item["timestamp"],
                            "class_name": "unknown_portable_object",
                            "object_class": "unknown_portable_object",
                            "class_id": 99,
                            "confidence": self.confidence,
                            "bounding_box": {
                                "x1": round(float(ox1), 2),
                                "y1": round(float(oy1), 2),
                                "x2": round(float(ox2), 2),
                                "y2": round(float(oy2), 2),
                            },
                            "detector_name": "micro_object_recovery_engine",
                            "detector_version": "1.0.0",
                            "validation_status": "VALID",
                            "observation_source": "micro_object_recovery_engine",
                            "model_or_heuristic": "heuristic_cv",
                            "image_width": frame_width,
                            "image_height": frame_height,
                            "is_edge_clipped": False,
                            "edge_clip_boundaries": [],
                            "validation_score": 0.85,
                            "validation_reason": "Temporal spatio-temporal micro-object interaction contour",
                            "track_id": None,
                            "metrics": {
                                "contour_area": item["area"],
                                "interaction_displacement": round(best_disp, 2),
                            },
                        })

        finally:
            if cap:
                cap.release()

        logger.info(f"Micro-object recovery engine generated {len(recovered)} coherent micro-object detections.")
        return recovered
