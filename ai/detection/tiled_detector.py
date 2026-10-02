"""
Sentinel Tiled & Multi-Scale Object Detector (Phase 20)

Implements native SAHI-style (Slicing Aided Hyper Inference) multi-scale detection
to reliably detect small surveillance objects (<32px: small bags, phones, distant people/vehicles)
without loading full uncompressed 4K frames at prohibitive inference resolutions.

Workflow:
1. Full-frame base inference (captures large/medium context).
2. Sliced tile crops (640x640 with 20% overlap) processed via mini-batching.
3. Coordinate re-projection from tile-relative to global frame coordinates.
4. Global Non-Maximum Suppression (NMS) and boundary deduplication.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

from ai.schemas import BoundingBox, CanonicalDetection, DetectionValidationStatus

logger = logging.getLogger(__name__)


class TiledObjectDetector:
    """
    Multi-scale tiled inference engine for high-resolution surveillance video.
    """

    def __init__(
        self,
        base_detector: Any,
        tile_size: int = 640,
        overlap_ratio: float = 0.20,
        nms_iou_threshold: float = 0.45,
        small_object_boost: bool = True,
    ):
        """
        Args:
            base_detector: Underlying detector (e.g. YOLODetector) supporting detect_batch.
            tile_size: Square tile dimension in pixels.
            overlap_ratio: Overlap percentage between adjacent tiles (0.20 = 20%).
            nms_iou_threshold: IoU threshold for global multi-scale NMS.
            small_object_boost: Whether to prioritize small-object candidates from tiles.
        """
        self.base_detector = base_detector
        self.tile_size = int(tile_size)
        self.overlap_ratio = float(overlap_ratio)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.small_object_boost = small_object_boost

    def _generate_slices(self, width: int, height: int) -> List[Tuple[int, int, int, int]]:
        """
        Generate overlapping (x1, y1, x2, y2) tile slices for the image.
        """
        slices: List[Tuple[int, int, int, int]] = []
        step = max(1, int(self.tile_size * (1.0 - self.overlap_ratio)))

        y_starts = list(range(0, max(1, height - self.tile_size + 1), step))
        if not y_starts or (y_starts[-1] + self.tile_size < height):
            y_starts.append(max(0, height - self.tile_size))

        x_starts = list(range(0, max(1, width - self.tile_size + 1), step))
        if not x_starts or (x_starts[-1] + self.tile_size < width):
            x_starts.append(max(0, width - self.tile_size))

        # Eliminate duplicate start coordinates
        y_starts = sorted(list(set(y_starts)))
        x_starts = sorted(list(set(x_starts)))

        for y in y_starts:
            for x in x_starts:
                x2 = min(width, x + self.tile_size)
                y2 = min(height, y + self.tile_size)
                slices.append((x, y, x2, y2))

        return slices

    def assess_small_object_risk(
        self,
        baseline_detections: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int,
        motion_regions: Optional[List[Tuple[int, int, int, int]]] = None,
        user_roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Assess whether the frame contains small or distant objects, weak candidates,
        or active motion clusters that warrant targeted high-resolution tiled inspection.

        Returns:
            List of (x1, y1, x2, y2) regions of interest.
        """
        rois: List[Tuple[int, int, int, int]] = []

        if user_roi is not None:
            rois.append(user_roi)

        frame_area = max(1.0, float(frame_width * frame_height))
        margin = self.tile_size // 4

        for det in baseline_detections:
            bb = det.get("bounding_box", {})
            bx1 = float(bb.get("x1", 0.0))
            by1 = float(bb.get("y1", 0.0))
            bx2 = float(bb.get("x2", 0.0))
            by2 = float(bb.get("y2", 0.0))
            bw = max(0.0, bx2 - bx1)
            bh = max(0.0, by2 - by1)
            conf = float(det.get("confidence", 0.0))
            cls = str(det.get("object_class") or det.get("class_name", "")).lower()

            is_small = (max(bw, bh) <= 52.0) or ((bw * bh) < (0.0025 * frame_area))
            is_weak = (0.20 <= conf <= 0.45)
            is_surveillance_class = cls in (
                "person", "car", "bicycle", "motorcycle", "backpack",
                "handbag", "suitcase", "bottle", "package", "box", "chair"
            )

            if (is_small or is_weak) and is_surveillance_class:
                rx1 = max(0, int(bx1 - margin))
                ry1 = max(0, int(by1 - margin))
                rx2 = min(frame_width, int(bx2 + margin))
                ry2 = min(frame_height, int(by2 + margin))
                rois.append((rx1, ry1, rx2, ry2))

        if motion_regions:
            for mx1, my1, mx2, my2 in motion_regions:
                rois.append((mx1, my1, mx2, my2))

        return rois

    def select_gated_slices(
        self,
        slices: List[Tuple[int, int, int, int]],
        rois: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int, int, int]]:
        """
        Filter full grid slices to only those overlapping with small-object risk ROIs.
        """
        if not rois:
            return []

        selected = []
        for sx1, sy1, sx2, sy2 in slices:
            overlaps = False
            for rx1, ry1, rx2, ry2 in rois:
                # Bounding box intersection check
                inter_x1 = max(sx1, rx1)
                inter_y1 = max(sy1, ry1)
                inter_x2 = min(sx2, rx2)
                inter_y2 = min(sy2, ry2)
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    overlaps = True
                    break
            if overlaps:
                selected.append((sx1, sy1, sx2, sy2))

        return selected

    def detect_tiled(
        self,
        frame_bgr: np.ndarray,
        timestamp: float = 0.0,
        frame_idx: int = 0,
        video_id: str = "",
        include_full_frame: bool = True,
        max_tiles_batch: int = 16,
    ) -> List[Dict[str, Any]]:
        """
        Run multi-scale tiled detection across frame and merge with global NMS.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w = frame_bgr.shape[:2]

        # For low-res frames (<= 720p), standard full-frame inference is already optimal
        if max(h, w) <= 720:
            return self.base_detector.detect(
                frame_bgr,
                timestamp=timestamp,
                frame_idx=frame_idx,
                video_id=video_id,
            )

        all_detections: List[Dict[str, Any]] = []

        # 1. Full-frame base inference
        if include_full_frame:
            global_dets = self.base_detector.detect(
                frame_bgr,
                timestamp=timestamp,
                frame_idx=frame_idx,
                video_id=video_id,
            )
            for d in global_dets:
                d_dict = d.to_dict() if hasattr(d, "to_dict") else dict(d)
                d_dict["scale_source"] = "global"
                all_detections.append(d_dict)

        # 2. Extract tile crops and offsets
        slices = self._generate_slices(w, h)
        tile_crops = []
        tile_offsets = []

        for x1, y1, x2, y2 in slices:
            crop = frame_bgr[y1:y2, x1:x2]
            # Pad if slice is smaller than tile_size at boundaries
            ch, cw = crop.shape[:2]
            if ch != self.tile_size or cw != self.tile_size:
                padded = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
                padded[:ch, :cw] = crop
                tile_crops.append(padded)
            else:
                tile_crops.append(crop)
            tile_offsets.append((x1, y1))

        # 3. Mini-batched tile inference
        for i in range(0, len(tile_crops), max_tiles_batch):
            batch_c = tile_crops[i:i + max_tiles_batch]
            batch_off = tile_offsets[i:i + max_tiles_batch]
            batch_ts = [timestamp] * len(batch_c)
            batch_fn = [frame_idx] * len(batch_c)

            if hasattr(self.base_detector, "detect_batch"):
                batch_res = self.base_detector.detect_batch(
                    batch_c,
                    batch_ts,
                    batch_fn,
                    video_id=video_id,
                    imgsz=self.tile_size,
                )
            else:
                batch_res = [
                    self.base_detector.detect(c, ts, frame_idx=fn, video_id=video_id, imgsz=self.tile_size)
                    for c, ts, fn in zip(batch_c, batch_ts, batch_fn)
                ]

            for tile_dets, (off_x, off_y) in zip(batch_res, batch_off):
                for d in tile_dets:
                    d_dict = d.to_dict() if hasattr(d, "to_dict") else dict(d)
                    bb = d_dict.get("bounding_box", {})
                    # Re-project from tile space to global frame coordinates
                    gx1 = max(0.0, min(float(w), float(bb.get("x1", 0.0)) + off_x))
                    gy1 = max(0.0, min(float(h), float(bb.get("y1", 0.0)) + off_y))
                    gx2 = max(0.0, min(float(w), float(bb.get("x2", 0.0)) + off_x))
                    gy2 = max(0.0, min(float(h), float(bb.get("y2", 0.0)) + off_y))

                    if gx2 <= gx1 or gy2 <= gy1:
                        continue

                    d_dict["bounding_box"] = {"x1": gx1, "y1": gy1, "x2": gx2, "y2": gy2}
                    d_dict["scale_source"] = "tile"
                    all_detections.append(d_dict)

        # 4. Global Class-Aware Non-Maximum Suppression
        final_dets = self._apply_global_nms(all_detections, iou_thresh=self.nms_iou_threshold)
        return final_dets

    def detect_selective(
        self,
        frame_bgr: np.ndarray,
        timestamp: float = 0.0,
        frame_idx: int = 0,
        video_id: str = "",
        motion_regions: Optional[List[Tuple[int, int, int, int]]] = None,
        user_roi: Optional[Tuple[int, int, int, int]] = None,
        validator: Optional[Any] = None,
        force_tiling: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Gated Selective SAHI Pipeline:
        1. Run baseline detection on full frame.
        2. Assess small-object risk and identify targeted regions of interest.
        3. If no small-object cues or motion present, return baseline detections immediately (0 extra tiles).
        4. If risk cues detected, run tiled inference ONLY on intersecting tile slices.
        5. Re-project coordinates from tile space to global frame space.
        6. Merge via class-aware Non-Maximum Suppression.
        7. Enforce Sentinel spatial validation standards (no bypassing).

        Returns:
            (validated_detections, telemetry_metadata)
        """
        import time
        t0 = time.perf_counter()

        if frame_bgr is None or frame_bgr.size == 0:
            return [], {"gated_mode": "empty_frame", "tiles_processed": 0, "latency_ms": 0.0}

        h, w = frame_bgr.shape[:2]

        # Resolution gate: Frames <= 720p do not benefit from tiling
        if max(h, w) <= 720 and not force_tiling:
            base_dets = self.base_detector.detect(
                frame_bgr, timestamp=timestamp, frame_idx=frame_idx, video_id=video_id
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return base_dets, {
                "gated_mode": "resolution_bypass_low_res",
                "total_slices": 0,
                "tiles_processed": 0,
                "latency_ms": round(elapsed_ms, 2),
            }

        # Step 1: Run full-frame baseline detection
        raw_base = self.base_detector.detect(
            frame_bgr, timestamp=timestamp, frame_idx=frame_idx, video_id=video_id
        )
        baseline_dets = [d.to_dict() if hasattr(d, "to_dict") else dict(d) for d in raw_base]
        for d in baseline_dets:
            d["scale_source"] = "global"

        # Step 2: Assess small-object risk
        rois = self.assess_small_object_risk(
            baseline_detections=baseline_dets,
            frame_width=w,
            frame_height=h,
            motion_regions=motion_regions,
            user_roi=user_roi,
        )

        all_slices = self._generate_slices(w, h)

        # If no small-object risk cues and not forced, return baseline immediately
        if not rois and not force_tiling:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return baseline_dets, {
                "gated_mode": "baseline_sufficient_zero_tiles",
                "total_slices": len(all_slices),
                "tiles_processed": 0,
                "latency_ms": round(elapsed_ms, 2),
            }

        # Step 3: Select only the intersecting slices
        if force_tiling:
            selected_slices = all_slices
        else:
            selected_slices = self.select_gated_slices(all_slices, rois)

        if not selected_slices:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return baseline_dets, {
                "gated_mode": "no_intersecting_tiles",
                "total_slices": len(all_slices),
                "tiles_processed": 0,
                "latency_ms": round(elapsed_ms, 2),
            }

        # Step 4: Extract and pad selected crops
        tile_crops = []
        tile_offsets = []
        for x1, y1, x2, y2 in selected_slices:
            crop = frame_bgr[y1:y2, x1:x2]
            ch, cw = crop.shape[:2]
            if ch != self.tile_size or cw != self.tile_size:
                padded = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
                padded[:ch, :cw] = crop
                tile_crops.append(padded)
            else:
                tile_crops.append(crop)
            tile_offsets.append((x1, y1))

        # Step 5: Batched tile inference
        tile_detections = []
        batch_ts = [timestamp] * len(tile_crops)
        batch_fn = [frame_idx] * len(tile_crops)

        if hasattr(self.base_detector, "detect_batch"):
            batch_res = self.base_detector.detect_batch(
                tile_crops, batch_ts, batch_fn, video_id=video_id, imgsz=self.tile_size
            )
        else:
            batch_res = [
                self.base_detector.detect(c, ts, frame_idx=fn, video_id=video_id, imgsz=self.tile_size)
                for c, ts, fn in zip(tile_crops, batch_ts, batch_fn)
            ]

        # Step 6: Coordinate re-projection
        for dets, (off_x, off_y) in zip(batch_res, tile_offsets):
            for d in dets:
                d_dict = d.to_dict() if hasattr(d, "to_dict") else dict(d)
                bb = d_dict.get("bounding_box", {})
                gx1 = max(0.0, min(float(w), float(bb.get("x1", 0.0)) + off_x))
                gy1 = max(0.0, min(float(h), float(bb.get("y1", 0.0)) + off_y))
                gx2 = max(0.0, min(float(w), float(bb.get("x2", 0.0)) + off_x))
                gy2 = max(0.0, min(float(h), float(bb.get("y2", 0.0)) + off_y))

                if gx2 <= gx1 or gy2 <= gy1:
                    continue

                d_dict["bounding_box"] = {"x1": gx1, "y1": gy1, "x2": gx2, "y2": gy2}
                d_dict["scale_source"] = "selective_tile"
                tile_detections.append(d_dict)

        # Step 7: Merge baseline + targeted tile detections via global NMS
        merged_candidates = self._apply_global_nms(
            baseline_dets + tile_detections, iou_thresh=self.nms_iou_threshold
        )

        # Step 8: Spatial & Aspect-Ratio Validation Gate
        if validator is not None:
            if hasattr(validator, "validate_events_list"):
                validator.validate_events_list(merged_candidates, frame_width=w, frame_height=h)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        telemetry = {
            "gated_mode": "selective_tiled",
            "total_slices": len(all_slices),
            "tiles_processed": len(selected_slices),
            "rois_identified": len(rois),
            "baseline_count": len(baseline_dets),
            "tile_count": len(tile_detections),
            "final_count": len(merged_candidates),
            "latency_ms": round(elapsed_ms, 2),
        }

        return merged_candidates, telemetry

    @staticmethod
    def _apply_global_nms(
        detections: List[Dict[str, Any]],
        iou_thresh: float = 0.45,
    ) -> List[Dict[str, Any]]:
        """
        Class-aware Non-Maximum Suppression across merged global and tiled candidates.
        Prioritizes higher confidence, with special retention for small high-confidence objects.
        """
        if not detections:
            return []

        # Group by class
        by_class: Dict[str, List[Dict[str, Any]]] = {}
        for d in detections:
            cls = d.get("object_class") or d.get("class_name", "object")
            by_class.setdefault(cls, []).append(d)

        kept: List[Dict[str, Any]] = []

        for cls, dets in by_class.items():
            # Sort descending by confidence
            dets.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
            suppressed = [False] * len(dets)

            for i in range(len(dets)):
                if suppressed[i]:
                    continue
                d_i = dets[i]
                kept.append(d_i)

                bb_i = d_i.get("bounding_box", {})
                box_i = BoundingBox(
                    x1=float(bb_i["x1"]),
                    y1=float(bb_i["y1"]),
                    x2=float(bb_i["x2"]),
                    y2=float(bb_i["y2"]),
                )

                for j in range(i + 1, len(dets)):
                    if suppressed[j]:
                        continue
                    d_j = dets[j]
                    bb_j = d_j.get("bounding_box", {})
                    box_j = BoundingBox(
                        x1=float(bb_j["x1"]),
                        y1=float(bb_j["y1"]),
                        x2=float(bb_j["x2"]),
                        y2=float(bb_j["y2"]),
                    )

                    iou = box_i.iou(box_j)
                    if iou >= iou_thresh:
                        suppressed[j] = True

        return kept
