"""
YOLO Object Detection Module

Responsible for:
- Loading a pretrained Ultralytics YOLO model (once, lazily).
- Running inference on individual video frames.
- Returning structured detection results with class, confidence, and bounding box.

IMPORTANT ETHICAL BOUNDARIES:
- No facial recognition or biometric matching.
- No criminal or threat classification.
- No identity attribution.
- Detections are probabilistic computer-vision outputs only.
- Confidence scores must always be shown to the user.
"""

import logging
import math
from typing import List, Dict, Any, Optional

from ai.schemas import BoundingBox, CanonicalDetection, DetectionValidationStatus
from ai.common.numeric import is_finite_number, ensure_finite, clamp_finite

logger = logging.getLogger(__name__)

# Surveillance-relevant COCO and forensic object classes.
SURVEILLANCE_CLASSES = {
    # Persons
    "person",
    # Vehicles & transit
    "bicycle", "car", "motorcycle", "bus", "truck", "boat", "train", "airplane",
    # Portable property / personal belongings / retail merchandise
    "backpack", "handbag", "suitcase", "bottle", "cell phone", "laptop",
    "mouse", "remote", "keyboard", "book", "clock", "vase", "scissors",
    "umbrella", "cup", "knife", "baseball bat", "sports ball", "skateboard",
    # Furniture & Infrastructure
    "chair", "bench", "couch", "bed", "dining table", "traffic light", "stop sign", "fire hydrant",
    # Generic investigation object labels
    "general_object", "package", "box", "merchandise",
}


class YOLODetector:
    """
    Reusable YOLO object detector based on Ultralytics YOLO.

    The model is loaded lazily on first use and cached for the lifetime
    of the instance. Load once, call detect() many times.

    Usage:
        detector = YOLODetector(model_name="yolov8n.pt", confidence_threshold=0.25)
        for frame_number, timestamp, frame_bgr in processor.sample_frames():
            detections = detector.detect(frame_bgr, timestamp)
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        surveillance_classes: Optional[set] = None,
        class_confidence_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            model_name: Ultralytics model name or path. "yolov8n.pt" downloads
                        the YOLOv8 Nano model (~6MB) on first use.
            confidence_threshold: Minimum confidence score [0.0, 1.0] to include
                                  a detection in results.
            surveillance_classes: Optional set of class names to filter results.
                                  If None, uses the default SURVEILLANCE_CLASSES set.
            class_confidence_thresholds: Optional per-class confidence thresholds.
        """
        self.model_name = model_name
        self.confidence_threshold = float(confidence_threshold)
        self.surveillance_classes = surveillance_classes or SURVEILLANCE_CLASSES
        self.class_confidence_thresholds = class_confidence_thresholds or {}
        self._model = None  # Lazy-loaded

    def _load_model(self):
        """Load the YOLO model once and cache it on the instance with PyTorch 2.6+ compatibility."""
        if self._model is None:
            try:
                from ai.detection.torch_compat import load_trusted_yolo_model
                logger.info(f"Loading YOLO model: {self.model_name}")
                self._model = load_trusted_yolo_model(self.model_name)
                logger.info(f"YOLO model loaded: {self.model_name}")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load YOLO model '{self.model_name}': {exc}"
                ) from exc
        return self._model

    def _parse_single_result(
        self,
        result: Any,
        frame: Any,
        timestamp: float,
        frame_idx: int = 0,
        video_id: str = "",
    ) -> List[Dict[str, Any]]:
        """Parse YOLO result for a single frame into standardized canonical detection dicts."""
        detections: List[Dict[str, Any]] = []
        if result is None or getattr(result, "boxes", None) is None:
            return detections

        model = self._load_model()
        frame_h, frame_w = frame.shape[:2] if hasattr(frame, "shape") and len(frame.shape) >= 2 else (None, None)

        for box in result.boxes:
            # Extract raw values
            try:
                confidence = float(box.conf[0])
            except Exception:
                continue

            try:
                class_id = int(box.cls[0])
                class_name = model.names.get(class_id, f"class_{class_id}")
            except Exception:
                continue

            # Class-specific or global confidence threshold
            req_threshold = self.class_confidence_thresholds.get(class_name, self.confidence_threshold)
            if confidence < req_threshold:
                continue

            # Optionally filter to surveillance-relevant classes
            if self.surveillance_classes and class_name not in self.surveillance_classes:
                continue

            try:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
            except Exception:
                continue

            # Reject NaN or infinite coordinates
            if not is_finite_number(x1) or not is_finite_number(y1) or not is_finite_number(x2) or not is_finite_number(y2):
                continue

            # Reject technically invalid (zero/negative) dimensions
            if x2 <= x1 or y2 <= y1:
                continue

            # Frame boundary validation
            if frame_w is not None and frame_h is not None and frame_w > 0 and frame_h > 0:
                if x2 <= 0 or y2 <= 0 or x1 >= frame_w or y1 >= frame_h:
                    continue
                x1 = max(0.0, min(float(frame_w), x1))
                y1 = max(0.0, min(float(frame_h), y1))
                x2 = max(0.0, min(float(frame_w), x2))
                y2 = max(0.0, min(float(frame_h), y2))
                if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                    continue

            bbox = BoundingBox(
                x1=round(x1, 2),
                y1=round(y1, 2),
                x2=round(x2, 2),
                y2=round(y2, 2),
            )
            is_edge = bbox.is_edge_clipped(frame_w, frame_h, margin=4.0)
            boundaries = bbox.edge_clip_boundaries(frame_w, frame_h, margin=4.0)

            canonical = CanonicalDetection(
                video_id=video_id,
                frame_index=frame_idx,
                timestamp_seconds=round(ensure_finite(timestamp, 0.0), 4),
                class_name=class_name,
                class_id=class_id,
                confidence=round(clamp_finite(confidence, 0.0, 1.0, default=0.0), 4),
                bounding_box=bbox,
                detector_name="yolo_detector",
                detector_version="v8n",
                validation_status=DetectionValidationStatus.RAW,
                observation_source="yolo_detector",
                model_or_heuristic="trained_model",
                image_width=frame_w,
                image_height=frame_h,
                is_edge_clipped=is_edge,
                edge_clip_boundaries=boundaries,
            )
            detections.append(canonical.to_dict())

        # Intra-frame duplicate suppression (IoU >= 0.70 for identical class)
        if len(detections) > 1:
            detections = self._suppress_duplicate_detections(detections, iou_threshold=0.70)

        return detections

    def detect(
        self,
        frame: Any,
        timestamp: float,
        frame_idx: int = 0,
        video_id: str = "",
        imgsz: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run YOLO object detection on a single video frame.

        Args:
            frame: BGR numpy array (as returned by cv2.VideoCapture.read()).
            timestamp: Timestamp in seconds of this frame in the source video.
            frame_idx: Optional index of the processed frame.
            video_id: Optional ID of the parent video.
            imgsz: Optional inference image size dimension.

        Returns:
            List of standardized canonical detection dicts.
        """
        model = self._load_model()
        try:
            if imgsz:
                results = model(frame, imgsz=imgsz, verbose=False)
            else:
                results = model(frame, verbose=False)
        except Exception as exc:
            logger.error(f"YOLO inference failed at timestamp {timestamp:.2f}s: {exc}")
            raise RuntimeError(f"YOLO inference error: {exc}") from exc

        result = results[0] if results else None
        return self._parse_single_result(result, frame, timestamp, frame_idx, video_id)

    def detect_batch(
        self,
        frames: List[Any],
        timestamps: List[float],
        frame_indices: Optional[List[int]] = None,
        video_id: str = "",
        imgsz: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Run YOLO object detection on a batch of video frames for high throughput.

        Args:
            frames: List of BGR numpy arrays.
            timestamps: List of timestamp floats in seconds corresponding to each frame.
            frame_indices: Optional list of frame index numbers.
            video_id: Optional ID of the parent video.
            imgsz: Optional inference image size dimension.

        Returns:
            List of detection lists, one per input frame.
        """
        if not frames:
            return []
        model = self._load_model()
        try:
            if imgsz:
                results = model(frames, imgsz=imgsz, verbose=False)
            else:
                results = model(frames, verbose=False)
        except Exception as exc:
            logger.error(f"YOLO batch inference failed: {exc}")
            raise RuntimeError(f"YOLO batch inference error: {exc}") from exc

        indices = frame_indices or [0] * len(frames)
        batch_detections: List[List[Dict[str, Any]]] = []
        for idx, (frame, timestamp, f_idx) in enumerate(zip(frames, timestamps, indices)):
            res = results[idx] if idx < len(results) else None
            batch_detections.append(
                self._parse_single_result(res, frame, timestamp, f_idx, video_id)
            )
        return batch_detections

    @staticmethod
    def _compute_iou(b1: Dict[str, float], b2: Dict[str, float]) -> float:
        ix1 = max(b1["x1"], b2["x1"])
        iy1 = max(b1["y1"], b2["y1"])
        ix2 = min(b1["x2"], b2["x2"])
        iy2 = min(b1["y2"], b2["y2"])
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        a1 = max(0.0, b1["x2"] - b1["x1"]) * max(0.0, b1["y2"] - b1["y1"])
        a2 = max(0.0, b2["x2"] - b2["x1"]) * max(0.0, b2["y2"] - b2["y1"])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    @classmethod
    def _suppress_duplicate_detections(
        cls, detections: List[Dict[str, Any]], iou_threshold: float = 0.70
    ) -> List[Dict[str, Any]]:
        """
        Suppresses redundant duplicate bounding boxes for the exact same physical object
        on a single frame, preserving the detection with the highest confidence score.
        Legitimate separate objects of the same class (low IoU) are strictly preserved.
        """
        sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)
        kept: List[Dict[str, Any]] = []
        for det in sorted_dets:
            det_class = det.get("object_class") or det.get("class_name")
            det_box = det.get("bounding_box") or {}
            is_dup = False
            for k in kept:
                k_class = k.get("object_class") or k.get("class_name")
                if det_class and k_class and det_class == k_class:
                    k_box = k.get("bounding_box") or {}
                    iou = cls._compute_iou(det_box, k_box)
                    if iou >= iou_threshold:
                        is_dup = True
                        break
            if not is_dup:
                kept.append(det)
        return kept

    def is_loaded(self) -> bool:
        """Return True if the YOLO model has been loaded."""
        return self._model is not None
