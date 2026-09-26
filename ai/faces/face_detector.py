"""
Face Detection Module for Sentinel Surveillance Intelligence

CRITICAL SAFETY & ETHICAL CONSTRAINTS:
1. DETECTS VISUAL REGIONS ONLY.
2. ABSOLUTELY ZERO FACIAL RECOGNITION.
3. ABSOLUTELY ZERO IDENTITY RECOGNITION OR NAME INFERENCE.
4. ABSOLUTELY ZERO FACE EMBEDDING EXTRACTION OR DATABASE COMPARISON.
5. NO BIOMETRIC DATA PERSISTENCE.

Outputs visual bounding box coordinates of face regions for security audit logging.
Uses anatomical head-region localization and chromatic skin-tone verification.
"""
import logging
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

from ai.schemas import BoundingBox, FaceDetection

logger = logging.getLogger(__name__)


_cached_cascade = None
_cascade_initialized = False


def _get_shared_cascade():
    global _cached_cascade, _cascade_initialized
    if not _cascade_initialized:
        _cascade_initialized = True
        if hasattr(cv2, "CascadeClassifier"):
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                casc = cv2.CascadeClassifier(cascade_path)
                if not casc.empty():
                    _cached_cascade = casc
            except Exception:
                _cached_cascade = None
    return _cached_cascade


class FaceDetector:
    """
    Lightweight visual face detector for surveillance video.
    Enforces strict safety: visual region localization only.
    """

    def __init__(self, min_skin_ratio: float = 0.10, min_size: int = 16):
        self.min_skin_ratio = min_skin_ratio
        self.min_size = min_size
        self.cascade = _get_shared_cascade()

    def detect_in_person_crop(
        self,
        frame_bgr: np.ndarray,
        person_bbox: BoundingBox,
        timestamp: float,
        track_id: Optional[str] = None,
    ) -> List[FaceDetection]:
        """
        Detect visual face region within a detected person's upper body region.
        """
        h_frame, w_frame = frame_bgr.shape[:2]
        px1 = max(0, min(int(person_bbox.x1), w_frame - 1))
        py1 = max(0, min(int(person_bbox.y1), h_frame - 1))
        px2 = max(px1 + 1, min(int(person_bbox.x2), w_frame))
        py2 = max(py1 + 1, min(int(person_bbox.y2), h_frame))

        pw = px2 - px1
        ph = py2 - py1

        # Person bounding box must be at least min_size
        if pw < self.min_size or ph < self.min_size * 2:
            return []

        # Anatomical head region is located in the upper 25% of the person height, centered horizontally
        head_top = py1
        head_bottom = py1 + max(int(ph * 0.28), self.min_size)
        head_left = px1 + int(pw * 0.20)
        head_right = px1 + int(pw * 0.80)

        head_crop = frame_bgr[head_top:head_bottom, head_left:head_right]
        if head_crop.size == 0 or head_crop.shape[0] < 8 or head_crop.shape[1] < 8:
            return []

        # Try cascade if available
        if self.cascade is not None and not self.cascade.empty():
            gray = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(16, 16))
            if len(faces) > 0:
                results = []
                for fx, fy, fw, fh in faces:
                    results.append(
                        FaceDetection(
                            timestamp=timestamp,
                            bounding_box=BoundingBox(
                                x1=float(head_left + fx),
                                y1=float(head_top + fy),
                                x2=float(head_left + fx + fw),
                                y2=float(head_top + fy + fh),
                            ),
                            confidence=0.88,
                            track_id=track_id,
                        )
                    )
                return results

        # Chromatic skin-tone verification in HSV color space
        hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
        # HSV skin tone mask: Hue [0, 25] or [165, 180], Saturation [25, 180], Value [40, 255]
        lower_skin1 = np.array([0, 25, 40], dtype=np.uint8)
        upper_skin1 = np.array([25, 180, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)

        lower_skin2 = np.array([165, 25, 40], dtype=np.uint8)
        upper_skin2 = np.array([180, 180, 255], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)

        skin_mask = cv2.bitwise_or(mask1, mask2)
        skin_count = int(np.count_nonzero(skin_mask))
        total_head_pixels = head_crop.shape[0] * head_crop.shape[1]

        skin_ratio = skin_count / float(total_head_pixels) if total_head_pixels > 0 else 0.0

        # If skin-tone density confirms visual face presence
        if skin_ratio >= self.min_skin_ratio:
            # Tightly bound the face inside the head region
            fx1 = max(float(head_left), float(px1 + pw * 0.25))
            fy1 = float(head_top + int(ph * 0.02))
            fx2 = min(float(head_right), float(px1 + pw * 0.75))
            fy2 = float(head_top + int(ph * 0.24))

            return [
                FaceDetection(
                    timestamp=timestamp,
                    bounding_box=BoundingBox(x1=round(fx1, 2), y1=round(fy1, 2), x2=round(fx2, 2), y2=round(fy2, 2)),
                    confidence=round(min(0.95, 0.50 + skin_ratio), 3),
                    track_id=track_id,
                )
            ]

        return []

    def detect_full_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp: float,
    ) -> List[FaceDetection]:
        """
        Detect face regions across the entire frame if cascade is available.
        """
        if self.cascade is None or self.cascade.empty():
            return []

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(24, 24),
        )

        results: List[FaceDetection] = []
        for fx, fy, fw, fh in faces:
            results.append(
                FaceDetection(
                    timestamp=timestamp,
                    bounding_box=BoundingBox(
                        x1=float(fx),
                        y1=float(fy),
                        x2=float(fx + fw),
                        y2=float(fy + fh),
                    ),
                    confidence=0.85,
                )
            )
        return results
