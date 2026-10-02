"""
Face Detection and Non-Biometric Visual Telemetry Module for Sentinel (Phase 20.2)

CRITICAL SAFETY & ETHICAL CONSTRAINTS:
1. DETECTS VISUAL REGIONS ONLY.
2. ABSOLUTELY ZERO FACIAL RECOGNITION.
3. ABSOLUTELY ZERO IDENTITY RECOGNITION OR NAME INFERENCE.
4. ABSOLUTELY ZERO FACE EMBEDDING EXTRACTION OR DATABASE COMPARISON.
5. NO BIOMETRIC DATA PERSISTENCE.

Outputs visual bounding box coordinates, optical sharpness, resolution,
approximate orientation, and visibility quality metrics for security audit logging.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

from ai.schemas import BoundingBox, FaceDetection, FaceRegionTelemetry
from ai.common.numeric import clamp_finite, ensure_finite

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
    Lightweight visual face detector and non-biometric region telemetry engine.
    Enforces strict safety: visual region localization and optical quality only.
    """

    def __init__(self, min_skin_ratio: float = 0.10, min_size: int = 16):
        self.min_skin_ratio = min_skin_ratio
        self.min_size = min_size
        self.cascade = _get_shared_cascade()

    def analyze_face_telemetry(
        self,
        frame_bgr: np.ndarray,
        person_bbox: BoundingBox,
        timestamp: float,
        track_id: Optional[str] = None,
    ) -> FaceRegionTelemetry:
        """
        Analyze head region for face presence and non-biometric optical quality telemetry.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return FaceRegionTelemetry(face_present=False, approximate_orientation="unknown")

        h_frame, w_frame = frame_bgr.shape[:2]
        px1 = max(0, min(int(person_bbox.x1), w_frame - 1))
        py1 = max(0, min(int(person_bbox.y1), h_frame - 1))
        px2 = max(px1 + 1, min(int(person_bbox.x2), w_frame))
        py2 = max(py1 + 1, min(int(person_bbox.y2), h_frame))

        pw = px2 - px1
        ph = py2 - py1

        if pw < self.min_size or ph < self.min_size * 2:
            return FaceRegionTelemetry(face_present=False, approximate_orientation="unknown")

        is_top_clipped = (py1 <= 3)

        # Head region is upper 28% of person height, center 60% of width
        head_top = py1
        head_bottom = py1 + max(int(ph * 0.28), self.min_size)
        head_left = px1 + int(pw * 0.20)
        head_right = px1 + int(pw * 0.80)

        head_crop = frame_bgr[head_top:head_bottom, head_left:head_right]
        if head_crop.size == 0 or head_crop.shape[0] < 8 or head_crop.shape[1] < 8:
            return FaceRegionTelemetry(
                face_present=False,
                is_occluded=is_top_clipped,
                approximate_orientation="unknown",
            )

        gray_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray_head, cv2.CV_64F).var())

        # 1. Try Cascade Classifier
        if self.cascade is not None and not self.cascade.empty():
            faces = self.cascade.detectMultiScale(gray_head, scaleFactor=1.1, minNeighbors=3, minSize=(16, 16))
            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]
                face_bbox = BoundingBox(
                    x1=float(head_left + fx),
                    y1=float(head_top + fy),
                    x2=float(head_left + fx + fw),
                    y2=float(head_top + fy + fh),
                )
                res_score = min(1.0, float(max(fw, fh)) / 48.0)
                sharp_score = min(1.0, sharpness / 120.0)
                quality = round(0.40 * res_score + 0.60 * sharp_score, 4)

                return FaceRegionTelemetry(
                    face_present=True,
                    visibility_score=0.90,
                    quality_score=quality,
                    approximate_orientation="frontal",
                    is_occluded=is_top_clipped,
                    sharpness_score=round(sharpness, 2),
                    resolution=(int(fw), int(fh)),
                    face_bbox=face_bbox,
                )

        # 2. Chromatic Skin-Tone Verification Fallback
        hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
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

        if skin_ratio >= self.min_skin_ratio:
            # Estimate orientation from horizontal skin center of mass
            M = cv2.moments(skin_mask)
            orient = "unknown"
            if M["m00"] > 0:
                cx = float(M["m10"] / M["m00"])
                half_w = float(head_crop.shape[1]) / 2.0
                if cx < half_w * 0.70:
                    orient = "profile_left"
                elif cx > half_w * 1.30:
                    orient = "profile_right"
                else:
                    orient = "angled"

            fx1 = max(float(head_left), float(px1 + pw * 0.25))
            fy1 = float(head_top + int(ph * 0.02))
            fx2 = min(float(head_right), float(px1 + pw * 0.75))
            fy2 = float(head_top + int(ph * 0.24))

            face_w = int(fx2 - fx1)
            face_h = int(fy2 - fy1)
            res_score = min(1.0, float(max(face_w, face_h)) / 48.0)
            sharp_score = min(1.0, sharpness / 120.0)
            quality = round(0.40 * res_score + 0.60 * sharp_score, 4)

            return FaceRegionTelemetry(
                face_present=True,
                visibility_score=round(min(0.85, 0.45 + skin_ratio), 4),
                quality_score=quality,
                approximate_orientation=orient,
                is_occluded=is_top_clipped,
                sharpness_score=round(sharpness, 2),
                resolution=(face_w, face_h),
                face_bbox=BoundingBox(x1=round(fx1, 2), y1=round(fy1, 2), x2=round(fx2, 2), y2=round(fy2, 2)),
            )

        return FaceRegionTelemetry(
            face_present=False,
            visibility_score=0.0,
            quality_score=0.0,
            approximate_orientation="unknown",
            is_occluded=is_top_clipped,
            sharpness_score=round(sharpness, 2),
            resolution=(0, 0),
            face_bbox=None,
        )

    def detect_in_person_crop(
        self,
        frame_bgr: np.ndarray,
        person_bbox: BoundingBox,
        timestamp: float,
        track_id: Optional[str] = None,
    ) -> List[FaceDetection]:
        """
        Detect visual face region within a detected person's upper body region.
        Preserves 100% backward compatibility with existing pipeline contracts.
        """
        telem = self.analyze_face_telemetry(frame_bgr, person_bbox, timestamp, track_id)
        if telem.face_present and telem.face_bbox:
            return [
                FaceDetection(
                    timestamp=timestamp,
                    bounding_box=telem.face_bbox,
                    confidence=telem.visibility_score,
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
