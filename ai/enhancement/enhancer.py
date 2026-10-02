"""
Adaptive Low-Light Video Enhancement Engine (Phase 20)

Provides controlled illumination and contrast enhancement for low-light / night
surveillance footage to boost object detector recall without distorting normal daylight.

SAFETY INVARIANTS:
1. Non-destructive: Original frame is NEVER modified in-place or replaced as authoritative evidence.
2. Color-preservation: CLAHE is applied exclusively to the L (Luminance) channel in CIE-LAB space;
   chromatic channels (A, B) are preserved to avoid synthetic color artifacts.
3. Selective activation: Only runs when SceneConditionAnalyzer determines `needs_enhancement == True`.
4. Detection consistency: Derived detections must have underlying physical edges in original frame.
"""
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
from ai.enhancement.scene_condition import SceneConditionReport, SceneConditionAnalyzer


class AdaptiveLowLightEnhancer:
    """
    Adaptive low-light and night vision enhancement engine for surveillance video.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        enabled: bool = True,
    ):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.enabled = enabled
        self._clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def enhance_if_needed(
        self,
        frame_bgr: np.ndarray,
        report: Optional[SceneConditionReport] = None,
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """
        Conditionally enhance frame if low-light or underexposed conditions are detected.

        Returns:
            (processed_frame, was_enhanced, telemetry_metadata)
            If enhancement not needed, returns (original_frame, False, metadata) directly.
        """
        if not self.enabled or frame_bgr is None:
            return frame_bgr, False, {"enhanced": False, "reason": "disabled_or_none"}

        if report is None:
            report = SceneConditionAnalyzer.analyze(frame_bgr)

        if not report.needs_enhancement or not report.is_usable:
            return frame_bgr, False, {
                "enhanced": False,
                "illumination_type": report.illumination_type.value,
                "reason": "illumination_adequate",
            }

        # Safe non-destructive enhancement
        enhanced_bgr = self._apply_clahe_lab(frame_bgr, report)

        telemetry = {
            "enhanced": True,
            "illumination_type": report.illumination_type.value,
            "original_mean_luminance": report.mean_luminance,
            "clip_limit": self.clip_limit,
            "applied_technique": "LAB_CLAHE_GAMMA",
        }
        return enhanced_bgr, True, telemetry

    def _apply_clahe_lab(self, frame_bgr: np.ndarray, report: SceneConditionReport) -> np.ndarray:
        """Apply CLAHE to L channel in LAB color space with gentle adaptive gamma."""
        # Convert BGR to LAB
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to luminance
        l_clahe = self._clahe.apply(l)

        # Gentle adaptive gamma correction for severe low-light (mean_lum < 50)
        if report.mean_luminance < 50.0:
            gamma = max(0.60, min(0.85, 0.50 + (report.mean_luminance / 150.0)))
            table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            l_clahe = cv2.LUT(l_clahe, table)

        # Recombine with original color channels
        merged_lab = cv2.merge((l_clahe, a, b))
        enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        return enhanced_bgr

    @staticmethod
    def verify_detection_consistency(
        original_frame: np.ndarray,
        bbox_dict: Dict[str, float],
        min_contrast_threshold: float = 4.0,
    ) -> bool:
        """
        Verify that a candidate detection found in an enhanced frame corresponds
        to genuine physical image features rather than amplified sensor noise.
        """
        if original_frame is None:
            return True

        h, w = original_frame.shape[:2]
        x1 = max(0, int(round(bbox_dict.get("x1", 0.0))))
        y1 = max(0, int(round(bbox_dict.get("y1", 0.0))))
        x2 = min(w, int(round(bbox_dict.get("x2", 0.0))))
        y2 = min(h, int(round(bbox_dict.get("y2", 0.0))))

        if x2 <= x1 or y2 <= y1:
            return False

        roi = original_frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        contrast = float(np.std(gray_roi))
        return contrast >= min_contrast_threshold
