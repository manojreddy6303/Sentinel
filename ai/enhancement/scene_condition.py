"""
Scene Condition and Illumination Profiling Engine (Phase 20)

Analyzes video frame lighting, contrast, saturation, blur, and noise characteristics
to determine scene condition (daylight, indoor, low-light, night/IR) and decide
whether adaptive preprocessing is warranted without modifying normal footage.
"""
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np


class SceneIlluminationType(str, Enum):
    DAYLIGHT = "daylight"
    INDOOR_NORMAL = "indoor_normal"
    LOW_LIGHT = "low_light"
    NIGHT_IR = "night_ir"
    OVEREXPOSED = "overexposed"
    DEGRADED_UNUSABLE = "degraded_unusable"


@dataclass
class SceneConditionReport:
    """Structured telemetry on frame lighting and visual quality."""
    illumination_type: SceneIlluminationType
    mean_luminance: float
    std_luminance: float
    mean_saturation: float
    blur_score: float
    noise_sigma: float
    needs_enhancement: bool
    is_usable: bool
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "illumination_type": self.illumination_type.value,
            "mean_luminance": round(self.mean_luminance, 2),
            "std_luminance": round(self.std_luminance, 2),
            "mean_saturation": round(self.mean_saturation, 2),
            "blur_score": round(self.blur_score, 2),
            "noise_sigma": round(self.noise_sigma, 2),
            "needs_enhancement": self.needs_enhancement,
            "is_usable": self.is_usable,
            "explanation": self.explanation,
        }


class SceneConditionAnalyzer:
    """
    Stateless evaluator for scene illumination and optical quality.
    Operates on downsampled thumbnails (<0.1ms per frame).
    """

    # Lighting classification thresholds
    DARK_LUMINANCE_THRESHOLD = 45.0
    LOW_LIGHT_THRESHOLD = 75.0
    OVEREXPOSED_THRESHOLD = 230.0
    IR_SATURATION_THRESHOLD = 25.0
    BLUR_LAPLACIAN_MIN = 8.0

    @classmethod
    def analyze(cls, frame_bgr: np.ndarray) -> SceneConditionReport:
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3:
            return SceneConditionReport(
                illumination_type=SceneIlluminationType.DEGRADED_UNUSABLE,
                mean_luminance=0.0,
                std_luminance=0.0,
                mean_saturation=0.0,
                blur_score=0.0,
                noise_sigma=0.0,
                needs_enhancement=False,
                is_usable=False,
                explanation="Invalid or empty frame buffer",
            )

        h, w = frame_bgr.shape[:2]
        if h < 32 or w < 32:
            return SceneConditionReport(
                illumination_type=SceneIlluminationType.DEGRADED_UNUSABLE,
                mean_luminance=0.0,
                std_luminance=0.0,
                mean_saturation=0.0,
                blur_score=0.0,
                noise_sigma=0.0,
                needs_enhancement=False,
                is_usable=False,
                explanation="Sub-minimum frame resolution",
            )

        # Ultra-fast downsampled thumbnail analysis (160x90)
        thumb = cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)

        mean_lum = float(np.mean(gray))
        std_lum = float(np.std(gray))
        mean_sat = float(np.mean(hsv[:, :, 1]))

        # Blur estimation via Laplacian variance
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = float(lap.var())

        # Noise estimation via median filter difference
        denoised_med = cv2.medianBlur(gray, 3)
        noise_diff = np.abs(gray.astype(np.float32) - denoised_med.astype(np.float32))
        noise_sigma = float(np.mean(noise_diff))

        # Classification logic
        if mean_lum < 15.0 and std_lum < 5.0:
            ill_type = SceneIlluminationType.DEGRADED_UNUSABLE
            needs_enh = False
            usable = False
            expl = "Severely underexposed or black frame"
        elif mean_lum > cls.OVEREXPOSED_THRESHOLD:
            ill_type = SceneIlluminationType.OVEREXPOSED
            needs_enh = False
            usable = True
            expl = "High overexposure / optical glare"
        elif mean_sat < cls.IR_SATURATION_THRESHOLD and mean_lum < 120.0:
            ill_type = SceneIlluminationType.NIGHT_IR
            needs_enh = True
            usable = True
            expl = "Monochromatic night vision / IR camera mode"
        elif mean_lum < cls.DARK_LUMINANCE_THRESHOLD:
            ill_type = SceneIlluminationType.LOW_LIGHT
            needs_enh = True
            usable = True
            expl = "Low-light / nighttime surveillance scene"
        elif mean_lum < cls.LOW_LIGHT_THRESHOLD and std_lum < 35.0:
            ill_type = SceneIlluminationType.LOW_LIGHT
            needs_enh = True
            usable = True
            expl = "Dim lighting with low dynamic range"
        elif mean_lum > 140.0 and mean_sat > 45.0:
            ill_type = SceneIlluminationType.DAYLIGHT
            needs_enh = False
            usable = True
            expl = "Standard well-lit daylight conditions"
        else:
            ill_type = SceneIlluminationType.INDOOR_NORMAL
            needs_enh = False
            usable = True
            expl = "Standard indoor lighting conditions"

        return SceneConditionReport(
            illumination_type=ill_type,
            mean_luminance=mean_lum,
            std_luminance=std_lum,
            mean_saturation=mean_sat,
            blur_score=blur_score,
            noise_sigma=noise_sigma,
            needs_enhancement=needs_enh,
            is_usable=usable,
            explanation=expl,
        )
