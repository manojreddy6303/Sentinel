"""
Frame-Level Media Quality Gate (Phase 15.3)

Evaluates individual video frames before specialized visual inference to prevent
detectors from processing unusable media. Eliminates false positives caused by:

  - Motion-blur or focus-blur frames (compressed CCTV transitions)
  - Extremely dark frames (night-mode transitions, scene cuts to black)
  - Overexposed / blown-out frames (flash, direct sunlight, camera flare)
  - Micro-resolution frames (thumbnails, heavily downsampled crops)
  - Heavily compressed frames (JPEG macroblock artifact storms)

Design constraints:
  - Entirely stateless: no cross-frame or cross-video state is stored.
  - Returns a structured result so callers can log reasons for abstention.
  - All thresholds are named constants — never buried magic numbers.
  - cv2 is the only dependency (already used throughout the pipeline).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Threshold constants (tuned for CCTV / surveillance footage)
# ---------------------------------------------------------------------------

# Laplacian variance below this indicates motion blur or focus blur.
# CCTV streams at 1–5 fps can have sharp frames even at low fps;
# genuine blur produces variance < 10.0 reliably.
BLUR_LAPLACIAN_VARIANCE_THRESHOLD: float = 10.0

# Mean 8-bit pixel intensity below this → effectively a dark / black frame.
DARKNESS_MEAN_THRESHOLD: float = 18.0

# Mean 8-bit pixel intensity above this → blown-out / overexposed frame.
OVEREXPOSURE_MEAN_THRESHOLD: float = 245.0

# Minimum frame dimension (width or height in pixels).
# Frames smaller than this cannot yield reliable specialized visual evidence.
MIN_FRAME_DIMENSION_PX: int = 48

# Fraction of the frame that must not be uniform (std-dev gate).
# A frame where >95% of pixels share nearly the same value is a solid-colour
# transition frame (black-to-black cut, camera cap-on, etc.).
MIN_FRAME_STD_DEV: float = 4.0


@dataclass
class FrameQualityResult:
    """
    Structured result from FrameQualityGate.evaluate().

    Attributes:
        is_usable: True if the frame passes all quality checks.
        reasons:   Human-readable list of reasons for abstention (empty when usable).
        blur_variance: Laplacian variance of the frame (higher = sharper).
        mean_brightness: Mean 8-bit pixel intensity.
        std_dev: Standard deviation of pixel intensities.
        frame_shape: (height, width, channels) of the evaluated frame.
    """
    is_usable: bool
    reasons: List[str] = field(default_factory=list)
    blur_variance: float = 0.0
    mean_brightness: float = 0.0
    std_dev: float = 0.0
    frame_shape: tuple = field(default_factory=tuple)

    def __bool__(self) -> bool:
        return self.is_usable


class FrameQualityGate:
    """
    Stateless per-frame quality evaluator.

    Usage::

        result = FrameQualityGate.evaluate(frame_bgr)
        if not result.is_usable:
            logger.debug("Frame skipped: %s", result.reasons)
            return []

    All methods are class-methods — the gate holds no instance state.
    """

    @classmethod
    def evaluate(
        cls,
        frame: np.ndarray,
        blur_threshold: float = BLUR_LAPLACIAN_VARIANCE_THRESHOLD,
        darkness_threshold: float = DARKNESS_MEAN_THRESHOLD,
        overexposure_threshold: float = OVEREXPOSURE_MEAN_THRESHOLD,
        min_dimension: int = MIN_FRAME_DIMENSION_PX,
        min_std_dev: float = MIN_FRAME_STD_DEV,
    ) -> FrameQualityResult:
        """
        Evaluate a single BGR frame for usability by specialized detectors.

        Parameters
        ----------
        frame:
            A BGR numpy array as returned by cv2.VideoCapture.read().
        blur_threshold:
            Minimum Laplacian variance for the frame to be considered sharp.
        darkness_threshold:
            Minimum mean pixel value (0–255) for the frame to be considered
            sufficiently lit.
        overexposure_threshold:
            Maximum mean pixel value above which the frame is considered blown-out.
        min_dimension:
            Minimum frame width or height in pixels.
        min_std_dev:
            Minimum pixel standard deviation; below this the frame is a solid
            near-uniform colour (transition / dead frame).

        Returns
        -------
        FrameQualityResult
            Structured result; ``is_usable`` is ``True`` only when all checks pass.
        """
        reasons: List[str] = []

        # Guard: null / empty frame
        if frame is None or frame.size == 0:
            return FrameQualityResult(
                is_usable=False,
                reasons=["Null or empty frame array"],
            )

        shape = frame.shape
        h = shape[0]
        w = shape[1] if len(shape) > 1 else 1

        result = FrameQualityResult(
            is_usable=True,
            frame_shape=shape,
        )

        # 1. Resolution gate
        if h < min_dimension or w < min_dimension:
            reasons.append(
                f"Frame resolution ({w}x{h}px) is below minimum "
                f"({min_dimension}px); specialized inference not reliable."
            )

        # Compute grayscale metrics once for efficiency
        if len(shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        mean_mat, std_mat = cv2.meanStdDev(gray)
        mean_val = float(mean_mat[0][0])
        std_val = float(std_mat[0][0])
        result.mean_brightness = round(mean_val, 2)
        result.std_dev = round(std_val, 2)

        # 2. Darkness gate
        if mean_val < darkness_threshold:
            reasons.append(
                f"Frame is too dark (mean brightness {mean_val:.1f} < "
                f"{darkness_threshold}); specialized detectors would "
                "operate on noise rather than genuine visual content."
            )

        # 3. Overexposure gate
        if mean_val > overexposure_threshold:
            reasons.append(
                f"Frame is overexposed / blown-out (mean brightness "
                f"{mean_val:.1f} > {overexposure_threshold}); chroma "
                "channels are saturated and unreliable for color analysis."
            )

        # 4. Uniform / transition frame gate
        if std_val < min_std_dev:
            reasons.append(
                f"Frame has near-uniform pixel distribution "
                f"(std-dev {std_val:.2f} < {min_std_dev}); likely a "
                "transition, camera-cap-on, or solid-colour frame."
            )

        # 5. Blur gate — only if frame has sufficient contrast to measure
        if std_val >= min_std_dev and mean_val >= darkness_threshold:
            lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            result.blur_variance = round(lap_var, 3)
            if lap_var < blur_threshold:
                reasons.append(
                    f"Frame is blurry (Laplacian variance {lap_var:.2f} < "
                    f"{blur_threshold}); motion-blur or focus-blur degrades "
                    "specialized visual detector reliability."
                )
        else:
            # Cannot meaningfully assess blur on a dark/uniform frame
            result.blur_variance = 0.0

        result.reasons = reasons
        result.is_usable = len(reasons) == 0

        if not result.is_usable:
            logger.debug(
                "FrameQualityGate: frame abstained — %s",
                "; ".join(reasons),
            )

        return result
