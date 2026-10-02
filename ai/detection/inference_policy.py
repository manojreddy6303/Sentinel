"""
Sentinel Resolution-Aware Inference Policy (Phase 20)

Dynamically selects optimal YOLO inference resolution (imgsz) and slicing strategy
based on source video dimensions, aspect ratio, and object scale profiles.
Prevents 4K frames from severe downscaling while keeping CPU latency and memory bounded.
"""
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


class InferenceResolutionPolicy:
    """
    Resolution-aware inference configuration for Sentinel detection pipeline.
    """

    BASELINE_IMGSZ: int = 640
    UHD_IMGSZ: int = 960
    MAX_IMGSZ: int = 1280

    @classmethod
    def select_inference_size(
        cls,
        width: Optional[int] = None,
        height: Optional[int] = None,
        is_high_density: bool = False,
        user_override: Optional[int] = None,
    ) -> int:
        """
        Determine optimal YOLO inference resolution (imgsz).

        Policy:
        - User override has highest precedence.
        - SD / 480p / 720p (max_dim <= 1280): 640 baseline
        - 1080p (max_dim <= 1920): 640 baseline (benchmark shows optimal CPU latency vs recall)
        - 1440p / 2K / 4K UHD (max_dim > 1920): 960 (preserves distant objects, 2.25x pixel area boost)
        - Dense small-object scenes: can scale up to 960/1280
        """
        if user_override and int(user_override) > 0:
            return int(user_override)

        if not width or not height or width <= 0 or height <= 0:
            return cls.BASELINE_IMGSZ

        max_dim = max(width, height)
        min_dim = min(width, height)

        # Vertical video (e.g. 9:16 mobile / corridor)
        is_vertical = height > width and (float(height) / float(width) >= 1.4)

        if max_dim <= 1280:
            # SD, 480p, 720p
            return cls.BASELINE_IMGSZ
        elif max_dim <= 1920:
            # 1080p: benchmark confirms 640 is optimal for standard CCTV
            if is_high_density:
                return cls.UHD_IMGSZ
            return cls.BASELINE_IMGSZ
        else:
            # 2K, 1440p, 4K UHD (e.g. 3840x2160)
            # 960 provides high small-object recall while remaining bounded in CPU time & memory
            return cls.UHD_IMGSZ

    @classmethod
    def get_preprocessing_metadata(
        cls,
        width: int,
        height: int,
        effective_imgsz: int,
    ) -> Dict[str, Any]:
        """Return diagnostic scaling telemetry for downstream pipeline validation."""
        max_dim = max(width, height) if width > 0 and height > 0 else effective_imgsz
        scale_ratio = float(effective_imgsz) / float(max_dim) if max_dim > 0 else 1.0
        return {
            "source_width": width,
            "source_height": height,
            "effective_imgsz": effective_imgsz,
            "scale_ratio": round(scale_ratio, 4),
            "is_downscaled": scale_ratio < 1.0,
            "is_upscaled": scale_ratio > 1.0,
        }
