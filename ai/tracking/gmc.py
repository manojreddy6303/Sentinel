"""
Global Motion Compensation (GMC) for Camera Motion Handling (Phase 21A)

Estimates inter-frame camera motion using sparse optical flow (Lucas-Kanade)
and fits an affine transformation via RANSAC. Used to warp predicted bounding
box positions before association, compensating for camera pan/tilt/zoom.

Falls back gracefully to identity transform on:
- First frame (no previous reference)
- Insufficient feature points
- Failed optical flow
- Failed RANSAC affine estimation
- OpenCV errors

Memory-safe: downsamples large frames (>1280px) for optical flow computation.
Stores only one previous grayscale frame (~1-2 MB for 1280px max).

SAFETY: No biometrics, no identity inference — purely geometric camera motion.
"""
import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class GlobalMotionCompensation:
    """
    Estimates and applies camera motion compensation between consecutive frames.

    Uses Shi-Tomasi corner detection + Lucas-Kanade optical flow to estimate
    a 2D partial affine transformation (rotation + translation + scale),
    then warps predicted bounding box positions.
    """

    def __init__(
        self,
        max_corners: int = 200,
        quality_level: float = 0.01,
        min_distance: int = 30,
        block_size: int = 3,
        min_inliers: int = 10,
        ransac_threshold: float = 3.0,
        downsample_max: int = 1280,
    ):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        self.min_inliers = min_inliers
        self.ransac_threshold = ransac_threshold
        self.downsample_max = downsample_max

        self._prev_gray: Optional[np.ndarray] = None
        self._scale: float = 1.0  # downsample scale factor

        # Lucas-Kanade optical flow parameters
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        # Shi-Tomasi corner detection parameters
        self._feature_params = dict(
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
        )

    def reset(self) -> None:
        """Reset internal state (call between videos)."""
        self._prev_gray = None
        self._scale = 1.0

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Estimate the affine transformation from the previous frame to the current frame.

        Args:
            frame_bgr: Current frame in BGR format. Can be None.

        Returns:
            2x3 affine matrix mapping prev-frame coords to current-frame coords.
            Returns identity matrix if estimation fails or no previous frame exists.
        """
        identity = np.eye(2, 3, dtype=np.float64)

        if frame_bgr is None:
            return identity

        # Convert to grayscale
        try:
            if len(frame_bgr.shape) == 3:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame_bgr.copy()
        except Exception:
            return identity

        # Downsample large frames for performance
        h, w = gray.shape[:2]
        self._scale = 1.0
        if max(h, w) > self.downsample_max:
            self._scale = self.downsample_max / max(h, w)
            gray = cv2.resize(
                gray, None, fx=self._scale, fy=self._scale,
                interpolation=cv2.INTER_AREA,
            )

        # First frame: just store and return identity
        if self._prev_gray is None:
            self._prev_gray = gray
            return identity

        # Ensure dimensions match (resolution might change between frames)
        if self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return identity

        # Detect features in previous frame
        prev_keypoints = cv2.goodFeaturesToTrack(self._prev_gray, **self._feature_params)
        if prev_keypoints is None or len(prev_keypoints) < self.min_inliers:
            self._prev_gray = gray
            return identity

        # Compute optical flow
        try:
            matched_kp, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, prev_keypoints, None, **self._lk_params
            )
        except cv2.error:
            self._prev_gray = gray
            return identity

        if matched_kp is None or status is None:
            self._prev_gray = gray
            return identity

        # Filter by tracking status
        mask = status.flatten().astype(bool)
        prev_pts = prev_keypoints[mask].reshape(-1, 2)
        curr_pts = matched_kp[mask].reshape(-1, 2)

        if len(prev_pts) < self.min_inliers:
            self._prev_gray = gray
            return identity

        # Estimate partial affine (rotation + translation + uniform scale) with RANSAC
        try:
            affine, inlier_mask = cv2.estimateAffinePartial2D(
                prev_pts, curr_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold,
            )
        except cv2.error:
            self._prev_gray = gray
            return identity

        if affine is None:
            self._prev_gray = gray
            return identity

        num_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        if num_inliers < self.min_inliers:
            logger.debug(
                "GMC: insufficient inliers (%d < %d), using identity",
                num_inliers, self.min_inliers,
            )
            self._prev_gray = gray
            return identity

        # Scale translation back to original resolution if downsampled
        if self._scale != 1.0:
            affine[0, 2] /= self._scale
            affine[1, 2] /= self._scale

        # Update state for next frame
        self._prev_gray = gray
        return affine

    @staticmethod
    def apply_affine_to_tlbr(
        affine: np.ndarray,
        x1: float, y1: float, x2: float, y2: float,
    ) -> Tuple[float, float, float, float]:
        """
        Apply affine transformation to a bounding box [x1, y1, x2, y2].
        Transforms center point and preserves width/height.

        Returns:
            (new_x1, new_y1, new_x2, new_y2)
        """
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1

        # Transform center point
        pt = np.array([cx, cy, 1.0], dtype=np.float64)
        new_cx = float(affine[0] @ pt)
        new_cy = float(affine[1] @ pt)

        return (new_cx - w / 2.0, new_cy - h / 2.0, new_cx + w / 2.0, new_cy + h / 2.0)

    @staticmethod
    def apply_affine_to_cxcywh(
        affine: np.ndarray,
        cx: float, cy: float, w: float, h: float,
    ) -> Tuple[float, float, float, float]:
        """
        Apply affine transformation to center-format bbox [cx, cy, w, h].
        Transforms center point and preserves width/height.

        Returns:
            (new_cx, new_cy, w, h)
        """
        pt = np.array([cx, cy, 1.0], dtype=np.float64)
        new_cx = float(affine[0] @ pt)
        new_cy = float(affine[1] @ pt)
        return (new_cx, new_cy, w, h)

    @staticmethod
    def is_identity(affine: np.ndarray, tol: float = 1e-6) -> bool:
        """Check if an affine matrix is approximately the identity transform."""
        identity = np.eye(2, 3, dtype=np.float64)
        return bool(np.allclose(affine, identity, atol=tol))
