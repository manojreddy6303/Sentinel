"""
Object Tracking Module

Provides:
- ObjectTracker: ByteTrack-style tracker (production default)
- BoTSORTTracker: BoT-SORT camera-motion compensated tracker (Phase 21A candidate)
- KalmanFilter: 8-dim state estimation for bounding box tracking
- GlobalMotionCompensation: sparse optical flow camera motion estimator
"""
from ai.tracking.tracker import ObjectTracker
from ai.tracking.botsort_tracker import BoTSORTTracker
from ai.tracking.kalman_filter import KalmanFilter
from ai.tracking.gmc import GlobalMotionCompensation

__all__ = ["ObjectTracker", "BoTSORTTracker", "KalmanFilter", "GlobalMotionCompensation"]
