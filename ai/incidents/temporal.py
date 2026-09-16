"""
Temporal Analysis Engine (Phase 10)

Provides reusable temporal reasoning utilities:
- event windows & observation persistence
- onset detection & disappearance analysis
- temporal ordering & before/after relationship validation
- state dwell duration measurement
- sudden behavioral change detection (abrupt stops, speed spikes, directional reversals)
"""
from typing import List, Dict, Any, Optional, Tuple, Callable
import math

from ai.schemas import TrackedObject
from ai.incidents.schemas import TrackMotion


class TemporalAnalysisEngine:
    """
    Time-series and temporal sequence analysis for Sentinel incident intelligence.
    """

    @staticmethod
    def is_temporally_overlapping(
        t1_start: float, t1_end: float,
        t2_start: float, t2_end: float,
        tolerance_seconds: float = 0.5,
    ) -> bool:
        """Check if two time intervals overlap within an optional tolerance window."""
        return not (
            t1_end + tolerance_seconds < t2_start or
            t2_end + tolerance_seconds < t1_start
        )

    @staticmethod
    def temporal_intersection(
        t1_start: float, t1_end: float,
        t2_start: float, t2_end: float,
    ) -> Optional[Tuple[float, float]]:
        """Return the overlapping sub-interval [start, end] or None."""
        start = max(t1_start, t2_start)
        end = min(t1_end, t2_end)
        if start <= end:
            return (start, end)
        return None

    @staticmethod
    def verify_temporal_sequence(
        events: List[Dict[str, Any]],
        time_key: str = "timestamp",
        max_gap_seconds: float = 10.0,
    ) -> bool:
        """
        Verify that a list of sequential stages occurs in strictly non-decreasing chronological order
        with no gap exceeding max_gap_seconds.
        """
        if len(events) < 2:
            return True

        for i in range(len(events) - 1):
            t1 = float(events[i].get(time_key, 0.0))
            t2 = float(events[i + 1].get(time_key, 0.0))
            if t2 < t1:
                return False
            if (t2 - t1) > max_gap_seconds:
                return False

        return True

    @staticmethod
    def detect_disappearance(
        target_track: TrackedObject,
        reference_track: TrackedObject,
        loss_window_seconds: float = 4.0,
    ) -> bool:
        """
        Check if `target_track` disappeared (stopped being detected) while `reference_track`
        continued to exist in the scene.
        """
        if not target_track or not reference_track:
            return False

        # Target must terminate before or shortly after reference
        if target_track.last_seen > reference_track.last_seen:
            return False

        # Reference must outlast the target by at least a fraction of the loss window
        outlast_duration = reference_track.last_seen - target_track.last_seen
        return outlast_duration >= 0.5 and (reference_track.first_seen <= target_track.last_seen + loss_window_seconds)

    @staticmethod
    def detect_sudden_deceleration(
        motions: List[TrackMotion],
        deceleration_threshold: float = 100.0,  # pixels/sec^2
        window_seconds: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """
        Identify moments where a track undergoes abrupt deceleration (e.g. sudden braking / impact).
        """
        sudden_stops: List[Dict[str, Any]] = []
        if len(motions) < 2:
            return sudden_stops

        for i in range(1, len(motions)):
            m = motions[i]
            prev = motions[i - 1]
            dt = max(1e-3, m.timestamp - prev.timestamp)
            if dt > window_seconds:
                continue

            # Deceleration is negative acceleration
            if m.acceleration_estimate <= -deceleration_threshold or (prev.velocity_estimate - m.velocity_estimate) >= 80.0:
                sudden_stops.append({
                    "timestamp": m.timestamp,
                    "prior_velocity": prev.velocity_estimate,
                    "post_velocity": m.velocity_estimate,
                    "speed_drop": prev.velocity_estimate - m.velocity_estimate,
                    "acceleration": m.acceleration_estimate,
                    "position": m.position,
                })

        return sudden_stops

    @staticmethod
    def calculate_persistence(
        timestamps: List[float],
        expected_interval: float = 1.0,
    ) -> float:
        """
        Measure the temporal continuity / persistence score (0.0 to 1.0) of observations.
        Penalizes large gaps.
        """
        if len(timestamps) < 2:
            return 1.0

        sorted_ts = sorted(timestamps)
        total_span = max(1e-3, sorted_ts[-1] - sorted_ts[0])
        expected_count = max(1, int(round(total_span / expected_interval)) + 1)
        actual_count = len(sorted_ts)

        score = min(1.0, actual_count / expected_count)
        return round(score, 4)
