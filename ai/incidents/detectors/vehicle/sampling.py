"""
Adaptive Temporal Sampling Engine (Sentinel Phase 11)

Provides on-demand localized burst analysis for high-speed vehicle events.
Distinguishes between scenarios with sufficient temporal evidence and those
with temporal resolution limitations (TEMPORAL_EVIDENCE_LIMITED).
"""
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AdaptiveTemporalSamplingEngine:
    """
    Manages dynamic burst temporal sampling for high-velocity or rapid-approach incidents.
    Avoids processing entire videos at high FPS by targeting localized temporal windows.
    """

    def __init__(
        self,
        base_sampling_fps: float = 1.0,
        burst_sampling_rates: Optional[List[int]] = None,
        min_rapid_approach_rate: float = 35.0,
        min_proximity_threshold: float = 70.0,
    ):
        self.base_sampling_fps = base_sampling_fps
        self.burst_sampling_rates = burst_sampling_rates or [5, 10, 15, 30]
        self.min_rapid_approach_rate = min_rapid_approach_rate
        self.min_proximity_threshold = min_proximity_threshold

    def evaluate_burst_trigger(
        self,
        interaction_metrics: Dict[str, Any],
    ) -> Tuple[bool, Optional[int], Optional[Tuple[float, float]]]:
        """
        Determines if an interaction triggers localized burst analysis.
        Returns: (needs_burst, recommended_fps, burst_window_seconds)
        """
        approach_rate = interaction_metrics.get("max_approach_rate_px_s", 0.0)
        min_dist = interaction_metrics.get("min_distance_px", float("inf"))
        event_time = interaction_metrics.get("event_time", 0.0)

        # Trigger condition: high approach rate + close proximity
        if approach_rate >= self.min_rapid_approach_rate and min_dist <= self.min_proximity_threshold:
            # Pick recommended burst FPS based on velocity
            if approach_rate > 80.0:
                rec_fps = 30
            elif approach_rate > 50.0:
                rec_fps = 15
            else:
                rec_fps = 10

            burst_window = (max(0.0, event_time - 1.5), event_time + 1.5)
            return True, rec_fps, burst_window

        return False, None, None

    def assess_temporal_sufficiency(
        self,
        actual_sample_rate_fps: float,
        target_velocity: float,
        contact_duration_seconds: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Checks if the available sampling rate provides sufficient temporal Nyquist
        coverage for the observed physical speed.
        """
        # A vehicle moving at 100 px/s sampled at 1 FPS covers 100 pixels between frames.
        # If sampling cannot guarantee at least 2 observations during convergence,
        # it is marked TEMPORAL_EVIDENCE_LIMITED.
        displacement_per_frame = target_velocity / max(1.0, actual_sample_rate_fps)

        is_limited = False
        limitation_reason = None

        if actual_sample_rate_fps < 5.0 and target_velocity > 40.0:
            is_limited = True
            limitation_reason = (
                f"Sampling rate ({actual_sample_rate_fps:.1f} FPS) is insufficient for high-speed motion "
                f"({target_velocity:.1f}px/s, ~{displacement_per_frame:.0f}px displacement/frame); "
                f"high-speed contact cannot be confirmed with temporal certainty."
            )

        return {
            "is_limited": is_limited,
            "temporal_evidence_tag": "TEMPORAL_EVIDENCE_LIMITED" if is_limited else "TEMPORAL_EVIDENCE_SUFFICIENT",
            "limitation_reason": limitation_reason,
            "sample_rate_fps": actual_sample_rate_fps,
            "displacement_per_frame_px": round(displacement_per_frame, 1),
        }
