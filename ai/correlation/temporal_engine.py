"""
ai/correlation/temporal_engine.py
=================================
Temporal correlation and interval calculus for Phase 16 incident fusion.
"""

from typing import Tuple, Optional
from ai.correlation.models import TemporalRelationType
from ai.common.numeric import ensure_finite


class CorrelationTemporalEngine:
    """
    Computes rigorous temporal interval relationships between candidates and tracks.
    """

    def __init__(self, max_temporal_gap: float = 3.0):
        self.max_temporal_gap = max_temporal_gap

    def determine_relation(
        self,
        start_a: float,
        end_a: float,
        start_b: float,
        end_b: float,
    ) -> TemporalRelationType:
        return self.evaluate_relation(start_a, end_a, start_b, end_b, self.max_temporal_gap)

    def are_temporally_compatible(
        self,
        start_a: float,
        end_a: float,
        start_b: float,
        end_b: float,
    ) -> bool:
        rel = self.determine_relation(start_a, end_a, start_b, end_b)
        return rel != TemporalRelationType.SEPARATED

    @staticmethod
    def evaluate_relation(
        start_a: float,
        end_a: float,
        start_b: float,
        end_b: float,
        adjacency_tolerance_s: float = 3.0,
    ) -> TemporalRelationType:
        """
        Classify the temporal relationship between interval A and interval B.
        """
        s_a = ensure_finite(start_a, 0.0)
        e_a = max(s_a, ensure_finite(end_a, s_a))
        s_b = ensure_finite(start_b, 0.0)
        e_b = max(s_b, ensure_finite(end_b, s_b))

        # Check overlap
        overlap = not (e_a < s_b or e_b < s_a)
        if overlap:
            # If start and end are practically identical, CONCURRENT
            if abs(s_a - s_b) <= 0.5 and abs(e_a - e_b) <= 0.5:
                return TemporalRelationType.CONCURRENT
            return TemporalRelationType.OVERLAPPING

        # A precedes B
        if e_a <= s_b:
            gap = s_b - e_a
            if gap <= adjacency_tolerance_s:
                return TemporalRelationType.IMMEDIATELY_PRECEDING
            return TemporalRelationType.SEPARATED

        # B precedes A
        gap = s_a - e_b
        if gap <= adjacency_tolerance_s:
            return TemporalRelationType.FOLLOWING
        return TemporalRelationType.SEPARATED

    @staticmethod
    def compute_temporal_envelope(
        intervals: list,
    ) -> Tuple[float, float, float]:
        """
        Returns (min_start, max_end, duration).
        """
        if not intervals:
            return 0.0, 0.0, 0.0
        min_s = min(ensure_finite(s, 0.0) for s, e in intervals)
        max_e = max(ensure_finite(e, 0.0) for s, e in intervals)
        dur = max(0.0, max_e - min_s)
        return min_s, max_e, dur

    @staticmethod
    def calculate_temporal_gap(
        start_a: float,
        end_a: float,
        start_b: float,
        end_b: float,
    ) -> float:
        """
        Computes absolute temporal gap between two intervals in seconds.
        Returns 0.0 if intervals overlap.
        """
        s_a = ensure_finite(start_a, 0.0)
        e_a = max(s_a, ensure_finite(end_a, s_a))
        s_b = ensure_finite(start_b, 0.0)
        e_b = max(s_b, ensure_finite(end_b, s_b))

        # Check overlap
        if not (e_a < s_b or e_b < s_a):
            return 0.0

        if e_a < s_b:
            return s_b - e_a
        return s_a - e_b

    @classmethod
    def is_temporally_continuous(
        cls,
        start_a: float,
        end_a: float,
        start_b: float,
        end_b: float,
        max_gap_seconds: float = 5.0,
    ) -> bool:
        """
        Evaluates whether two intervals are temporally contiguous or overlapping within max_gap_seconds.
        Guarantees that observation density does not fragment continuous episodes.
        """
        gap = cls.calculate_temporal_gap(start_a, end_a, start_b, end_b)
        return gap <= max_gap_seconds

    @staticmethod
    def adaptive_temporal_gap(sample_rate_fps: float = 1.0, baseline_gap: float = 5.0) -> float:
        """
        Principled temporal gap calculation invariant to sampling rate.
        Maintains constant real-time interval semantics across 1 FPS, 3 FPS, and VFR streams.
        """
        return float(baseline_gap)

