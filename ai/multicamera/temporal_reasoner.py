"""
ai/multicamera/temporal_reasoner.py — Phase 18

Computes temporal plausibility for a cross-camera track association.

The core question: given that a track disappeared from Camera A at time T_exit,
is it plausible that it appeared in Camera B at time T_entry?

Validity rules:
  1. T_entry must be AFTER T_exit (no time-travel).
  2. The gap (T_entry - T_exit) should be positive and within a plausible range.
  3. Plausibility is highest when the gap matches typical transit time between
     adjacent cameras.

Default plausible gap range:   [1.0s, 300.0s]
  (subject needs at least 1s to move; beyond 5 min is unlikely same transit)

Adjacency-hinted cameras use a tighter preferred window:   [0.5s, 60.0s]
  (next-door cameras — subject should arrive within a minute)

PRIVACY: No identity signals. Gap scoring is purely temporal arithmetic.
"""

from __future__ import annotations
import logging
from typing import List, Optional, Tuple

from ai.multicamera.schemas import AssociationEvidence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default plausibility windows (seconds)
# ---------------------------------------------------------------------------
_MIN_PLAUSIBLE_GAP: float = 1.0      # too small → likely same camera glitch
_MAX_PLAUSIBLE_GAP: float = 300.0    # 5 minutes

_ADJACENT_MIN_GAP: float = 0.5
_ADJACENT_MAX_GAP: float = 60.0      # 1 minute for physically adjacent cameras

# Optimal gap (peak plausibility) for non-adjacent cameras
_OPTIMAL_GAP: float = 15.0           # 15 seconds is the most plausible transit
_ADJACENT_OPTIMAL_GAP: float = 5.0   # 5 seconds for adjacent cameras


class TemporalCompatibilityReasoner:
    """
    Scores the temporal plausibility of a cross-camera track reappearance.

    Usage:
        reasoner = TemporalCompatibilityReasoner()
        score, gap_seconds, evidence = reasoner.compute(
            source_last_seen=120.5,
            target_first_seen=135.0,
            cameras_are_adjacent=True,
        )
    """

    def __init__(
        self,
        min_gap: float = _MIN_PLAUSIBLE_GAP,
        max_gap: float = _MAX_PLAUSIBLE_GAP,
        adjacent_min_gap: float = _ADJACENT_MIN_GAP,
        adjacent_max_gap: float = _ADJACENT_MAX_GAP,
        optimal_gap: float = _OPTIMAL_GAP,
        adjacent_optimal_gap: float = _ADJACENT_OPTIMAL_GAP,
    ):
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.adjacent_min_gap = adjacent_min_gap
        self.adjacent_max_gap = adjacent_max_gap
        self.optimal_gap = optimal_gap
        self.adjacent_optimal_gap = adjacent_optimal_gap

    def compute(
        self,
        source_last_seen: Optional[float],
        target_first_seen: Optional[float],
        cameras_are_adjacent: bool = False,
    ) -> Tuple[float, Optional[float], AssociationEvidence]:
        """
        Compute temporal plausibility.

        Returns:
            (plausibility_score, gap_seconds, evidence)

        score = 0.0 on immediate disqualification (negative gap, missing data).
        """
        # Missing timestamps → neutral (cannot reason)
        if source_last_seen is None or target_first_seen is None:
            return 0.40, None, AssociationEvidence(
                signal_type="temporal_gap_plausibility",
                description="Missing timestamps — temporal reasoning unavailable",
                score=0.40,
                metadata={"source_last_seen": source_last_seen, "target_first_seen": target_first_seen},
            )

        gap = target_first_seen - source_last_seen

        # Hard disqualification: negative gap (target appeared before source left)
        if gap < 0.0:
            return 0.0, gap, AssociationEvidence(
                signal_type="temporal_gap_plausibility",
                description=(
                    f"Negative temporal gap: target appeared {abs(gap):.2f}s BEFORE "
                    f"source left frame — association impossible"
                ),
                score=0.0,
                metadata={
                    "source_last_seen": source_last_seen,
                    "target_first_seen": target_first_seen,
                    "gap_seconds": gap,
                    "reason": "negative_gap",
                },
            )

        # Zero gap (simultaneous) with non-overlapping cameras → impossible
        if gap == 0.0 and not cameras_are_adjacent:
            return 0.0, 0.0, AssociationEvidence(
                signal_type="temporal_gap_plausibility",
                description="Zero temporal gap for non-overlapping cameras — impossible transit",
                score=0.0,
                metadata={
                    "source_last_seen": source_last_seen,
                    "target_first_seen": target_first_seen,
                    "gap_seconds": 0.0,
                    "reason": "zero_gap_non_adjacent",
                },
            )

        # Select applicable window
        if cameras_are_adjacent:
            lo = self.adjacent_min_gap
            hi = self.adjacent_max_gap
            optimal = self.adjacent_optimal_gap
            window_label = "adjacent"
        else:
            lo = self.min_gap
            hi = self.max_gap
            optimal = self.optimal_gap
            window_label = "general"

        # Out-of-window → score 0.0
        if gap < lo or gap > hi:
            score = 0.0
            reason = "below_minimum" if gap < lo else "above_maximum"
            description = (
                f"Gap {gap:.2f}s outside {window_label} plausibility window "
                f"[{lo:.1f}s, {hi:.1f}s] — {reason}"
            )
        else:
            # Triangular plausibility: peak at optimal, linear falloff to edges
            score = _triangular_score(gap, lo, optimal, hi)
            description = (
                f"Gap {gap:.2f}s within {window_label} window "
                f"[{lo:.1f}s, {hi:.1f}s] optimal={optimal:.1f}s → score {score:.3f}"
            )
            reason = "within_window"

        adjacency_note = "adjacent cameras" if cameras_are_adjacent else "non-adjacent cameras"
        return score, gap, AssociationEvidence(
            signal_type="temporal_gap_plausibility",
            description=f"{description} ({adjacency_note})",
            score=score,
            metadata={
                "source_last_seen": source_last_seen,
                "target_first_seen": target_first_seen,
                "gap_seconds": gap,
                "window_lo": lo,
                "window_hi": hi,
                "optimal": optimal,
                "reason": reason,
                "cameras_are_adjacent": cameras_are_adjacent,
            },
        )


def _triangular_score(x: float, lo: float, peak: float, hi: float) -> float:
    """
    Triangular scoring function:
      - 0.0 at x <= lo or x >= hi
      - 1.0 at x == peak
      - Linear interpolation between lo→peak and peak→hi

    Clipped to [0.0, 1.0].
    """
    if x <= lo or x >= hi:
        return 0.0
    if x <= peak:
        return (x - lo) / (peak - lo) if peak != lo else 1.0
    else:
        return (hi - x) / (hi - peak) if hi != peak else 1.0
