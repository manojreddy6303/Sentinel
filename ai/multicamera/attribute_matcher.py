"""
ai/multicamera/attribute_matcher.py — Phase 18

Computes observable-attribute similarity between two CameraObservation instances.

All scoring is based solely on:
  - Object class identity
  - Color vocabulary match
  - Bounding-box size-ratio proximity
  - Trajectory direction compatibility (exit vs entry angle)

PRIVACY: No biometric features, no facial data, no identity signals.
Correct abstention (returning 0.0 on cross-class pairs) is enforced at the
class-match gate: no association is possible between different object classes.
"""

from __future__ import annotations
import math
import logging
from typing import List, Optional, Tuple

from ai.multicamera.schemas import CameraObservation, AssociationEvidence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color vocabulary
# ---------------------------------------------------------------------------
# Controlled vocabulary with perceptual group mapping for fuzzy color matching.
# E.g. "dark_blue" and "navy" belong to the same group.
_COLOR_GROUPS: dict[str, str] = {
    "red": "red", "dark_red": "red", "maroon": "red",
    "blue": "blue", "dark_blue": "blue", "navy": "blue", "light_blue": "blue",
    "green": "green", "dark_green": "green", "olive": "green",
    "black": "black", "dark": "black",
    "white": "white", "light": "white", "cream": "white", "silver": "silver",
    "grey": "grey", "gray": "grey", "dark_grey": "grey",
    "yellow": "yellow", "gold": "yellow",
    "orange": "orange", "brown": "brown",
    "purple": "purple", "violet": "purple",
    "pink": "pink",
    "unknown": "unknown",
}

_SAME_COLOR_SCORE = 1.0
_SAME_GROUP_SCORE = 0.70
_DIFFERENT_COLOR_SCORE = 0.0
_UNKNOWN_COLOR_SCORE = 0.30   # neither penalise nor reward unknown color

# ---------------------------------------------------------------------------
# Size-ratio tolerance
# ---------------------------------------------------------------------------
_SIZE_RATIO_TOLERANCE = 0.40   # 40% relative difference → score 0; 0% → score 1

# ---------------------------------------------------------------------------
# Direction compatibility
# ---------------------------------------------------------------------------
# Angular delta within this range is considered compatible
_DIRECTION_COMPAT_THRESHOLD_DEG = 60.0


def _color_group(color: Optional[str]) -> str:
    if color is None:
        return "unknown"
    return _COLOR_GROUPS.get(color.lower().replace(" ", "_"), "unknown")


def _angular_delta(a_deg: float, b_deg: float) -> float:
    """Smallest unsigned angular difference between two compass bearings (0–360)."""
    delta = abs(a_deg - b_deg) % 360.0
    if delta > 180.0:
        delta = 360.0 - delta
    return delta


class AttributeMatcher:
    """
    Computes observable-attribute similarity between two CameraObservation records.

    Usage:
        matcher = AttributeMatcher()
        score, evidence = matcher.compute(obs_source, obs_target)
        # score ∈ [0.0, 1.0]; evidence is List[AssociationEvidence]
    """

    def compute(
        self,
        source: CameraObservation,
        target: CameraObservation,
    ) -> Tuple[float, List[AssociationEvidence]]:
        """
        Compute attribute match score between source and target observations.

        Returns:
            (attribute_match_score, evidence_signals)

        Returns (0.0, []) immediately if object classes differ — no association
        is possible across classes.
        """
        evidence: List[AssociationEvidence] = []

        # ----------------------------------------------------------------
        # Gate 1: Object class must match (hard requirement)
        # ----------------------------------------------------------------
        if source.object_class.lower() != target.object_class.lower():
            return 0.0, []

        evidence.append(AssociationEvidence(
            signal_type="object_class_match",
            description=f"Both tracks classified as '{source.object_class}'",
            score=1.0,
            metadata={"class": source.object_class},
        ))

        # ----------------------------------------------------------------
        # Signal 2: Color similarity
        # ----------------------------------------------------------------
        color_score, color_ev = self._score_color(source, target)
        evidence.append(color_ev)

        # ----------------------------------------------------------------
        # Signal 3: Bounding-box size-ratio proximity
        # ----------------------------------------------------------------
        size_score, size_ev = self._score_size_ratio(source, target)
        if size_ev is not None:
            evidence.append(size_ev)

        # ----------------------------------------------------------------
        # Signal 4: Trajectory direction compatibility
        # ----------------------------------------------------------------
        dir_score, dir_ev = self._score_direction(source, target)
        if dir_ev is not None:
            evidence.append(dir_ev)

        # ----------------------------------------------------------------
        # Weighted aggregate
        # Weights: color 0.40, size 0.20, direction 0.40
        # (class match is a gate, not a weight contributor)
        # ----------------------------------------------------------------
        weights = {
            "color": 0.40,
            "size": 0.20,
            "direction": 0.40,
        }
        scores = {
            "color": color_score,
            "size": size_score,
            "direction": dir_score,
        }

        # If direction signal is unavailable, redistribute its weight to color
        if dir_ev is None:
            weights["color"] += weights["direction"]
            weights["direction"] = 0.0
        if size_ev is None:
            weights["color"] += weights["size"]
            weights["size"] = 0.0

        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0, evidence

        aggregate = sum(scores[k] * weights[k] for k in weights) / total_weight
        aggregate = max(0.0, min(1.0, aggregate))

        return aggregate, evidence

    # ------------------------------------------------------------------
    # Internal scoring helpers
    # ------------------------------------------------------------------

    def _score_color(
        self,
        source: CameraObservation,
        target: CameraObservation,
    ) -> Tuple[float, AssociationEvidence]:
        sc = source.color
        tc = target.color

        # Unknown on either side → neutral score
        if sc is None or tc is None:
            return _UNKNOWN_COLOR_SCORE, AssociationEvidence(
                signal_type="color_match",
                description="Color unknown for one or both tracks — neutral",
                score=_UNKNOWN_COLOR_SCORE,
                metadata={"source_color": sc, "target_color": tc},
            )

        if sc.lower() == tc.lower():
            return _SAME_COLOR_SCORE, AssociationEvidence(
                signal_type="color_match",
                description=f"Exact color match: '{sc}'",
                score=_SAME_COLOR_SCORE,
                metadata={"source_color": sc, "target_color": tc},
            )

        sg, tg = _color_group(sc), _color_group(tc)
        if sg != "unknown" and sg == tg:
            return _SAME_GROUP_SCORE, AssociationEvidence(
                signal_type="color_match",
                description=f"Same color group: '{sc}' ≈ '{tc}'",
                score=_SAME_GROUP_SCORE,
                metadata={"source_color": sc, "target_color": tc, "group": sg},
            )

        return _DIFFERENT_COLOR_SCORE, AssociationEvidence(
            signal_type="color_match",
            description=f"Color mismatch: '{sc}' vs '{tc}'",
            score=_DIFFERENT_COLOR_SCORE,
            metadata={"source_color": sc, "target_color": tc},
        )

    def _score_size_ratio(
        self,
        source: CameraObservation,
        target: CameraObservation,
    ) -> Tuple[float, Optional[AssociationEvidence]]:
        sh = source.estimated_height_ratio
        th = target.estimated_height_ratio
        if sh is None or th is None or sh <= 0 or th <= 0:
            return 0.50, None   # neutral, evidence not appended

        ratio = abs(sh - th) / max(sh, th)
        score = max(0.0, 1.0 - ratio / _SIZE_RATIO_TOLERANCE)

        return score, AssociationEvidence(
            signal_type="size_ratio_match",
            description=(
                f"Height ratio proximity: src={sh:.3f} tgt={th:.3f} "
                f"delta={ratio:.3f} → score {score:.3f}"
            ),
            score=score,
            metadata={"source_height_ratio": sh, "target_height_ratio": th, "relative_delta": ratio},
        )

    def _score_direction(
        self,
        source: CameraObservation,
        target: CameraObservation,
    ) -> Tuple[float, Optional[AssociationEvidence]]:
        """
        Exit direction of source should be roughly opposite to entry direction
        of target if cameras face each other, or similar if cameras form a path.

        We compute compatibility as: angular_delta ≤ threshold → high score.
        We test both:
          (a) direct continuation: exit ≈ entry (same heading)
          (b) reflective continuation: exit ≈ 180° + entry (walked toward camera)

        Best of the two is used.
        """
        ex = source.exit_direction_degrees
        en = target.entry_direction_degrees
        if ex is None or en is None:
            return 0.50, None  # neutral, evidence not appended

        delta_direct = _angular_delta(ex, en)
        delta_reflect = _angular_delta(ex, (en + 180.0) % 360.0)
        best_delta = min(delta_direct, delta_reflect)
        mode = "direct" if delta_direct < delta_reflect else "reflective"

        score = max(0.0, 1.0 - best_delta / _DIRECTION_COMPAT_THRESHOLD_DEG)

        return score, AssociationEvidence(
            signal_type="trajectory_direction_compatibility",
            description=(
                f"Exit {ex:.1f}° → Entry {en:.1f}° "
                f"({mode} delta {best_delta:.1f}°) → score {score:.3f}"
            ),
            score=score,
            metadata={
                "source_exit_degrees": ex,
                "target_entry_degrees": en,
                "best_delta_degrees": best_delta,
                "mode": mode,
            },
        )
