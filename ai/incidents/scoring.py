"""
Incident Scoring Framework (Phase 10)

Calculates standardized confidence scores and evidence strength based on:
- detection confidence
- track persistence and reliability
- spatial and motion consistency
- diversity and count of independent physical supporting signals

CRITICAL OBSERVATIONAL SAFETY CONSTRAINT:
Scores are calibrated as evidence strength / confidence of observable physical patterns.
Never presented as mathematical certainty of guilt, crime, or malicious intent.
"""
from typing import List, Dict, Any, Optional

from ai.schemas import TrackedObject
from ai.incidents.schemas import SupportingSignal


class IncidentScorer:
    """
    Standardized multi-signal evidence scoring utility.
    """

    @staticmethod
    def _categorize_signal_family(signal_type: str) -> str:
        """Map signal type to independent physical measurement family to prevent double-counting."""
        st = signal_type.lower()
        if any(k in st for k in ["proximity", "distance", "separation", "clearance", "cohesion", "cluster"]):
            return "SPATIAL_PROXIMITY"
        if any(k in st for k in ["velocity", "speed", "acceleration", "deceleration", "oscillation", "heading", "motion", "vector"]):
            return "KINEMATIC_DYNAMICS"
        if any(k in st for k in ["aspect ratio", "geometry", "height", "width", "bounding-box"]):
            return "BOUNDING_BOX_GEOMETRY"
        if any(k in st for k in ["pose", "keypoint", "skeleton"]):
            return "POSE_ANATOMY"
        if any(k in st for k in ["zone", "corridor", "scene", "baseline", "density"]):
            return "SCENE_CONTEXT"
        if any(k in st for k in ["dwell", "persistence", "duration", "lag", "arrest"]):
            return "TEMPORAL_PERSISTENCE"
        return "GENERAL_OBSERVATION"

    @staticmethod
    def calculate_evidence_score(
        base_confidence: float,
        supporting_signals: List[SupportingSignal],
        tracks: Optional[List[TrackedObject]] = None,
        duration_seconds: float = 0.0,
        expected_duration_threshold: float = 4.0,
        contradictory_signals: Optional[List[SupportingSignal]] = None,
        validation_decision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calibrate an incident's evidence strength score and reliability tier.
        Avoids inflating confidence via correlated features or aliases of the same physical measurement.
        """
        score = base_confidence

        # Signal family deduplication: award bonuses for independent measurement modalities
        signal_families = set()
        for s in supporting_signals:
            family = IncidentScorer._categorize_signal_family(s.signal_type)
            signal_families.add(family)

        unique_families_count = len(signal_families)
        total_signals_count = len(supporting_signals)
        # Unique family bonus: 0.04 per independent modality; minor 0.01 per intra-family reinforcing signal
        family_bonus = min(0.12, unique_families_count * 0.04)
        intra_family_bonus = min(0.04, max(0, total_signals_count - unique_families_count) * 0.01)
        score += family_bonus + intra_family_bonus

        # Track reliability bonus
        if tracks:
            validated_tracks = [t for t in tracks if t.is_validated]
            if len(validated_tracks) == len(tracks) and len(tracks) > 0:
                score += 0.04
            # Multi-frame depth bonus
            avg_det_count = sum(t.detection_count for t in tracks) / len(tracks)
            if avg_det_count >= 5:
                score += 0.03

        # Temporal duration bonus
        if duration_seconds > 0 and expected_duration_threshold > 0:
            duration_ratio = min(2.0, duration_seconds / expected_duration_threshold)
            score += min(0.06, duration_ratio * 0.03)

        # Contradictory signals penalty
        neg_count = len(contradictory_signals) if contradictory_signals else 0
        if neg_count > 0:
            score -= (neg_count * 0.15)

        # Uncapped pattern evidence strength based on observed signals
        pattern_strength = round(max(0.10, min(0.95, score)), 4)
        pattern_pct = int(round(pattern_strength * 100))

        # Cap based on validation status: review-required candidates cannot claim high certainty
        max_allowed = 0.95
        if validation_decision == "REVIEW_REQUIRED":
            max_allowed = 0.65
        elif validation_decision == "REJECTED":
            max_allowed = 0.30

        final_score = round(max(0.10, min(max_allowed, score)), 4)
        final_pct = int(round(final_score * 100))

        # Strength tier reflects final assessment score to strictly respect the REVIEW_REQUIRED <= 0.65 ceiling
        if final_pct >= 80:
            tier = "High Evidence Strength"
            reliability_rating = "Validated Observation"
        elif final_pct >= 60:
            tier = "Moderate Evidence Strength"
            reliability_rating = "Review Required" if validation_decision == "REVIEW_REQUIRED" else "Substantiated Pattern"
        else:
            tier = "Initial Observation"
            reliability_rating = "Review Required"

        verification_note = (
            f"Pattern Evidence Strength: {pattern_pct}% ({tier}) • Final Assessment: {final_pct}% ({validation_decision}) — "
            f"{'Human verification required' if validation_decision == 'REVIEW_REQUIRED' else 'Sensor-grounded pattern'}"
        )

        return {
            "score": final_score,
            "confidence": final_score,
            "assessment_score": final_score,
            "percentage": final_pct,
            "pattern_strength": pattern_strength,
            "pattern_percentage": pattern_pct,
            "strength_tier": tier,
            "evidence_strength": tier,
            "reliability_rating": reliability_rating,
            "signal_count": total_signals_count,
            "unique_signal_families": len(signal_families),
            "verification_note": verification_note,
        }
