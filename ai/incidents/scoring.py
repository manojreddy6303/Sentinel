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
        incident_type: Optional[str] = None,
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

        # Dynamic evidence-grounded assessment calculation:
        # Avoid flattening every REVIEW_REQUIRED finding to a single hardcoded 0.65
        if validation_decision == "ACCEPTED" or validation_decision is None:
            final_score = pattern_strength
        elif validation_decision == "REJECTED":
            final_score = round(max(0.10, min(0.30, pattern_strength * 0.35)), 4)
        else:
            # REVIEW_REQUIRED: Calibrate assessment based on duration persistence, track validation, and signal diversity
            persistence_factor = min(0.08, (duration_seconds / max(1.0, expected_duration_threshold)) * 0.02) if expected_duration_threshold > 0 else 0.0
            track_factor = 0.03 if (tracks and all(t.is_validated for t in tracks)) else 0.0
            signal_factor = min(0.06, unique_families_count * 0.02)
            
            # Type-specific baseline assessment (strictly capped at 0.65 for unverified review-required patterns)
            if incident_type in ("POTENTIAL_THEFT", "theft_and_takeaway"):
                # For canonical theft (3 families, 16s dwell, validated tracks): calibrated to 0.65
                base_review = 0.58 + signal_factor + persistence_factor + track_factor
                final_score = min(0.65, base_review)
            elif incident_type in ("PROLONGED_PRESENCE", "prolonged_presence"):
                # Dwell persistence distinguishes 41s vs 16s dwell
                final_score = min(0.65, max(0.50, 0.54 + persistence_factor * 1.5 + track_factor))
            elif incident_type in ("CROWD_DISPERSAL", "crowd_density", "crowd_movement"):
                # Track volume and duration scaling
                track_vol = min(0.04, len(tracks) * 0.01) if tracks else 0.0
                final_score = min(0.65, max(0.52, 0.55 + track_vol + persistence_factor))
            elif incident_type in ("POTENTIAL_FORCED_MOVEMENT", "forced_movement"):
                final_score = min(0.62, max(0.48, 0.52 + track_factor + persistence_factor))
            else:
                final_score = min(0.65, max(0.45, score * 0.70))

            final_score = round(final_score, 4)

        final_pct = int(round(final_score * 100))

        # Pattern tier correctly reflects pattern evidence strength percentage (e.g. 95% -> High Evidence Strength)
        if pattern_pct >= 80:
            pattern_tier = "High Evidence Strength"
        elif pattern_pct >= 60:
            pattern_tier = "Moderate Evidence Strength"
        else:
            pattern_tier = "Initial Observation"

        # Final assessment tier reflects calibrated assessment score
        if final_pct >= 80:
            final_tier = "High Evidence Strength"
            reliability_rating = "Validated Observation"
        elif final_pct >= 60:
            final_tier = "Moderate Evidence Strength"
            reliability_rating = "Review Required" if validation_decision == "REVIEW_REQUIRED" else "Substantiated Pattern"
        else:
            final_tier = "Initial Observation"
            reliability_rating = "Review Required"

        verification_note = (
            f"Pattern Evidence Strength: {pattern_pct}% ({pattern_tier}) • Final Assessment: {final_pct}% ({validation_decision}) — "
            f"{'Human verification required' if validation_decision == 'REVIEW_REQUIRED' else 'Sensor-grounded pattern'}"
        )

        return {
            "score": final_score,
            "confidence": final_score,
            "assessment_score": final_score,
            "percentage": final_pct,
            "pattern_strength": pattern_strength,
            "pattern_percentage": pattern_pct,
            "strength_tier": final_tier,
            "evidence_strength": final_tier,
            "pattern_tier": pattern_tier,
            "final_tier": final_tier,
            "reliability_rating": reliability_rating,
            "signal_count": total_signals_count,
            "unique_signal_families": len(signal_families),
            "verification_note": verification_note,
        }
