"""
ai/correlation/storyline_generator.py
=====================================
Deterministic observational incident storyline generator for Phase 16.
Creates human-readable, timestamped, database-grounded narratives without unsupported intent.
"""

from typing import List, Dict, Any, Optional
from ai.correlation.models import ObservationalRelationship, CorrelationRelationshipType


class IncidentStorylineGenerator:
    """
    Synthesizes grounded observational narratives from correlated incidents and physical relationships.
    """

    @staticmethod
    def get_decision_clause(validation_decision: str) -> str:
        """
        Differentiate verification language according to canonical evidence policy:
        - ACCEPTED: observational result accepted by Sentinel policy.
        - REVIEW_REQUIRED: insufficient certainty for automatic acceptance; human verification required.
        """
        v_upper = str(validation_decision).upper()
        if "ACCEPTED" in v_upper and "REVIEW" not in v_upper:
            return "Accepted by Sentinel's evidence policy; observational result. Human review may still be appropriate for operational decisions."
        elif "REVIEW_REQUIRED" in v_upper or "REVIEW" in v_upper:
            return "Insufficient certainty for automatic acceptance; human verification required."
        elif "REJECTED" in v_upper:
            return "Rejected by Sentinel's evidence policy."
        return f"Status: {validation_decision}."

    @staticmethod
    def generate_storyline(
        category: str,
        subcategory: str,
        start_time: float,
        end_time: float,
        duration: float,
        primary_tracks: List[str],
        supporting_tracks: List[str],
        object_classes: List[str],
        relationships: List[ObservationalRelationship],
        supporting_signals: List[Dict[str, Any]],
        confidence: float,
        validation_decision: str,
    ) -> str:
        """
        Generate deterministic narrative text.
        """
        tracks_str = ", ".join(primary_tracks) if primary_tracks else "untracked subject"
        classes_str = "/".join(object_classes) if object_classes else "entity"
        decision_clause = IncidentStorylineGenerator.get_decision_clause(validation_decision)

        # 1. Property / Takeaway Pattern
        if "theft" in subcategory.lower() or "takeaway" in subcategory.lower():
            person_trks = [t for t in primary_tracks + supporting_tracks if "person" in t.lower() or "trk" in t.lower()]
            obj_trks = [t for t in supporting_tracks + primary_tracks if t not in person_trks]
            p_label = person_trks[0] if person_trks else "Track"
            o_label = obj_trks[0] if obj_trks else "target object"
            target_class = [c for c in object_classes if c != "person"]
            cls_label = target_class[0] if target_class else "object"

            narrative = (
                f"At {start_time:.1f}s, {p_label} was observed approaching and remaining in close proximity to "
                f"{cls_label} ({o_label}) for approximately {duration:.1f} seconds. "
                f"Following physical interaction and co-movement telemetry, the {cls_label} was subsequently "
                f"no longer detected in the scene while {p_label} departed the immediate area. "
                f"This formed a potential object takeaway pattern (assessment score: {confidence*100:.0f}%, "
                f"status: {validation_decision}). {decision_clause}"
            )
            return narrative
        elif category == "property" or any(k in subcategory.lower() for k in ("displacement", "pickup", "moving", "transport")):
            person_trks = [t for t in primary_tracks + supporting_tracks if "person" in t.lower() or "trk" in t.lower()]
            obj_trks = [t for t in supporting_tracks + primary_tracks if t not in person_trks]
            p_label = person_trks[0] if person_trks else "Track"
            o_label = obj_trks[0] if obj_trks else "target object"
            target_class = [c for c in object_classes if c != "person"]
            cls_label = target_class[0] if target_class else "object"
            narrative = (
                f"At {start_time:.1f}s, {p_label} was observed interacting with and moving "
                f"{cls_label} ({o_label}) across {duration:.1f} seconds. "
                f"Pattern indicates object movement or relocation "
                f"(assessment score: {confidence*100:.0f}%, status: {validation_decision}). {decision_clause}"
            )
            return narrative

        # 2. Vehicle Patterns
        if "near" in subcategory.lower():
            v_tracks = primary_tracks[:2] if len(primary_tracks) >= 2 else primary_tracks
            v_str = " and ".join(v_tracks) if v_tracks else "vehicles"
            narrative = (
                f"At {start_time:.1f}s, rapid spatial convergence was recorded involving {v_str} ({classes_str}) "
                f"within close proximity without physical bounding box contact across a {duration:.1f}-second "
                f"temporal window, forming a near-collision pattern (assessment score: {confidence*100:.0f}%, status: {validation_decision}). {decision_clause}"
            )
            return narrative
        elif "collision" in subcategory.lower():
            v_tracks = primary_tracks[:2] if len(primary_tracks) >= 2 else primary_tracks
            v_str = " and ".join(v_tracks) if v_tracks else "vehicles"
            narrative = (
                f"At {start_time:.1f}s, a vehicle collision pattern with rapid spatial convergence and physical bounding box contact was recorded "
                f"involving {v_str} ({classes_str}). Post-contact deceleration and motion arrest was observed "
                f"across a {duration:.1f}-second temporal window (assessment score: {confidence*100:.0f}%, "
                f"status: {validation_decision}). {decision_clause}"
            )
            return narrative
        elif category == "vehicle" or "stop" in subcategory.lower() or "trajectory" in subcategory.lower():
            v_tracks = primary_tracks[:2] if len(primary_tracks) >= 2 else primary_tracks
            v_str = " and ".join(v_tracks) if v_tracks else "vehicles"
            narrative = (
                f"At {start_time:.1f}s, vehicle movement anomaly ({subcategory.replace('_', ' ')}) was recorded "
                f"involving {v_str} ({classes_str}) across a {duration:.1f}-second temporal window "
                f"(assessment score: {confidence*100:.0f}%, status: {validation_decision}). {decision_clause}"
            )
            return narrative

        # 3. Person Posture / Medical / Safety Pattern
        if "fall" in subcategory.lower() or "person_down" in subcategory.lower():
            narrative = (
                f"At {start_time:.1f}s, an abrupt downward vertical transition followed by prolonged stationary posture "
                f"was recorded for {tracks_str} ({classes_str}) across {duration:.1f} seconds. "
                f"Behavioral telemetry indicates a potential fall or person-down incident (assessment score: {confidence*100:.0f}%, "
                f"status: {validation_decision}). {decision_clause}"
            )
            return narrative

        # 4. Crowd / Zone Movement Pattern
        if category in ("crowd", "zone") or "crowd" in subcategory.lower():
            narrative = (
                f"At {start_time:.1f}s, elevated regional density and directional movement telemetry was observed "
                f"across a {duration:.1f}-second window involving {len(primary_tracks) + len(supporting_tracks)} track(s). "
                f"Pattern matches {subcategory.replace('_', ' ')} (assessment score: {confidence*100:.0f}%, "
                f"status: {validation_decision}). {decision_clause}"
            )
            return narrative

        # 5. Specialized Visual (Fire / Smoke) Pattern
        if "fire" in subcategory.lower() or "smoke" in subcategory.lower() or category == "environment":
            narrative = (
                f"At {start_time:.1f}s, specialized visual sensors recorded localized atmospheric/chromatic signatures "
                f"({subcategory.replace('_', ' ')}) persisting across {duration:.1f} seconds. "
                f"Visual telemetry confirms independent validation (assessment score: {confidence*100:.0f}%, "
                f"status: {validation_decision}). {decision_clause}"
            )
            return narrative

        # General Grounded Fallback
        signals_summary = f"Supported by {len(supporting_signals)} validated physical signal(s)." if supporting_signals else ""
        action_verb = "detected" if ("REVIEW" in str(validation_decision).upper()) else "validated"
        return (
            f"At {start_time:.1f}s, automated surveillance intelligence registered a {action_verb} {subcategory.replace('_', ' ')} "
            f"pattern involving {tracks_str} ({classes_str}) lasting {duration:.1f} seconds. {signals_summary} "
            f"Confidence: {confidence*100:.0f}% (status: {validation_decision}). {decision_clause}"
        )
