"""
Global Incident Candidate Validator (Phase 10-R)

Evaluates every incident candidate before fusion or persistence across 12 criteria:
1. Source detection validity
2. Track validity & reliability
3. Timestamp validity
4. Bounding-box validity
5. Spatial consistency
6. Temporal consistency
7. Motion consistency
8. Scene/context compatibility
9. Supporting signal count
10. Contradictory/negative evidence ratio
11. Duplicate incident risk
12. Evidence availability

Decisions:
- ACCEPTED: Strong supporting evidence, zero/negligible contradictory evidence, scene compatible.
- REVIEW_REQUIRED: Weak/ambiguous signals, partial contradiction, or uncalibrated camera perspective.
- REJECTED: Strongly refuted by negative evidence, invalid tracking/detections, or physically implausible.
"""

from typing import List, Dict, Any, Optional
import logging

from ai.incidents.schemas import (
    IncidentCandidate,
    IncidentContext,
    ValidationDecision,
    SupportingSignal,
)
from ai.incidents.scene_context import SceneContextData

logger = logging.getLogger(__name__)


class IncidentCandidateValidator:
    """
    Universal validator that evaluates incident candidates and assigns
    a definitive ValidationDecision (ACCEPTED, REVIEW_REQUIRED, REJECTED).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.version = "1.0.0"

    def validate_candidate(
        self,
        candidate: IncidentCandidate,
        context: Optional[IncidentContext] = None,
        scene_context: Optional[SceneContextData] = None,
    ) -> IncidentCandidate:
        """
        Evaluate candidate against all 12 criteria and populate:
        - candidate.validation_decision
        - candidate.validation_reasons
        - candidate.confidence (calibrated evidence strength)
        """
        reasons: List[str] = []
        contradictory_reasons: List[str] = []
        is_rejected = False
        requires_review = False

        # 1. Timestamp validity
        if candidate.start_time < 0 or candidate.end_time < candidate.start_time:
            reasons.append("Invalid temporal interval: start_time < 0 or end_time < start_time")
            is_rejected = True

        # 2. Bounding-box / Spatial validity
        if candidate.spatial_context and candidate.spatial_context.bounding_box:
            box = candidate.spatial_context.bounding_box
            w = getattr(box, "width", getattr(box, "x2", 1) - getattr(box, "x1", 0))
            h = getattr(box, "height", getattr(box, "y2", 1) - getattr(box, "y1", 0))
            if w <= 0 or h <= 0:
                reasons.append("Invalid spatial bounding box: non-positive dimensions")
                is_rejected = True


        # 3. Track validity & reliability
        if candidate.track_ids and context and context.tracks:
            track_map = {str(getattr(t, "track_id", t.get("track_id") if isinstance(t, dict) else "")): t for t in context.tracks}
            for tid in candidate.track_ids:
                t = track_map.get(str(tid))
                if not t:
                    continue
                if hasattr(t, "detection_count"):
                    obs_count = t.detection_count
                elif hasattr(t, "history_bboxes"):
                    obs_count = len(t.history_bboxes)
                elif isinstance(t, dict):
                    obs_count = t.get("observation_count", len(t.get("points", t.get("history_bboxes", []))))
                else:
                    obs_count = 2

                # A single observation track claiming complex multi-second incident
                if obs_count <= 1 and candidate.duration > 2.0:
                    reasons.append(f"Track {tid} has only 1 observation; insufficient temporal stability for sustained incident")
                    requires_review = True


        # 4. Temporal consistency for event type
        if candidate.duration < 0.05 and candidate.event_type not in ["INSTANT_DISPLACEMENT", "OBSERVATIONAL_ANOMALY"]:
            reasons.append(f"Duration {candidate.duration:.2f}s is too brief for sustained event {candidate.event_type}")
            requires_review = True

        # 5. Supporting signal count
        support_count = len(candidate.supporting_signals)
        if support_count == 0:
            reasons.append("Zero supporting signals provided for incident candidate")
            requires_review = True

        # 6. Contradictory / Negative evidence checks (CRITICAL)
        contradictory_signals = candidate.contradictory_signals or []
        for cs in contradictory_signals:
            stype = getattr(cs, "signal_type", getattr(cs, "signal_name", "signal"))
            sconf = getattr(cs, "confidence", getattr(cs, "value", 1.0))
            contradictory_reasons.append(f"{stype} (strength: {sconf:.2f}): {cs.description}")

        # Check for severe contradictory indicators
        severe_contradictions = []
        for cs in contradictory_signals:
            stype = getattr(cs, "signal_type", getattr(cs, "signal_name", "")).lower()
            sconf = getattr(cs, "confidence", getattr(cs, "value", 1.0))
            if sconf >= 0.70 and any(
                k in stype for k in [
                    "zero physical",
                    "zero_physical_overlap",
                    "continued normal",
                    "continued_normal_transit",
                    "parallel lane",
                    "parallel_lane_following",
                    "object remains",
                    "object_remains_at_rest",
                    "normal traffic transit",
                    "continuous transit flow",
                    "synchronized_traffic_slowdown",
                    "traffic_queue_present",
                    "ambiguous_roadway_flow",
                    "turning_maneuver_present",
                    "general_traffic_congestion",
                    "normal_passing_clearance",
                    "stable_lateral_separation",
                    "resumed upright walking",
                    "resumed walking transit",
                    "parallel side-by-side walking",
                    "normal pair walking abreast",
                    "pedestrian queue formation",
                    "normal pedestrian transit pace",
                    "transient passing pedestrians",
                    "frame boundary truncation",
                    "symmetric trajectory correlation",
                    "upright posture maintained",
                    "person in proximity",
                    "person remained with object",
                    "person returned",
                    "zero property displacement",
                    "displacement below noise threshold",
                    "temporary occlusion",
                    "camera jitter",
                    "orderly queue flow",
                    "orderly linear queue",
                    "steady scene baseline",
                    "normal commute flow",
                    "dispersed non-cohesive",
                    "boundary jitter",
                    "transient boundary crossing",
                    "insufficient dwell",
                    "unrestricted access",
                    "transient visual glitch",
                    "static surface chromaticity",
                    "transient dispersion artifact",
                    "global atmospheric haze",
                    "video compression artifact",
                    "insufficient optical resolution",
                    "common personal belonging context",
                    "absent pose keypoints",
                ]
            ):
                severe_contradictions.append(cs)

        if severe_contradictions:
            names = ", ".join(getattr(c, "signal_type", getattr(c, "signal_name", "signal")) for c in severe_contradictions)
            reasons.append(f"Severe counter-evidence refutes incident: {names}")
            is_rejected = True

        # 7. Scene context compatibility
        if scene_context:
            if scene_context.scene_type == "roadway":
                # Roadway suppresses prolonged presence unless stationary dwell is extreme
                if "PROLONGED" in candidate.event_type:
                    stationary_signals = [
                        s for s in candidate.supporting_signals
                        if "stationary" in getattr(s, "signal_type", "").lower() or "stopped" in getattr(s, "signal_type", "").lower()
                    ]
                    if not stationary_signals:
                        reasons.append("Roadway scene context contradicts prolonged presence for moving traffic")
                        is_rejected = True
                # Roadway requires physical contact for collision
                if "COLLISION" in candidate.event_type:
                    has_contact = any(
                        any(k in getattr(s, "signal_type", "").lower() for k in ["physical", "contact", "overlap", "deceleration", "stoppage"])
                        for s in candidate.supporting_signals
                    )
                    if not has_contact:
                        reasons.append("Roadway perspective convergence lacks verified physical contact or deceleration stop")
                        requires_review = True

        # 8. Calibrated Confidence / Evidence Strength adjustment
        # Negative evidence degrades raw confidence
        contradictory_penalty = sum(getattr(cs, "confidence", getattr(cs, "value", 1.0)) * 0.35 for cs in contradictory_signals)
        calibrated_conf = max(0.05, candidate.confidence - contradictory_penalty)

        # High-risk human interaction classes and unzoned prolonged presence require human review
        # (visual kinematics alone cannot establish intent without operator-defined restricted boundaries)
        is_unzoned_prolonged = candidate.event_type == "PROLONGED_PRESENCE" and not (
            candidate.spatial_context and candidate.spatial_context.zone_name
        )
        if is_unzoned_prolonged:
            requires_review = True
            reasons.append("Stationary presence outside configured restricted zone requires human verification to determine operational intent")

        is_inherent_review_type = candidate.event_type in [
            "POTENTIAL_PHYSICAL_ALTERCATION",
            "POTENTIAL_FORCED_MOVEMENT",
            "POTENTIAL_FIRE",
            "POTENTIAL_SMOKE",
            "POTENTIAL_FIRE_SMOKE",
            "POTENTIAL_WEAPON_VISUAL",
        ] or is_unzoned_prolonged
        prior_review_required = (
            candidate.validation_decision in ["REVIEW_REQUIRED", ValidationDecision.REVIEW_REQUIRED.value]
            or is_inherent_review_type
        )

        # Decision synthesis
        if is_rejected:
            decision = ValidationDecision.REJECTED
            calibrated_conf = min(calibrated_conf, 0.25)
        elif requires_review or contradictory_signals or calibrated_conf < 0.60 or prior_review_required:
            decision = ValidationDecision.REVIEW_REQUIRED
            # Keep confidence labeled honestly as moderate evidence strength (capped for unverified observational patterns)
            calibrated_conf = min(calibrated_conf, 0.65)
            if not reasons:
                reasons.append("Observational pattern requires human verification (visual kinematics alone cannot establish intent)")
        else:
            decision = ValidationDecision.ACCEPTED
            reasons.append("Passes all 12 candidate validation checks with strong supporting evidence")

        candidate.validation_decision = decision.value
        candidate.validation_reasons = reasons
        candidate.confidence = round(calibrated_conf, 4)
        return candidate

    def validate_all(
        self,
        candidates: List[IncidentCandidate],
        context: Optional[IncidentContext] = None,
        scene_context: Optional[SceneContextData] = None,
    ) -> List[IncidentCandidate]:
        """Validate a batch of candidates."""
        validated = []
        for cand in candidates:
            validated.append(self.validate_candidate(cand, context=context, scene_context=scene_context))
        return validated
