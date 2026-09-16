"""
Potential Near-Collision Detector (Sentinel Phase 11)

Detects critical near-miss vehicle encounters:
- Extremely close proximity with rapid convergence
- Evasive trajectory swerve or abrupt lateral deflection
- Strictly ZERO physical contact (IoU == 0.0)
- Continued movement post-encounter

Distinct from physical collision and distinct from normal passing/overtaking.
"""
import math
from typing import List, Dict, Any, Optional, Tuple

from ai.incidents.base_detector import BaseIncidentDetector
from ai.incidents.schemas import (
    IncidentCandidate,
    IncidentContext,
    IncidentCategory,
    SupportingSignal,
    EvidenceCandidate,
    SpatialContext,
)
from ai.incidents.scoring import IncidentScorer
from ai.incidents.detectors.vehicle.interaction import VehicleInteractionModel


class NearCollisionDetector(BaseIncidentDetector):
    """
    Evaluates close-proximity vehicle encounters for near-collision / evasive maneuver patterns.
    """

    detector_name: str = "near_collision_detector"
    detector_version: str = "1.0.0"
    category: IncidentCategory = IncidentCategory.VEHICLE

    # Phase Z Contract Declarations
    required_signals: List[str] = ["Close Proximity Approach", "Evasive Trajectory Response"]
    supporting_signals_declared: List[str] = [
        "Close Proximity Approach",
        "Evasive Trajectory Response",
        "Absence of Physical Contact",
        "Post-Encounter Separation",
    ]
    contradictory_signals_declared: List[str] = [
        "Negative: Physical Contact Detected",
        "Negative: Normal Parallel Lane Passing",
        "Negative: Stable Separation Distance",
    ]
    context_requirements: Dict[str, Any] = {"scene_type_priors": {"roadway": "requires_evasive_deflection"}}
    evidence_requirements: Dict[str, Any] = {"verified_tracks": True, "bounding_box_grounding": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        min_approach_rate: float = 30.0,
        max_proximity_threshold_px: float = 65.0,
        min_evasive_angle_deg: float = 20.0,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.min_approach_rate = min_approach_rate
        self.max_proximity_threshold_px = max_proximity_threshold_px
        self.min_evasive_angle_deg = min_evasive_angle_deg
        self.vehicle_classes = {"car", "bus", "truck", "motorcycle", "van"}

    def _detect_evasive_deflection(self, v: Any, event_time: float) -> Tuple[bool, float]:
        """Check if vehicle exhibited an abrupt heading change near event_time."""
        pts = v.trajectory or []
        if len(pts) < 3:
            return False, 0.0

        # Compare heading before event vs heading after event
        pre_pts = [p for p in pts if p[0] <= event_time]
        post_pts = [p for p in pts if p[0] >= event_time]

        if len(pre_pts) < 2 or len(post_pts) < 2:
            return False, 0.0

        dx_pre = pre_pts[-1][1] - pre_pts[0][1]
        dy_pre = pre_pts[-1][2] - pre_pts[0][2]
        dx_post = post_pts[-1][1] - post_pts[0][1]
        dy_post = post_pts[-1][2] - post_pts[0][2]

        ang_pre = math.atan2(dy_pre, dx_pre)
        ang_post = math.atan2(dy_post, dx_post)

        diff = abs(math.degrees(ang_pre - ang_post)) % 360.0
        if diff > 180.0:
            diff = 360.0 - diff

        return (diff >= self.min_evasive_angle_deg), round(diff, 1)

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        vehicles = [t for t in context.tracks if t.object_class in self.vehicle_classes and t.is_validated]
        if len(vehicles) < 2:
            return candidates

        checked_pairs = set()
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                v1, v2 = vehicles[i], vehicles[j]
                pair_key = (min(v1.track_id, v2.track_id), max(v1.track_id, v2.track_id))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                interaction = VehicleInteractionModel.analyze_interaction(
                    v1, v2, context_motions=context.track_motions
                )
                if not interaction:
                    continue

                min_dist = interaction["min_distance_px"]
                max_approach = interaction["max_approach_rate_px_s"]
                event_t = interaction["event_time"]
                has_physical_contact = interaction["has_physical_contact"]
                is_parallel = interaction["is_parallel_flow"]

                # 1. Negative evidence: if physical contact occurred, it belongs to CollisionDetector, not NearCollision!
                if has_physical_contact:
                    continue

                # 2. Must approach rapidly to close proximity
                if min_dist > self.max_proximity_threshold_px or max_approach < self.min_approach_rate:
                    continue

                # 3. Check for evasive swerve / trajectory deflection
                swerved_1, angle_1 = self._detect_evasive_deflection(v1, event_t)
                swerved_2, angle_2 = self._detect_evasive_deflection(v2, event_t)
                has_evasive_maneuver = swerved_1 or swerved_2

                # 4. Negative evidence: normal parallel passing without evasive swerve
                if is_parallel and not has_evasive_maneuver:
                    # Vehicles passed normally in their respective lanes. ABSTAIN.
                    continue

                if not has_evasive_maneuver:
                    # No evasive swerve observed; distance alone is not a near collision
                    continue

                max_swerve_angle = max(angle_1, angle_2)
                signals = [
                    SupportingSignal(
                        signal_type="Close Proximity Approach",
                        description=(
                            f"Vehicles [{v1.track_id}] and [{v2.track_id}] approached to {min_dist:.1f}px "
                            f"(approach rate: {max_approach:.1f}px/s) without physical contact."
                        ),
                        confidence=0.85,
                        timestamp=event_t,
                    ),
                    SupportingSignal(
                        signal_type="Evasive Trajectory Response",
                        description=f"Lateral evasive trajectory deflection ({max_swerve_angle:.1f}°) observed during encounter",
                        confidence=0.88,
                        timestamp=event_t,
                    ),
                    SupportingSignal(
                        signal_type="Absence of Physical Contact",
                        description="Clearance maintained throughout encounter (0% bounding box overlap)",
                        confidence=0.92,
                        timestamp=event_t,
                    ),
                ]

                scoring = IncidentScorer.calculate_evidence_score(
                    base_confidence=0.72,
                    supporting_signals=signals,
                    tracks=[v1, v2],
                )

                cpa_box = interaction["cpa_box1"]

                cand = self.build_candidate(
                    video_id=context.video_id,
                    event_type="POTENTIAL_NEAR_COLLISION",
                    start_time=max(0.0, event_t - 1.5),
                    end_time=event_t + 2.0,
                    severity="NORMAL",
                    confidence=scoring["score"],
                    explanation=(
                        f"Potential Near-Collision pattern: {v1.object_class} [{v1.track_id}] and "
                        f"{v2.object_class} [{v2.track_id}] approached rapidly ({max_approach:.1f}px/s to {min_dist:.1f}px) "
                        f"with evasive deflection ({max_swerve_angle:.1f}°) and zero physical contact."
                    ),
                    track_ids=[v1.track_id, v2.track_id],
                    object_classes=[v1.object_class, v2.object_class],
                    supporting_signals=signals,
                    spatial_context=SpatialContext(
                        centroid=cpa_box.centroid if cpa_box else None,
                        bounding_box=cpa_box,
                        inter_track_distances={v2.track_id: round(min_dist, 2)},
                    ),
                    evidence_candidates=[
                        EvidenceCandidate(
                            timestamp=event_t,
                            bounding_box=cpa_box,
                            target_track_id=v1.track_id,
                            reason="Closest approach point of potential near-collision encounter",
                        )
                    ],
                    prefix="NEAR",
                )
                candidates.append(cand)

        return candidates
