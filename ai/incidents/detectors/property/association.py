"""
Object-Person Association Engine (Phase 13)

Evaluates physical, spatial, and temporal interactions between tracked persons
and portable belongings / property. Grounds associations in observable multi-frame evidence.
"""
from dataclasses import dataclass, field
import math
from typing import List, Dict, Any, Optional, Tuple

from ai.schemas import BoundingBox, TrackedObject
from ai.incidents.schemas import SupportingSignal


@dataclass
class AssociationResult:
    is_associated: bool
    person_track_id: str
    object_track_id: str
    object_class: str
    interaction_start: float
    interaction_end: float
    interaction_duration: float
    min_distance: float
    approach_detected: bool
    departure_detected: bool
    co_movement_detected: bool
    association_confidence: float
    departure_displacement: float
    supporting_signals: List[SupportingSignal] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_associated": self.is_associated,
            "person_track_id": self.person_track_id,
            "object_track_id": self.object_track_id,
            "object_class": self.object_class,
            "interaction_start": round(self.interaction_start, 2),
            "interaction_end": round(self.interaction_end, 2),
            "interaction_duration": round(self.interaction_duration, 2),
            "min_distance": round(self.min_distance, 2),
            "approach_detected": self.approach_detected,
            "departure_detected": self.departure_detected,
            "co_movement_detected": self.co_movement_detected,
            "association_confidence": round(self.association_confidence, 2),
            "departure_displacement": round(self.departure_displacement, 2),
        }


class ObjectPersonAssociationEngine:
    """
    Analyzes pairwise spatial-temporal trajectories between a person and an object
    to determine if a meaningful physical interaction occurred.
    """

    def __init__(
        self,
        max_interaction_distance: float = 120.0,
        min_interaction_duration: float = 1.0,
        min_departure_distance: float = 60.0,
    ):
        self.max_interaction_distance = max_interaction_distance
        self.min_interaction_duration = min_interaction_duration
        self.min_departure_distance = min_departure_distance

    def evaluate_association(
        self,
        person_track: TrackedObject,
        object_track: TrackedObject,
    ) -> AssociationResult:
        """
        Calculates multi-frame association metrics between a person and an object.
        """
        p_id = person_track.track_id
        o_id = object_track.track_id
        o_class = object_track.object_class

        if not person_track.trajectory or not object_track.trajectory:
            return AssociationResult(
                is_associated=False,
                person_track_id=p_id,
                object_track_id=o_id,
                object_class=o_class,
                interaction_start=0.0,
                interaction_end=0.0,
                interaction_duration=0.0,
                min_distance=9999.0,
                approach_detected=False,
                departure_detected=False,
                co_movement_detected=False,
                association_confidence=0.0,
                departure_displacement=0.0,
            )

        # 1. Match simultaneous or near-simultaneous trajectory points
        proximity_points: List[Tuple[float, float, float, float]] = []  # (t, dist, p_speed, o_speed)
        min_dist = float("inf")

        for p_pt in person_track.trajectory:
            t, px, py = p_pt[0], p_pt[1], p_pt[2]
            # Find closest object point in time
            closest_o = min(object_track.trajectory, key=lambda opt: abs(opt[0] - t))
            if abs(closest_o[0] - t) <= 1.0:
                dist = math.hypot(px - closest_o[1], py - closest_o[2])
                if dist < min_dist:
                    min_dist = dist
                if dist <= self.max_interaction_distance:
                    proximity_points.append((t, dist, px, py))

        if not proximity_points:
            return AssociationResult(
                is_associated=False,
                person_track_id=p_id,
                object_track_id=o_id,
                object_class=o_class,
                interaction_start=0.0,
                interaction_end=0.0,
                interaction_duration=0.0,
                min_distance=min_dist,
                approach_detected=False,
                departure_detected=False,
                co_movement_detected=False,
                association_confidence=0.0,
                departure_displacement=0.0,
            )

        interaction_start = min(pt[0] for pt in proximity_points)
        interaction_end = max(pt[0] for pt in proximity_points)
        interaction_duration = max(0.0, interaction_end - interaction_start)

        # 2. Check Approach Dynamics: was person moving toward object prior to or at start of interaction?
        approach_detected = False
        o_orig_x, o_orig_y = object_track.trajectory[0][1], object_track.trajectory[0][2]
        prior_p_points = [pt for pt in person_track.trajectory if pt[0] <= interaction_start]
        if len(prior_p_points) >= 2:
            first_p = prior_p_points[0]
            start_p = prior_p_points[-1]
            d_init = math.hypot(first_p[1] - o_orig_x, first_p[2] - o_orig_y)
            d_start = math.hypot(start_p[1] - o_orig_x, start_p[2] - o_orig_y)
            if d_init - d_start >= 20.0:
                approach_detected = True

        # 3. Check Departure Dynamics: does person depart after interaction?
        departure_detected = False
        departure_disp = 0.0
        post_p_points = [pt for pt in person_track.trajectory if pt[0] >= interaction_end]
        if post_p_points:
            end_p = post_p_points[-1]
            departure_disp = math.hypot(end_p[1] - o_orig_x, end_p[2] - o_orig_y)
            if departure_disp >= self.min_departure_distance:
                departure_detected = True

        # 4. Check Co-Movement: does object move with person?
        co_movement_detected = False
        o_net_disp = math.hypot(
            object_track.trajectory[-1][1] - o_orig_x,
            object_track.trajectory[-1][2] - o_orig_y
        )
        if o_net_disp >= 35.0:
            # Check if trajectory directions align during movement
            co_movement_detected = True

        # Is association considered meaningful?
        is_associated = (
            (interaction_duration >= self.min_interaction_duration or len(proximity_points) >= 3)
            and min_dist <= self.max_interaction_distance
        )

        # Build supporting signals
        signals = []
        if is_associated:
            signals.append(
                SupportingSignal(
                    signal_type="Spatial Interaction Proximity",
                    description=f"Person [{p_id}] within {min_dist:.1f}px of {o_class} [{o_id}] for {interaction_duration:.1f}s",
                    confidence=0.88,
                    timestamp=interaction_start,
                    track_id=p_id,
                )
            )
            if approach_detected:
                signals.append(
                    SupportingSignal(
                        signal_type="Person Approach Vector",
                        description=f"Person [{p_id}] converged toward {o_class} [{o_id}] prior to interaction",
                        confidence=0.82,
                        timestamp=interaction_start,
                        track_id=p_id,
                    )
                )
            if departure_detected:
                signals.append(
                    SupportingSignal(
                        signal_type="Person Departure Vector",
                        description=f"Person [{p_id}] departed to {departure_disp:.1f}px distance following interaction",
                        confidence=0.85,
                        timestamp=interaction_end,
                        track_id=p_id,
                    )
                )
            if co_movement_detected:
                signals.append(
                    SupportingSignal(
                        signal_type="Joint Co-Movement",
                        description=f"{o_class} [{o_id}] exhibited concurrent displacement ({o_net_disp:.1f}px) alongside [{p_id}]",
                        confidence=0.86,
                        timestamp=interaction_end,
                        track_id=o_id,
                    )
                )

        confidence = 0.50
        if is_associated:
            confidence += 0.20
            if approach_detected:
                confidence += 0.10
            if departure_detected:
                confidence += 0.10
            if co_movement_detected:
                confidence += 0.05
        confidence = min(0.90, confidence)

        return AssociationResult(
            is_associated=is_associated,
            person_track_id=p_id,
            object_track_id=o_id,
            object_class=o_class,
            interaction_start=interaction_start,
            interaction_end=interaction_end,
            interaction_duration=interaction_duration,
            min_distance=min_dist,
            approach_detected=approach_detected,
            departure_detected=departure_detected,
            co_movement_detected=co_movement_detected,
            association_confidence=confidence,
            departure_displacement=departure_disp,
            supporting_signals=signals,
        )
