"""
Negative Evidence Engine (Phase 10-R)

Evaluates counter-evidence that actively refutes or contradicts an incident hypothesis.
Ensures Sentinel asks not only "What evidence supports this incident?" but also
"What evidence contradicts or disproves this incident?"

CRITICAL PRINCIPLE:
A mathematical proximity or 2D intersection is not an incident if physical
counter-evidence demonstrates normal, uninterrupted behavior.
"""
import math
from typing import List, Dict, Any, Optional, Tuple

from ai.schemas import TrackedObject, ZoneDefinition, BoundingBox
from ai.incidents.schemas import SupportingSignal, IncidentContext
from ai.incidents.spatial import SpatialRelationshipEngine


class NegativeEvidenceEngine:
    """
    Evaluates physical counter-evidence for incident hypotheses.
    """

    @classmethod
    def evaluate_collision_negative_evidence(
        cls,
        track_v1: TrackedObject,
        track_v2: TrackedObject,
        context: IncidentContext,
        event_time: float,
        post_event_window_seconds: float = 3.0,
    ) -> List[SupportingSignal]:
        """
        Identify counter-evidence indicating two vehicles experienced normal traffic
        passage rather than a physical collision.
        """
        contradictory_signals: List[SupportingSignal] = []
        if not track_v1 or not track_v2:
            return contradictory_signals

        # 1. Negative Evidence: Zero Physical Contact (No Bounding Box Overlap)
        # Search all simultaneous observations near event_time
        had_bbox_contact = False
        min_observed_distance = float("inf")

        for b1 in track_v1.history_bboxes:
            t1 = b1["timestamp"]
            if abs(t1 - event_time) > 1.5:
                continue
            # Find closest observation in track_v2
            matching_b2 = [
                b2 for b2 in track_v2.history_bboxes
                if abs(b2["timestamp"] - t1) <= 0.5
            ]
            for b2 in matching_b2:
                box1 = BoundingBox(
                    x1=float(b1["bbox"]["x1"]),
                    y1=float(b1["bbox"]["y1"]),
                    x2=float(b1["bbox"]["x2"]),
                    y2=float(b1["bbox"]["y2"]),
                )
                box2 = BoundingBox(
                    x1=float(b2["bbox"]["x1"]),
                    y1=float(b2["bbox"]["y1"]),
                    x2=float(b2["bbox"]["x2"]),
                    y2=float(b2["bbox"]["y2"]),
                )
                dist = SpatialRelationshipEngine.bbox_distance(box1, box2)
                if dist < min_observed_distance:
                    min_observed_distance = dist
                if SpatialRelationshipEngine.bbox_overlap(box1, box2):
                    had_bbox_contact = True
                    break

        if not had_bbox_contact:
            contradictory_signals.append(
                SupportingSignal(
                    signal_type="Negative: Zero Physical Contact",
                    description=(
                        f"Vehicles [{track_v1.track_id}] and [{track_v2.track_id}] maintained physical "
                        f"clearance throughout convergence (minimum centroid distance: {min_observed_distance:.1f}px, 0% bbox overlap)."
                    ),
                    confidence=0.92,
                    timestamp=event_time,
                    metadata={"had_bbox_contact": False, "min_distance": min_observed_distance},
                )
            )

        # 2. Negative Evidence: Continued Normal Transit Post-Event
        # In an actual crash, vehicles stop, spin, deflect severely, or decelerate to zero.
        # In normal traffic, both vehicles continue traveling along the roadway.
        v1_motions = context.track_motions.get(track_v1.track_id, [])
        v2_motions = context.track_motions.get(track_v2.track_id, [])

        post_v1 = [m for m in v1_motions if m.timestamp > event_time]
        post_v2 = [m for m in v2_motions if m.timestamp > event_time]

        v1_continued = len(post_v1) >= 2 and any(m.velocity_estimate > 15.0 for m in post_v1)
        v2_continued = len(post_v2) >= 2 and any(m.velocity_estimate > 15.0 for m in post_v2)

        v1_stopped = any(m.is_stationary and m.stationary_duration >= 2.0 for m in post_v1)
        v2_stopped = any(m.is_stationary and m.stationary_duration >= 2.0 for m in post_v2)

        if v1_continued and v2_continued and not (v1_stopped or v2_stopped):
            contradictory_signals.append(
                SupportingSignal(
                    signal_type="Negative: Continued Normal Transit",
                    description=(
                        f"Both vehicles [{track_v1.track_id}] and [{track_v2.track_id}] continued uninterrupted "
                        f"motion at speed post-convergence with zero post-impact stoppage."
                    ),
                    confidence=0.95,
                    timestamp=event_time + 1.0,
                    metadata={"both_continued_transit": True},
                )
            )

        # 3. Negative Evidence: Parallel Trajectory Heading (Lane Following)
        # Check if vehicles are moving in approximately the same or opposite linear road direction
        if len(track_v1.trajectory) >= 2 and len(track_v2.trajectory) >= 2:
            dx1 = track_v1.trajectory[-1][1] - track_v1.trajectory[0][1]
            dy1 = track_v1.trajectory[-1][2] - track_v1.trajectory[0][2]
            dx2 = track_v2.trajectory[-1][1] - track_v2.trajectory[0][1]
            dy2 = track_v2.trajectory[-1][2] - track_v2.trajectory[0][2]
            ang1 = math.atan2(dy1, dx1)
            ang2 = math.atan2(dy2, dx2)
            angle_diff = abs(math.degrees(ang1 - ang2)) % 360.0
            if angle_diff > 180.0:
                angle_diff = 360.0 - angle_diff

            # Parallel same-direction (< 25 deg) or opposite-direction (> 155 deg) indicates lane transit
            if angle_diff <= 25.0 or angle_diff >= 155.0:
                contradictory_signals.append(
                    SupportingSignal(
                        signal_type="Negative: Parallel Lane Following",
                        description=(
                            f"Heading vectors are aligned with roadway traffic flow (relative angle: {angle_diff:.1f}°), "
                            f"characteristic of multi-lane vehicle passage rather than collision."
                        ),
                        confidence=0.88,
                        timestamp=event_time,
                        metadata={"heading_alignment_deg": angle_diff},
                    )
                )

        return contradictory_signals

    @classmethod
    def evaluate_theft_negative_evidence(
        cls,
        person_track: TrackedObject,
        object_track: TrackedObject,
        context: IncidentContext,
        interaction_end_time: float,
    ) -> List[SupportingSignal]:
        """
        Identify counter-evidence indicating an object was not stolen or taken away.
        """
        contradictory_signals: List[SupportingSignal] = []

        # 1. Negative Evidence: Object Continues to be Detected at Original Location
        if object_track.last_seen > interaction_end_time + 4.0:
            # Check displacement of object at the end of its track
            if object_track.trajectory:
                start_x, start_y = object_track.trajectory[0][1], object_track.trajectory[0][2]
                end_x, end_y = object_track.trajectory[-1][1], object_track.trajectory[-1][2]
                disp = math.hypot(end_x - start_x, end_y - start_y)
                if disp < 25.0:
                    contradictory_signals.append(
                        SupportingSignal(
                            signal_type="Negative: Object Remains Present",
                            description=(
                                f"Object [{object_track.object_class} • {object_track.track_id}] remained present at original "
                                f"location after person departed (displacement: {disp:.1f}px, visible until {object_track.last_seen:.1f}s)."
                            ),
                            confidence=0.92,
                            timestamp=interaction_end_time,
                        )
                    )

        # 2. Negative Evidence: Object Frame Boundary Exit
        # If the object disappeared while touching the frame boundary, it exited the field of view.
        if object_track.current_bbox:
            last_bbox = object_track.current_bbox
            is_clipped = False
            # Check if coordinates are normalized [0.0, 1.0]
            if last_bbox.x2 <= 1.0 and last_bbox.y2 <= 1.0:
                if last_bbox.x1 <= 0.02 or last_bbox.y1 <= 0.02 or last_bbox.x2 >= 0.98 or last_bbox.y2 >= 0.98:
                    is_clipped = True
            else:
                # Pixel space coordinates: left or top boundary
                if last_bbox.x1 <= 4.0 or last_bbox.y1 <= 4.0:
                    is_clipped = True
            if not is_clipped and object_track.history_bboxes:
                last_h = object_track.history_bboxes[-1]
                if last_h.get("is_edge_clipped"):
                    is_clipped = True
            if is_clipped:
                contradictory_signals.append(
                    SupportingSignal(
                        signal_type="Negative: Frame Boundary Exit",
                        description=(
                            f"Object [{object_track.object_class} • {object_track.track_id}] exited the camera field "
                            f"of view at frame boundary, indicating normal scene exit rather than disappearance."
                        ),
                        confidence=0.88,
                        timestamp=object_track.last_seen,
                    )
                )

        # 3. Negative Evidence: Transient Detection Drop (Unstable Object Track)
        # If object track had only 1 observation and duration < 0.5s, its disappearance is a detector dropout, not physical theft
        if object_track.detection_count <= 1 and object_track.duration_seconds < 0.5:
            contradictory_signals.append(
                SupportingSignal(
                    signal_type="Negative: Transient Object Detection",
                    description=(
                        f"Object [{object_track.object_class} • {object_track.track_id}] was only observed in "
                        f"{object_track.detection_count} frame ({object_track.duration_seconds:.2f}s); disappearance represents detector loss."
                    ),
                    confidence=0.90,
                    timestamp=object_track.last_seen,
                )
            )

        # 4. Negative Evidence: Object Lost Prior to Person Arrival
        # If object track ceased before person even reached interaction proximity
        if object_track.last_seen < interaction_end_time - 3.0:
            contradictory_signals.append(
                SupportingSignal(
                    signal_type="Negative: Object Ceased Prior to Proximity",
                    description=(
                        f"Object [{object_track.object_class} • {object_track.track_id}] ceased detection "
                        f"prior to person arrival (last seen at {object_track.last_seen:.1f}s vs interaction {interaction_end_time:.1f}s)."
                    ),
                    confidence=0.92,
                    timestamp=object_track.last_seen,
                )
            )

        return contradictory_signals

    @classmethod
    def evaluate_loitering_negative_evidence(
        cls,
        track: TrackedObject,
        context: IncidentContext,
    ) -> List[SupportingSignal]:
        """
        Identify counter-evidence indicating normal transit rather than prolonged presence.
        """
        contradictory_signals: List[SupportingSignal] = []
        summary = context.get_motion_summary(track.track_id) or {}

        # 1. Negative Evidence: High Velocity Transit
        avg_vel = summary.get("avg_velocity", 0.0)
        path_cons = summary.get("path_consistency", 1.0)
        net_disp = summary.get("net_displacement", 0.0)

        if avg_vel > 20.0 and path_cons > 0.35 and net_disp > 100.0:
            contradictory_signals.append(
                SupportingSignal(
                    signal_type="Negative: Continuous Transit Flow",
                    description=(
                        f"Subject [{track.track_id}] exhibited continuous directional transit (avg velocity: {avg_vel:.1f}px/s, "
                        f"displacement: {net_disp:.1f}px), characteristic of normal passage."
                    ),
                    confidence=0.90,
                    timestamp=track.first_seen,
                )
            )

        return contradictory_signals

    @classmethod
    def evaluate_intrusion_negative_evidence(
        cls,
        track: TrackedObject,
        zone: ZoneDefinition,
        context: IncidentContext,
    ) -> List[SupportingSignal]:
        """
        Identify counter-evidence refuting a zone breach.
        """
        contradictory_signals: List[SupportingSignal] = []
        if not zone.enabled or not zone.polygon or len(zone.polygon) < 3:
            contradictory_signals.append(
                SupportingSignal(
                    signal_type="Negative: Unconfigured Zone",
                    description=f"Zone '{zone.name}' is disabled or lacks valid polygon boundary coordinates.",
                    confidence=1.0,
                )
            )

        if not track.is_validated:
            contradictory_signals.append(
                SupportingSignal(
                    signal_type="Negative: Unverified Track",
                    description=f"Track [{track.track_id}] lacks multi-frame verification (detection count: {track.detection_count}).",
                    confidence=0.85,
                )
            )

        return contradictory_signals

    @classmethod
    def evaluate_near_collision_negative_evidence(
        cls,
        v1: TrackedObject,
        v2: TrackedObject,
        min_dist: float,
        is_parallel: bool,
        has_evasive_maneuver: bool,
    ) -> List[SupportingSignal]:
        """Identify counter-evidence refuting a near-collision hypothesis."""
        contradictory_signals: List[SupportingSignal] = []
        if is_parallel and not has_evasive_maneuver:
            contradictory_signals.append(
                SupportingSignal(
                    signal_type="Negative: Normal Parallel Lane Passing",
                    description=f"Vehicles [{v1.track_id}] and [{v2.track_id}] maintained parallel lane trajectory without evasive deflection.",
                    confidence=0.90,
                )
            )
        if min_dist > 80.0:
            contradictory_signals.append(
                SupportingSignal(
                    signal_type="Negative: Wide Clearance Separation",
                    description=f"Separation distance ({min_dist:.1f}px) exceeds near-collision proximity threshold.",
                    confidence=0.85,
                )
            )
        return contradictory_signals

    @classmethod
    def evaluate_sudden_stop_negative_evidence(
        cls,
        is_synchronized: bool,
    ) -> List[SupportingSignal]:
        """Identify counter-evidence refuting an isolated sudden vehicle stop."""
        contradictory_signals: List[SupportingSignal] = []
        if is_synchronized:
            contradictory_signals.append(
                SupportingSignal(
                    signal_type="Negative: Synchronized Traffic Slowdown",
                    description="Multiple neighboring vehicles decelerated simultaneously, indicating traffic light or queue.",
                    confidence=0.92,
                )
            )
        return contradictory_signals

    @classmethod
    def evaluate_wrong_way_negative_evidence(
        cls,
        is_turning: bool,
        flow_confidence: float,
    ) -> List[SupportingSignal]:
        """Identify counter-evidence refuting wrong-way vehicle movement."""
        contradictory_signals: List[SupportingSignal] = []
        if is_turning:
            contradictory_signals.append(
                SupportingSignal(
                    signal_type="Negative: Turning / Intersection Maneuver",
                    description="Vehicle trajectory exhibits a continuous angular sweep consistent with a turn or curve.",
                    confidence=0.88,
                )
            )
        if flow_confidence < 0.60:
            contradictory_signals.append(
                SupportingSignal(
                    signal_type="Negative: Ambiguous Roadway Flow",
                    description="Roadway flow direction cannot be determined with sufficient confidence from visual observations.",
                    confidence=0.90,
                )
            )
        return contradictory_signals

    @classmethod
    def evaluate_stationary_vehicle_negative_evidence(
        cls,
        is_congestion: bool,
    ) -> List[SupportingSignal]:
        """Identify counter-evidence refuting an isolated stationary vehicle stall."""
        contradictory_signals: List[SupportingSignal] = []
        if is_congestion:
            contradictory_signals.append(
                SupportingSignal(
                    signal_type="Negative: General Congestion Queue",
                    description="Majority of vehicles in scene are stationary, indicating general traffic congestion rather than isolated breakdown.",
                    confidence=0.92,
                )
            )
        return contradictory_signals

    @classmethod
    def evaluate_fall_negative_evidence(
        cls,
        track: Any,
        context: Any,
        event_time: float,
    ) -> List[SupportingSignal]:
        """Identify counter-evidence refuting a person fall."""
        contradictory_signals: List[SupportingSignal] = []
        motions = context.track_motions.get(track.track_id, []) if context else []
        post_motions = [m for m in motions if m.timestamp > event_time]

        # 1. Resumed walking check
        if len(post_motions) >= 3 and any(m.velocity_estimate > 25.0 for m in post_motions):
            contradictory_signals.append(
                SupportingSignal(
                    signal_type="Negative: Resumed Upright Walking",
                    description=f"Person [{track.track_id}] resumed upright walking transit at {post_motions[-1].velocity_estimate:.1f}px/s after descent.",
                    confidence=0.88,
                    timestamp=event_time,
                )
            )

        # 2. Camera Frame Boundary Clipping Check
        # Detect if bounding box touched frame borders (causing artificial width/height inversion)
        if track.history_bboxes:
            for b_entry in track.history_bboxes:
                b = b_entry.get("bbox", {})
                x1 = float(b.get("x1", 100))
                y1 = float(b.get("y1", 100))
                if x1 < 25.0 or y1 < 25.0:
                    contradictory_signals.append(
                        SupportingSignal(
                            signal_type="Negative: Frame Boundary Truncation",
                            description="Bounding box geometry was clipped by the camera frame border, causing artificial aspect-ratio distortion.",
                            confidence=0.92,
                            timestamp=event_time,
                        )
                    )
                    break

        # 3. Posture recovery check
        if track.history_bboxes and len(track.history_bboxes) >= 4:
            later_boxes = [b for b in track.history_bboxes if float(b.get("timestamp", 0)) > event_time + 1.0]
            if later_boxes:
                upright_count = 0
                for lb in later_boxes:
                    bb = lb.get("bbox", {})
                    w = max(1.0, float(bb.get("x2", 1)) - float(bb.get("x1", 0)))
                    h = max(1.0, float(bb.get("y2", 1)) - float(bb.get("y1", 0)))
                    if (w / h) < 0.65:
                        upright_count += 1
                if upright_count >= 2:
                    contradictory_signals.append(
                        SupportingSignal(
                            signal_type="Negative: Upright Posture Maintained",
                            description="Person posture returned to upright geometry shortly after event window.",
                            confidence=0.85,
                            timestamp=event_time,
                        )
                    )

        return contradictory_signals

    @classmethod
    def evaluate_person_down_negative_evidence(
        cls,
        track: Any,
        context: Any,
        event_time: float,
    ) -> List[SupportingSignal]:
        """Identify counter-evidence refuting a person down state."""
        contradictory_signals: List[SupportingSignal] = []
        motions = context.track_motions.get(track.track_id, []) if context else []
        post_motions = [m for m in motions if m.timestamp > event_time]

        if len(post_motions) >= 2 and any(m.velocity_estimate > 20.0 for m in post_motions):
            contradictory_signals.append(
                SupportingSignal(
                    signal_type="Negative: Resumed Walking Transit",
                    description=f"Person [{track.track_id}] resumed walking transit, refuting sustained person down state.",
                    confidence=0.90,
                    timestamp=event_time,
                )
            )

        if track.history_bboxes:
            for b_entry in track.history_bboxes:
                b = b_entry.get("bbox", {})
                x1 = float(b.get("x1", 100))
                y1 = float(b.get("y1", 100))
                if x1 < 25.0 or y1 < 25.0:
                    contradictory_signals.append(
                        SupportingSignal(
                            signal_type="Negative: Frame Boundary Truncation",
                            description="Subject was occluded or clipped by camera edge, precluding reliable low-mobility classification.",
                            confidence=0.90,
                            timestamp=event_time,
                        )
                    )
                    break

        return contradictory_signals

    @classmethod
    def evaluate_running_negative_evidence(
        cls,
        track: Any,
        context: Any,
    ) -> List[SupportingSignal]:
        """Identify counter-evidence refuting panic running."""
        contradictory_signals: List[SupportingSignal] = []
        # If all other people in the scene are moving at a similar pace, it's normal scene flow
        if context and len(context.tracks) >= 3:
            velocities = []
            for t in context.tracks:
                summary = context.get_motion_summary(t.track_id) or {}
                if summary.get("avg_velocity", 0.0) > 0.0:
                    velocities.append(summary["avg_velocity"])
            if velocities:
                mean_v = sum(velocities) / len(velocities)
                t_summary = context.get_motion_summary(track.track_id) or {}
                t_v = t_summary.get("avg_velocity", mean_v)
                if abs(t_v - mean_v) < (mean_v * 0.35):
                    contradictory_signals.append(
                        SupportingSignal(
                            signal_type="Negative: Normal Pedestrian Transit Pace",
                            description="Subject speed is consistent with overall scene pedestrian transit pace.",
                            confidence=0.85,
                        )
                    )
        return contradictory_signals

    @classmethod
    def evaluate_altercation_negative_evidence(
        cls,
        p1: Any,
        p2: Any,
        context: Any,
        event_time: float,
    ) -> List[SupportingSignal]:
        """Identify counter-evidence refuting a physical altercation."""
        contradictory_signals: List[SupportingSignal] = []
        pts1 = p1.trajectory or []
        pts2 = p2.trajectory or []

        if len(pts1) >= 3 and len(pts2) >= 3:
            # Check for parallel side-by-side walking
            dx1 = pts1[-1][1] - pts1[0][1]
            dy1 = pts1[-1][2] - pts1[0][2]
            dx2 = pts2[-1][1] - pts2[0][1]
            dy2 = pts2[-1][2] - pts2[0][2]
            m1 = math.hypot(dx1, dy1)
            m2 = math.hypot(dx2, dy2)

            if m1 > 30.0 and m2 > 30.0:
                ang1 = math.degrees(math.atan2(dy1, dx1)) % 360.0
                ang2 = math.degrees(math.atan2(dy2, dx2)) % 360.0
                diff = abs(ang1 - ang2) % 360.0
                if diff > 180.0:
                    diff = 360.0 - diff
                if diff <= 20.0:
                    contradictory_signals.append(
                        SupportingSignal(
                            signal_type="Negative: Parallel Side-by-Side Walking",
                            description=f"Persons [{p1.track_id}] and [{p2.track_id}] maintained parallel walking transit ({diff:.1f} deg alignment).",
                            confidence=0.92,
                            timestamp=event_time,
                        )
                    )

            # Transient passing pedestrians (opposing transit without scuffle)
            net_dot = (dx1 * dx2) + (dy1 * dy2)
            if net_dot < -200.0 and m1 > 35.0 and m2 > 35.0:
                contradictory_signals.append(
                    SupportingSignal(
                        signal_type="Negative: Transient Passing Pedestrians",
                        description=f"Persons [{p1.track_id}] and [{p2.track_id}] transited in opposing directions along normal walking vectors.",
                        confidence=0.91,
                        timestamp=event_time,
                    )
                )

        return contradictory_signals

    @classmethod
    def evaluate_forced_movement_negative_evidence(
        cls,
        p1: Any,
        p2: Any,
        context: Any,
        event_time: float,
    ) -> List[SupportingSignal]:
        """Identify counter-evidence refuting forced movement."""
        contradictory_signals: List[SupportingSignal] = []
        pts1 = p1.trajectory or []
        pts2 = p2.trajectory or []
        if len(pts1) >= 3 and len(pts2) >= 3:
            dx1 = pts1[-1][1] - pts1[0][1]
            dy1 = pts1[-1][2] - pts1[0][2]
            dx2 = pts2[-1][1] - pts2[0][1]
            dy2 = pts2[-1][2] - pts2[0][2]
            if math.hypot(dx1, dy1) > 25.0 and math.hypot(dx2, dy2) > 25.0:
                ang1 = math.degrees(math.atan2(dy1, dx1)) % 360.0
                ang2 = math.degrees(math.atan2(dy2, dx2)) % 360.0
                diff = abs(ang1 - ang2) % 360.0
                if diff > 180.0:
                    diff = 360.0 - diff

                # Check if trajectory is straight (smooth transit) vs abrupt deviation
                has_sharp_turn = False
                if len(pts1) >= 4:
                    headings = []
                    for k in range(len(pts1) - 1):
                        step_dx = pts1[k + 1][1] - pts1[k][1]
                        step_dy = pts1[k + 1][2] - pts1[k][2]
                        if math.hypot(step_dx, step_dy) > 10.0:
                            headings.append(math.degrees(math.atan2(step_dy, step_dx)))
                    for h_idx in range(len(headings) - 1):
                        h_diff = abs(headings[h_idx + 1] - headings[h_idx]) % 360.0
                        if h_diff > 180.0:
                            h_diff = 360.0 - h_diff
                        if h_diff >= 40.0:
                            has_sharp_turn = True
                            break

                if diff <= 15.0 and not has_sharp_turn:
                    contradictory_signals.append(
                        SupportingSignal(
                            signal_type="Negative: Normal Pair Walking Abreast",
                            description="Pair traveled in smooth parallel transit without constrained course deviations.",
                            confidence=0.88,
                            timestamp=event_time,
                        )
                    )
        return contradictory_signals

    @classmethod
    def evaluate_following_negative_evidence(
        cls,
        p1: Any,
        p2: Any,
        context: Any,
        event_time: float,
    ) -> List[SupportingSignal]:
        """Identify counter-evidence refuting aggressive following."""
        contradictory_signals: List[SupportingSignal] = []
        summary1 = context.get_motion_summary(p1.track_id) or {} if context else {}
        summary2 = context.get_motion_summary(p2.track_id) or {} if context else {}

        if summary1.get("is_predominantly_stationary") and summary2.get("is_predominantly_stationary"):
            contradictory_signals.append(
                SupportingSignal(
                    signal_type="Negative: Pedestrian Queue Formation",
                    description="Both individuals remained stationary, characteristic of a queue or service wait.",
                    confidence=0.90,
                    timestamp=event_time,
                )
            )

        pts1 = p1.trajectory or []
        pts2 = p2.trajectory or []
        if len(pts1) >= 3 and len(pts2) >= 3:
            dx1 = pts1[-1][1] - pts1[0][1]
            dy1 = pts1[-1][2] - pts1[0][2]
            dx2 = pts2[-1][1] - pts2[0][1]
            dy2 = pts2[-1][2] - pts2[0][2]
            m1 = math.hypot(dx1, dy1)
            m2 = math.hypot(dx2, dy2)
            if m1 > 25.0 and m2 > 25.0:
                ang1 = math.degrees(math.atan2(dy1, dx1)) % 360.0
                ang2 = math.degrees(math.atan2(dy2, dx2)) % 360.0
                diff = abs(ang1 - ang2) % 360.0
                if diff > 180.0:
                    diff = 360.0 - diff
                if diff <= 15.0 and abs(m1 - m2) < 20.0:
                    # Traveled with near-identical displacement concurrently
                    contradictory_signals.append(
                        SupportingSignal(
                            signal_type="Negative: Symmetric Trajectory Correlation",
                            description="Pair traveled in concurrent lockstep transit without asymmetric leader-follower temporal lag.",
                            confidence=0.86,
                            timestamp=event_time,
                        )
                    )

        return contradictory_signals

    @classmethod
    def evaluate_abandoned_object_negative_evidence(
        cls,
        obj_track: TrackedObject,
        context: IncidentContext,
    ) -> List[SupportingSignal]:
        """Identify counter-evidence refuting abandoned object hypothesis."""
        contradictory_signals: List[SupportingSignal] = []
        if not obj_track or not context:
            return contradictory_signals

        # 1. Negative Evidence: Person within immediate proximity
        for p in context.tracks:
            if p.object_class == "person" and p.trajectory and obj_track.trajectory:
                last_o = obj_track.trajectory[-1]
                # Check recent person positions near object
                recent_p = [pt for pt in p.trajectory if abs(pt[0] - last_o[0]) <= 2.0]
                for pt in recent_p:
                    d = math.hypot(pt[1] - last_o[1], pt[2] - last_o[2])
                    if d <= 75.0:
                        contradictory_signals.append(
                            SupportingSignal(
                                signal_type="Negative: Person In Proximity",
                                description=f"Tracked person [{p.track_id}] is within {d:.1f}px of {obj_track.object_class}",
                                confidence=0.90,
                                timestamp=last_o[0],
                            )
                        )
                        return contradictory_signals

        # 2. Negative Evidence: Continuous motion (being carried)
        summary = context.get_motion_summary(obj_track.track_id) or {}
        if summary.get("net_displacement", 0.0) > 40.0:
            contradictory_signals.append(
                SupportingSignal(
                    signal_type="Negative: Moving Belonging",
                    description=f"{obj_track.object_class} exhibited continuous displacement ({summary.get('net_displacement', 0.0):.1f}px)",
                    confidence=0.88,
                    timestamp=obj_track.first_seen,
                )
            )

        return contradictory_signals

    @classmethod
    def evaluate_object_left_behind_negative_evidence(
        cls,
        person_track: TrackedObject,
        obj_track: TrackedObject,
        context: IncidentContext,
    ) -> List[SupportingSignal]:
        """Identify counter-evidence indicating an object was not left behind."""
        contradictory_signals: List[SupportingSignal] = []
        if not person_track or not obj_track:
            return contradictory_signals

        # Person still near object at the end of track
        if person_track.trajectory and obj_track.trajectory:
            p_end = person_track.trajectory[-1]
            o_end = obj_track.trajectory[-1]
            d = math.hypot(p_end[1] - o_end[1], p_end[2] - o_end[2])
            if d < 60.0:
                contradictory_signals.append(
                    SupportingSignal(
                        signal_type="Negative: Person Remained With Object",
                        description=f"Person [{person_track.track_id}] remained within {d:.1f}px of {obj_track.object_class}",
                        confidence=0.88,
                        timestamp=p_end[0],
                    )
                )

        return contradictory_signals

    @classmethod
    def evaluate_object_displacement_negative_evidence(
        cls,
        obj_track: TrackedObject,
        context: IncidentContext,
    ) -> List[SupportingSignal]:
        """Identify counter-evidence refuting significant object displacement."""
        contradictory_signals: List[SupportingSignal] = []
        if not obj_track or not obj_track.trajectory or len(obj_track.trajectory) < 2:
            return contradictory_signals

        start_pt = obj_track.trajectory[0]
        end_pt = obj_track.trajectory[-1]
        disp = math.hypot(end_pt[1] - start_pt[1], end_pt[2] - start_pt[2])
        if disp < 25.0:
            contradictory_signals.append(
                SupportingSignal(
                    signal_type="Negative: Displacement Below Noise Threshold",
                    description=f"Observed net displacement ({disp:.1f}px) is within normal sensor jitter limits",
                    confidence=0.90,
                    timestamp=end_pt[0],
                )
            )

        return contradictory_signals

    @classmethod
    def evaluate_crowd_surge_negative_evidence(
        cls,
        person_tracks: List[TrackedObject],
        context: IncidentContext,
        is_queue: bool = False,
        directional_coherence: float = 0.0,
        mean_velocity: float = 0.0,
    ) -> List[SupportingSignal]:
        """Identify counter-evidence refuting a crowd surge hypothesis."""
        contradictory_signals: List[SupportingSignal] = []
        t_ref = context.video_metadata.get("current_timestamp", 0.0) if context.video_metadata else 0.0

        if is_queue:
            contradictory_signals.append(
                SupportingSignal(
                    signal_type="Negative: Orderly Queue Flow",
                    description="Observed pedestrian grouping forms an orderly linear queue arrangement with uniform heading",
                    confidence=0.92,
                    timestamp=t_ref,
                    metadata={"queue_detected": True},
                )
            )

        if directional_coherence < 0.40 and len(person_tracks) >= 3:
            contradictory_signals.append(
                SupportingSignal(
                    signal_type="Negative: Dispersed Non-Cohesive Movement",
                    description=f"Pedestrian movement lacks directional alignment (coherence: {directional_coherence:.2f} < 0.40)",
                    confidence=0.85,
                    timestamp=t_ref,
                )
            )

        if mean_velocity < 12.0 and len(person_tracks) >= 2:
            contradictory_signals.append(
                SupportingSignal(
                    signal_type="Negative: Steady Scene Baseline Transit",
                    description=f"Average group velocity ({mean_velocity:.1f}px/s) is within stationary/normal walking limits",
                    confidence=0.80,
                    timestamp=t_ref,
                )
            )

        return contradictory_signals

    @classmethod
    def evaluate_restricted_zone_crowding_negative_evidence(
        cls,
        zone: ZoneDefinition,
        occupant_tracks: List[TrackedObject],
        dwell_seconds: float,
        is_restricted: bool = True,
    ) -> List[SupportingSignal]:
        """Identify counter-evidence refuting restricted-zone crowding."""
        contradictory_signals: List[SupportingSignal] = []
        z_name = getattr(zone, "name", getattr(zone, "zone_name", "Zone"))
        z_type = getattr(zone, "zone_type", "").lower()

        if not is_restricted or z_type not in ["restricted", "sterile"]:
            contradictory_signals.append(
                SupportingSignal(
                    signal_type="Negative: Unrestricted Access Zone",
                    description=f"Zone '{z_name}' is not configured as restricted or sterile access",
                    confidence=0.95,
                )
            )

        if dwell_seconds < 2.5:
            contradictory_signals.append(
                SupportingSignal(
                    signal_type="Negative: Transient Boundary Crossing",
                    description=f"Occupant presence ({dwell_seconds:.1f}s) is brief transit/boundary crossing (< 2.5s minimum dwell threshold)",
                    confidence=0.88,
                )
            )

        if len(occupant_tracks) < 2:
            contradictory_signals.append(
                SupportingSignal(
                    signal_type="Negative: Insufficient Occupant Count",
                    description=f"Single occupant observed ({len(occupant_tracks)} person); does not satisfy multi-occupant crowding definition",
                    confidence=0.90,
                )
            )

        return contradictory_signals

    # -----------------------------------------------------------------------
    # Phase 15: Specialized Visual Negative Evidence Evaluators
    # -----------------------------------------------------------------------

    @classmethod
    def evaluate_fire_negative_evidence(
        cls,
        observation_count: int,
        persistence_duration: float,
        is_static_surface: bool = False,
        scene_context: Optional[Any] = None,
        is_roadway_lighting: bool = False,
        is_person_clothing: bool = False,
        has_smoke: bool = True,
        event_time: float = 0.0,
    ) -> List[SupportingSignal]:
        """Identify counter-evidence refuting a fire hypothesis."""
        from ai.specialized.negative_evidence import SpecializedNegativeEvidenceEngine
        contradictory: List[SupportingSignal] = []

        if observation_count <= 1 or persistence_duration < 0.4:
            contradictory.append(
                SupportingSignal(
                    signal_type="Negative: Transient Visual Glitch",
                    description=f"Fire visual observation observed in only {observation_count} frame(s) ({persistence_duration:.2f}s); lacks sustained flame persistence.",
                    confidence=0.90,
                    timestamp=event_time,
                )
            )

        if is_static_surface:
            contradictory.append(
                SupportingSignal(
                    signal_type="Negative: Static Surface Chromaticity",
                    description="Observed chromatic region shows negligible dynamic flicker; consistent with painted orange/red fixture.",
                    confidence=0.88,
                    timestamp=event_time,
                )
            )

        if is_person_clothing:
            contradictory.append(
                SupportingSignal(
                    signal_type="Negative: Person Clothing / Accessory Chromaticity",
                    description="Visual chromaticity co-located with pedestrian clothing/accessories without thermal combustion.",
                    confidence=0.95,
                    timestamp=event_time,
                )
            )

        if not has_smoke:
            contradictory.append(
                SupportingSignal(
                    signal_type="Negative: Absence of Smoke Plume",
                    description="Indoor environment shows zero corresponding aerosol smoke plume.",
                    confidence=0.82,
                    timestamp=event_time,
                )
            )

        if is_roadway_lighting or (scene_context and getattr(scene_context, "scene_type", "") == "roadway"):
            contradictory.append(
                SupportingSignal(
                    signal_type="Negative: Potential Vehicle Lighting Glare",
                    description="Roadway context luminescence consistent with vehicle tail lamps or specular reflections.",
                    confidence=0.75,
                    timestamp=event_time,
                )
            )

        return contradictory

    @classmethod
    def evaluate_smoke_negative_evidence(
        cls,
        observation_count: int,
        persistence_duration: float,
        is_global_haze: bool = False,
        is_compression_artifact: bool = False,
        is_static_surface: bool = False,
        event_time: float = 0.0,
    ) -> List[SupportingSignal]:
        """Identify counter-evidence refuting a smoke hypothesis."""
        contradictory: List[SupportingSignal] = []

        if observation_count <= 1 or persistence_duration < 0.5:
            contradictory.append(
                SupportingSignal(
                    signal_type="Negative: Transient Dispersion Artifact",
                    description=f"Smoke-like pattern observed in only {observation_count} frame(s) ({persistence_duration:.2f}s); lacks persistent plume dynamics.",
                    confidence=0.88,
                    timestamp=event_time,
                )
            )

        if is_static_surface:
            contradictory.append(
                SupportingSignal(
                    signal_type="Negative: Static Surface Texture",
                    description="Visual texture remains spatially anchored without volumetric plume deformation, dispersion, or drift; matches road surface, wall, or static architectural fixture.",
                    confidence=0.92,
                    timestamp=event_time,
                )
            )

        if is_global_haze:
            contradictory.append(
                SupportingSignal(
                    signal_type="Negative: Global Atmospheric Haze",
                    description="Atmospheric condition exhibits uniform low saturation across entire visual field.",
                    confidence=0.85,
                    timestamp=event_time,
                )
            )

        if is_compression_artifact:
            contradictory.append(
                SupportingSignal(
                    signal_type="Negative: Video Compression Artifact",
                    description="High macro-block compression artifact level in video stream matches pseudo-smoke texture.",
                    confidence=0.82,
                    timestamp=event_time,
                )
            )

        return contradictory

    @classmethod
    def evaluate_weapon_negative_evidence(
        cls,
        pixel_resolution: Tuple[int, int],
        associated_object_class: Optional[str] = None,
        event_time: float = 0.0,
    ) -> List[SupportingSignal]:
        """Identify counter-evidence refuting a weapon hypothesis."""
        contradictory: List[SupportingSignal] = []
        w, h = pixel_resolution

        if w < 32 or h < 32:
            contradictory.append(
                SupportingSignal(
                    signal_type="Negative: Insufficient Optical Resolution",
                    description=f"Object crop resolution ({w}x{h}px) is below forensic classification threshold (32x32px).",
                    confidence=0.92,
                    timestamp=event_time,
                )
            )

        if associated_object_class in ["cell phone", "bottle", "umbrella", "backpack", "handbag"]:
            contradictory.append(
                SupportingSignal(
                    signal_type="Negative: Common Personal Belonging Context",
                    description=f"Object localization aligns with non-threatening category '{associated_object_class}'.",
                    confidence=0.85,
                    timestamp=event_time,
                )
            )

        return contradictory
