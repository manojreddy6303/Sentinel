"""
Person Motion Feature Engine (Sentinel Phase 12)

Extracts reusable kinematic, geometric, and dynamic motion metrics for tracked persons:
- Aspect-ratio dynamics (upright walking ~0.3-0.5 vs horizontal/fallen >= 0.9)
- Vertical displacement & downward descent velocity
- Scale-normalized velocity (body-lengths per second)
- Heading oscillation and abrupt deflection rates
- Pairwise lagged trajectory correlation (person following)
- Pairwise reciprocal motion and rapid distance oscillation (physical altercation)
"""
import math
from typing import Dict, Any, List, Optional, Tuple

from ai.schemas import TrackedObject, BoundingBox
from ai.incidents.spatial import SpatialRelationshipEngine
from ai.incidents.temporal import TemporalAnalysisEngine


class PersonMotionFeatureEngine:
    """
    Computes single-track and pairwise person dynamic features for incident analysis.
    """

    @classmethod
    def compute_person_dynamics(
        cls,
        track: TrackedObject,
        context_motions: Optional[Dict[str, Any]] = None,
        frame_width: Optional[float] = None,
        frame_height: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Extract holistic posture and movement dynamics across a person track.
        Includes 4-border frame clipping detection and multi-signal temporal persistence.
        """
        if not track or not track.history_bboxes:
            return {
                "track_id": getattr(track, "track_id", "unknown"),
                "duration_seconds": 0.0,
                "observation_count": 0,
                "has_aspect_ratio_transition": False,
                "has_posture_persistence": False,
                "max_downward_velocity_px_s": 0.0,
                "max_downward_velocity_bl_s": 0.0,
                "max_normalized_speed": 0.0,
                "avg_normalized_speed": 0.0,
                "is_low_mobility": False,
                "is_edge_clipped": False,
                "is_transition_edge_clipped": False,
                "is_track_terminating_at_boundary": False,
            }

        sorted_history = sorted(track.history_bboxes, key=lambda b: float(b["timestamp"]))
        observations: List[Dict[str, Any]] = []

        margin_x = max(18.0, (frame_width * 0.035) if frame_width else 20.0)
        margin_y = max(18.0, (frame_height * 0.035) if frame_height else 20.0)

        for item in sorted_history:
            t = float(item["timestamp"])
            bbox_dict = item["bbox"]
            x1 = float(bbox_dict["x1"])
            y1 = float(bbox_dict["y1"])
            x2 = float(bbox_dict["x2"])
            y2 = float(bbox_dict["y2"])
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            ar = w / h  # width / height (aspect ratio)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # 4-edge boundary clipping detection
            clipped_edges = []
            if x1 <= margin_x:
                clipped_edges.append("left")
            if y1 <= margin_y:
                clipped_edges.append("top")
            if frame_width and x2 >= (frame_width - margin_x):
                clipped_edges.append("right")
            if frame_height and y2 >= (frame_height - margin_y):
                clipped_edges.append("bottom")
            if w < 15.0 or h < 15.0:
                clipped_edges.append("collapsed")

            is_clipped = len(clipped_edges) > 0

            observations.append({
                "timestamp": t,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": w,
                "height": h,
                "aspect_ratio": ar,
                "cx": cx,
                "cy": cy,
                "ground_y": y2,
                "is_border_clipped": is_clipped,
                "clipped_edges": clipped_edges,
            })

        if not observations:
            return {"track_id": track.track_id, "duration_seconds": 0.0, "observation_count": 0}

        # 1. Aspect Ratio Dynamics & Temporal Sequence Verification
        # True fall requires:
        # UPRIGHT (ar < 0.65) -> genuine rapid descent -> horizontal (ar >= 0.85) -> persistent low posture
        aspect_ratios = [o["aspect_ratio"] for o in observations]
        initial_ar = aspect_ratios[0]
        final_ar = aspect_ratios[-1]
        min_ar = min(aspect_ratios)
        max_ar = max(aspect_ratios)

        has_aspect_ratio_transition = False
        has_posture_persistence = False
        is_transition_edge_clipped = False
        transition_timestamp = None
        transition_duration = 0.0

        for i in range(len(observations)):
            if observations[i]["aspect_ratio"] < 0.65:
                # Look forward up to 2.5 seconds
                for j in range(i + 1, len(observations)):
                    dt = observations[j]["timestamp"] - observations[i]["timestamp"]
                    if dt > 2.5:
                        break
                    if observations[j]["aspect_ratio"] >= 0.85:
                        # Check 1: Was transition observation clipped by an image boundary?
                        if observations[j]["is_border_clipped"]:
                            is_transition_edge_clipped = True
                            continue

                        # Check 2: Verify head descent (y1 increases downward in image coordinates)
                        head_descent = observations[j]["y1"] - observations[i]["y1"]
                        if head_descent < (observations[i]["height"] * 0.35):
                            continue

                        # Check 3: Temporal Persistence (must not be a 1-frame anomaly or terminal disappearing frame)
                        # Does horizontal/low posture persist for at least 1 follow-up observation or dwell time?
                        if j >= len(observations) - 1:
                            # Track ends immediately at transition with ZERO post-event evidence!
                            continue

                        subsequent = observations[j + 1:]
                        if not subsequent:
                            continue

                        # Check if subsequent observation sustains low posture or if person immediately bounces upright
                        next_ar = subsequent[0]["aspect_ratio"]
                        # Crouch/bend quick recovery check: returns to upright within 1.5s
                        quick_recovery = any(
                            s["aspect_ratio"] < 0.60
                            for s in subsequent
                            if (s["timestamp"] - observations[j]["timestamp"]) <= 1.5
                        )
                        if quick_recovery and next_ar < 0.70:
                            continue

                        # Persistence confirmed if next observation is >= 0.70 or average remaining >= 0.70
                        avg_subsequent_ar = sum(s["aspect_ratio"] for s in subsequent) / len(subsequent)
                        if next_ar >= 0.70 or avg_subsequent_ar >= 0.70:
                            has_aspect_ratio_transition = True
                            has_posture_persistence = True
                            transition_timestamp = observations[j]["timestamp"]
                            transition_duration = dt
                            break

            if has_aspect_ratio_transition:
                break

        # 2. Downward Vertical Displacement & Velocity
        max_downward_disp = 0.0
        max_downward_vel = 0.0
        downward_event_time = None

        for i in range(len(observations) - 1):
            o_curr = observations[i]
            for j in range(i + 1, len(observations)):
                dt = observations[j]["timestamp"] - o_curr["timestamp"]
                if dt <= 0.0:
                    continue
                if dt > 2.5:
                    break
                d_cy = observations[j]["cy"] - o_curr["cy"]
                if d_cy > 0:
                    v_down = d_cy / dt
                    if v_down > max_downward_vel:
                        max_downward_vel = v_down
                        max_downward_disp = d_cy
                        downward_event_time = observations[j]["timestamp"]

        # 3. Normalized Velocities (Body-Lengths / Second)
        avg_height = sum(o["height"] for o in observations) / len(observations)
        normalized_speeds = []

        for i in range(len(observations) - 1):
            o1 = observations[i]
            o2 = observations[i + 1]
            dt = max(1e-3, o2["timestamp"] - o1["timestamp"])
            dist_px = math.hypot(o2["cx"] - o1["cx"], o2["cy"] - o1["cy"])
            vel_px_s = dist_px / dt
            body_lengths_s = vel_px_s / max(10.0, avg_height)
            normalized_speeds.append(body_lengths_s)

        max_norm_speed = max(normalized_speeds) if normalized_speeds else 0.0
        avg_norm_speed = (sum(normalized_speeds) / len(normalized_speeds)) if normalized_speeds else 0.0

        # 4. Low Mobility / Ground Dwell
        is_low_mobility = False
        horizontal_dwell_seconds = 0.0

        if len(observations) >= 3:
            final_segment = [o for o in observations if o["timestamp"] >= observations[-1]["timestamp"] - 3.5]
            if final_segment:
                final_ar_avg = sum(o["aspect_ratio"] for o in final_segment) / len(final_segment)
                disp_final = math.hypot(
                    final_segment[-1]["cx"] - final_segment[0]["cx"],
                    final_segment[-1]["cy"] - final_segment[0]["cy"],
                )
                if final_ar_avg >= 0.75 and disp_final < (avg_height * 0.40):
                    is_low_mobility = True
                    horizontal_dwell_seconds = final_segment[-1]["timestamp"] - final_segment[0]["timestamp"]

        # 5. Image Boundary Clipping / Exit Suppression
        # Detect if person bounding box is clipped by camera frame edges
        is_edge_clipped = any(o["is_border_clipped"] for o in observations)
        # Check if track terminates at or near an image boundary (exit rather than fall)
        is_track_terminating_at_boundary = False
        if observations:
            last_obs = observations[-1]
            if last_obs["is_border_clipped"]:
                is_track_terminating_at_boundary = True

        return {
            "track_id": track.track_id,
            "duration_seconds": round(track.duration_seconds, 2),
            "observation_count": len(observations),
            "avg_height": round(avg_height, 1),
            "initial_aspect_ratio": round(initial_ar, 3),
            "final_aspect_ratio": round(final_ar, 3),
            "min_aspect_ratio": round(min_ar, 3),
            "max_aspect_ratio": round(max_ar, 3),
            "has_aspect_ratio_transition": has_aspect_ratio_transition,
            "has_posture_persistence": has_posture_persistence,
            "transition_timestamp": transition_timestamp,
            "transition_duration": round(transition_duration, 2),
            "max_downward_displacement_px": round(max_downward_disp, 1),
            "max_downward_velocity_px_s": round(max_downward_vel, 1),
            "max_downward_velocity_bl_s": round(max_downward_vel / max(10.0, avg_height), 2),
            "downward_event_time": downward_event_time,
            "max_normalized_speed_bl_s": round(max_norm_speed, 2),
            "avg_normalized_speed_bl_s": round(avg_norm_speed, 2),
            "is_low_mobility": is_low_mobility,
            "horizontal_dwell_seconds": round(horizontal_dwell_seconds, 2),
            "is_edge_clipped": is_edge_clipped,
            "is_transition_edge_clipped": is_transition_edge_clipped,
            "is_track_terminating_at_boundary": is_track_terminating_at_boundary,
            "observations": observations,
        }

    @classmethod
    def compute_lagged_trajectory_similarity(
        cls,
        p1: TrackedObject,
        p2: TrackedObject,
        max_lag_seconds: float = 3.0,
    ) -> Dict[str, Any]:
        """
        Evaluate if Person 2 is persistently following Person 1's trajectory with a time lag.
        Enforces true directional asymmetry:
        - Person 1 must be spatially ahead of Person 2 along movement heading.
        - Lagged trajectory error must be significantly lower than simultaneous zero-lag error (not side-by-side).
        - Both tracks must exhibit meaningful spatial transit.
        """
        pts1 = p1.trajectory or []
        pts2 = p2.trajectory or []
        if len(pts1) < 4 or len(pts2) < 4:
            return {"is_following": False, "confidence": 0.0, "reason": "Insufficient trajectory points"}

        overlap = TemporalAnalysisEngine.temporal_intersection(
            p1.first_seen, p1.last_seen,
            p2.first_seen, p2.last_seen,
        )
        if not overlap or (overlap[1] - overlap[0]) < 2.0:
            return {"is_following": False, "confidence": 0.0, "reason": "Insufficient temporal overlap"}

        # Net displacement check: following requires active movement across space
        dx1 = pts1[-1][1] - pts1[0][1]
        dy1 = pts1[-1][2] - pts1[0][2]
        disp1 = math.hypot(dx1, dy1)
        dx2 = pts2[-1][1] - pts2[0][1]
        dy2 = pts2[-1][2] - pts2[0][2]
        disp2 = math.hypot(dx2, dy2)
        if disp1 < 25.0 or disp2 < 25.0:
            return {"is_following": False, "confidence": 0.0, "reason": "Insufficient spatial transit"}

        # Compute simultaneous (zero-lag) separation
        simul_errors = []
        precedence_dots = []
        heading_unit = (dx1 / max(1e-5, disp1), dy1 / max(1e-5, disp1))

        for t2, cx2, cy2 in pts2:
            near_p1 = [p for p in pts1 if abs(p[0] - t2) <= 0.35]
            if near_p1:
                p1_match = min(near_p1, key=lambda p: abs(p[0] - t2))
                simul_errors.append(math.hypot(cx2 - p1_match[1], cy2 - p1_match[2]))
                # Relative vector from p2 (follower) to p1 (leader)
                vec_x = p1_match[1] - cx2
                vec_y = p1_match[2] - cy2
                precedence_dots.append((vec_x * heading_unit[0]) + (vec_y * heading_unit[1]))

        zero_lag_error = (sum(simul_errors) / len(simul_errors)) if simul_errors else float("inf")
        avg_precedence = (sum(precedence_dots) / len(precedence_dots)) if precedence_dots else -1.0

        # Spatial precedence check: p1 must be physically ahead of p2 along path of travel!
        if avg_precedence < 10.0:
            return {"is_following": False, "confidence": 0.0, "reason": "Negative or ambiguous spatial precedence"}

        # Test lags from 0.5s to max_lag_seconds in 0.5s increments
        best_lag = 0.0
        best_mean_error = float("inf")
        matched_points_count = 0

        for lag in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            errors = []
            for t2, cx2, cy2 in pts2:
                target_t1 = t2 - lag
                near_p1 = [p for p in pts1 if abs(p[0] - target_t1) <= 0.35]
                if near_p1:
                    p1_match = min(near_p1, key=lambda p: abs(p[0] - target_t1))
                    err = math.hypot(cx2 - p1_match[1], cy2 - p1_match[2])
                    errors.append(err)

            if len(errors) >= 3:
                mean_err = sum(errors) / len(errors)
                if mean_err < best_mean_error:
                    best_mean_error = mean_err
                    best_lag = lag
                    matched_points_count = len(errors)

        # Scale tolerance relative to person dimension
        char_dim_1 = math.sqrt(p1.current_bbox.width * p1.current_bbox.height) if p1.current_bbox else 50.0
        char_dim_2 = math.sqrt(p2.current_bbox.width * p2.current_bbox.height) if p2.current_bbox else 50.0
        avg_scale = (char_dim_1 + char_dim_2) / 2.0

        # True following requires:
        # 1. Path deviation at lag is small (< 1.25 body scale)
        # 2. Lagged error is significantly BETTER than simultaneous walking side-by-side (< 0.75 * zero_lag_error)
        #    OR zero-lag separation was substantial (> 1.2 * avg_scale) while lagged deviation is tight
        is_side_by_side = (zero_lag_error < (avg_scale * 1.5)) and (best_mean_error >= (zero_lag_error * 0.75))
        is_following = (best_mean_error < (avg_scale * 1.25)) and (matched_points_count >= 3) and not is_side_by_side

        conf = 0.0
        if is_following:
            conf = max(0.50, min(0.85, 1.0 - (best_mean_error / (avg_scale * 2.0))))

        return {
            "is_following": is_following,
            "best_lag_seconds": round(best_lag, 2),
            "mean_path_deviation_px": round(best_mean_error, 1),
            "zero_lag_error_px": round(zero_lag_error, 1),
            "spatial_precedence_px": round(avg_precedence, 1),
            "matched_points_count": matched_points_count,
            "confidence": round(conf, 3),
        }

    @classmethod
    def compute_reciprocal_motion(
        cls,
        p1: TrackedObject,
        p2: TrackedObject,
    ) -> Dict[str, Any]:
        """
        Evaluate physical altercation motion patterns:
        - Sustained close proximity
        - Repeated reciprocal movement (opposing heading vectors and rapid distance expansions/contractions)
        - Excludes transient pedestrian walk-pasts and static proximity
        """
        pts1 = p1.trajectory or []
        pts2 = p2.trajectory or []
        if len(pts1) < 4 or len(pts2) < 4:
            return {"reciprocal_score": 0.0, "is_reciprocal": False, "min_distance_px": float("inf"), "is_passing": False}

        # Find simultaneous observations
        simultaneous: List[Tuple[float, Tuple[float, float], Tuple[float, float]]] = []
        for t1, x1, y1 in pts1:
            near_p2 = [p for p in pts2 if abs(p[0] - t1) <= 0.40]
            if near_p2:
                p2_match = min(near_p2, key=lambda p: abs(p[0] - t1))
                simultaneous.append((t1, (x1, y1), (p2_match[1], p2_match[2])))

        if len(simultaneous) < 4:
            return {"reciprocal_score": 0.0, "is_reciprocal": False, "min_distance_px": float("inf"), "is_passing": False}

        distances = [math.hypot(p1_pos[0] - p2_pos[0], p1_pos[1] - p2_pos[1]) for _, p1_pos, p2_pos in simultaneous]
        min_dist = min(distances)

        char_dim_1 = math.sqrt(p1.current_bbox.width * p1.current_bbox.height) if p1.current_bbox else 50.0
        char_dim_2 = math.sqrt(p2.current_bbox.width * p2.current_bbox.height) if p2.current_bbox else 50.0
        mean_scale = (char_dim_1 + char_dim_2) / 2.0

        # Transient passing check: pedestrians traveling in opposite directions whose distance
        # monotonically decreases then increases without localized scuffle
        dx1 = pts1[-1][1] - pts1[0][1]
        dy1 = pts1[-1][2] - pts1[0][2]
        dx2 = pts2[-1][1] - pts2[0][1]
        dy2 = pts2[-1][2] - pts2[0][2]
        disp1 = math.hypot(dx1, dy1)
        disp2 = math.hypot(dx2, dy2)
        net_dot = (dx1 * dx2) + (dy1 * dy2)

        # Oscillation count: requires at least 15px change to avoid detection noise
        oscillations = 0
        deltas = [distances[i + 1] - distances[i] for i in range(len(distances) - 1)]
        for i in range(len(deltas) - 1):
            if (deltas[i] * deltas[i + 1]) < -100.0 and abs(deltas[i]) >= 10.0 and abs(deltas[i + 1]) >= 10.0:
                oscillations += 1

        # Opposing velocity vectors when close
        opposing_count = 0
        agitated_velocity_count = 0
        for i in range(len(simultaneous) - 1):
            dt = max(1e-3, simultaneous[i + 1][0] - simultaneous[i][0])
            v1_dx = (simultaneous[i + 1][1][0] - simultaneous[i][1][0]) / dt
            v1_dy = (simultaneous[i + 1][1][1] - simultaneous[i][1][1]) / dt
            v2_dx = (simultaneous[i + 1][2][0] - simultaneous[i][2][0]) / dt
            v2_dy = (simultaneous[i + 1][2][1] - simultaneous[i][2][1]) / dt
            spd1 = math.hypot(v1_dx, v1_dy)
            spd2 = math.hypot(v2_dx, v2_dy)
            if spd1 > 40.0 or spd2 > 40.0:
                agitated_velocity_count += 1

            dot = (v1_dx * v2_dx) + (v1_dy * v2_dy)
            if dot < -400.0:  # actively moving in opposing directions
                opposing_count += 1

        # Check for normal passing walk-by
        is_passing = False
        if net_dot < -200.0 and disp1 > 40.0 and disp2 > 40.0 and oscillations <= 1:
            is_passing = True

        reciprocal_score = (oscillations * 0.35) + (opposing_count * 0.20) + (min(2, agitated_velocity_count) * 0.15)
        # Real altercation requires: close proximity, multiple oscillations (not a single pass),
        # genuine opposing vectors or elevated kinetic agitation, and NOT a simple pass-by.
        # Proximity or ordinary movement alone must NOT become physical altercation.
        has_active_conflict = (opposing_count >= 1 or agitated_velocity_count >= 1)
        is_reciprocal = (
            (min_dist <= mean_scale * 1.20)
            and (oscillations >= 2)
            and (reciprocal_score >= 0.65)
            and has_active_conflict
            and not is_passing
        )

        return {
            "reciprocal_score": round(reciprocal_score, 2),
            "is_reciprocal": is_reciprocal,
            "min_distance_px": round(min_dist, 1),
            "oscillations": oscillations,
            "opposing_count": opposing_count,
            "mean_scale_px": round(mean_scale, 1),
            "is_passing": is_passing,
        }
