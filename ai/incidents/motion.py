"""
Universal Motion Engine (Phase 10)

Calculates derived motion telemetry from multi-frame tracking data:
- displacement & cumulative distance traveled
- instantaneous direction, velocity estimate, and acceleration estimate
- speed change & sudden deceleration/acceleration
- stationary vs moving state tracking & dwell durations
- path consistency (straightness ratio)
- inter-track relative distance, approach rate, and separation rate

CRITICAL SAFETY CONSTRAINT:
All metrics are in pixel/normalized coordinate space. Never claims real-world km/h
or mph unless explicit camera calibration matrices are present.
"""
import math
import logging
from typing import List, Dict, Any, Optional, Tuple

from ai.schemas import TrackedObject
from ai.incidents.schemas import TrackMotion

logger = logging.getLogger(__name__)


class UniversalMotionEngine:
    """
    Modular motion analysis engine for Sentinel tracks.
    """

    def __init__(
        self,
        stationary_speed_threshold: float = 12.0,   # pixels/sec below which an object is considered stationary
        stationary_disp_threshold: float = 20.0,    # maximum displacement over window to be stationary
    ):
        self.stationary_speed_threshold = stationary_speed_threshold
        self.stationary_disp_threshold = stationary_disp_threshold

    def compute_track_motion(
        self,
        track: TrackedObject,
        frame_width: Optional[float] = None,
        frame_height: Optional[float] = None,
    ) -> List[TrackMotion]:
        """
        Compute frame-by-frame TrackMotion telemetry along a track's trajectory.
        Trajectory format: [(timestamp, cx, cy), ...]
        Applies resolution-normalized speed and displacement thresholds.
        """
        if not track or not track.trajectory:
            return []

        trajectory = track.trajectory
        n = len(trajectory)
        if n == 0:
            return []

        # Resolution scale factor relative to 1080p canonical diagonal (2202.906)
        CANONICAL_REF_DIAGONAL = 2202.906
        if frame_width and frame_height and frame_width > 0 and frame_height > 0:
            scale_factor = math.hypot(frame_width, frame_height) / CANONICAL_REF_DIAGONAL
        elif track and track.current_bbox and (track.current_bbox.x2 > 1920 or track.current_bbox.y2 > 1080):
            scale_factor = math.hypot(3840, 2160) / CANONICAL_REF_DIAGONAL
        else:
            scale_factor = 1.0

        effective_speed_threshold = self.stationary_speed_threshold * scale_factor
        effective_min_step = 2.0 * scale_factor

        motions: List[TrackMotion] = []
        origin_t, origin_x, origin_y = trajectory[0]

        cumulative_distance = 0.0
        consecutive_stationary_sec = 0.0
        consecutive_moving_sec = 0.0
        prev_velocity = 0.0

        for i in range(n):
            curr_t, curr_x, curr_y = trajectory[i]

            # Net displacement from origin
            net_disp = math.hypot(curr_x - origin_x, curr_y - origin_y)

            if i == 0:
                # Initial frame baseline
                motion = TrackMotion(
                    track_id=track.track_id,
                    timestamp=curr_t,
                    position=(curr_x, curr_y),
                    displacement=0.0,
                    distance_traveled=0.0,
                    direction_radians=0.0,
                    direction_degrees=0.0,
                    velocity_estimate=0.0,
                    acceleration_estimate=0.0,
                    speed_change=0.0,
                    is_stationary=True,
                    stationary_duration=0.0,
                    movement_duration=0.0,
                    path_consistency=1.0,
                    confidence=track.confidence,
                )
                motions.append(motion)
                continue

            prev_t, prev_x, prev_y = trajectory[i - 1]
            raw_dt = curr_t - prev_t
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            step_distance = math.hypot(dx, dy)
            cumulative_distance += step_distance

            # Direction & Angle
            direction_rad = math.atan2(dy, dx)
            direction_deg = (math.degrees(direction_rad) + 360.0) % 360.0

            # Guard against invalid dt (<= 0 or not finite): return safe zero motion
            if raw_dt <= 0.0 or not math.isfinite(raw_dt):
                instant_velocity = 0.0
                acceleration = 0.0
                speed_delta = 0.0
                dt = 0.0
            else:
                dt = raw_dt
                instant_velocity = step_distance / dt
                acceleration = (instant_velocity - prev_velocity) / dt
                speed_delta = abs(instant_velocity - prev_velocity)

            # Ensure all metrics are strictly finite
            if not math.isfinite(instant_velocity):
                instant_velocity = 0.0
            if not math.isfinite(acceleration):
                acceleration = 0.0
            if not math.isfinite(speed_delta):
                speed_delta = 0.0

            # Stationary evaluation with resolution-scaled thresholds
            is_stat = instant_velocity < effective_speed_threshold or step_distance < effective_min_step
            if is_stat:
                consecutive_stationary_sec += dt
                consecutive_moving_sec = 0.0
            else:
                consecutive_moving_sec += dt
                consecutive_stationary_sec = 0.0

            # Path consistency (straightness ratio: net_displacement / cumulative_distance)
            consistency = (net_disp / cumulative_distance) if cumulative_distance > 1e-3 else 1.0
            consistency = max(0.0, min(1.0, consistency))

            motion = TrackMotion(
                track_id=track.track_id,
                timestamp=curr_t,
                position=(curr_x, curr_y),
                displacement=net_disp,
                distance_traveled=cumulative_distance,
                direction_radians=direction_rad,
                direction_degrees=direction_deg,
                velocity_estimate=instant_velocity,
                acceleration_estimate=acceleration,
                speed_change=speed_delta,
                is_stationary=is_stat,
                stationary_duration=consecutive_stationary_sec,
                movement_duration=consecutive_moving_sec,
                path_consistency=consistency,
                confidence=track.confidence,
            )
            motions.append(motion)
            prev_velocity = instant_velocity

        return motions

    def summarize_track_motion(self, motions: List[TrackMotion]) -> Dict[str, Any]:
        """
        Generate aggregate statistical summary for a track's motion lifecycle.
        """
        if not motions:
            return {
                "total_duration": 0.0,
                "net_displacement": 0.0,
                "total_distance": 0.0,
                "avg_velocity": 0.0,
                "max_velocity": 0.0,
                "max_deceleration": 0.0,
                "path_consistency": 1.0,
                "is_predominantly_stationary": True,
                "max_stationary_duration": 0.0,
                "max_movement_duration": 0.0,
            }

        velocities = [m.velocity_estimate for m in motions]
        decelerations = [abs(m.acceleration_estimate) for m in motions if m.acceleration_estimate < 0]
        stationary_durations = [m.stationary_duration for m in motions]
        movement_durations = [m.movement_duration for m in motions]

        total_distance = motions[-1].distance_traveled
        net_disp = motions[-1].displacement
        consistency = motions[-1].path_consistency

        stationary_frames = sum(1 for m in motions if m.is_stationary)
        predominantly_stationary = (stationary_frames / len(motions)) >= 0.70

        return {
            "total_duration": round(motions[-1].timestamp - motions[0].timestamp, 3),
            "net_displacement": round(net_disp, 2),
            "total_distance": round(total_distance, 2),
            "avg_velocity": round(sum(velocities) / len(velocities), 2) if velocities else 0.0,
            "max_velocity": round(max(velocities), 2) if velocities else 0.0,
            "max_deceleration": round(max(decelerations), 2) if decelerations else 0.0,
            "path_consistency": round(consistency, 4),
            "is_predominantly_stationary": predominantly_stationary,
            "max_stationary_duration": round(max(stationary_durations), 2) if stationary_durations else 0.0,
            "max_movement_duration": round(max(movement_durations), 2) if movement_durations else 0.0,
        }

    @staticmethod
    def compute_inter_track_relative_motion(
        track_a: TrackedObject,
        track_b: TrackedObject,
        temporal_tolerance: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Compute relative distance, approach rate, and separation rate between two tracks
        at overlapping moments in time.
        """
        if not track_a or not track_b or not track_a.trajectory or not track_b.trajectory:
            return []

        # Find overlapping time range
        t_start = max(track_a.first_seen, track_b.first_seen)
        t_end = min(track_a.last_seen, track_b.last_seen)
        if t_start > t_end:
            return []

        rel_metrics: List[Dict[str, Any]] = []
        prev_dist: Optional[float] = None
        prev_t: Optional[float] = None

        # Sample at timestamps present in track_a
        for pt_a in track_a.trajectory:
            t, ax, ay = pt_a
            if t < t_start or t > t_end:
                continue

            # Find closest point in track_b
            closest_b = min(track_b.trajectory, key=lambda p: abs(p[0] - t))
            if abs(closest_b[0] - t) > temporal_tolerance:
                continue

            bx, by = closest_b[1], closest_b[2]
            dist = math.hypot(ax - bx, ay - by)

            approach_rate = 0.0
            separation_rate = 0.0
            if prev_dist is not None and prev_t is not None:
                raw_dt = t - prev_t
                if raw_dt > 0.0 and math.isfinite(raw_dt):
                    delta_dist = dist - prev_dist
                    rate = delta_dist / raw_dt  # negative if approaching, positive if separating
                    if rate < 0:
                        approach_rate = abs(rate)
                    else:
                        separation_rate = rate
                else:
                    approach_rate = 0.0
                    separation_rate = 0.0

            rel_metrics.append({
                "timestamp": round(t, 4),
                "distance": round(dist, 2),
                "approach_rate": round(approach_rate, 2),
                "separation_rate": round(separation_rate, 2),
                "pos_a": (round(ax, 2), round(ay, 2)),
                "pos_b": (round(bx, 2), round(by, 2)),
            })

            prev_dist = dist
            prev_t = t

        return rel_metrics
