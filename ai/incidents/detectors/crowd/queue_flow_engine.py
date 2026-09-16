"""
Queue and Orderly Flow Engine (Phase 14)

Evaluates pedestrian groupings for linear spatial alignment, uniform headings,
and low velocity variance indicative of orderly queueing or structured commute flow.
Operates primarily as a negative evidence engine to prevent false crowd surge alarms.
"""
from dataclasses import dataclass
import math
from typing import List, Dict, Any, Optional, Tuple

from ai.schemas import TrackedObject
from ai.incidents.schemas import SupportingSignal


@dataclass
class QueueAssessment:
    is_queue_detected: bool
    queue_member_count: int
    linear_fit_score: float
    heading_coherence: float
    member_track_ids: List[str]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_queue_detected": self.is_queue_detected,
            "queue_member_count": self.queue_member_count,
            "linear_fit_score": round(self.linear_fit_score, 2),
            "heading_coherence": round(self.heading_coherence, 2),
            "member_track_ids": self.member_track_ids,
            "explanation": self.explanation,
        }


class QueueAndFlowEngine:
    """
    Evaluates groups of simultaneous pedestrian tracks for queue / transit structure.
    """

    def __init__(
        self,
        min_queue_persons: int = 3,
        max_collinear_scatter_px: float = 35.0,
        max_heading_diff_deg: float = 30.0,
    ):
        self.min_queue_persons = min_queue_persons
        self.max_collinear_scatter_px = max_collinear_scatter_px
        self.max_heading_diff_deg = max_heading_diff_deg

    def evaluate_queue_pattern(
        self,
        person_tracks: List[TrackedObject],
        eval_time: float,
    ) -> QueueAssessment:
        """
        Determines if active pedestrians at eval_time form an orderly queue or linear stream.
        """
        active_points: List[Tuple[str, float, float, float]] = []  # (track_id, cx, cy, heading)

        for pt in person_tracks:
            if not pt.trajectory:
                continue
            # Find point near eval_time
            matching = [p for p in pt.trajectory if abs(p[0] - eval_time) <= 1.0]
            if matching:
                p_curr = matching[0]
                # Calculate heading if trajectory has history
                heading = 0.0
                if len(pt.trajectory) >= 2:
                    dx = pt.trajectory[-1][1] - pt.trajectory[0][1]
                    dy = pt.trajectory[-1][2] - pt.trajectory[0][2]
                    heading = math.degrees(math.atan2(dy, dx)) % 360.0
                active_points.append((pt.track_id, p_curr[1], p_curr[2], heading))

        if len(active_points) < self.min_queue_persons:
            return QueueAssessment(
                is_queue_detected=False,
                queue_member_count=len(active_points),
                linear_fit_score=0.0,
                heading_coherence=0.0,
                member_track_ids=[p[0] for p in active_points],
                explanation="Insufficient simultaneous pedestrians for queue assessment.",
            )

        # 1. Evaluate Linearity via Principal Axis Scatter
        xs = [p[1] for p in active_points]
        ys = [p[2] for p in active_points]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)

        # Approximate line orientation via covariance
        sxx = sum((x - mean_x) ** 2 for x in xs)
        syy = sum((y - mean_y) ** 2 for y in ys)
        sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))

        # Angle of primary line
        theta = 0.5 * math.atan2(2 * sxy, sxx - syy)
        # Normal vector to line: (-sin theta, cos theta)
        nx = -math.sin(theta)
        ny = math.cos(theta)

        # Perpendicular distances to line
        perp_dists = [abs((x - mean_x) * nx + (y - mean_y) * ny) for x, y in zip(xs, ys)]
        mean_scatter = sum(perp_dists) / len(perp_dists)

        # Normalized linear score (1.0 = perfectly collinear)
        linear_score = max(0.0, min(1.0, 1.0 - (mean_scatter / self.max_collinear_scatter_px)))

        # Parallel span along the principal line to verify elongation
        par_coords = [(x - mean_x) * math.cos(theta) + (y - mean_y) * math.sin(theta) for x, y in zip(xs, ys)]
        span_length = max(par_coords) - min(par_coords) if par_coords else 0.0
        is_elongated = span_length >= max(60.0, 2.2 * mean_scatter)

        # 2. Evaluate Heading Coherence
        headings = [p[3] for p in active_points]
        # Angular variance check
        cos_sum = sum(math.cos(math.radians(h)) for h in headings)
        sin_sum = sum(math.sin(math.radians(h)) for h in headings)
        r = math.hypot(cos_sum, sin_sum) / len(headings)  # 1.0 = identical headings

        # Velocity check: an orderly queue moves at slow/moderate advance speed (<= 22px/s)
        velocities = []
        for pt in person_tracks:
            if pt.trajectory and len(pt.trajectory) >= 2:
                dt = pt.trajectory[-1][0] - pt.trajectory[0][0]
                if dt > 0:
                    dist = math.hypot(pt.trajectory[-1][1] - pt.trajectory[0][1], pt.trajectory[-1][2] - pt.trajectory[0][2])
                    velocities.append(dist / dt)
        mean_speed = (sum(velocities) / len(velocities)) if velocities else 0.0

        is_queue = (
            linear_score >= 0.70
            and is_elongated
            and (r >= 0.70 or mean_scatter <= 15.0)
            and (mean_speed <= 22.0 or not velocities)
        )

        explanation = (
            f"Linear scatter: {mean_scatter:.1f}px (fit score: {linear_score:.2f}, elongation: {span_length:.1f}px, speed: {mean_speed:.1f}px/s), "
            f"heading coherence: {r:.2f}. "
            + ("Orderly queue / transit alignment detected." if is_queue else "Non-queue 2D cluster, rapid movement, or dispersion.")
        )

        return QueueAssessment(
            is_queue_detected=is_queue,
            queue_member_count=len(active_points),
            linear_fit_score=linear_score,
            heading_coherence=r,
            member_track_ids=[p[0] for p in active_points],
            explanation=explanation,
        )
