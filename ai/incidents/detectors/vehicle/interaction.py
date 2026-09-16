"""
Vehicle Interaction Model (Sentinel Phase 11)

Reusable pairwise vehicle interaction engine computing:
- Normalized perspective distance (invariant to camera distance)
- Approach & separation dynamics (approach rate, separation rate)
- Contact geometry (bounding-box IoU, physical clearance)
- Heading & trajectory alignment (parallel lanes vs lateral convergence)
- Pre-event and post-event kinematics (velocities, decelerations)
"""
import math
from typing import Dict, Any, List, Optional, Tuple

from ai.schemas import TrackedObject, BoundingBox
from ai.incidents.spatial import SpatialRelationshipEngine
from ai.incidents.temporal import TemporalAnalysisEngine


class VehicleInteractionModel:
    """
    Computes holistic pairwise kinematic, spatial, and perspective metrics between two vehicles.
    """

    @classmethod
    def perspective_normalized_distance(cls, box1: Any, box2: Any) -> float:
        if isinstance(box1, (list, tuple)):
            b1 = BoundingBox(x1=box1[0], y1=box1[1], x2=box1[2], y2=box1[3])
        else:
            b1 = box1
        if isinstance(box2, (list, tuple)):
            b2 = BoundingBox(x1=box2[0], y1=box2[1], x2=box2[2], y2=box2[3])
        else:
            b2 = box2
        dist = SpatialRelationshipEngine.bbox_distance(b1, b2)
        char_dim_1 = math.sqrt(max(1.0, b1.width * b1.height))
        char_dim_2 = math.sqrt(max(1.0, b2.width * b2.height))
        mean_vehicle_scale = (char_dim_1 + char_dim_2) / 2.0
        return dist / max(1.0, mean_vehicle_scale)

    @classmethod
    def compute_iou(cls, box1: Any, box2: Any) -> float:
        if isinstance(box1, (list, tuple)):
            b1 = BoundingBox(x1=box1[0], y1=box1[1], x2=box1[2], y2=box1[3])
        else:
            b1 = box1
        if isinstance(box2, (list, tuple)):
            b2 = BoundingBox(x1=box2[0], y1=box2[1], x2=box2[2], y2=box2[3])
        else:
            b2 = box2
        return b1.iou(b2)

    @classmethod
    def relative_heading_degrees(cls, v1_heading: Tuple[float, float], v2_heading: Tuple[float, float]) -> float:
        dot = v1_heading[0] * v2_heading[0] + v1_heading[1] * v2_heading[1]
        mag1 = math.hypot(v1_heading[0], v1_heading[1])
        mag2 = math.hypot(v2_heading[0], v2_heading[1])
        if mag1 < 1e-4 or mag2 < 1e-4:
            return 0.0
        cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_theta))

    @classmethod
    def analyze_interaction(
        cls,
        v1: TrackedObject,
        v2: TrackedObject,
        context_motions: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze the full interaction between two vehicles across their overlapping temporal window.
        Returns a rich metrics dictionary or None if tracks do not overlap.
        """
        if not v1 or not v2:
            return None

        # Check temporal overlap
        overlap_window = TemporalAnalysisEngine.temporal_intersection(
            v1.first_seen, v1.last_seen,
            v2.first_seen, v2.last_seen,
        )
        if not overlap_window:
            return None

        # 1. Match simultaneous bounding boxes across time
        observations: List[Dict[str, Any]] = []
        for b1 in v1.history_bboxes:
            t1 = float(b1["timestamp"])
            # Find closest observation in v2 within 0.5s tolerance
            matching_b2 = [
                b2 for b2 in v2.history_bboxes
                if abs(float(b2["timestamp"]) - t1) <= 0.5
            ]
            if not matching_b2:
                continue

            b2 = min(matching_b2, key=lambda b: abs(float(b["timestamp"]) - t1))
            t2 = float(b2["timestamp"])
            mean_t = round((t1 + t2) / 2.0, 4)

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

            # Centroid distance
            dist = SpatialRelationshipEngine.bbox_distance(box1, box2)
            iou = box1.iou(box2)
            has_overlap = SpatialRelationshipEngine.bbox_overlap(box1, box2)

            # Perspective normalization: scale factor based on geometric mean of vehicle bbox dimensions
            char_dim_1 = math.sqrt(max(1.0, box1.width * box1.height))
            char_dim_2 = math.sqrt(max(1.0, box2.width * box2.height))
            mean_vehicle_scale = (char_dim_1 + char_dim_2) / 2.0
            norm_dist = dist / max(1.0, mean_vehicle_scale)

            observations.append({
                "timestamp": mean_t,
                "distance": dist,
                "normalized_distance": norm_dist,
                "iou": iou,
                "overlap": has_overlap,
                "scale": mean_vehicle_scale,
                "box1": box1,
                "box2": box2,
            })

        if not observations:
            return None

        # Sort chronologically
        observations.sort(key=lambda o: o["timestamp"])

        # 2. Min distance & closest point of approach (CPA)
        cpa_entry = min(observations, key=lambda o: o["distance"])
        min_dist = cpa_entry["distance"]
        min_norm_dist = cpa_entry["normalized_distance"]
        event_time = cpa_entry["timestamp"]
        has_physical_contact = any(o["overlap"] or o["iou"] > 0.0 for o in observations)
        max_iou = max(o["iou"] for o in observations)

        # 3. Dynamic approach & separation rates
        approach_rates = []
        separation_rates = []
        for i in range(len(observations) - 1):
            o_curr = observations[i]
            o_next = observations[i + 1]
            dt = max(1e-3, o_next["timestamp"] - o_curr["timestamp"])
            d_dist = o_next["distance"] - o_curr["distance"]
            rate = -d_dist / dt  # Positive rate means distance is decreasing (approaching)
            if rate > 0:
                approach_rates.append(rate)
            else:
                separation_rates.append(-rate)

        max_approach_rate = max(approach_rates) if approach_rates else 0.0
        max_separation_rate = max(separation_rates) if separation_rates else 0.0

        # 4. Heading vectors & angle difference
        relative_heading_deg = 0.0
        is_parallel_flow = False
        if len(v1.trajectory) >= 2 and len(v2.trajectory) >= 2:
            dx1 = v1.trajectory[-1][1] - v1.trajectory[0][1]
            dy1 = v1.trajectory[-1][2] - v1.trajectory[0][2]
            dx2 = v2.trajectory[-1][1] - v2.trajectory[0][1]
            dy2 = v2.trajectory[-1][2] - v2.trajectory[0][2]
            ang1 = math.atan2(dy1, dx1)
            ang2 = math.atan2(dy2, dx2)
            angle_diff = abs(math.degrees(ang1 - ang2)) % 360.0
            if angle_diff > 180.0:
                angle_diff = 360.0 - angle_diff
            relative_heading_deg = angle_diff
            is_parallel_flow = (angle_diff <= 25.0) or (angle_diff >= 155.0)

        # 5. Pre-event vs Post-event velocities
        pre_v1_vels = []
        post_v1_vels = []
        pre_v2_vels = []
        post_v2_vels = []

        if context_motions:
            m1_list = context_motions.get(v1.track_id, [])
            m2_list = context_motions.get(v2.track_id, [])
            for m in m1_list:
                if m.timestamp < event_time:
                    pre_v1_vels.append(m.velocity_estimate)
                elif m.timestamp > event_time:
                    post_v1_vels.append(m.velocity_estimate)
            for m in m2_list:
                if m.timestamp < event_time:
                    pre_v2_vels.append(m.velocity_estimate)
                elif m.timestamp > event_time:
                    post_v2_vels.append(m.velocity_estimate)

        pre_v1_avg = sum(pre_v1_vels) / len(pre_v1_vels) if pre_v1_vels else 0.0
        post_v1_avg = sum(post_v1_vels) / len(post_v1_vels) if post_v1_vels else 0.0
        pre_v2_avg = sum(pre_v2_vels) / len(pre_v2_vels) if pre_v2_vels else 0.0
        post_v2_avg = sum(post_v2_vels) / len(post_v2_vels) if post_v2_vels else 0.0

        v1_velocity_drop = max(0.0, pre_v1_avg - post_v1_avg)
        v2_velocity_drop = max(0.0, pre_v2_avg - post_v2_avg)

        return {
            "v1_track_id": v1.track_id,
            "v2_track_id": v2.track_id,
            "event_time": event_time,
            "overlap_window": overlap_window,
            "min_distance_px": round(min_dist, 2),
            "min_normalized_distance": round(min_norm_dist, 3),
            "has_physical_contact": has_physical_contact,
            "max_iou": round(max_iou, 4),
            "max_approach_rate_px_s": round(max_approach_rate, 2),
            "max_separation_rate_px_s": round(max_separation_rate, 2),
            "relative_heading_deg": round(relative_heading_deg, 1),
            "is_parallel_flow": is_parallel_flow,
            "pre_v1_velocity": round(pre_v1_avg, 2),
            "post_v1_velocity": round(post_v1_avg, 2),
            "pre_v2_velocity": round(pre_v2_avg, 2),
            "post_v2_velocity": round(post_v2_avg, 2),
            "v1_velocity_drop": round(v1_velocity_drop, 2),
            "v2_velocity_drop": round(v2_velocity_drop, 2),
            "cpa_box1": cpa_entry["box1"],
            "cpa_box2": cpa_entry["box2"],
            "observations_count": len(observations),
        }
