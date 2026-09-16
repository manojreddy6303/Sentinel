"""
Spatial Relationship Engine (Phase 10)

Provides reusable, high-performance 2D spatial reasoning:
- bounding-box overlap & IoU
- centroid distance & relative directional orientation
- proximity & interaction radius checks
- 2D trajectory segment intersection
- polygon zone membership (ray casting), entry, exit, and dwell tracking
"""
import math
from typing import List, Tuple, Optional, Dict, Any

from ai.schemas import BoundingBox, TrackedObject, ZoneDefinition


class SpatialRelationshipEngine:
    """
    Core spatial calculation utility for Sentinel incident detectors.
    """

    @staticmethod
    def iou(box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        if not box1 or not box2:
            return 0.0
        ix1 = max(box1.x1, box2.x1)
        iy1 = max(box1.y1, box2.y1)
        ix2 = min(box1.x2, box2.x2)
        iy2 = min(box1.y2, box2.y2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter_area = iw * ih

        box1_area = max(0.0, box1.x2 - box1.x1) * max(0.0, box1.y2 - box1.y1)
        box2_area = max(0.0, box2.x2 - box2.x1) * max(0.0, box2.y2 - box2.y1)
        union_area = box1_area + box2_area - inter_area
        if union_area <= 0.0:
            return 0.0
        return inter_area / union_area

    @staticmethod
    def bbox_overlap(box1: BoundingBox, box2: BoundingBox) -> bool:
        """Check if two bounding boxes have any overlapping area."""
        if not box1 or not box2:
            return False
        return not (
            box1.x2 < box2.x1 or
            box1.x1 > box2.x2 or
            box1.y2 < box2.y1 or
            box1.y1 > box2.y2
        )

    @staticmethod
    def centroid_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Euclidean distance between two 2D points."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def bbox_distance(box1: BoundingBox, box2: BoundingBox) -> float:
        """Distance between the centroids of two bounding boxes."""
        c1 = box1.centroid
        c2 = box2.centroid
        return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

    @staticmethod
    def relative_orientation(from_point: Tuple[float, float], to_point: Tuple[float, float]) -> str:
        """
        Determine coarse relative directional relationship of `to_point` from `from_point`.
        Returns: 'north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest'.
        """
        dx = to_point[0] - from_point[0]
        dy = to_point[1] - from_point[1]
        if abs(dx) < 1e-3 and abs(dy) < 1e-3:
            return "coincident"

        angle = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
        # 0 deg is east (+x), 90 deg is south (+y in image coordinates)
        if 337.5 <= angle or angle < 22.5:
            return "east"
        elif 22.5 <= angle < 67.5:
            return "southeast"
        elif 67.5 <= angle < 112.5:
            return "south"
        elif 112.5 <= angle < 157.5:
            return "southwest"
        elif 157.5 <= angle < 202.5:
            return "west"
        elif 202.5 <= angle < 247.5:
            return "northwest"
        elif 247.5 <= angle < 292.5:
            return "north"
        else:
            return "northeast"

    @staticmethod
    def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """
        Ray-casting algorithm to test if a point (x, y) is inside a polygon.
        """
        if not polygon or len(polygon) < 3:
            return False

        x, y = point
        inside = False
        n = len(polygon)
        p1x, p1y = polygon[0]

        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if min(p1y, p2y) < y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        x_inters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= x_inters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    @classmethod
    def track_zone_dwell_analysis(
        cls,
        track: TrackedObject,
        zone: ZoneDefinition,
    ) -> Dict[str, Any]:
        """
        Analyze a track's spatial lifecycle with respect to a defined security zone:
        - entered: bool
        - exited: bool
        - entry_time: Optional[float]
        - exit_time: Optional[float]
        - total_dwell_seconds: float
        - inside_observations_count: int
        """
        if not track.trajectory or not zone.polygon:
            return {
                "entered": False,
                "exited": False,
                "entry_time": None,
                "exit_time": None,
                "total_dwell_seconds": 0.0,
                "inside_observations_count": 0,
            }

        poly = [(float(p[0]), float(p[1])) for p in zone.polygon]
        inside_flags = []
        for i, pt in enumerate(track.trajectory):
            t, cx, cy = pt
            inside = cls.point_in_polygon((cx, cy), poly)
            if not inside and i < len(track.history_bboxes):
                b = track.history_bboxes[i].get("bbox", {})
                y2 = float(b.get("y2", cy))
                inside = cls.point_in_polygon((cx, y2), poly)
            inside_flags.append(inside)

        inside_count = sum(1 for f in inside_flags if f)
        if inside_count == 0:
            return {
                "entered": False,
                "exited": False,
                "entry_time": None,
                "exit_time": None,
                "total_dwell_seconds": 0.0,
                "inside_observations_count": 0,
            }

        first_in_idx = inside_flags.index(True)
        last_in_idx = len(inside_flags) - 1 - inside_flags[::-1].index(True)

        entry_t = track.trajectory[first_in_idx][0]
        exit_t = track.trajectory[last_in_idx][0]
        dwell_time = max(0.0, exit_t - entry_t)

        exited = (last_in_idx < len(inside_flags) - 1) and not inside_flags[-1]

        return {
            "entered": True,
            "exited": exited,
            "entry_time": entry_t,
            "exit_time": exit_t if exited else None,
            "total_dwell_seconds": dwell_time,
            "inside_observations_count": inside_count,
        }

    @staticmethod
    def check_segments_intersect(
        p1: Tuple[float, float], p2: Tuple[float, float],
        p3: Tuple[float, float], p4: Tuple[float, float],
    ) -> bool:
        """Check whether line segment (p1-p2) intersects with (p3-p4)."""
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

    @classmethod
    def check_trajectory_intersection(
        cls,
        track_a: TrackedObject,
        track_b: TrackedObject,
        temporal_window: Optional[Tuple[float, float]] = None,
    ) -> bool:
        """
        Determine if the 2D trajectories of two tracks cross each other.
        Optionally bounded by a temporal window (start_t, end_t).
        """
        traj_a = track_a.trajectory
        traj_b = track_b.trajectory
        if len(traj_a) < 2 or len(traj_b) < 2:
            return False

        for i in range(len(traj_a) - 1):
            t1, x1, y1 = traj_a[i]
            t2, x2, y2 = traj_a[i + 1]
            if temporal_window:
                if t2 < temporal_window[0] or t1 > temporal_window[1]:
                    continue

            for j in range(len(traj_b) - 1):
                tb1, xb1, yb1 = traj_b[j]
                tb2, xb2, yb2 = traj_b[j + 1]
                if temporal_window:
                    if tb2 < temporal_window[0] or tb1 > temporal_window[1]:
                        continue

                if cls.check_segments_intersect((x1, y1), (x2, y2), (xb1, yb1), (xb2, yb2)):
                    return True

        return False
