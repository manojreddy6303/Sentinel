"""
ai/correlation/spatial_engine.py
================================
Spatial reasoning, bounding box overlap, centroid metrics, and trajectory convergence
for Phase 16 incident correlation.
"""

import math
from typing import Dict, Any, Optional, Tuple
from ai.schemas import BoundingBox
from ai.common.numeric import ensure_finite


class CorrelationSpatialEngine:
    """
    Computes spatial proximity, bounding box contact, and normalized distance metrics.
    """

    @staticmethod
    def box_iou(b1: Any, b2: Any) -> float:
        """Calculate Intersection over Union of two bounding boxes or dicts."""
        if not b1 or not b2:
            return 0.0

        if isinstance(b1, dict):
            x1_a, y1_a, x2_a, y2_a = b1.get("x1", 0.0), b1.get("y1", 0.0), b1.get("x2", 0.0), b1.get("y2", 0.0)
        else:
            x1_a, y1_a, x2_a, y2_a = getattr(b1, "x1", 0.0), getattr(b1, "y1", 0.0), getattr(b1, "x2", 0.0), getattr(b1, "y2", 0.0)

        if isinstance(b2, dict):
            x1_b, y1_b, x2_b, y2_b = b2.get("x1", 0.0), b2.get("y1", 0.0), b2.get("x2", 0.0), b2.get("y2", 0.0)
        else:
            x1_b, y1_b, x2_b, y2_b = getattr(b2, "x1", 0.0), getattr(b2, "y1", 0.0), getattr(b2, "x2", 0.0), getattr(b2, "y2", 0.0)

        inter_x1 = max(x1_a, x1_b)
        inter_y1 = max(y1_a, y1_b)
        inter_x2 = min(x2_a, x2_b)
        inter_y2 = min(y2_a, y2_b)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = max(0.0, (x2_a - x1_a) * (y2_a - y1_a))
        area_b = max(0.0, (x2_b - x1_b) * (y2_b - y1_b))
        union_area = area_a + area_b - inter_area
        if union_area <= 0.0:
            return 0.0
        return ensure_finite(inter_area / union_area, 0.0)

    @staticmethod
    def centroid_distance(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
        """Euclidean distance between two centroids."""
        if not c1 or not c2:
            return 999999.0
        return ensure_finite(math.hypot(c1[0] - c2[0], c1[1] - c2[1]), 999999.0)

    @staticmethod
    def are_spatially_proximate(
        b1: Any,
        b2: Any,
        max_centroid_dist: float = 120.0,
        min_iou: float = 0.01,
        threshold: Optional[float] = None,
    ) -> bool:
        """Determines if two entities are close enough to interact."""
        eff_iou = threshold if threshold is not None else min_iou
        iou = CorrelationSpatialEngine.box_iou(b1, b2)
        if iou >= eff_iou and iou > 0.0:
            return True

        if isinstance(b1, dict):
            x1_a, y1_a, x2_a, y2_a = b1.get("x1", 0.0), b1.get("y1", 0.0), b1.get("x2", 0.0), b1.get("y2", 0.0)
        else:
            x1_a, y1_a, x2_a, y2_a = getattr(b1, "x1", 0.0), getattr(b1, "y1", 0.0), getattr(b1, "x2", 0.0), getattr(b1, "y2", 0.0)

        if isinstance(b2, dict):
            x1_b, y1_b, x2_b, y2_b = b2.get("x1", 0.0), b2.get("y1", 0.0), b2.get("x2", 0.0), b2.get("y2", 0.0)
        else:
            x1_b, y1_b, x2_b, y2_b = getattr(b2, "x1", 0.0), getattr(b2, "y1", 0.0), getattr(b2, "x2", 0.0), getattr(b2, "y2", 0.0)

        c1 = ((x1_a + x2_a) / 2.0, (y1_a + y2_a) / 2.0)
        c2 = ((x1_b + x2_b) / 2.0, (y1_b + y2_b) / 2.0)

        dist = CorrelationSpatialEngine.centroid_distance(c1, c2)

        # Scale-invariant characteristic dimension for resolution generalization
        diag_a = math.hypot(abs(x2_a - x1_a), abs(y2_a - y1_a))
        diag_b = math.hypot(abs(x2_b - x1_b), abs(y2_b - y1_b))
        char_scale = (diag_a + diag_b) / 2.0

        if char_scale > 0.0 and (dist / char_scale) <= 2.0:
            return True

        return dist <= max_centroid_dist

    @classmethod
    def compute_iou(cls, b1: Any, b2: Any) -> float:
        return cls.box_iou(b1, b2)

    @staticmethod
    def is_in_zone(b: Any, polygon: list) -> bool:
        """Determines if bounding box centroid is within a polygon (Ray-casting algorithm)."""
        if not b or not polygon or len(polygon) < 3:
            return False
        if isinstance(b, dict):
            cx = (b.get("x1", 0.0) + b.get("x2", 0.0)) / 2.0
            cy = (b.get("y1", 0.0) + b.get("y2", 0.0)) / 2.0
        else:
            cx = (getattr(b, "x1", 0.0) + getattr(b, "x2", 0.0)) / 2.0
            cy = (getattr(b, "y1", 0.0) + getattr(b, "y2", 0.0)) / 2.0

        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if cy > min(p1y, p2y):
                if cy <= max(p1y, p2y):
                    if cx <= max(p1x, p2x):
                        if p1y != p2y:
                            xints = (cy - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or cx <= xints:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
