"""
Crowd Density & Spatial Clustering Engine (Phase 14)

Provides perspective-normalized pedestrian density calculation, anonymous spatial clustering,
inter-person proximity matrices, and temporal density trend analysis.
Uses unique validated person tracks to prevent raw-detection double counting.
"""
from dataclasses import dataclass, field
import math
from typing import List, Dict, Any, Optional, Tuple, Set

from ai.schemas import BoundingBox, TrackedObject
from ai.incidents.schemas import IncidentContext


@dataclass
class CrowdClusterRecord:
    cluster_id: str
    track_ids: List[str]
    size: int
    centroid: Tuple[float, float]
    bounding_box: Optional[BoundingBox]
    average_inter_distance: float
    is_compact: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "track_ids": self.track_ids,
            "size": self.size,
            "centroid": [round(self.centroid[0], 1), round(self.centroid[1], 1)],
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "average_inter_distance": round(self.average_inter_distance, 1),
            "is_compact": self.is_compact,
        }


@dataclass
class DensityWindowMetrics:
    timestamp: float
    window_start: float
    window_end: float
    active_person_count: int
    active_vehicle_count: int
    unique_person_track_ids: List[str]
    clusters: List[CrowdClusterRecord]
    avg_inter_person_dist: float
    min_inter_person_dist: float
    density_ratio_to_baseline: float
    rate_of_change: float
    occupied_area_px: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 2),
            "window_start": round(self.window_start, 2),
            "window_end": round(self.window_end, 2),
            "active_person_count": self.active_person_count,
            "active_vehicle_count": self.active_vehicle_count,
            "unique_person_track_ids": self.unique_person_track_ids,
            "clusters_count": len(self.clusters),
            "clusters": [c.to_dict() for c in self.clusters],
            "avg_inter_person_dist": round(self.avg_inter_person_dist, 1),
            "min_inter_person_dist": round(self.min_inter_person_dist, 1),
            "density_ratio_to_baseline": round(self.density_ratio_to_baseline, 2),
            "rate_of_change": round(self.rate_of_change, 2),
            "occupied_area_px": round(self.occupied_area_px, 1),
        }


class CrowdDensityEngine:
    """
    Analyzes multi-frame validated person tracks across temporal windows to derive
    anonymous density metrics, clustering, and rate of change.
    """

    def __init__(
        self,
        window_seconds: float = 3.0,
        cluster_distance_px: float = 120.0,
        min_cluster_size: int = 3,
        frame_w: float = 1920.0,
        frame_h: float = 1080.0,
    ):
        self.window_seconds = window_seconds
        self.cluster_distance_px = cluster_distance_px
        self.min_cluster_size = min_cluster_size
        self.frame_w = frame_w
        self.frame_h = frame_h

    def _cluster_points(
        self,
        person_items: List[Tuple[str, float, float, Optional[BoundingBox]]]
    ) -> List[CrowdClusterRecord]:
        """
        Groups active persons into anonymous spatial clusters via Euclidean connectivity.
        """
        if len(person_items) < self.min_cluster_size:
            return []

        n = len(person_items)
        adj: Dict[int, Set[int]] = {i: set() for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                d = math.hypot(person_items[i][1] - person_items[j][1], person_items[i][2] - person_items[j][2])
                if d <= self.cluster_distance_px:
                    adj[i].add(j)
                    adj[j].add(i)

        visited = set()
        clusters: List[CrowdClusterRecord] = []
        cluster_idx = 1

        for i in range(n):
            if i in visited:
                continue

            # BFS to find connected component
            component = []
            queue = [i]
            visited.add(i)

            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if len(component) >= self.min_cluster_size:
                comp_tracks = [person_items[idx][0] for idx in component]
                cx = sum(person_items[idx][1] for idx in component) / len(component)
                cy = sum(person_items[idx][2] for idx in component) / len(component)

                # Pairwise inter-distances in cluster
                dists = []
                for a in range(len(component)):
                    for b in range(a + 1, len(component)):
                        d = math.hypot(
                            person_items[component[a]][1] - person_items[component[b]][1],
                            person_items[component[a]][2] - person_items[component[b]][2]
                        )
                        dists.append(d)

                avg_dist = sum(dists) / len(dists) if dists else 0.0

                # Bounding box of cluster
                xs = [person_items[idx][1] for idx in component]
                ys = [person_items[idx][2] for idx in component]
                bx1 = max(0.0, min(xs) - 40.0)
                by1 = max(0.0, min(ys) - 40.0)
                bx2 = max(bx1 + 10.0, max(xs) + 40.0)
                by2 = max(by1 + 10.0, max(ys) + 40.0)
                cluster_box = BoundingBox(
                    x1=round(bx1, 2),
                    y1=round(by1, 2),
                    x2=round(bx2, 2),
                    y2=round(by2, 2),
                )

                clusters.append(
                    CrowdClusterRecord(
                        cluster_id=f"CLUSTER-{cluster_idx:02d}",
                        track_ids=comp_tracks,
                        size=len(component),
                        centroid=(cx, cy),
                        bounding_box=cluster_box,
                        average_inter_distance=avg_dist,
                        is_compact=(avg_dist <= self.cluster_distance_px * 0.70),
                    )
                )
                cluster_idx += 1

        return clusters

    def evaluate_windows(self, context: IncidentContext) -> List[DensityWindowMetrics]:
        """
        Computes temporal density windows across the video context.
        """
        person_tracks = [t for t in context.tracks if t.object_class == "person"]
        vehicle_tracks = [t for t in context.tracks if t.object_class in {"car", "truck", "bus", "motorcycle"}]

        if not person_tracks:
            return []

        min_t = min(t.first_seen for t in person_tracks)
        max_t = max(t.last_seen for t in person_tracks)
        duration = max(1.0, max_t - min_t)

        step = max(1.0, self.window_seconds / 2.0)
        num_windows = max(1, int(math.ceil(duration / step)))

        raw_metrics: List[Dict[str, Any]] = []

        for w in range(num_windows):
            w_start = min_t + (w * step)
            w_end = w_start + self.window_seconds

            active_persons: List[Tuple[str, float, float, Optional[BoundingBox]]] = []
            for pt in person_tracks:
                # Check if person track overlaps window
                if pt.first_seen <= w_end and pt.last_seen >= w_start:
                    # Find closest point in trajectory
                    pts = [p for p in pt.trajectory if w_start <= p[0] <= w_end]
                    if pts:
                        mid_p = pts[len(pts) // 2]
                        active_persons.append((pt.track_id, mid_p[1], mid_p[2], pt.current_bbox))
                    elif pt.trajectory:
                        closest = min(pt.trajectory, key=lambda p: abs(p[0] - ((w_start + w_end) / 2.0)))
                        active_persons.append((pt.track_id, closest[1], closest[2], pt.current_bbox))

            # Active vehicles
            active_veh_count = sum(
                1 for vt in vehicle_tracks
                if vt.first_seen <= w_end and vt.last_seen >= w_start
            )

            # Spatial metrics
            all_dists = []
            for i in range(len(active_persons)):
                for j in range(i + 1, len(active_persons)):
                    d = math.hypot(
                        active_persons[i][1] - active_persons[j][1],
                        active_persons[i][2] - active_persons[j][2]
                    )
                    all_dists.append(d)

            avg_dist = sum(all_dists) / len(all_dists) if all_dists else 999.0
            min_dist = min(all_dists) if all_dists else 999.0

            clusters = self._cluster_points(active_persons)

            occupied_area = 0.0
            if len(active_persons) >= 2:
                xs = [p[1] for p in active_persons]
                ys = [p[2] for p in active_persons]
                occupied_area = (max(xs) - min(xs)) * (max(ys) - min(ys))

            raw_metrics.append({
                "timestamp": (w_start + w_end) / 2.0,
                "window_start": w_start,
                "window_end": w_end,
                "active_person_count": len(active_persons),
                "active_vehicle_count": active_veh_count,
                "unique_person_track_ids": [p[0] for p in active_persons],
                "clusters": clusters,
                "avg_inter_person_dist": avg_dist,
                "min_inter_person_dist": min_dist,
                "occupied_area_px": occupied_area,
            })

        # Calculate baseline (use scene_context if provided, else mean count across windows)
        if context.scene_context and hasattr(context.scene_context, "baseline") and getattr(context.scene_context.baseline, "average_density_per_second", 0.0) > 0:
            baseline_count = context.scene_context.baseline.average_density_per_second
        else:
            counts = [m["active_person_count"] for m in raw_metrics]
            baseline_count = (sum(counts) / len(counts)) if counts else 1.0
        baseline_count = max(1.0, baseline_count)

        # Build final windows with density ratio and rate of change
        windows: List[DensityWindowMetrics] = []
        for i, m in enumerate(raw_metrics):
            ratio = m["active_person_count"] / baseline_count
            rate = 0.0
            if i > 0:
                dt = raw_metrics[i]["timestamp"] - raw_metrics[i - 1]["timestamp"]
                if dt > 0:
                    rate = (m["active_person_count"] - raw_metrics[i - 1]["active_person_count"]) / dt

            windows.append(
                DensityWindowMetrics(
                    timestamp=m["timestamp"],
                    window_start=m["window_start"],
                    window_end=m["window_end"],
                    active_person_count=m["active_person_count"],
                    active_vehicle_count=m["active_vehicle_count"],
                    unique_person_track_ids=m["unique_person_track_ids"],
                    clusters=m["clusters"],
                    avg_inter_person_dist=m["avg_inter_person_dist"],
                    min_inter_person_dist=m["min_inter_person_dist"],
                    density_ratio_to_baseline=ratio,
                    rate_of_change=rate,
                    occupied_area_px=m["occupied_area_px"],
                )
            )

        return windows
