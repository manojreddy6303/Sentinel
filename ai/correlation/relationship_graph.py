"""
ai/correlation/relationship_graph.py
====================================
Observational relationship graph layer for Phase 16 incident correlation.
Detects and models physical entity interactions strictly based on geometric telemetry.
"""

from typing import List, Dict, Any, Optional, Tuple
from ai.correlation.models import (
    ObservationalRelationship,
    CorrelationRelationshipType,
)
from ai.correlation.spatial_engine import CorrelationSpatialEngine
from ai.correlation.temporal_engine import CorrelationTemporalEngine


class IncidentRelationshipGraph:
    """
    Extracts and stores observational relationships between tracks and candidates.
    """

    def __init__(self, proximity_threshold: float = 120.0):
        self.proximity_threshold = proximity_threshold

    def build_relationships(
        self,
        tracks: List[Any],
        fps: float = 30.0,
    ) -> List[ObservationalRelationship]:
        return self.build_track_relationships(tracks, fps)

    def build_track_relationships(
        self,
        tracks: List[Any],
        fps: float = 30.0,
    ) -> List[ObservationalRelationship]:
        """
        Derive pairwise observational relationships from persistent tracking histories.
        """
        relationships: List[ObservationalRelationship] = []
        if len(tracks) < 2:
            return relationships

        for i in range(len(tracks)):
            t1 = tracks[i]
            cls1 = getattr(t1, "object_class", "unknown").lower()
            tid1 = getattr(t1, "track_id", f"trk_{i}")

            for j in range(i + 1, len(tracks)):
                t2 = tracks[j]
                cls2 = getattr(t2, "object_class", "unknown").lower()
                tid2 = getattr(t2, "track_id", f"trk_{j}")

                # Check temporal overlap
                start1, end1 = getattr(t1, "first_seen", 0.0), getattr(t1, "last_seen", 0.0)
                start2, end2 = getattr(t2, "first_seen", 0.0), getattr(t2, "last_seen", 0.0)

                overlap_start = max(start1, start2)
                overlap_end = min(end1, end2)
                if overlap_end < overlap_start:
                    continue  # No temporal intersection

                # Check spatial closeness during overlap
                b1 = getattr(t1, "current_bbox", None) or (getattr(t1, "bounding_boxes", [None])[-1] if hasattr(t1, "bounding_boxes") and t1.bounding_boxes else None)
                b2 = getattr(t2, "current_bbox", None) or (getattr(t2, "bounding_boxes", [None])[-1] if hasattr(t2, "bounding_boxes") and t2.bounding_boxes else None)

                if not b1 or not b2:
                    continue

                iou = CorrelationSpatialEngine.box_iou(b1, b2)
                is_near = CorrelationSpatialEngine.are_spatially_proximate(b1, b2, max_centroid_dist=self.proximity_threshold)

                if is_near or iou > 0.0:
                    # Determine relationship type
                    if (cls1 == "person" and cls2 in ["suitcase", "backpack", "handbag", "bag", "box", "package"]) or \
                       (cls2 == "person" and cls1 in ["suitcase", "backpack", "handbag", "bag", "box", "package"]):
                        p_id = tid1 if cls1 == "person" else tid2
                        o_id = tid2 if cls1 == "person" else tid1
                        rel_type = CorrelationRelationshipType.INTERACTS_WITH.value if iou > 0.15 else CorrelationRelationshipType.REMAINS_NEAR.value
                        relationships.append(
                            ObservationalRelationship(
                                subject_track_id=p_id,
                                target_track_id=o_id,
                                relationship_type=rel_type,
                                confidence=min(1.0, 0.70 + iou * 0.3),
                                start_time=overlap_start,
                                end_time=overlap_end,
                                spatial_proximity=iou,
                                metadata={"object_class": cls2 if cls1 == "person" else cls1},
                            )
                        )

                    elif cls1 in ["car", "bus", "truck"] and cls2 in ["car", "bus", "truck"]:
                        rel_type = CorrelationRelationshipType.COLLIDES_WITH.value if iou > 0.25 else CorrelationRelationshipType.APPROACHES.value
                        relationships.append(
                            ObservationalRelationship(
                                subject_track_id=tid1,
                                target_track_id=tid2,
                                relationship_type=rel_type,
                                confidence=min(1.0, 0.75 + iou * 0.25),
                                start_time=overlap_start,
                                end_time=overlap_end,
                                spatial_proximity=iou,
                                metadata={"vehicle_types": [cls1, cls2]},
                            )
                        )

                    elif cls1 == "person" and cls2 == "person":
                        rel_type = CorrelationRelationshipType.MOVES_WITH.value if iou > 0.10 else CorrelationRelationshipType.REMAINS_NEAR.value
                        relationships.append(
                            ObservationalRelationship(
                                subject_track_id=tid1,
                                target_track_id=tid2,
                                relationship_type=rel_type,
                                confidence=0.70,
                                start_time=overlap_start,
                                end_time=overlap_end,
                                spatial_proximity=iou,
                                metadata={"social_grouping": True},
                            )
                        )

        return relationships
