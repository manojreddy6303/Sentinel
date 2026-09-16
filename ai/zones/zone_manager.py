"""
Restricted Zone & Intrusion Detection Module for Sentinel

Evaluates polygon spatial regions to detect unauthorized or unexpected object presence.
Produces structured POTENTIAL_INTRUSION events with strictly observational wording.

SAFETY CONSTRAINT:
Uses neutral observational wording: 'Potential restricted-zone intrusion detected'.
Never asserts criminal intent, illegal entry, or guilt.
"""
import uuid
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

from ai.schemas import BoundingBox, TrackedObject, ZoneDefinition, SecurityEvent


class ZoneManager:
    """
    Manages spatial restricted zones and detects intrusions by tracked surveillance objects.
    """

    def __init__(self, zones: Optional[List[ZoneDefinition]] = None):
        self._zones: Dict[str, ZoneDefinition] = {}
        if zones:
            for z in zones:
                self._zones[z.zone_id] = z

    def add_zone(
        self,
        name: str,
        polygon: List[Tuple[float, float]],
        target_classes: Optional[List[str]] = None,
        zone_id: Optional[str] = None,
    ) -> ZoneDefinition:
        """Create and register a new security zone."""
        zid = zone_id or f"ZONE-{uuid.uuid4().hex[:8]}"
        zone = ZoneDefinition(
            zone_id=zid,
            name=name,
            polygon=polygon,
            target_classes=target_classes or ["person"],
            enabled=True,
        )
        self._zones[zid] = zone
        return zone

    def remove_zone(self, zone_id: str) -> bool:
        """Remove a zone by ID."""
        return self._zones.pop(zone_id, None) is not None

    def list_zones(self) -> List[ZoneDefinition]:
        """Return all registered zones."""
        return list(self._zones.values())

    @staticmethod
    def is_point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """
        Check whether a 2D point (x, y) is inside or on the boundary of a polygon.
        Uses OpenCV's robust pointPolygonTest.
        """
        if len(polygon) < 3:
            return False
        poly_np = np.array(polygon, dtype=np.float32)
        # measureDist=False returns +1 (inside), -1 (outside), 0 (on edge)
        res = cv2.pointPolygonTest(poly_np, (float(point[0]), float(point[1])), measureDist=False)
        return res >= 0.0

    def evaluate_track_intrusions(
        self,
        timestamp: float,
        tracks: List[TrackedObject],
    ) -> List[SecurityEvent]:
        """
        Evaluate active tracks against all enabled zones.
        Generates POTENTIAL_INTRUSION events when a target class enters a zone.
        """
        events: List[SecurityEvent] = []

        for zone in self._zones.values():
            if not zone.enabled or len(zone.polygon) < 3:
                continue

            for track in tracks:
                if not track.active:
                    continue
                if track.object_class not in zone.target_classes:
                    continue

                cx, cy = track.current_bbox.centroid
                # Bottom-center is particularly relevant for ground-plane contact (feet / tires)
                bx = cx
                by = track.current_bbox.y2

                is_inside = self.is_point_in_polygon((cx, cy), zone.polygon) or self.is_point_in_polygon(
                    (bx, by), zone.polygon
                )

                if is_inside:
                    event = SecurityEvent(
                        event_id=f"EV-INTR-{uuid.uuid4().hex[:8]}",
                        event_type="POTENTIAL_INTRUSION",
                        severity="HIGH",
                        timestamp=timestamp,
                        duration_seconds=track.duration_seconds,
                        confidence=track.confidence,
                        description=(
                            f"Potential restricted-zone intrusion detected: {track.object_class} "
                            f"[{track.track_id}] entered '{zone.name}' at {timestamp:.2f}s."
                        ),
                        observable_signals=[
                            f"Object class: {track.object_class}",
                            f"Track ID: {track.track_id}",
                            f"Zone: {zone.name}",
                            f"Position: ({cx:.1f}, {cy:.1f})",
                            f"First seen: {track.first_seen:.2f}s",
                        ],
                        track_id=track.track_id,
                        object_class=track.object_class,
                        zone_name=zone.name,
                        bounding_box=track.current_bbox,
                    )
                    events.append(event)

        return events
