"""
Event Generator & Intelligence Aggregator Module

Responsible for:
- Defining the structured detection event schema.
- Creating individual timestamped detection events from raw YOLO output.
- Grouping raw detections into aggregated, high-level timeline events.
"""

import uuid
from typing import List, Dict, Any, Optional

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "bicycle"}
PERSON_CLASSES = {"person"}

def determine_event_type(objects_count: Dict[str, int]) -> str:
    """Determine event label based on detected object composition."""
    has_person = any(cls in PERSON_CLASSES for cls in objects_count)
    has_vehicle = any(cls in VEHICLE_CLASSES for cls in objects_count)
    other_objects = any(cls not in PERSON_CLASSES and cls not in VEHICLE_CLASSES for cls in objects_count)

    if has_person and has_vehicle:
        return "MULTIPLE_OBJECTS_DETECTED"
    elif has_person and not other_objects:
        return "PERSON_ACTIVITY"
    elif has_vehicle and not other_objects:
        return "VEHICLE_ACTIVITY"
    elif sum(objects_count.values()) > 1:
        return "MULTIPLE_OBJECTS_DETECTED"
    elif objects_count:
        first_cls = list(objects_count.keys())[0].upper()
        return f"{first_cls}_DETECTED"
    return "OBJECT_ACTIVITY"



def calculate_investigation_priority(total_detections: int, max_confidence: float, event_type: str) -> str:
    """
    Calculate an objective investigation priority rating (LOW, NORMAL, HIGH).
    Note: This measures analytical interest for investigators, NOT threat probability.
    """
    if total_detections >= 10 or (max_confidence >= 0.85 and "MULTIPLE" in event_type):
        return "HIGH"
    elif total_detections >= 4 or max_confidence >= 0.70:
        return "NORMAL"
    return "LOW"


def create_detection_event(
    video_id: str,
    video_filename: str,
    fps: float,
    video_duration: float,
    frame_number: int,
    detection: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a single structured detection event from a raw/validated YOLO detection."""
    return {
        "event_id": str(uuid.uuid4()),
        "video_id": video_id,
        "event_type": "object_detected",
        "object_class": detection["object_class"],
        "class_id": detection["class_id"],
        "confidence": round(float(detection["confidence"]), 4),
        "bounding_box": detection["bounding_box"],
        "timestamp": round(float(detection["timestamp"]), 4),
        "frame_number": frame_number,
        "video_filename": video_filename,
        "fps": round(fps, 4),
        "video_duration": round(video_duration, 4),
        "validation_status": detection.get("validation_status", "VALID"),
        "validation_score": detection.get("validation_score", round(float(detection.get("confidence", 0.0)), 4)),
        "validation_reason": detection.get("validation_reason"),
    }


class EventGenerator:
    """Converts raw per-frame YOLO detections into structured raw events & grouped timeline events."""

    def __init__(self, window_seconds: float = 2.0):
        self.window_seconds = window_seconds

    def generate_events(
        self,
        video_id: str,
        video_filename: str,
        fps: float,
        video_duration: float,
        frame_detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build flat list of detection event dicts."""
        events: List[Dict[str, Any]] = []
        for entry in frame_detections:
            frame_number = entry.get("frame_number", 0)
            raw_detections = entry.get("detections", [])
            for detection in raw_detections:
                event = create_detection_event(
                    video_id=video_id,
                    video_filename=video_filename,
                    fps=fps,
                    video_duration=video_duration,
                    frame_number=frame_number,
                    detection=detection,
                )
                events.append(event)
        return events

    def group_events(
        self,
        video_id: str,
        raw_events: List[Dict[str, Any]],
        window_seconds: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Group validated detections temporally into structured investigation events.

        Only events with validation_status == 'VALID' are included in grouped events.
        Detections within `window_seconds` of the group start time are merged.
        """
        win = window_seconds if window_seconds is not None else self.window_seconds
        if not raw_events:
            return []

        # Filter out REJECTED or UNCERTAIN events — timeline groupings should only reflect confirmed objects
        valid_events = [
            e for e in raw_events
            if e.get("validation_status", "VALID") == "VALID"
        ]
        if not valid_events:
            return []

        # Sort validated events chronologically
        sorted_events = sorted(valid_events, key=lambda e: e["timestamp"])

        grouped: List[Dict[str, Any]] = []
        current_cluster: List[Dict[str, Any]] = []
        cluster_start_time = sorted_events[0]["timestamp"]

        for event in sorted_events:
            if event["timestamp"] - cluster_start_time <= win:
                current_cluster.append(event)
            else:
                # Flush cluster
                grouped.append(self._build_grouped_event(video_id, current_cluster))
                current_cluster = [event]
                cluster_start_time = event["timestamp"]

        if current_cluster:
            grouped.append(self._build_grouped_event(video_id, current_cluster))

        return grouped

    def _build_grouped_event(self, video_id: str, cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate a cluster of raw events into a single GroupedEvent dict."""
        start_time = round(cluster[0]["timestamp"], 2)
        end_time = round(cluster[-1]["timestamp"], 2)
        duration = round(max(0.1, end_time - start_time), 2)

        # Count objects by class and collect associated track IDs
        objects_count: Dict[str, int] = {}
        objects_tracks: Dict[str, set] = {}
        max_conf = 0.0
        for ev in cluster:
            cls = ev["object_class"]
            objects_count[cls] = objects_count.get(cls, 0) + 1
            if cls not in objects_tracks:
                objects_tracks[cls] = set()
            t_id = ev.get("track_id")
            if t_id:
                objects_tracks[cls].add(t_id)
            if ev["confidence"] > max_conf:
                max_conf = ev["confidence"]

        objects_summary = []
        for cls, count in sorted(objects_count.items()):
            c_tracks = sorted(list(objects_tracks.get(cls, set())))
            objects_summary.append({
                "class": cls,
                "count": count,  # preserved for backward compatibility
                "detection_count": count,
                "track_count": len(c_tracks),
                "track_ids": c_tracks,
            })

        all_cluster_tracks = sorted(list(set(
            ev.get("track_id") for ev in cluster if ev.get("track_id")
        )))
        event_type = determine_event_type(objects_count)
        priority = calculate_investigation_priority(len(cluster), max_conf, event_type)

        return {
            "event_id": str(uuid.uuid4()),
            "video_id": video_id,
            "event_type": event_type,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration,
            "objects": objects_summary,
            "total_detections": len(cluster),
            "unique_tracks_count": len(all_cluster_tracks),
            "involved_tracks": all_cluster_tracks,
            "max_confidence": round(max_conf, 4),
            "priority": priority,
        }
