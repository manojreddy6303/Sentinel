"""
Crowd & Activity Density Analyzer for Sentinel Surveillance Intelligence

Calculates temporal detection density, monitors pedestrian and vehicle volumes across
sliding time windows, identifies activity peaks, and outputs HIGH_ACTIVITY_PERIOD events.

SAFETY CONSTRAINT:
Elevated activity or crowd density is an analytical observation, NEVER criminal attribution.
"""
import uuid
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from ai.schemas import SecurityEvent, TrackedObject, BoundingBox


class ActivityAnalyzer:
    """
    Computes temporal activity density and identifies elevated activity windows.
    """

    def __init__(
        self,
        window_seconds: float = 2.0,
        high_activity_threshold: int = 15,
        person_density_threshold: int = 6,
    ):
        self.window_seconds = window_seconds
        self.high_activity_threshold = high_activity_threshold
        self.person_density_threshold = person_density_threshold

    def analyze_timeline_density(
        self,
        events: List[Dict[str, Any]],
        video_duration: float,
    ) -> List[Dict[str, Any]]:
        """
        Compute density buckets across the video duration.
        """
        if not events or video_duration <= 0:
            return []

        num_buckets = max(1, int(video_duration / self.window_seconds) + 1)
        buckets = [
            {
                "window_start": round(i * self.window_seconds, 2),
                "window_end": round(min(video_duration, (i + 1) * self.window_seconds), 2),
                "total_detections": 0,
                "person_count": 0,
                "vehicle_count": 0,
                "other_count": 0,
                "classes": defaultdict(int),
            }
            for i in range(num_buckets)
        ]

        for ev in events:
            t = ev.get("timestamp") or ev.get("timestamp_seconds") or 0.0
            idx = min(num_buckets - 1, int(t / self.window_seconds))
            cls = ev.get("object_class", "unknown")
            b = buckets[idx]
            b["total_detections"] += 1
            b["classes"][cls] += 1
            if cls == "person":
                b["person_count"] += 1
            elif cls in ["car", "bus", "truck", "motorcycle", "bicycle"]:
                b["vehicle_count"] += 1
            else:
                b["other_count"] += 1

        # Convert defaultdict to regular dict
        for b in buckets:
            b["classes"] = dict(b["classes"])

        return buckets

    def detect_activity_peaks(
        self,
        events: List[Dict[str, Any]],
        video_duration: float,
    ) -> List[SecurityEvent]:
        """
        Detect periods where overall detection density or person density exceeds thresholds.
        """
        buckets = self.analyze_timeline_density(events, video_duration)
        peak_events: List[SecurityEvent] = []

        for b in buckets:
            total = b["total_detections"]
            persons = b["person_count"]
            vehicles = b["vehicle_count"]

            is_high_activity = total >= self.high_activity_threshold or persons >= self.person_density_threshold

            if is_high_activity:
                w_start = b["window_start"]
                w_end = b["window_end"]
                dur = round(w_end - w_start, 2)

                peak_events.append(
                    SecurityEvent(
                        event_id=f"EV-ACT-{uuid.uuid4().hex[:8]}",
                        event_type="HIGH_ACTIVITY_PERIOD",
                        severity="NORMAL" if total < (self.high_activity_threshold * 1.5) else "HIGH",
                        timestamp=w_start,
                        duration_seconds=dur,
                        confidence=min(0.99, round(total / (self.high_activity_threshold + 5), 2)),
                        description=(
                            f"Elevated activity detected between {w_start:.2f}s and {w_end:.2f}s: "
                            f"{total} detections ({persons} persons, {vehicles} vehicles)."
                        ),
                        observable_signals=[
                            f"Total detections in {dur}s window: {total}",
                            f"Person count: {persons}",
                            f"Vehicle count: {vehicles}",
                            f"Window: [{w_start:.2f}s – {w_end:.2f}s]",
                        ],
                    )
                )

        return peak_events
