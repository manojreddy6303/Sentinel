"""
Incident Context Engine (Phase 10)

Assembles rich, structured IncidentContext from raw video events, tracks,
zones, and derived motion metrics. Provides scene-level density awareness
so detectors avoid misinterpreting high-traffic scenes as generic threats.
"""
from collections import defaultdict
from typing import List, Dict, Any, Optional

from ai.schemas import TrackedObject, VehicleAttribute, FaceDetection, ZoneDefinition
from ai.incidents.schemas import IncidentContext, TrackMotion
from ai.incidents.motion import UniversalMotionEngine


class IncidentContextBuilder:
    """
    Constructs an immutable IncidentContext for an analysis session.
    """

    def __init__(self, motion_engine: Optional[UniversalMotionEngine] = None):
        self.motion_engine = motion_engine or UniversalMotionEngine()

    def build_context(
        self,
        video_id: str,
        tracks: List[TrackedObject],
        validated_detections: List[Dict[str, Any]],
        fps: float = 30.0,
        duration_seconds: float = 0.0,
        sample_rate_fps: float = 1.0,
        zones: Optional[List[ZoneDefinition]] = None,
        vehicle_attributes: Optional[List[VehicleAttribute]] = None,
        face_detections: Optional[List[FaceDetection]] = None,
        video_metadata: Optional[Dict[str, Any]] = None,
    ) -> IncidentContext:
        """
        Build fully calculated IncidentContext.
        """
        all_zones = zones or []
        attrs = vehicle_attributes or []
        faces = face_detections or []
        meta = video_metadata or {}
        # Ensure incident candidate generation only evaluates validated observations
        clean_validated = [
            d for d in (validated_detections or [])
            if d.get("validation_status") not in ("REJECTED", "rejected")
        ]

        # 1. Compute frame-by-frame motion for each track
        track_motions: Dict[str, List[TrackMotion]] = {}
        motion_summaries: Dict[str, Dict[str, Any]] = {}

        for trk in tracks:
            motions = self.motion_engine.compute_track_motion(trk)
            track_motions[trk.track_id] = motions
            summary = self.motion_engine.summarize_track_motion(motions)
            motion_summaries[trk.track_id] = summary

        # 2. Scene density analysis across time
        density_by_sec = defaultdict(int)
        for det in clean_validated:
            t = round(float(det.get("timestamp") or det.get("timestamp_seconds") or 0.0))
            density_by_sec[t] += 1

        peak_density = max(density_by_sec.values()) if density_by_sec else 0
        avg_density = (sum(density_by_sec.values()) / len(density_by_sec)) if density_by_sec else 0.0

        scene_density = {
            "peak_detections_per_second": peak_density,
            "average_detections_per_second": round(avg_density, 2),
            "is_high_density_crowd": avg_density >= 12.0 or peak_density >= 25,
            "total_detections": len(clean_validated),
            "total_tracks": len(tracks),
        }

        return IncidentContext(
            video_id=video_id,
            fps=fps,
            duration_seconds=duration_seconds,
            sample_rate_fps=sample_rate_fps,
            validated_detections=clean_validated,
            tracks=tracks,
            vehicle_attributes=attrs,
            face_detections=faces,
            zones=all_zones,
            track_motions=track_motions,
            motion_summaries=motion_summaries,
            scene_density=scene_density,
            video_metadata=meta,
        )
