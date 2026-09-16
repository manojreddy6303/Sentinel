"""
Scene Context & Normal Behavior Baseline Engine (Phase 10-R)

Dynamically infers scene context from collective track observations without hardcoding:
- Roadway / Highway (high-speed parallel directional vehicle flows)
- Pedestrian Zone (walking pace, pedestrian dominant)
- Crowd Scene (high spatial-temporal detection density)
- Perimeter / Security Zone (defined boundary monitoring)
- Mixed Surveillance Area

Establishes normal-behavior baselines (typical velocities, flow angles, density)
so detectors and validators do not mistake normal scene physics for security incidents.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import math

from ai.schemas import TrackedObject, ZoneDefinition


@dataclass
class NormalBehaviorBaseline:
    """
    Statistical baseline of normal activity observed in the video stream.
    """
    average_velocity: float = 0.0
    velocity_variance: float = 0.0
    dominant_heading_degrees: Optional[float] = None
    vehicle_ratio: float = 0.0
    person_ratio: float = 0.0
    average_density_per_second: float = 0.0
    priors: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "average_velocity": round(self.average_velocity, 2),
            "velocity_variance": round(self.velocity_variance, 2),
            "dominant_heading_degrees": round(self.dominant_heading_degrees, 1) if self.dominant_heading_degrees is not None else None,
            "vehicle_ratio": round(self.vehicle_ratio, 3),
            "person_ratio": round(self.person_ratio, 3),
            "average_density_per_second": round(self.average_density_per_second, 2),
            "priors": self.priors,
        }


@dataclass
class SceneContextData:
    """
    Semantic scene context dynamically inferred from visual track observations.
    """
    scene_type: str = "mixed_surveillance"  # roadway, pedestrian_area, crowd_scene, perimeter_zone, mixed_surveillance
    confidence: float = 0.80
    description: str = "Standard mixed surveillance environment"
    baseline: NormalBehaviorBaseline = field(default_factory=NormalBehaviorBaseline)
    inferred_attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_type": self.scene_type,
            "confidence": round(self.confidence, 4),
            "description": self.description,
            "baseline": self.baseline.to_dict(),
            "inferred_attributes": self.inferred_attributes,
        }


class SceneContextEngine:
    """
    Infers scene context and normal behavior baselines across tracks.
    """

    @classmethod
    def infer_scene_context(
        cls,
        tracks: Any,
        validated_detections: Optional[List[Dict[str, Any]]] = None,
        zones: Optional[List[ZoneDefinition]] = None,
        duration_seconds: float = 0.0,
    ) -> SceneContextData:
        """
        Dynamically infer scene context from visual telemetry without hardcoded assumptions.
        Accepts either an IncidentContext object or individual lists.
        """
        if hasattr(tracks, "tracks"):
            ctx = tracks
            actual_tracks = ctx.tracks or []
            actual_dets = ctx.validated_detections or []
            actual_zones = ctx.zones or []
            actual_dur = ctx.duration_seconds
        else:
            actual_tracks = tracks or []
            actual_dets = validated_detections or []
            actual_zones = zones or []
            actual_dur = duration_seconds

        if not actual_tracks and not actual_dets:
            return SceneContextData()

        tracks = actual_tracks
        validated_detections = actual_dets
        zones = actual_zones
        duration_seconds = actual_dur


        # 1. Class distributions
        total_tracks = max(1, len(tracks))
        vehicle_classes = {"car", "truck", "bus", "motorcycle", "van"}
        vehicle_tracks = [t for t in tracks if t.object_class in vehicle_classes]
        person_tracks = [t for t in tracks if t.object_class == "person"]

        vehicle_ratio = len(vehicle_tracks) / total_tracks
        person_ratio = len(person_tracks) / total_tracks

        # 2. Velocity & Motion distributions
        velocities = []
        headings = []
        for t in tracks:
            if len(t.trajectory) >= 2:
                dt = max(1e-3, t.last_seen - t.first_seen)
                start_x, start_y = t.trajectory[0][1], t.trajectory[0][2]
                end_x, end_y = t.trajectory[-1][1], t.trajectory[-1][2]
                disp = math.hypot(end_x - start_x, end_y - start_y)
                vel = disp / dt
                velocities.append(vel)
                heading = (math.degrees(math.atan2(end_y - start_y, end_x - start_x)) + 360.0) % 360.0
                headings.append(heading)

        avg_vel = (sum(velocities) / len(velocities)) if velocities else 0.0
        vel_var = (sum((v - avg_vel) ** 2 for v in velocities) / len(velocities)) if velocities else 0.0

        # Dominant heading & flow consistency
        dominant_heading = None
        flow_confidence = 0.0
        if headings:
            # Bucket headings into 8 sectors (45-degree each)
            sector_counts = [0] * 8
            for h in headings:
                sec = int(((h + 22.5) % 360.0) // 45.0)
                sector_counts[sec] += 1
            max_sector = sector_counts.index(max(sector_counts))
            flow_confidence = max(sector_counts) / len(headings)
            if flow_confidence >= 0.40:
                dominant_heading = (max_sector * 45.0) % 360.0

        # 3. Density
        duration = max(1.0, duration_seconds)
        avg_density = len(validated_detections) / duration

        # 4. Synthesize Baseline
        baseline = NormalBehaviorBaseline(
            average_velocity=avg_vel,
            velocity_variance=vel_var,
            dominant_heading_degrees=dominant_heading,
            vehicle_ratio=vehicle_ratio,
            person_ratio=person_ratio,
            average_density_per_second=avg_density,
        )

        # 5. Determine Scene Type
        if zones and any(z.enabled for z in zones):
            scene_type = "perimeter_zone"
            desc = "Controlled perimeter with active restricted security zones"
            conf = 0.90
        elif vehicle_ratio >= 0.60 and avg_vel >= 15.0:
            scene_type = "roadway"
            desc = "Active multi-vehicle roadway with high-velocity transit flows"
            conf = 0.92
            baseline.priors["prior_against_collision"] = True
            baseline.priors["prior_against_loitering"] = True
            baseline.priors["expected_parallel_flow"] = True
        elif person_ratio >= 0.60 and avg_density >= 10.0:
            scene_type = "crowd_scene"
            desc = "High-density pedestrian crowd scene"
            conf = 0.88
            baseline.priors["normal_high_activity"] = True
        elif person_ratio >= 0.60:
            scene_type = "pedestrian_area"
            desc = "Pedestrian walkway or public concourse"
            conf = 0.85
        else:
            scene_type = "mixed_surveillance"
            desc = "Mixed urban or commercial surveillance environment"
            conf = 0.80

        inferred_attrs = {
            "vehicle_count": len(vehicle_tracks),
            "person_count": len(person_tracks),
            "average_velocity_px_s": round(avg_vel, 2),
            "dominant_flow_angle": dominant_heading,
            "flow_confidence": round(flow_confidence, 3),
        }

        return SceneContextData(
            scene_type=scene_type,
            confidence=conf,
            description=desc,
            baseline=baseline,
            inferred_attributes=inferred_attrs,
        )

