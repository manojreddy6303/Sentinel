"""
Specialized Negative Evidence Engine (Phase 15)

Evaluates physical counter-evidence actively refuting specialized visual hypotheses:
- Fire: static orange/red surface, vehicle taillight, sunset/glare, single-frame flicker
- Smoke: global fog/haze, dust, exhaust steam, compression blocking
- Weapon: low resolution, ambiguous handheld object, high occlusion
- Pose: low keypoint confidence, boundary cutoff, seated/transit posture
"""
import math
from typing import List, Dict, Any, Optional, Tuple

from ai.incidents.schemas import SupportingSignal
from ai.specialized.schemas import SpecializedTemporalTrack, SpecializedObservation


class SpecializedNegativeEvidenceEngine:
    """
    Evaluates counter-evidence refuting specialized visual candidates.
    """

    @classmethod
    def evaluate_fire_negative_evidence(
        cls,
        track: SpecializedTemporalTrack,
        scene_context: Optional[Any] = None,
        camera_metrics: Optional[Dict[str, Any]] = None,
    ) -> List[SupportingSignal]:
        """
        Evaluate counter-evidence indicating visual pattern is an ordinary red/orange object,
        vehicle taillight, sunset, or lighting reflection rather than combustion flame.
        """
        contradictory: List[SupportingSignal] = []

        # 1. Negative Evidence: Single-Frame or Brief Transient Glitch
        if track.observation_count <= 1 or track.persistence_duration < 0.4:
            contradictory.append(
                SupportingSignal(
                    signal_type="Negative: Transient Visual Glitch",
                    description=(
                        f"Fire visual observation observed in only {track.observation_count} frame(s) "
                        f"({track.persistence_duration:.2f}s duration); lacks continuous flame persistence."
                    ),
                    confidence=0.90,
                    timestamp=track.first_seen,
                )
            )

        # 2. Negative Evidence: Zero Boundary Dynamics / Perfectly Static Object
        # Real fire exhibits continuous boundary variation (flicker). Static objects have zero variation.
        if len(track.visual_metrics_history) >= 3:
            areas = [m.get("area_pixels", 0.0) for m in track.visual_metrics_history]
            mean_area = sum(areas) / len(areas) if areas else 1.0
            if mean_area > 0:
                area_variance = sum((a - mean_area) ** 2 for a in areas) / len(areas)
                area_cv = math.sqrt(area_variance) / mean_area
                if area_cv < 0.04:  # Almost completely stationary size
                    contradictory.append(
                        SupportingSignal(
                            signal_type="Negative: Static Surface Chromaticity",
                            description=(
                                f"Observed chromatic region shows negligible dynamic fluctuation (coefficient of variation {area_cv:.3f}); "
                                f"characteristic of painted orange/red object or static fixture rather than flame combustion."
                            ),
                            confidence=0.88,
                            timestamp=track.first_seen,
                        )
                    )

        # 3. Negative Evidence: Roadway Vehicle Light Correlation
        if scene_context and getattr(scene_context, "scene_type", "") == "roadway":
            # If region is small and at typical vehicle height
            if track.bounding_boxes:
                b = track.bounding_boxes[0]
                w = b.get("x2", 0) - b.get("x1", 0)
                h = b.get("y2", 0) - b.get("y1", 0)
                if w < 50 and h < 50:
                    contradictory.append(
                        SupportingSignal(
                            signal_type="Negative: Potential Vehicle Lighting Glare",
                            description="Roadway context with compact localized luminescence consistent with vehicle tail lamps or reflections.",
                            confidence=0.75,
                            timestamp=track.first_seen,
                        )
                    )

        # 4. Negative Evidence: Pedestrian Clothing / Personal Belonging Context
        has_clothing_tag = any(m.get("is_person_clothing", False) for m in track.visual_metrics_history)
        if has_clothing_tag:
            contradictory.append(
                SupportingSignal(
                    signal_type="Negative: Person Clothing / Accessory Chromaticity",
                    description="Visual chromaticity co-located with pedestrian clothing/accessories without thermal combustion.",
                    confidence=0.92,
                    timestamp=track.first_seen,
                )
            )

        return contradictory

    @classmethod
    def evaluate_smoke_negative_evidence(
        cls,
        track: SpecializedTemporalTrack,
        scene_context: Optional[Any] = None,
        camera_metrics: Optional[Dict[str, Any]] = None,
        is_static_surface: bool = False,
    ) -> List[SupportingSignal]:
        """
        Evaluate counter-evidence indicating visual pattern is global fog, dust storm,
        steam condensation, or compression artifacts.
        """
        contradictory: List[SupportingSignal] = []

        # 1. Negative Evidence: Single-frame transient anomaly
        if track.observation_count <= 1 or track.persistence_duration < 0.5:
            contradictory.append(
                SupportingSignal(
                    signal_type="Negative: Transient Dispersion Artifact",
                    description=(
                        f"Smoke-like pattern observed in only {track.observation_count} frame(s) "
                        f"({track.persistence_duration:.2f}s duration); lacks persistent plume dynamics."
                    ),
                    confidence=0.88,
                    timestamp=track.first_seen,
                )
            )

        # 2. Negative Evidence: Static Surface Texture
        if is_static_surface:
            contradictory.append(
                SupportingSignal(
                    signal_type="Negative: Static Surface Texture",
                    description="Visual texture remains spatially anchored without volumetric plume deformation, dispersion, or drift; matches road surface, wall, or static architectural fixture.",
                    confidence=0.92,
                    timestamp=track.first_seen,
                )
            )

        # 3. Negative Evidence: Global Scene Haze / Low Contrast Weather
        if scene_context:
            lighting = getattr(scene_context, "lighting_condition", "")
            weather = getattr(scene_context, "weather_condition", "")
            if weather in ["foggy", "hazy", "rainy"] or lighting in ["overcast", "dim"]:
                contradictory.append(
                    SupportingSignal(
                        signal_type="Negative: Global Atmospheric Haze",
                        description=f"Atmospheric condition ({weather or lighting}) exhibits uniform low saturation across entire visual field.",
                        confidence=0.85,
                        timestamp=track.first_seen,
                    )
                )

        # 4. Negative Evidence: Compression / Encoding Blocking Artifact
        if camera_metrics and camera_metrics.get("compression_blockiness", 0.0) > 0.65:
            contradictory.append(
                SupportingSignal(
                    signal_type="Negative: Video Compression Artifact",
                    description="High macro-block compression artifact level detected in video stream; may cause pseudo-smoke textures.",
                    confidence=0.82,
                    timestamp=track.first_seen,
                )
            )

        return contradictory

    @classmethod
    def evaluate_weapon_negative_evidence(
        cls,
        observation: SpecializedObservation,
        target_crop_resolution: Optional[Tuple[int, int]] = None,
        track_object_class: Optional[str] = None,
    ) -> List[SupportingSignal]:
        """
        Evaluate counter-evidence indicating visual pattern is an ordinary handheld item
        or lacks optical resolution for reliable identification.
        """
        contradictory: List[SupportingSignal] = []

        # 1. Negative Evidence: Insufficient Pixel Resolution
        if target_crop_resolution:
            w, h = target_crop_resolution
            if w < 32 or h < 32:
                contradictory.append(
                    SupportingSignal(
                        signal_type="Negative: Insufficient Optical Resolution",
                        description=f"Object crop resolution ({w}x{h}px) is below minimum forensic threshold (32x32px) for object classification.",
                        confidence=0.92,
                        timestamp=observation.timestamp,
                    )
                )

        # 2. Negative Evidence: Common Handheld Belonging Context
        if track_object_class in ["cell phone", "bottle", "umbrella", "backpack", "handbag"]:
            contradictory.append(
                SupportingSignal(
                    signal_type="Negative: Common Personal Belonging Context",
                    description=f"Object is associated with known non-threatening category '{track_object_class}'.",
                    confidence=0.85,
                    timestamp=observation.timestamp,
                )
            )

        return contradictory

    @classmethod
    def evaluate_pose_negative_evidence(
        cls,
        keypoints: Optional[List[Any]],
        bounding_box: Optional[Any] = None,
        frame_dimensions: Optional[Tuple[int, int]] = None,
    ) -> List[SupportingSignal]:
        """
        Evaluate counter-evidence indicating pose telemetry is unreliable due to
        occlusion, frame clipping, or low keypoint confidence.
        """
        contradictory: List[SupportingSignal] = []

        if not keypoints or len(keypoints) == 0:
            contradictory.append(
                SupportingSignal(
                    signal_type="Negative: Absent Pose Keypoints",
                    description="Zero skeletal keypoints extracted for subject; pose classification unsupported.",
                    confidence=0.95,
                )
            )
            return contradictory

        # Check keypoint confidence if available
        low_conf_kpts = 0
        total_kpts = 0
        for pt in keypoints:
            if isinstance(pt, (list, tuple)) and len(pt) >= 3:
                total_kpts += 1
                if pt[2] < 0.35:
                    low_conf_kpts += 1

        if total_kpts > 0 and (low_conf_kpts / total_kpts) > 0.5:
            contradictory.append(
                SupportingSignal(
                    signal_type="Negative: Low Keypoint Confidence",
                    description=f"{low_conf_kpts}/{total_kpts} keypoints fall below confidence threshold 0.35.",
                    confidence=0.85,
                )
            )

        # Frame boundary clipping
        if bounding_box and frame_dimensions:
            fw, fh = frame_dimensions
            bx1, by1 = getattr(bounding_box, "x1", 0), getattr(bounding_box, "y1", 0)
            bx2, by2 = getattr(bounding_box, "x2", fw), getattr(bounding_box, "y2", fh)
            if bx1 <= 5 or by1 <= 5 or bx2 >= fw - 5 or by2 >= fh - 5:
                contradictory.append(
                    SupportingSignal(
                        signal_type="Negative: Frame Boundary Truncation",
                        description="Person bounding box intersects frame perimeter; keypoint angles may be distorted by camera edge.",
                        confidence=0.80,
                    )
                )

        return contradictory
