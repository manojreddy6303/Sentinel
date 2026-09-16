"""
Camera Stability & Scene Motion Protection Engine (Phase 13)

Evaluates global scene motion and frame-level stability to prevent false property incidents
caused by camera shakes, panning, optical zoom changes, or perspective shifts.
"""
from dataclasses import dataclass
import math
from typing import List, Dict, Any, Optional, Tuple

from ai.schemas import BoundingBox, TrackedObject
from ai.incidents.schemas import IncidentContext, SupportingSignal


@dataclass
class CameraStabilityAssessment:
    is_camera_stable: bool
    global_translation_magnitude: float
    coherent_motion_ratio: float
    is_jitter_detected: bool
    is_frame_boundary_event: bool
    confidence: float
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_camera_stable": self.is_camera_stable,
            "global_translation_magnitude": round(self.global_translation_magnitude, 2),
            "coherent_motion_ratio": round(self.coherent_motion_ratio, 2),
            "is_jitter_detected": self.is_jitter_detected,
            "is_frame_boundary_event": self.is_frame_boundary_event,
            "confidence": round(self.confidence, 2),
            "explanation": self.explanation,
        }


class CameraStabilityEngine:
    """
    Infers camera and scene stability from collective track telemetry.
    """

    def __init__(
        self,
        coherent_motion_threshold: float = 0.70,
        translation_noise_threshold: float = 18.0,
        frame_margin_px: float = 25.0,
    ):
        self.coherent_motion_threshold = coherent_motion_threshold
        self.translation_noise_threshold = translation_noise_threshold
        self.frame_margin_px = frame_margin_px

    def assess_stability(
        self,
        context: Optional[IncidentContext],
        target_track: Optional[TrackedObject] = None,
        frame_w: float = 1920.0,
        frame_h: float = 1080.0,
    ) -> CameraStabilityAssessment:
        """
        Determines whether the camera platform exhibited stability during the video.
        """
        if not context or not context.tracks or len(context.tracks) < 2:
            # Insufficient tracks to compute collective motion; default to stable
            is_edge = False
            if target_track and target_track.current_bbox:
                b = target_track.current_bbox
                is_edge = (
                    b.x1 <= self.frame_margin_px
                    or b.y1 <= self.frame_margin_px
                    or b.x2 >= frame_w - self.frame_margin_px
                    or b.y2 >= frame_h - self.frame_margin_px
                )
            return CameraStabilityAssessment(
                is_camera_stable=True,
                global_translation_magnitude=0.0,
                coherent_motion_ratio=0.0,
                is_jitter_detected=False,
                is_frame_boundary_event=is_edge,
                confidence=0.85,
                explanation="Camera stable (baseline static platform assumptions hold).",
            )

        # Evaluate vectors across all tracks with length >= 3
        dx_list = []
        dy_list = []
        for t in context.tracks:
            if t.trajectory and len(t.trajectory) >= 3:
                start_p = t.trajectory[0]
                end_p = t.trajectory[-1]
                dx = end_p[1] - start_p[1]
                dy = end_p[2] - start_p[2]
                dx_list.append(dx)
                dy_list.append(dy)

        if len(dx_list) < 3:
            return CameraStabilityAssessment(
                is_camera_stable=True,
                global_translation_magnitude=0.0,
                coherent_motion_ratio=0.0,
                is_jitter_detected=False,
                is_frame_boundary_event=False,
                confidence=0.85,
                explanation="Camera platform considered stable.",
            )

        mean_dx = sum(dx_list) / len(dx_list)
        mean_dy = sum(dy_list) / len(dy_list)
        global_trans = math.hypot(mean_dx, mean_dy)

        # Check directional alignment ratio with global vector
        coherent_count = 0
        for dx, dy in zip(dx_list, dy_list):
            dot = (dx * mean_dx) + (dy * mean_dy)
            mag = math.hypot(dx, dy) * global_trans
            if mag > 0 and (dot / mag) > 0.60:
                coherent_count += 1

        coherent_ratio = coherent_count / len(dx_list)
        is_unstable = (global_trans > self.translation_noise_threshold and coherent_ratio >= self.coherent_motion_threshold)

        # Check target track edge boundary
        is_edge = False
        if target_track and target_track.current_bbox:
            b = target_track.current_bbox
            is_edge = (
                b.x1 <= self.frame_margin_px
                or b.y1 <= self.frame_margin_px
                or b.x2 >= frame_w - self.frame_margin_px
                or b.y2 >= frame_h - self.frame_margin_px
            )

        explanation = (
            f"Global scene motion: mean drift {global_trans:.1f}px, coherence: {coherent_ratio:.0%}. "
            + ("Camera pan / motion detected." if is_unstable else "Platform stationary.")
        )

        return CameraStabilityAssessment(
            is_camera_stable=not is_unstable,
            global_translation_magnitude=global_trans,
            coherent_motion_ratio=coherent_ratio,
            is_jitter_detected=is_unstable,
            is_frame_boundary_event=is_edge,
            confidence=0.90 if is_unstable else 0.85,
            explanation=explanation,
        )
