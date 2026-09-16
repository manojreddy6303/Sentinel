"""
Pose and Action Specialized Detector (Phase 15)

Reuses Phase 12 PoseFeatureEngine without duplicate code or surprise downloads.
Extracts observational posture evidence (standing, crouching, lying/down, raised-arm)
and feeds it to the incident engine strictly as supporting signals.
"""
import uuid
import logging
from typing import Dict, Any, List, Optional
import numpy as np

from ai.schemas import BoundingBox
from ai.specialized.base import BaseSpecializedDetector
from ai.specialized.schemas import (
    SpecializedDetectorStatus,
    SpecializedObservation,
    SpecializedValidationStatus,
    SpecializedModelInfo,
)
from ai.incidents.detectors.person.pose_features import PoseFeatureEngine

logger = logging.getLogger(__name__)


class PoseActionDetector(BaseSpecializedDetector):
    """
    Observational posture and action evidence extractor.
    Integrates directly with Phase 12 PoseFeatureEngine.
    """

    detector_name: str = "pose_action_detector"
    detector_version: str = "1.0.0"
    supported_classes: List[str] = [
        "pose_standing",
        "pose_crouching",
        "pose_lying_down",
        "pose_raised_arms",
    ]

    def __init__(self, enabled: bool = True, config: Optional[Dict[str, Any]] = None):
        super().__init__(enabled=enabled, config=config)
        self._pose_engine = PoseFeatureEngine()

    def initialize(self) -> bool:
        if self._is_initialized:
            return self._status == SpecializedDetectorStatus.AVAILABLE

        # Attempt to initialize pose engine (respects SENTINEL_ENABLE_POSE_ESTIMATION and local weights)
        is_avail = self._pose_engine.initialize()
        status_data = self._pose_engine.get_status()

        if is_avail:
            self._status = SpecializedDetectorStatus.AVAILABLE
            self._status_reason = "YOLOv8-Pose model loaded and active"
        elif not status_data.get("enabled_by_config", False):
            self._status = SpecializedDetectorStatus.NOT_CONFIGURED
            self._status_reason = "Pose estimation not enabled in config (SENTINEL_ENABLE_POSE_ESTIMATION=0)"
        else:
            self._status = SpecializedDetectorStatus.UNAVAILABLE
            self._status_reason = "Pose model weights absent; falling back to bounding-box kinematic heuristics"

        self._is_initialized = True
        return True

    def detect_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_idx: int = 0,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[SpecializedObservation]:
        """
        Extract posture telemetry for people in frame.
        Uses keypoints if model is active; uses bounding-box aspect ratio kinematics if inactive.
        """
        observations: List[SpecializedObservation] = []
        if context is None:
            context = {}

        person_boxes = context.get("person_bounding_boxes", [])
        # If no person boxes in context, return empty
        if not person_boxes:
            return []

        for pbox in person_boxes:
            bx1, by1, bx2, by2 = pbox.get("x1", 0), pbox.get("y1", 0), pbox.get("x2", 0), pbox.get("y2", 0)
            bw = max(1.0, bx2 - bx1)
            bh = max(1.0, by2 - by1)
            aspect_ratio = bh / bw

            # Check if keypoint engine is active
            pose_data = self._pose_engine.extract_pose(frame, bounding_box=[bx1, by1, bx2, by2])

            detected_posture = "pose_standing"
            conf = 0.65
            evidence_str = 0.60
            metrics = {
                "aspect_ratio": round(aspect_ratio, 2),
                "pose_engine_active": pose_data.get("pose_available", False),
            }

            if pose_data.get("pose_available") and pose_data.get("keypoints"):
                kpts = pose_data["keypoints"]
                # Torso horizontal check
                if pose_data.get("is_horizontal_torso"):
                    detected_posture = "pose_lying_down"
                    conf = 0.85
                    evidence_str = 0.80
                else:
                    # Check raised arms: COCO wrists (9=left_wrist, 10=right_wrist) vs shoulders (5, 6)
                    try:
                        if len(kpts) > 10:
                            lw_y = kpts[9][1]
                            rw_y = kpts[10][1]
                            ls_y = kpts[5][1]
                            rs_y = kpts[6][1]
                            if (lw_y < ls_y or rw_y < rs_y) and (lw_y > 0 and ls_y > 0):
                                detected_posture = "pose_raised_arms"
                                conf = 0.80
                                evidence_str = 0.75
                    except Exception:
                        pass
            else:
                # Kinematic aspect ratio fallback (honest, zero synthetic keypoints)
                if aspect_ratio < 0.75:
                    detected_posture = "pose_lying_down"
                    conf = 0.65
                    evidence_str = 0.55
                elif aspect_ratio < 1.3:
                    detected_posture = "pose_crouching"
                    conf = 0.60
                    evidence_str = 0.50
                else:
                    detected_posture = "pose_standing"
                    conf = 0.70
                    evidence_str = 0.60

            obs = SpecializedObservation(
                observation_id=f"OBS-POSE-{uuid.uuid4().hex[:6]}",
                detector_name=self.detector_name,
                detector_version=self.detector_version,
                class_name=detected_posture,
                timestamp=timestamp,
                confidence=round(conf, 4),
                evidence_strength=round(evidence_str, 4),
                bounding_box=BoundingBox(x1=bx1, y1=by1, x2=bx2, y2=by2),
                validation_status=SpecializedValidationStatus.VALID,
                frame_number=frame_idx,
                visual_metrics=metrics,
            )
            observations.append(obs)

        return observations
