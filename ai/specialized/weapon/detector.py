"""
Weapon and Suspicious Object Visual Detector Foundation (Phase 15)

Provides a modular foundation for specialized weapon / suspicious object detection.
CRITICAL ACCURACY & SAFETY RULES:
- Never label arbitrary objects as weapons.
- Never state "weapon detected" definitively; use "POTENTIAL_WEAPON_VISUAL".
- If no custom trained model is configured or available:
  Explicitly report NOT_CONFIGURED / UNAVAILABLE.
  Never fabricate synthetic weapon detections.
"""
import os
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

logger = logging.getLogger(__name__)


class WeaponVisualDetector(BaseSpecializedDetector):
    """
    Modular foundation for specialized weapon and suspicious object visual detection.
    Strictly abstains from fabricating detections if no validated model is provided.
    """

    detector_name: str = "weapon_visual_detector"
    detector_version: str = "1.0.0"
    supported_classes: List[str] = [
        "handgun-like object",
        "rifle-like object",
        "knife-like object",
        "suspicious_object",
    ]

    def __init__(self, enabled: bool = True, config: Optional[Dict[str, Any]] = None):
        super().__init__(enabled=enabled, config=config)
        self.model_path = os.environ.get("WEAPON_MODEL_PATH") or self.config.get("model_path")
        self._model = None

    def initialize(self) -> bool:
        if self._is_initialized:
            return self._status == SpecializedDetectorStatus.AVAILABLE

        # Look for configured local model path
        possible_paths = []
        if self.model_path:
            possible_paths.append(self.model_path)
        possible_paths.extend([
            os.path.join(os.getcwd(), "models", "weapon_detector.pt"),
            os.path.join(os.getcwd(), "weapon_detector.pt"),
        ])

        found_path = None
        for p in possible_paths:
            if os.path.isfile(p):
                found_path = p
                break

        if not found_path:
            self._status = SpecializedDetectorStatus.NOT_CONFIGURED
            self._status_reason = (
                "No specialized weapon detection weights configured or found locally. "
                "Set WEAPON_MODEL_PATH or place weapon_detector.pt in models/. "
                "Detector will safely abstain with zero fabricated detections."
            )
            self._is_initialized = True
            logger.info("WeaponVisualDetector: No local model found. Safely in NOT_CONFIGURED status.")
            return False

        try:
            from ultralytics import YOLO
            self._model = YOLO(found_path)
            self._status = SpecializedDetectorStatus.AVAILABLE
            self._status_reason = f"Loaded weapon weights from {found_path}"
            logger.info(f"Weapon detector successfully initialized from {found_path}")
        except Exception as e:
            self._status = SpecializedDetectorStatus.FAILED
            self._status_reason = f"Failed loading weapon model from {found_path}: {e}"
            logger.warning(self._status_reason)

        self._is_initialized = True
        return self._status == SpecializedDetectorStatus.AVAILABLE

    def detect_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_idx: int = 0,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[SpecializedObservation]:
        """
        Inference on frame. If model is absent or not configured, returns [] immediately.
        """
        if self._status != SpecializedDetectorStatus.AVAILABLE or self._model is None:
            return []

        observations: List[SpecializedObservation] = []
        try:
            results = self._model.predict(frame, verbose=False, conf=0.45)
            if results and len(results) > 0:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()

                    # Class name mapping
                    names = getattr(self._model, "names", {})
                    raw_name = names.get(cls_id, "suspicious_object")
                    # Map to forensic label
                    if "gun" in raw_name.lower() or "pistol" in raw_name.lower():
                        class_name = "handgun-like object"
                    elif "rifle" in raw_name.lower():
                        class_name = "rifle-like object"
                    elif "knife" in raw_name.lower() or "blade" in raw_name.lower():
                        class_name = "knife-like object"
                    else:
                        class_name = "suspicious_object"

                    bbox = BoundingBox(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3])
                    obs = SpecializedObservation(
                        observation_id=f"OBS-WEAPON-{uuid.uuid4().hex[:6]}",
                        detector_name=self.detector_name,
                        detector_version=self.detector_version,
                        class_name=class_name,
                        timestamp=timestamp,
                        confidence=conf,
                        evidence_strength=round(min(1.0, conf * 0.85), 4),
                        bounding_box=bbox,
                        validation_status=SpecializedValidationStatus.RAW,
                        frame_number=frame_idx,
                        visual_metrics={"raw_class": raw_name},
                    )
                    observations.append(obs)
        except Exception as e:
            logger.debug(f"Weapon detector inference error: {e}")

        return observations

    def get_model_info(self) -> SpecializedModelInfo:
        return SpecializedModelInfo(
            name=self.detector_name,
            version=self.detector_version,
            source="custom_yolo_weights" if self._status == SpecializedDetectorStatus.AVAILABLE else "none",
            license="Custom / User Provided",
            input_format="BGR Frame",
            output_format="BoundingBox + Forensic Object Class",
            status=self._status,
            status_reason=self._status_reason,
        )
