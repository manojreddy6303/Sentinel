"""
Specialized Visual Observation Validation Engine (Phase 15.4)

Enforces strict validation boundaries between raw detector inferences and
downstream incident intelligence components (tracker, episode aggregator, incident engine).

Core Invariant:
RAW
  ↓
VALIDATION ENGINE (Hard boundary)
  ↓
ONLY VALID OBSERVATIONS (validation_status == SpecializedValidationStatus.VALID)
  ↓
TRACKER & EPISODE AGGREGATION
  ↓
INCIDENTS & EVIDENCE
"""
import logging
from typing import Dict, Any, List, Optional
import numpy as np

from ai.specialized.schemas import (
    SpecializedObservation,
    SpecializedValidationStatus,
)
from ai.common.numeric import is_finite_number

logger = logging.getLogger(__name__)


class SpecializedValidationEngine:
    """
    Validates specialized visual observations against physical, optical,
    and temporal plausibility criteria before tracking or episode aggregation.
    """

    MIN_CONFIDENCE_SMOKE: float = 0.40
    MIN_EVIDENCE_STRENGTH_SMOKE: float = 0.30
    MIN_SCENE_SATURATION_SMOKE: float = 35.0
    MIN_TEMPORAL_MOTION_SMOKE: float = 2.0
    MAX_ROUGHNESS_SMOKE: float = 260.0

    MIN_CONFIDENCE_FIRE: float = 0.40
    MIN_EVIDENCE_STRENGTH_FIRE: float = 0.35

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def validate_observation(
        self,
        obs: SpecializedObservation,
        frame: Optional[np.ndarray] = None,
        timestamp: float = 0.0,
    ) -> SpecializedObservation:
        """
        Evaluate a single raw specialized observation and assign definitive
        validation_status (VALID or REJECTED) with diagnostic reason.
        """
        if obs is None:
            return obs

        # 1. Bounding box validity
        bbox = getattr(obs, "bounding_box", None)
        if bbox is None:
            obs.validation_status = SpecializedValidationStatus.REJECTED
            obs.validation_reason = "Missing spatial bounding box"
            return obs

        if not (is_finite_number(bbox.x1) and is_finite_number(bbox.y1) and is_finite_number(bbox.x2) and is_finite_number(bbox.y2)):
            obs.validation_status = SpecializedValidationStatus.REJECTED
            obs.validation_reason = "Non-finite bounding box coordinates"
            return obs

        bw = bbox.x2 - bbox.x1
        bh = bbox.y2 - bbox.y1
        if bw <= 0 or bh <= 0:
            obs.validation_status = SpecializedValidationStatus.REJECTED
            obs.validation_reason = "Degenerate zero or negative bounding box dimensions"
            return obs

        cls_name = (getattr(obs, "class_name", "") or "").lower()
        metrics = getattr(obs, "visual_metrics", {}) or {}

        # 2. Class-specific validation rules
        if cls_name == "smoke":
            return self._validate_smoke(obs, bw, bh, metrics, frame)
        elif cls_name == "fire":
            return self._validate_fire(obs, bw, bh, metrics, frame)
        elif "weapon" in cls_name:
            return self._validate_weapon(obs, metrics)
        elif "pose" in cls_name:
            return self._validate_pose(obs, metrics)

        # Default: accept if confidence meets standard threshold
        if obs.confidence >= 0.40:
            obs.validation_status = SpecializedValidationStatus.VALID
            obs.validation_reason = "General specialized observation verified"
        else:
            obs.validation_status = SpecializedValidationStatus.REJECTED
            obs.validation_reason = f"Confidence {obs.confidence:.2f} below minimum threshold 0.40"

        return obs

    def _validate_smoke(
        self,
        obs: SpecializedObservation,
        bw: float,
        bh: float,
        metrics: Dict[str, Any],
        frame: Optional[np.ndarray] = None,
    ) -> SpecializedObservation:
        # Minimum confidence & evidence strength
        if obs.confidence < self.MIN_CONFIDENCE_SMOKE:
            obs.validation_status = SpecializedValidationStatus.REJECTED
            obs.validation_reason = f"Smoke confidence {obs.confidence:.2f} below minimum {self.MIN_CONFIDENCE_SMOKE}"
            return obs

        if obs.evidence_strength < self.MIN_EVIDENCE_STRENGTH_SMOKE:
            obs.validation_status = SpecializedValidationStatus.REJECTED
            obs.validation_reason = f"Smoke evidence strength {obs.evidence_strength:.2f} below minimum {self.MIN_EVIDENCE_STRENGTH_SMOKE}"
            return obs

        # IR / Monochromatic scene rejection
        scene_sat = metrics.get("scene_mean_saturation")
        if scene_sat is not None and scene_sat < self.MIN_SCENE_SATURATION_SMOKE:
            obs.validation_status = SpecializedValidationStatus.REJECTED
            obs.validation_reason = f"Monochrome/IR scene (mean sat {scene_sat:.1f} < {self.MIN_SCENE_SATURATION_SMOKE}): heuristic chromatic smoke cannot be validated"
            return obs

        # Temporal motion check (reject static background surfaces)
        temporal_motion = metrics.get("temporal_motion")
        if temporal_motion is not None and temporal_motion < self.MIN_TEMPORAL_MOTION_SMOKE:
            obs.validation_status = SpecializedValidationStatus.REJECTED
            obs.validation_reason = f"Static background surface (temporal motion {temporal_motion:.2f} < {self.MIN_TEMPORAL_MOTION_SMOKE})"
            return obs

        # High-frequency solid surface rejection (asphalt gravel, brickwork, concrete pavement)
        roughness = metrics.get("texture_roughness")
        if roughness is not None and roughness > self.MAX_ROUGHNESS_SMOKE:
            obs.validation_status = SpecializedValidationStatus.REJECTED
            obs.validation_reason = f"High-frequency solid surface texture (roughness {roughness:.1f} > {self.MAX_ROUGHNESS_SMOKE})"
            return obs

        # Static surface rejection
        if metrics.get("is_static_surface", False):
            obs.validation_status = SpecializedValidationStatus.REJECTED
            obs.validation_reason = "Static background surface without plume deformation or dispersion"
            return obs

        # Border-clamped boundary fixture
        if metrics.get("is_border_anchored", False):
            obs.validation_status = SpecializedValidationStatus.REJECTED
            obs.validation_reason = "Static border-clamped boundary fixture"
            return obs

        # Aspect ratio check (reject narrow vertical poles, wall seams, horizontal lines)
        aspect = bw / max(1.0, bh)
        if aspect < 0.30 or aspect > 3.0:
            obs.validation_status = SpecializedValidationStatus.REJECTED
            obs.validation_reason = f"Non-plume geometric aspect ratio {aspect:.2f}"
            return obs

        # Passed all smoke validation gates
        obs.validation_status = SpecializedValidationStatus.VALID
        obs.validation_reason = "Smoke plume visual evidence validated (turbulent, dynamic low-saturation region)"
        return obs

    def _validate_fire(
        self,
        obs: SpecializedObservation,
        bw: float,
        bh: float,
        metrics: Dict[str, Any],
        frame: Optional[np.ndarray] = None,
    ) -> SpecializedObservation:
        if obs.confidence < self.MIN_CONFIDENCE_FIRE:
            obs.validation_status = SpecializedValidationStatus.REJECTED
            obs.validation_reason = f"Fire confidence {obs.confidence:.2f} below minimum {self.MIN_CONFIDENCE_FIRE}"
            return obs

        # Solid static surface rejection (vest, cone, sign)
        std_lum = metrics.get("std_luminance", 10.0)
        circularity = metrics.get("circularity", 0.0)
        if std_lum < 6.5 and circularity > 0.65:
            obs.validation_status = SpecializedValidationStatus.REJECTED
            obs.validation_reason = "Static solid-color reflective surface (uniform luminance, high circularity)"
            return obs

        # Specular reflection rejection (glossy product packaging, merchandise displays, reflective signs)
        core_ratio = metrics.get("incandescent_core_ratio", 1.0)
        if metrics.get("inference_source") == "forensic_chromatic_rules":
            if core_ratio < 0.08 or metrics.get("is_specular_glare", False):
                obs.validation_status = SpecializedValidationStatus.REJECTED
                obs.validation_reason = f"Specular reflection on non-combustion surface: core ratio ({core_ratio:.1%}) below thermal combustion threshold (8%)"
                return obs

        # Rigid background motion rejection (stationary merchandise moving solely with camera shake/pan)
        if metrics.get("is_rigid_background", False):
            obs.validation_status = SpecializedValidationStatus.REJECTED
            obs.validation_reason = "Static background fixture / merchandise: region motion is rigid with background camera movement, lacking flame fluid dynamics"
            return obs

        # Geometric aspect ratio check for linear fixtures / shelf edges
        aspect = bw / max(1.0, bh)
        if aspect > 3.5 or aspect < 0.25:
            obs.validation_status = SpecializedValidationStatus.REJECTED
            obs.validation_reason = f"Non-flame geometric aspect ratio {aspect:.2f} consistent with linear fixture or shelf edge"
            return obs

        # Pedestrian clothing / accessory rejection
        if metrics.get("is_person_clothing", False):
            obs.validation_status = SpecializedValidationStatus.REJECTED
            obs.validation_reason = "Chromaticity co-located with pedestrian clothing/accessory without flame turbulence"
            return obs

        # Incandescence core check: Combustion flames produce intense sensor luminance saturation (>= 235.0)
        # Warm floor tiles, wooden panels, and ambient interior lighting (luminance 180-225) lack thermal combustion
        max_lum = metrics.get("max_luminance", 200.0)
        if max_lum < 235.0:
            obs.validation_status = SpecializedValidationStatus.REJECTED
            obs.validation_reason = f"Sub-incandescent optical luminance ({max_lum:.1f} < 235.0): consistent with warm ambient surface or lighting reflection, lacks combustion core"
            return obs

        obs.validation_status = SpecializedValidationStatus.VALID
        obs.validation_reason = "Fire chromaticity and luminance thermal profile validated"
        return obs

    def _validate_weapon(self, obs: SpecializedObservation, metrics: Dict[str, Any]) -> SpecializedObservation:
        # Foundation detectors without trained models must abstain
        inference_src = metrics.get("inference_source", "")
        if "foundation" in inference_src or obs.confidence <= 0.0:
            obs.validation_status = SpecializedValidationStatus.REJECTED
            obs.validation_reason = "Weapon foundation mode abstains from unvalidated detection"
            return obs

        if obs.confidence >= 0.50:
            obs.validation_status = SpecializedValidationStatus.VALID
            obs.validation_reason = "Weapon model detection verified"
        else:
            obs.validation_status = SpecializedValidationStatus.REJECTED
            obs.validation_reason = f"Weapon confidence {obs.confidence:.2f} below 0.50"
        return obs

    def _validate_pose(self, obs: SpecializedObservation, metrics: Dict[str, Any]) -> SpecializedObservation:
        if obs.confidence >= 0.40:
            obs.validation_status = SpecializedValidationStatus.VALID
            obs.validation_reason = "Pose keypoint action verified"
        else:
            obs.validation_status = SpecializedValidationStatus.REJECTED
            obs.validation_reason = f"Pose confidence {obs.confidence:.2f} below 0.40"
        return obs

    def validate_all(
        self,
        observations: List[SpecializedObservation],
        frame: Optional[np.ndarray] = None,
        timestamp: float = 0.0,
    ) -> List[SpecializedObservation]:
        """Validate a list of observations in-place and return them."""
        return [self.validate_observation(o, frame=frame, timestamp=timestamp) for o in observations]
