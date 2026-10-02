"""
Fire and Smoke Visual Detectors (Phase 15)

Implements dedicated visual detectors for:
1. Fire / Flame Visual Evidence (POTENTIAL_FIRE)
2. Smoke Plume Visual Evidence (POTENTIAL_SMOKE)

Supports optional custom YOLO weights if provided locally, with a robust,
scientifically validated computer vision feature extraction fallback (YCbCr/HSV
chromaticity rules + spatial morphology) when external models are absent.
"""
import os
import uuid
import logging
from typing import Dict, Any, List, Optional, Tuple
import cv2
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


class FireVisualDetector(BaseSpecializedDetector):
    """
    Dedicated visual flame and fire evidence detector.
    Does NOT equate generic red/orange objects with combustion fire.
    Evaluates:
    - Flame chromaticity in YCbCr & HSV color spaces
    - Localized high-intensity thermal cores
    - Spatial clustering and area thresholds
    """

    detector_name: str = "fire_visual_detector"
    detector_version: str = "1.0.0"
    supported_classes: List[str] = ["fire", "flame"]

    def __init__(self, enabled: bool = True, config: Optional[Dict[str, Any]] = None):
        super().__init__(enabled=enabled, config=config)
        self.model_path = os.environ.get("FIRE_MODEL_PATH") or self.config.get("model_path")
        self._custom_model = None
        self._use_cv_fallback = True
        self._prev_frame_gray: Optional[np.ndarray] = None
        self._prev_timestamp: Optional[float] = None
        self._prev_candidates: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Reset temporal state across video transitions."""
        self._prev_frame_gray = None
        self._prev_timestamp = None
        self._prev_candidates.clear()

    def initialize(self) -> bool:
        if self._is_initialized:
            return self._status == SpecializedDetectorStatus.AVAILABLE

        # Check if custom model file exists
        if self.model_path and os.path.isfile(self.model_path):
            try:
                from ultralytics import YOLO
                self._custom_model = YOLO(self.model_path)
                self._use_cv_fallback = False
                self._status = SpecializedDetectorStatus.AVAILABLE
                self._status_reason = f"Loaded custom weights from {self.model_path}"
                logger.info(f"Fire detector initialized with custom weights: {self.model_path}")
            except Exception as e:
                logger.warning(f"Could not load custom fire weights ({e}). Falling back to heuristic CV engine.")
                self._use_cv_fallback = True
                self._status = SpecializedDetectorStatus.AVAILABLE
                self._status_reason = "Operating in heuristic chromatic visual mode"
        else:
            # Operational via scientifically grounded heuristic CV engine
            self._use_cv_fallback = True
            self._status = SpecializedDetectorStatus.AVAILABLE
            self._status_reason = "Operating in forensic chromatic visual analysis mode (no custom weights required)"

        self._is_initialized = True
        return True

    def detect_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_idx: int = 0,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[SpecializedObservation]:
        observations: List[SpecializedObservation] = []

        if self._custom_model is not None and not self._use_cv_fallback:
            # Run custom YOLO inference
            results = self._custom_model.predict(frame, verbose=False, conf=0.35)
            if results and len(results) > 0:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()
                    bbox = BoundingBox(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3])
                    obs = SpecializedObservation(
                        observation_id=f"OBS-FIRE-{uuid.uuid4().hex[:6]}",
                        detector_name=self.detector_name,
                        detector_version=self.detector_version,
                        class_name="fire",
                        timestamp=timestamp,
                        confidence=conf,
                        evidence_strength=round(min(1.0, conf * 0.95), 4),
                        bounding_box=bbox,
                        validation_status=SpecializedValidationStatus.RAW,
                        frame_number=frame_idx,
                        visual_metrics={"inference_source": "custom_yolo"},
                    )
                    observations.append(obs)
            return observations

        # Heuristic Chromatic Flame Analysis Fallback
        # Flame rule: Y >= Cb, Cr >= Cb, and |Cb - Cr| >= threshold
        # In HSV: Hue in [0, 32] (red-yellow) or [170, 180] (red wrap), Saturation >= 70, Value >= 160
        h, w = frame.shape[:2]
        if h < 32 or w < 32:
            return []

        # Downsample large frames for performance if needed
        scale = 1.0
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        work_frame = frame
        if max(h, w) > 960:
            scale = 960.0 / max(h, w)
            work_frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        ycrcb = cv2.cvtColor(work_frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)

        # Condition 1: Y(x,y) >= Cb(x,y) and Cr(x,y) >= Cb(x,y)
        rule_y_cb = (y >= cb)
        rule_cr_cb = (cr >= cb)
        rule_diff = (cv2.absdiff(cr, cb) >= 32)

        # Condition 2: Flame HSV range (Incandescent bright orange-red-yellow)
        rule_h = ((hue <= 30) | (hue >= 170))
        rule_s = (sat >= 80)
        rule_v = (val >= 210)

        # Combined flame mask
        flame_mask = (rule_y_cb & rule_cr_cb & rule_diff & rule_h & rule_s & rule_v).astype(np.uint8) * 255

        # Morphological opening to remove isolated noise pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_mask = cv2.morphologyEx(flame_mask, cv2.MORPH_OPEN, kernel)

        # Find connected candidate flame contours
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_flame_area = (h * w) * 0.00015  # Minimum 0.015% of frame

        from ai.common.numeric import clamp_finite, ensure_finite

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_flame_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue

            # Circularity / compactness: 4*pi*area / (perimeter^2)
            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)

            bx, by, bw, bh = cv2.boundingRect(cnt)
            # Rescale to original frame dimensions
            orig_x1 = max(0.0, bx / scale)
            orig_y1 = max(0.0, by / scale)
            orig_x2 = min(float(w), (bx + bw) / scale)
            orig_y2 = min(float(h), (by + bh) / scale)

            if orig_x2 <= orig_x1 or orig_y2 <= orig_y1:
                continue

            # Intensity metrics
            roi_y = y[by:by+bh, bx:bx+bw]
            roi_val = val[by:by+bh, bx:bx+bw]
            mean_lum = float(np.mean(roi_val))
            max_lum = float(np.max(roi_val))
            std_lum = float(np.std(roi_val))

            # Thermal core requirement: Genuine combustion fire exhibits an incandescent core
            # (>232 in 8-bit scale). Non-combustion red/orange surfaces lack this core.
            if max_lum < 232.0:
                continue

            # Core saturation ratio check:
            # Combustion flames possess an incandescent core occupying >= 8% of the flame area.
            # Isolated specular highlights from overhead store lighting on glossy packaging cover < 5% of the patch.
            roi_cleaned_mask = cleaned_mask[by:by+bh, bx:bx+bw]
            flame_pixels = int(np.count_nonzero(roi_cleaned_mask))
            core_pixels = int(np.count_nonzero((roi_cleaned_mask > 0) & (roi_val >= 235)))
            core_ratio = core_pixels / max(1, flame_pixels)
            is_specular = (core_ratio < 0.08)

            # Specular reflection check on stationary merchandise/print:
            # If peak luminance reached threshold solely due to pinpoint specular reflection (< 5% core ratio),
            # reject as non-combustion reflective surface
            if core_ratio < 0.05:
                continue

            # Suppress static solid orange surfaces (safety vests, traffic cones, signs, posters)
            # Solid painted surfaces have low internal variance (std < 9.0) and high geometric compactness
            is_static_surface = (std_lum < 9.0 and circularity > 0.50) or (std_lum < 6.5)
            if is_static_surface:
                continue

            # Motion / Rigid Background Consistency check:
            # Genuine combustion flames flicker and exhibit turbulent non-rigid fluid dynamics.
            # Stationary store shelves, merchandise displays, posters, and signs move in rigid lockstep with camera motion.
            is_rigid_bg = False
            if self._prev_frame_gray is not None and self._prev_frame_gray.shape == curr_gray.shape:
                try:
                    px1 = max(0, int(orig_x1))
                    py1 = max(0, int(orig_y1))
                    px2 = min(w, int(orig_x2))
                    py2 = min(h, int(orig_y2))
                    if (px2 - px1) >= 8 and (py2 - py1) >= 8:
                        p_prev = self._prev_frame_gray[py1:py2, px1:px2]
                        p_curr = curr_gray[py1:py2, px1:px2]
                        (dx_p, dy_p), r_p = cv2.phaseCorrelate(np.float32(p_prev), np.float32(p_curr))
                        (dx_bg, dy_bg), r_bg = cv2.phaseCorrelate(np.float32(self._prev_frame_gray), np.float32(curr_gray))
                        motion_diff = math.hypot(dx_p - dx_bg, dy_p - dy_bg)
                        if motion_diff <= 1.5 and r_p >= 0.65:
                            is_rigid_bg = True
                except Exception:
                    pass

            if is_rigid_bg and core_ratio < 0.15:
                # Stationary shelf / packaging moving purely with camera background motion
                continue

            # Person clothing / accessory suppression:
            # Prevents human clothing (red dresses, orange jackets, red bags/shoes) from triggering false fire
            person_boxes = (context or {}).get("person_bounding_boxes", [])
            is_person_clothing = False
            cnt_cx = (orig_x1 + orig_x2) / 2.0
            cnt_cy = (orig_y1 + orig_y2) / 2.0
            cnt_area = (orig_x2 - orig_x1) * (orig_y2 - orig_y1)

            for pb in person_boxes:
                px1 = float(pb.get("x1", 0.0))
                py1 = float(pb.get("y1", 0.0))
                px2 = float(pb.get("x2", 0.0))
                py2 = float(pb.get("y2", 0.0))
                pw = max(1.0, px2 - px1)
                ph = max(1.0, py2 - py1)
                p_area = pw * ph

                # Check if centroid is within expanded person bounding box (15% margin)
                if (px1 - pw * 0.15) <= cnt_cx <= (px2 + pw * 0.15) and (py1 - ph * 0.05) <= cnt_cy <= (py2 + ph * 0.05):
                    # Unless area has extreme thermal luminance (>250) and violent turbulence (>28), it's clothing/accessory
                    if max_lum < 250.0 or std_lum < 28.0 or (cnt_area < 0.45 * p_area):
                        is_person_clothing = True
                        break

            if is_person_clothing:
                continue

            # Vehicle lighting / tail lamp suppression:
            # Prevents vehicle brake lights, blinkers, and red body paint from triggering false fire
            vehicle_boxes = (context or {}).get("vehicle_bounding_boxes", [])
            is_vehicle_light = False
            for vb in vehicle_boxes:
                vx1 = float(vb.get("x1", 0.0))
                vy1 = float(vb.get("y1", 0.0))
                vx2 = float(vb.get("x2", 0.0))
                vy2 = float(vb.get("y2", 0.0))
                vw = max(1.0, vx2 - vx1)
                vh = max(1.0, vy2 - vy1)
                if vx1 <= cnt_cx <= vx2 and vy1 <= cnt_cy <= vy2:
                    if max_lum < 252.0 or std_lum < 28.0 or (cnt_area < 0.15 * (vw * vh)):
                        is_vehicle_light = True
                        break

            if is_vehicle_light:
                continue

            # Confidence based on flame luminance saturation consistency & texture turbulence
            base_conf = (mean_lum / 255.0) * 0.75 + (area / (h * w * scale * scale)) * 5.0 + min(0.15, std_lum / 100.0)
            conf = clamp_finite(base_conf, 0.40, 0.92, default=0.50)
            evidence_str = clamp_finite(conf * 0.90, 0.35, 0.90, default=0.45)

            obs = SpecializedObservation(
                observation_id=f"OBS-FIRE-{uuid.uuid4().hex[:6]}",
                detector_name=self.detector_name,
                detector_version=self.detector_version,
                class_name="fire",
                timestamp=ensure_finite(timestamp, 0.0),
                confidence=round(conf, 4),
                evidence_strength=round(evidence_str, 4),
                bounding_box=BoundingBox(
                    x1=round(orig_x1, 2),
                    y1=round(orig_y1, 2),
                    x2=round(orig_x2, 2),
                    y2=round(orig_y2, 2),
                ),
                validation_status=SpecializedValidationStatus.RAW,
                frame_number=frame_idx,
                visual_metrics={
                    "area_pixels": round(area / (scale * scale), 1),
                    "mean_luminance": round(mean_lum, 1),
                    "max_luminance": round(max_lum, 1),
                    "std_luminance": round(std_lum, 1),
                    "circularity": round(circularity, 3),
                    "aspect_ratio": round(bw / max(1, bh), 2),
                    "incandescent_core_ratio": round(core_ratio, 4),
                    "is_specular_glare": is_specular,
                    "is_rigid_background": is_rigid_bg,
                    "inference_source": "forensic_chromatic_rules",
                },
            )
            observations.append(obs)

        self._prev_frame_gray = curr_gray
        self._prev_timestamp = timestamp
        return observations

    detect = detect_frame


class SmokeVisualDetector(BaseSpecializedDetector):
    """
    Dedicated visual smoke plume evidence detector.
    Identifies smoke-like low-saturation, moderate/high-luminance regions
    with spatial expansion characteristics while rejecting global fog and dust.
    """

    detector_name: str = "smoke_visual_detector"
    detector_version: str = "1.0.0"
    supported_classes: List[str] = ["smoke", "plume"]

    def __init__(self, enabled: bool = True, config: Optional[Dict[str, Any]] = None):
        super().__init__(enabled=enabled, config=config)
        self.model_path = os.environ.get("SMOKE_MODEL_PATH") or self.config.get("model_path")
        self._custom_model = None
        self._use_cv_fallback = True
        self._prev_frame_gray: Optional[np.ndarray] = None
        self._prev_timestamp: Optional[float] = None
        self._prev_candidates: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Reset temporal state across video transitions."""
        self._prev_frame_gray = None
        self._prev_timestamp = None
        self._prev_candidates = []

    def initialize(self) -> bool:
        if self._is_initialized:
            return self._status == SpecializedDetectorStatus.AVAILABLE

        if self.model_path and os.path.isfile(self.model_path):
            try:
                from ultralytics import YOLO
                self._custom_model = YOLO(self.model_path)
                self._use_cv_fallback = False
                self._status = SpecializedDetectorStatus.AVAILABLE
                self._status_reason = f"Loaded custom smoke weights from {self.model_path}"
                logger.info(f"Smoke detector initialized with custom weights: {self.model_path}")
            except Exception as e:
                logger.warning(f"Could not load custom smoke weights ({e}). Falling back to heuristic CV.")
                self._use_cv_fallback = True
                self._status = SpecializedDetectorStatus.AVAILABLE
                self._status_reason = "Operating in heuristic chromatic visual mode"
        else:
            self._use_cv_fallback = True
            self._status = SpecializedDetectorStatus.AVAILABLE
            self._status_reason = "Operating in forensic smoke texture/saturation mode (no custom weights required)"

        self._is_initialized = True
        return True

    def detect_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_idx: int = 0,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[SpecializedObservation]:
        observations: List[SpecializedObservation] = []

        if self._custom_model is not None and not self._use_cv_fallback:
            results = self._custom_model.predict(frame, verbose=False, conf=0.35)
            if results and len(results) > 0:
                for box in results[0].boxes:
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()
                    bbox = BoundingBox(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3])
                    obs = SpecializedObservation(
                        observation_id=f"OBS-SMOKE-{uuid.uuid4().hex[:6]}",
                        detector_name=self.detector_name,
                        detector_version=self.detector_version,
                        class_name="smoke",
                        timestamp=timestamp,
                        confidence=conf,
                        evidence_strength=round(min(1.0, conf * 0.90), 4),
                        bounding_box=bbox,
                        validation_status=SpecializedValidationStatus.RAW,
                        frame_number=frame_idx,
                        visual_metrics={"inference_source": "custom_yolo"},
                    )
                    observations.append(obs)
            return observations

        # Heuristic Smoke Plume Texture & Saturation Analysis
        # Smoke characteristics: Low saturation (gray/white/light-brown), moderate-to-high luminance
        # Distinct from global fog (smoke is localized, fog is uniform across whole frame)
        h, w = frame.shape[:2]
        if h < 32 or w < 32:
            return []

        scale = 1.0
        work_frame = frame
        if max(h, w) > 960:
            scale = 960.0 / max(h, w)
            work_frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)

        # Global saturation & IR/monochrome check (if whole scene is desaturated fog, night vision, or IR monochrome CCTV)
        scene_mean_sat = float(np.mean(sat))
        p90_sat = float(np.percentile(sat, 90))
        pct_low_sat = float(np.mean(sat <= 55))
        if scene_mean_sat < 35.0 or pct_low_sat > 0.85 or p90_sat < 65.0:
            # Whole scene is gray/fog/night/IR monochrome — abstain from heuristic chromatic smoke detection
            self._prev_frame_gray = cv2.cvtColor(work_frame, cv2.COLOR_BGR2GRAY)
            self._prev_timestamp = timestamp
            return []

        # Smoke condition: Low Saturation (<= 55) and Moderate-to-High Value (110 <= V <= 235)
        smoke_mask = ((sat <= 55) & (val >= 110) & (val <= 235)).astype(np.uint8) * 255

        # Morphological filter to remove isolated noise and connect localized plumes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_smoke_area = (h * w) * 0.0015  # Minimum 0.15% of frame (filters tiny digital noise speckles)
        max_smoke_area = (h * w) * 0.25    # Maximum 25% of frame (localized plume; not global room background)

        # Scale dimensions for guard computations
        sh = h * scale
        sw = w * scale

        curr_frame_gray = cv2.cvtColor(work_frame, cv2.COLOR_BGR2GRAY)

        # Estimate global camera motion across background
        global_motion = 0.0
        if self._prev_frame_gray is not None and self._prev_frame_gray.shape == curr_frame_gray.shape:
            global_motion = float(np.mean(np.abs(curr_frame_gray.astype(float) - self._prev_frame_gray.astype(float))))
            if global_motion > 18.0:
                # Global camera pan, tilt, or macro-shake: local plume motion is ambiguous
                self._prev_frame_gray = curr_frame_gray
                self._prev_timestamp = timestamp
                self._prev_candidates = []
                return []

        curr_candidates: List[Dict[str, Any]] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_smoke_area or area > max_smoke_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            orig_x1 = max(0.0, bx / scale)
            orig_y1 = max(0.0, by / scale)
            orig_x2 = min(float(w), (bx + bw) / scale)
            orig_y2 = min(float(h), (by + bh) / scale)

            if orig_x2 <= orig_x1 or orig_y2 <= orig_y1:
                continue

            # Minimum dimensions for plausible smoke plume
            if bw < 16 or bh < 16:
                continue

            # Plume cannot span the vast majority of camera width or height
            if bw > (sw * 0.55) or bh > (sh * 0.55):
                continue

            # Aspect ratio check: plumes expand diffusely; reject extreme needle or strip geometries
            aspect = bw / max(1, bh)
            if aspect < 0.32 or aspect > 2.8:
                continue

            # Suppress static boundary strips & edge-clamped architectural borders
            is_border_anchored = (bx <= 4 or (bx + bw) >= (int(sw) - 4) or by <= 4 or (by + bh) >= (int(sh) - 4))
            if is_border_anchored and (bw < 48 or bh < 48 or by <= 4):
                continue

            roi_sat = sat[by:by+bh, bx:bx+bw]
            roi_val = val[by:by+bh, bx:bx+bw]
            mean_s = float(np.mean(roi_sat))
            mean_v = float(np.mean(roi_val))

            # 1. Suppress ceiling lighting fixtures (top 12% of frame with extreme brightness)
            if by < (sh * 0.12) and mean_v > 230:
                continue

            # 2. Suppress direct specular glare (near-white saturated lamps)
            if mean_v > 248 and mean_s < 8.0:
                continue

            # 3. Suppress bottom-strip floor reflections and flat ground surfaces
            if (by + bh) > (sh * 0.82) and by > (sh * 0.65):
                continue

            # 4. Suppress near-square compact blobs (ceiling panels, light covers, signs)
            normalized_area = area / max(1.0, sh * sw)
            if 0.80 <= aspect <= 1.25 and normalized_area < 0.018:
                continue

            # 5. Texture roughness analysis:
            # Real smoke plumes are soft, diffuse volumetric clouds with low-to-moderate spatial frequency (18.0 <= roughness <= 260.0).
            # Extreme roughness (> 260.0) is the signature of high-frequency solid textures: asphalt gravel, concrete pavement, brickwork, roof tiles.
            roi_gray = curr_frame_gray[by:by+bh, bx:bx+bw]
            texture_roughness = float(cv2.Laplacian(roi_gray, cv2.CV_64F).var())
            if texture_roughness < 18.0 or texture_roughness > 260.0:
                continue

            # 6. Person bounding box overlap check:
            # Prevent moving people's desaturated clothing or luggage from being flagged as smoke
            if context and "person_bounding_boxes" in context:
                person_bboxes = context.get("person_bounding_boxes") or []
                is_person_part = False
                scx = orig_x1 + (orig_x2 - orig_x1) / 2.0
                scy = orig_y1 + (orig_y2 - orig_y1) / 2.0
                for pbox in person_bboxes:
                    if isinstance(pbox, dict):
                        px1, py1, px2, py2 = pbox.get("x1", 0), pbox.get("y1", 0), pbox.get("x2", 0), pbox.get("y2", 0)
                    elif hasattr(pbox, "x1"):
                        px1, py1, px2, py2 = pbox.x1, pbox.y1, pbox.x2, pbox.y2
                    else:
                        continue
                    if px1 <= scx <= px2 and py1 <= scy <= py2:
                        is_person_part = True
                        break
                    ix1, iy1 = max(orig_x1, px1), max(orig_y1, py1)
                    ix2, iy2 = min(orig_x2, px2), min(orig_y2, py2)
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        if inter / max(1.0, (orig_x2 - orig_x1) * (orig_y2 - orig_y1)) > 0.30:
                            is_person_part = True
                            break
                if is_person_part:
                    continue

            # 7. Temporal motion evidence:
            # Smoke plumes are dynamic, billowing, and fluctuating fluids.
            # Static background surfaces have negligible inter-frame difference (< 2.2 intensity levels).
            temporal_motion = 5.0  # Default nominal for first frame
            if self._prev_frame_gray is not None and self._prev_timestamp is not None:
                dt = abs(timestamp - self._prev_timestamp)
                if 0.05 <= dt <= 3.0:
                    prev_roi = self._prev_frame_gray[by:by+bh, bx:bx+bw]
                    if prev_roi.shape == roi_gray.shape:
                        temporal_motion = float(np.mean(np.abs(roi_gray.astype(float) - prev_roi.astype(float))))
                        required_motion = max(2.2, global_motion * 0.5 + 1.2)
                        if temporal_motion < required_motion:
                            # Static surface with negligible inter-frame change
                            continue

            # 8. Check spatial anchoring against previous candidate history
            cx = bx + bw / 2.0
            cy = by + bh / 2.0
            is_static_surface = False
            for prev_c in self._prev_candidates:
                dist = math.hypot(cx - prev_c["cx"], cy - prev_c["cy"])
                area_ratio = abs(area - prev_c["area"]) / max(1.0, prev_c["area"])
                if dist < 6.0 and area_ratio < 0.08:
                    is_static_surface = True
                    break

            if is_static_surface:
                continue

            curr_candidates.append({"cx": cx, "cy": cy, "area": area, "bx": bx, "by": by, "bw": bw, "bh": bh})

            # Smoke confidence inversely proportional to saturation and positively to plume size
            from ai.common.numeric import clamp_finite, ensure_finite
            base_conf = 0.50 + (1.0 - mean_s / 55.0) * 0.20 + (area / (h * w * scale * scale)) * 3.0 + min(0.08, texture_roughness / 300.0)
            conf = clamp_finite(base_conf, 0.35, 0.85, default=0.50)
            evidence_str = clamp_finite(conf * 0.85, 0.30, 0.80, default=0.45)

            obs = SpecializedObservation(
                observation_id=f"OBS-SMOKE-{uuid.uuid4().hex[:6]}",
                detector_name=self.detector_name,
                detector_version=self.detector_version,
                class_name="smoke",
                timestamp=ensure_finite(timestamp, 0.0),
                confidence=round(conf, 4),
                evidence_strength=round(evidence_str, 4),
                bounding_box=BoundingBox(
                    x1=round(orig_x1, 2),
                    y1=round(orig_y1, 2),
                    x2=round(orig_x2, 2),
                    y2=round(orig_y2, 2),
                ),
                validation_status=SpecializedValidationStatus.RAW,
                frame_number=frame_idx,
                visual_metrics={
                    "area_pixels": round(area / (scale * scale), 1),
                    "mean_saturation": round(mean_s, 1),
                    "mean_luminance": round(mean_v, 1),
                    "texture_roughness": round(texture_roughness, 1),
                    "temporal_motion": round(temporal_motion, 2),
                    "aspect_ratio": round(aspect, 2),
                    "scene_mean_saturation": round(scene_mean_sat, 1),
                    "global_motion": round(global_motion, 2),
                    "is_border_anchored": is_border_anchored,
                    "inference_source": "forensic_smoke_texture",
                },
            )
            observations.append(obs)

        self._prev_frame_gray = curr_frame_gray
        self._prev_timestamp = timestamp
        self._prev_candidates = curr_candidates
        return observations

