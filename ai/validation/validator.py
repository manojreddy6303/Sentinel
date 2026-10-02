"""
Sentinel Detection Validation Engine

Multi-signal validator that evaluates raw YOLO detections against geometric sanity,
class-specific profiles, temporal persistence across neighboring frames, and spatial context.
Produces structured ValidationResult (VALID, UNCERTAIN, REJECTED) with diagnostic reasons.
"""

import math
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Set
from .policy import DetectionValidationPolicy, ValidationStatus, ClassValidationRule


@dataclass
class ValidationResult:
    status: ValidationStatus
    score: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "score": round(self.score, 4),
            "reason": self.reason,
        }


class DetectionValidator:
    """
    Centralized, multi-signal object detection validation engine.
    """

    def __init__(self, policy: Optional[DetectionValidationPolicy] = None):
        self.policy = policy or DetectionValidationPolicy()

    @staticmethod
    def _compute_iou(b1: Dict[str, float], b2: Dict[str, float]) -> float:
        ix1 = max(b1["x1"], b2["x1"])
        iy1 = max(b1["y1"], b2["y1"])
        ix2 = min(b1["x2"], b2["x2"])
        iy2 = min(b1["y2"], b2["y2"])
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        a1 = max(0.0, b1["x2"] - b1["x1"]) * max(0.0, b1["y2"] - b1["y1"])
        a2 = max(0.0, b2["x2"] - b2["x1"]) * max(0.0, b2["y2"] - b2["y1"])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _compute_centroid_distance(b1: Dict[str, float], b2: Dict[str, float]) -> float:
        c1x = (b1["x1"] + b1["x2"]) / 2.0
        c1y = (b1["y1"] + b1["y2"]) / 2.0
        c2x = (b2["x1"] + b2["x2"]) / 2.0
        c2y = (b2["y1"] + b2["y2"]) / 2.0
        return math.hypot(c1x - c2x, c1y - c2y)

    def validate_single_detection(
        self,
        detection: Dict[str, Any],
        temporal_neighbors: Optional[List[Dict[str, Any]]] = None,
        context_neighbors: Optional[List[Dict[str, Any]]] = None,
        frame_width: Optional[float] = None,
        frame_height: Optional[float] = None,
    ) -> ValidationResult:
        """
        Validate an individual detection using geometry, confidence, temporal support, and context.
        """
        obj_class = (detection.get("object_class") or "").lower().strip()
        rule = self.policy.get_rule(obj_class)
        conf = float(detection.get("confidence") or 0.0)
        bbox = detection.get("bounding_box") or {}

        # -------------------------------------------------------------------
        # 1. Bounding Box Geometric Sanity Checks
        # -------------------------------------------------------------------
        x1 = float(bbox.get("x1", 0.0))
        y1 = float(bbox.get("y1", 0.0))
        x2 = float(bbox.get("x2", 0.0))
        y2 = float(bbox.get("y2", 0.0))
        w = x2 - x1
        h = y2 - y1

        # Check for zero, negative, or inverted dimensions
        if w <= 0.0 or h <= 0.0 or x2 <= x1 or y2 <= y1:
            return ValidationResult(
                status=ValidationStatus.REJECTED,
                score=0.0,
                reason=f"Degenerate or inverted bounding box dimensions (width={w:.1f}, height={h:.1f})",
            )

        # Check bounding box aspect ratio
        aspect_ratio = w / h
        if aspect_ratio < rule.min_aspect_ratio or aspect_ratio > rule.max_aspect_ratio:
            return ValidationResult(
                status=ValidationStatus.REJECTED,
                score=0.10,
                reason=f"Implausible aspect ratio {aspect_ratio:.2f} for '{obj_class}' (expected {rule.min_aspect_ratio:.2f}-{rule.max_aspect_ratio:.2f})",
            )

        # Check absolute pixel bounds against single-pixel sensor noise
        area_pixels = w * h
        if w < rule.min_pixel_dimension or h < rule.min_pixel_dimension or area_pixels < rule.min_pixel_area:
            if conf < 0.50:
                return ValidationResult(
                    status=ValidationStatus.REJECTED,
                    score=0.05,
                    reason=f"Extremely small bounding box ({w:.0f}x{h:.0f} px, {area_pixels:.0f} px) below absolute minimum for '{obj_class}'",
                )

        # -------------------------------------------------------------------
        # 2. High Confidence Standalone Override
        # -------------------------------------------------------------------
        if conf >= self.policy.high_confidence_override:
            return ValidationResult(
                status=ValidationStatus.VALID,
                score=conf,
                reason=f"High confidence standalone detection ({conf:.2f})",
            )

        # -------------------------------------------------------------------
        # 3. Temporal Support Analysis
        # -------------------------------------------------------------------
        temporal_support = False
        matching_support_count = 0
        ref_diag = math.hypot(w, h)
        frame_scale = math.hypot(frame_width or 1920.0, frame_height or 1080.0)
        max_dist = max(ref_diag * 1.5, frame_scale * self.policy.temporal_max_centroid_disp)

        if temporal_neighbors:
            for nb in temporal_neighbors:
                nb_class = (nb.get("object_class") or "").lower().strip()
                # Check matching or compatible vehicle class
                is_compatible_class = (nb_class == obj_class)
                if not is_compatible_class and obj_class in {"bus", "truck", "car"} and nb_class in {"bus", "truck", "car"}:
                    is_compatible_class = True

                if is_compatible_class:
                    nb_box = nb.get("bounding_box") or {}
                    iou = self._compute_iou(bbox, nb_box)
                    dist = self._compute_centroid_distance(bbox, nb_box)
                    if iou >= self.policy.temporal_match_iou or dist <= max_dist:
                        temporal_support = True
                        matching_support_count += 1

        # -------------------------------------------------------------------
        # 4. Contextual Boost (e.g. luggage/portable items near persons)
        # -------------------------------------------------------------------
        context_boost = False
        if rule.context_boost_classes and context_neighbors:
            for ctx in context_neighbors:
                ctx_class = (ctx.get("object_class") or "").lower().strip()
                if ctx_class in rule.context_boost_classes:
                    ctx_box = ctx.get("bounding_box") or {}
                    dist = self._compute_centroid_distance(bbox, ctx_box)
                    if dist <= frame_scale * self.policy.context_proximity_distance:
                        context_boost = True
                        break

        # -------------------------------------------------------------------
        # 5. Multi-Signal Decision Logic
        # -------------------------------------------------------------------
        if temporal_support:
            if conf >= rule.min_confidence_with_temporal:
                boosted_score = min(1.0, conf + 0.10 + min(0.10, matching_support_count * 0.03))
                return ValidationResult(
                    status=ValidationStatus.VALID,
                    score=boosted_score,
                    reason=f"Validated by temporal persistence ({matching_support_count} matches) and confidence ({conf:.2f})",
                )
            elif conf >= rule.min_confidence_with_temporal * 0.85:
                return ValidationResult(
                    status=ValidationStatus.UNCERTAIN,
                    score=conf,
                    reason=f"Marginal confidence ({conf:.2f}) despite temporal support",
                )
            else:
                return ValidationResult(
                    status=ValidationStatus.REJECTED,
                    score=conf,
                    reason=f"Low confidence ({conf:.2f}) below temporal threshold ({rule.min_confidence_with_temporal:.2f})",
                )

        # Standalone evaluation (no temporal support)
        if context_boost and conf >= rule.min_confidence_with_temporal:
            return ValidationResult(
                status=ValidationStatus.VALID,
                score=min(1.0, conf + 0.08),
                reason=f"Validated by contextual association with person ({conf:.2f})",
            )

        # Area check normalized to canonical reference frame smoothly across all resolutions
        canonical_area = self.policy.canonical_reference_width * self.policy.canonical_reference_height
        if frame_width and frame_height and frame_width > 0 and frame_height > 0:
            frame_area = frame_width * frame_height
            scale_to_canonical = frame_area / canonical_area
            # Scale area pixels to 1080p-equivalent area to prevent sensor resolution bias
            equiv_area = area_pixels / scale_to_canonical if scale_to_canonical > 0 else area_pixels
            effective_area_frac = equiv_area / canonical_area
            if effective_area_frac < rule.min_area_fraction:
                if conf >= rule.min_confidence_standalone:
                    return ValidationResult(
                        status=ValidationStatus.UNCERTAIN,
                        score=conf,
                        reason=f"Small isolated detection (area fraction {effective_area_frac:.5f} < {rule.min_area_fraction:.5f}) with standalone confidence ({conf:.2f})",
                    )
                elif obj_class in {"backpack", "suitcase", "handbag"} and conf >= rule.min_confidence_with_temporal:
                    # Unattended luggage catch-22 fix: allow to reach UNCERTAIN for dwell evaluation
                    return ValidationResult(
                        status=ValidationStatus.UNCERTAIN,
                        score=conf,
                        reason=f"Candidate small unattended {obj_class} ({conf:.2f}) pending temporal dwell evaluation",
                    )
                else:
                    return ValidationResult(
                        status=ValidationStatus.REJECTED,
                        score=0.15,
                        reason=f"Area fraction {effective_area_frac:.5f} below minimum {rule.min_area_fraction:.5f} for '{obj_class}' without temporal support",
                    )

        if rule.requires_temporal_support:
            # Heavy classes like bus/truck require temporal support unless high confidence
            if conf >= rule.min_confidence_standalone:
                return ValidationResult(
                    status=ValidationStatus.UNCERTAIN,
                    score=conf,
                    reason=f"Isolated single-frame detection of {obj_class} without temporal persistence",
                )
            else:
                return ValidationResult(
                    status=ValidationStatus.REJECTED,
                    score=conf,
                    reason=f"Isolated low-confidence {obj_class} detection ({conf:.2f}) without temporal persistence",
                )
        else:
            if conf >= rule.min_confidence_standalone:
                return ValidationResult(
                    status=ValidationStatus.VALID,
                    score=conf,
                    reason=f"Sufficient standalone confidence ({conf:.2f}) for '{obj_class}'",
                )
            elif conf >= rule.min_confidence_with_temporal:
                return ValidationResult(
                    status=ValidationStatus.UNCERTAIN,
                    score=conf,
                    reason=f"Unconfirmed single-frame detection with intermediate confidence ({conf:.2f})",
                )
            elif obj_class in {"backpack", "suitcase", "handbag"} and conf >= (rule.min_confidence_with_temporal * 0.90):
                # Unattended luggage catch-22 fix: allow marginal confidence luggage to reach UNCERTAIN
                return ValidationResult(
                    status=ValidationStatus.UNCERTAIN,
                    score=conf,
                    reason=f"Candidate unattended {obj_class} ({conf:.2f}) pending temporal dwell evaluation",
                )
            else:
                return ValidationResult(
                    status=ValidationStatus.REJECTED,
                    score=conf,
                    reason=f"Low confidence ({conf:.2f}) below standalone threshold ({rule.min_confidence_standalone:.2f})",
                )

    def validate_sequence(
        self,
        frame_detections: List[Dict[str, Any]],
        frame_width: Optional[float] = None,
        frame_height: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Validate a sequence of per-frame detections across the video timeline.
        Annotates each detection in-place with:
          - 'validation_status': 'VALID' | 'UNCERTAIN' | 'REJECTED'
          - 'validation_score': float
          - 'validation_reason': str
        """
        # Flatten all detections with temporal index for fast temporal lookup
        all_dets: List[Tuple[float, Dict[str, Any]]] = []
        for entry in frame_detections:
            dets = entry.get("detections", [])
            for d in dets:
                t = float(d.get("timestamp", 0.0))
                all_dets.append((t, d))

        win = self.policy.temporal_window_seconds

        for idx, (t, d) in enumerate(all_dets):
            # Gather temporal neighbors in neighboring frames within window
            temporal_neighbors = []
            context_neighbors = []
            for jdx, (other_t, other_d) in enumerate(all_dets):
                if idx == jdx:
                    continue
                dt = abs(other_t - t)
                # Same frame context (e.g. luggage near person in current frame)
                if dt < 0.01:
                    context_neighbors.append(other_d)
                # Nearby frame temporal and context support
                elif dt <= win:
                    temporal_neighbors.append(other_d)
                    context_neighbors.append(other_d)

            res = self.validate_single_detection(
                detection=d,
                temporal_neighbors=temporal_neighbors,
                context_neighbors=context_neighbors,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            d["validation_status"] = res.status.value
            d["validation_score"] = round(res.score, 4)
            d["validation_reason"] = res.reason

        return frame_detections

    def validate_events_list(
        self,
        events: List[Dict[str, Any]],
        frame_width: Optional[float] = None,
        frame_height: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Validate a flat list of detection events.
        Annotates each event dict with 'validation_status', 'validation_score', 'validation_reason'.
        """
        win = self.policy.temporal_window_seconds
        for idx, ev in enumerate(events):
            t = float(ev.get("timestamp", 0.0))
            temporal_neighbors = []
            context_neighbors = []
            for jdx, other_ev in enumerate(events):
                if idx == jdx:
                    continue
                dt = abs(float(other_ev.get("timestamp", 0.0)) - t)
                if dt < 0.01:
                    context_neighbors.append(other_ev)
                elif dt <= win:
                    temporal_neighbors.append(other_ev)
                    context_neighbors.append(other_ev)

            res = self.validate_single_detection(
                detection=ev,
                temporal_neighbors=temporal_neighbors,
                context_neighbors=context_neighbors,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            ev["validation_status"] = res.status.value
            ev["validation_score"] = round(res.score, 4)
            ev["validation_reason"] = res.reason

        return events
