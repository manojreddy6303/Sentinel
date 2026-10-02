"""
Person Visual Attribute Analyzer Module for Sentinel (Phase 20.2)

Extracts clean, non-biometric person visual attributes:
- Upper-body clothing color (torso)
- Lower-body clothing color (legs/pants/skirt)
- Outerwear color (when distinct visual evidence exists)
- Carried-object category and color
- Headwear detection
- Face-region telemetry (non-biometric only)
- Posture and occlusion estimation

SAFETY & PRIVACY CONSTRAINTS:
1. Strict non-biometric boundaries: NO facial recognition, NO identity inference, NO embeddings.
2. Anatomic region partitioning: Never blindly sample color from the full person bounding box.
3. Illumination resilience: Dark/underexposed scenes yield "uncertain", never inventing attributes.
"""
from typing import Dict, Any, Optional, Tuple, List
import logging
import cv2
import numpy as np

from ai.schemas import (
    BoundingBox,
    ClothingColor,
    FaceRegionTelemetry,
    PersonVisualAttributes,
)
from ai.attributes.color_analyzer import RobustColorExtractor, TemporalColorFilter
from ai.faces.face_detector import FaceDetector

logger = logging.getLogger(__name__)


class PersonAttributeAnalyzer:
    """
    Region-aware visual attribute analyzer for human pedestrians.
    """

    def __init__(self, face_detector: Optional[FaceDetector] = None):
        self.face_detector = face_detector or FaceDetector()
        self.upper_color_filter = TemporalColorFilter(history_window=10)
        self.lower_color_filter = TemporalColorFilter(history_window=10)

    def reset(self, video_id: str = "") -> None:
        """Reset temporal state across video sessions."""
        self.upper_color_filter.reset(video_id)
        self.lower_color_filter.reset(video_id)

    def analyze(
        self,
        frame_bgr: np.ndarray,
        person_bbox: BoundingBox,
        timestamp: float,
        track_id: Optional[str] = None,
        scene_report: Optional[Any] = None,
        associated_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> PersonVisualAttributes:
        """
        Analyze person crop and return structured PersonVisualAttributes.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return PersonVisualAttributes(
                track_id=track_id or "ANON",
                timestamp=timestamp,
                bounding_box=person_bbox,
                upper_clothing_color=ClothingColor("unknown", 0.0),
                lower_clothing_color=ClothingColor("unknown", 0.0),
            )

        h_frame, w_frame = frame_bgr.shape[:2]
        px1 = max(0, min(int(person_bbox.x1), w_frame - 1))
        py1 = max(0, min(int(person_bbox.y1), h_frame - 1))
        px2 = max(px1 + 1, min(int(person_bbox.x2), w_frame))
        py2 = max(py1 + 1, min(int(person_bbox.y2), h_frame))

        pw = px2 - px1
        ph = py2 - py1

        # Check minimum resolvable dimensions
        if pw < 8 or ph < 16:
            return PersonVisualAttributes(
                track_id=track_id or "ANON",
                timestamp=timestamp,
                bounding_box=person_bbox,
                upper_clothing_color=ClothingColor("unknown", 0.0, is_illumination_uncertain=False),
                lower_clothing_color=ClothingColor("unknown", 0.0, is_illumination_uncertain=False),
                posture="unknown",
            )

        crop = frame_bgr[py1:py2, px1:px2]

        # Illumination / IR check from scene condition report
        is_ir = False
        if scene_report is not None:
            ill_type = getattr(scene_report, "illumination_type", None)
            ill_val = ill_type.value if hasattr(ill_type, "value") else str(ill_type or "")
            if "night_ir" in ill_val:
                is_ir = True

        # Posture & Aspect Ratio Estimation
        aspect_ratio = pw / float(max(1, ph))
        if aspect_ratio > 1.25:
            posture = "sitting_or_crouched"
        elif aspect_ratio > 0.85:
            posture = "bending_or_angled"
        else:
            posture = "standing"

        # Edge clipping / occlusion analysis
        is_bottom_clipped = (py2 >= h_frame - 3)
        is_top_clipped = (py1 <= 3)
        is_left_clipped = (px1 <= 3)
        is_right_clipped = (px2 >= w_frame - 3)

        occlusion_score = 0.0
        if is_bottom_clipped:
            occlusion_score += 0.35
        if is_top_clipped:
            occlusion_score += 0.20
        if is_left_clipped or is_right_clipped:
            occlusion_score += 0.25

        # =====================================================================
        # 1. UPPER BODY (TORSO) CLOTHING COLOR
        # Anatomic boundaries: 18% to 52% height, 20% to 80% width
        # =====================================================================
        uy1 = int(ph * 0.18)
        uy2 = int(ph * 0.52)
        ux1 = int(pw * 0.20)
        ux2 = int(pw * 0.80)

        upper_crop = crop[uy1:uy2, ux1:ux2]
        if upper_crop.size == 0:
            upper_crop = crop[:int(ph * 0.5), :]

        upper_metrics = RobustColorExtractor.extract_from_roi(upper_crop, filter_skin=True, scene_is_ir=is_ir)
        raw_up_color = upper_metrics["dominant_color"]
        raw_up_conf = upper_metrics["dominant_confidence"]
        up_unc = upper_metrics.get("is_illumination_uncertain", False)

        if track_id:
            st_up_col, st_up_conf, up_cnt, st_up_unc = self.upper_color_filter.update_track_color(
                track_id=track_id,
                color=raw_up_color,
                confidence=raw_up_conf,
                timestamp=timestamp,
                is_illumination_uncertain=up_unc,
            )
        else:
            st_up_col, st_up_conf, up_cnt, st_up_unc = raw_up_color, raw_up_conf, 1, up_unc

        upper_attr = ClothingColor(
            color_name=st_up_col,
            confidence=round(st_up_conf, 4),
            observation_count=up_cnt,
            is_illumination_uncertain=st_up_unc,
            region="upper",
            color_space_metrics={
                "mean_v": upper_metrics.get("mean_luminance"),
                "mean_s": upper_metrics.get("mean_saturation"),
                "counts": upper_metrics.get("color_counts", {}),
            },
        )

        # =====================================================================
        # 2. LOWER BODY (LEGS / PANTS) CLOTHING COLOR
        # Anatomic boundaries: 56% to 88% height, 25% to 75% width
        # =====================================================================
        if is_bottom_clipped and ph < 40:
            # Lower body completely cut off by frame boundary
            lower_attr = ClothingColor(
                color_name="uncertain",
                confidence=0.0,
                observation_count=0,
                is_illumination_uncertain=False,
                region="lower",
                color_space_metrics={"reason": "lower_body_clipped_by_frame_boundary"},
            )
        else:
            ly1 = int(ph * 0.56)
            ly2 = int(ph * 0.88)
            lx1 = int(pw * 0.25)
            lx2 = int(pw * 0.75)

            lower_crop = crop[ly1:ly2, lx1:lx2]
            if lower_crop.size == 0:
                lower_crop = crop[int(ph * 0.5):, :]

            lower_metrics = RobustColorExtractor.extract_from_roi(lower_crop, filter_skin=False, scene_is_ir=is_ir)
            raw_low_color = lower_metrics["dominant_color"]
            raw_low_conf = lower_metrics["dominant_confidence"]
            low_unc = lower_metrics.get("is_illumination_uncertain", False)

            if track_id:
                st_low_col, st_low_conf, low_cnt, st_low_unc = self.lower_color_filter.update_track_color(
                    track_id=track_id,
                    color=raw_low_color,
                    confidence=raw_low_conf,
                    timestamp=timestamp,
                    is_illumination_uncertain=low_unc,
                )
            else:
                st_low_col, st_low_conf, low_cnt, st_low_unc = raw_low_color, raw_low_conf, 1, low_unc

            lower_attr = ClothingColor(
                color_name=st_low_col,
                confidence=round(st_low_conf, 4),
                observation_count=low_cnt,
                is_illumination_uncertain=st_low_unc,
                region="lower",
                color_space_metrics={
                    "mean_v": lower_metrics.get("mean_luminance"),
                    "mean_s": lower_metrics.get("mean_saturation"),
                    "counts": lower_metrics.get("color_counts", {}),
                },
            )

        # =====================================================================
        # 3. OUTERWEAR DERIVATION (Only when visual evidence is distinct)
        # =====================================================================
        outerwear_attr: Optional[ClothingColor] = None
        sec_up = upper_metrics.get("secondary_color")
        sec_up_conf = upper_metrics.get("secondary_confidence")
        if sec_up and sec_up_conf and sec_up_conf >= 0.22 and sec_up != st_up_col:
            # Upper body exhibits dual distinct color modes (e.g. jacket over shirt)
            outerwear_attr = ClothingColor(
                color_name=sec_up,
                confidence=round(sec_up_conf, 4),
                observation_count=1,
                is_illumination_uncertain=up_unc,
                region="outerwear",
            )

        # =====================================================================
        # 4. CARRIED OBJECT ASSOCIATION
        # =====================================================================
        carried_cat = None
        carried_col = None
        if associated_objects:
            p_diag = np.hypot(pw, ph)
            for obj in associated_objects:
                o_cls = obj.get("object_class") or obj.get("class_name")
                if o_cls in ("backpack", "handbag", "suitcase", "bottle", "box", "package", "umbrella"):
                    obb = obj.get("bounding_box", {})
                    ox1 = float(obb.get("x1", 0))
                    oy1 = float(obb.get("y1", 0))
                    ox2 = float(obb.get("x2", 0))
                    oy2 = float(obb.get("y2", 0))
                    ocx = (ox1 + ox2) / 2.0
                    ocy = (oy1 + oy2) / 2.0
                    # Check if center of object is inside or adjacent to person
                    if (px1 - pw * 0.2) <= ocx <= (px2 + pw * 0.2) and (py1 - ph * 0.1) <= ocy <= (py2 + ph * 0.1):
                        carried_cat = o_cls
                        # Extract carried object color if image crop available
                        ox1_c = max(0, min(int(ox1), w_frame - 1))
                        oy1_c = max(0, min(int(oy1), h_frame - 1))
                        ox2_c = max(ox1_c + 1, min(int(ox2), w_frame))
                        oy2_c = max(oy1_c + 1, min(int(oy2), h_frame))
                        o_crop = frame_bgr[oy1_c:oy2_c, ox1_c:ox2_c]
                        if o_crop.size > 0:
                            o_metrics = RobustColorExtractor.extract_from_roi(o_crop, filter_skin=True, scene_is_ir=is_ir)
                            carried_col = o_metrics.get("dominant_color")
                        break

        # =====================================================================
        # 5. HEADWEAR DETECTION (Top 0%–14% of person height)
        # =====================================================================
        headwear = None
        head_top = int(ph * 0.0)
        head_bottom = int(ph * 0.14)
        head_crop = crop[head_top:head_bottom, int(pw * 0.25):int(pw * 0.75)]
        if head_crop.size > 20 and not is_top_clipped:
            h_metrics = RobustColorExtractor.extract_from_roi(head_crop, filter_skin=True, scene_is_ir=is_ir)
            h_col = h_metrics.get("dominant_color")
            # If headwear crop shows prominent non-skin color with high confidence
            if h_col in ("black", "white", "red", "yellow", "blue") and h_metrics.get("dominant_confidence", 0.0) > 0.40:
                headwear = f"{h_col}_headwear"

        # =====================================================================
        # 6. FACE REGION TELEMETRY (Strictly non-biometric)
        # =====================================================================
        face_telem: Optional[FaceRegionTelemetry] = None
        try:
            if hasattr(self.face_detector, "analyze_face_telemetry"):
                face_telem = self.face_detector.analyze_face_telemetry(
                    frame_bgr=frame_bgr,
                    person_bbox=person_bbox,
                    timestamp=timestamp,
                    track_id=track_id,
                )
        except Exception as exc:
            logger.debug("Face telemetry extraction deferred: %s", exc)

        return PersonVisualAttributes(
            track_id=track_id or "ANON",
            timestamp=timestamp,
            bounding_box=person_bbox,
            upper_clothing_color=upper_attr,
            lower_clothing_color=lower_attr,
            outerwear_color=outerwear_attr,
            carried_object_category=carried_cat,
            carried_object_color=carried_col,
            headwear=headwear,
            face_telemetry=face_telem,
            occlusion_level=round(min(1.0, occlusion_score), 2),
            posture=posture,
        )
