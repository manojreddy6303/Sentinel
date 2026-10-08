"""
Universal Robust Color Analysis Module for Sentinel Surveillance Intelligence (Phase 20.2)

Provides illumination-resilient, region-aware visual color extraction and temporal
aggregation across anonymous track histories.

SAFETY & PRIVACY CONSTRAINTS:
1. Non-biometric: Color analysis applies to vehicle bodies, apparel, and portable objects only.
2. Strict honesty: Low-illumination, underexposed, or monochromatic IR scenes yield "uncertain",
   never inventing colors or guessing dark/black/grey due to poor lighting.
3. Temporal stability: Single-frame glitches, specular reflections, or shadows cannot
   overwrite an established, multi-frame track color.
"""
from typing import Dict, Any, Optional, Tuple, List
import logging
import cv2
import numpy as np

from ai.schemas import BoundingBox, VehicleAttribute, ClothingColor
from ai.common.numeric import clamp_finite, ensure_finite

logger = logging.getLogger(__name__)


class RobustColorExtractor:
    """
    Stateless, illumination-aware color extractor operating on calibrated ROIs.
    """

    COLOR_VOCABULARY = [
        "black",
        "white",
        "grey",
        "silver",
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "brown",
        "other",
        "unknown",
        "uncertain",
    ]

    CHROMATIC_FAMILIES = ["blue", "red", "orange", "yellow", "green", "brown"]

    MIN_LUMINANCE_THRESHOLD = 30.0
    MIN_SATURATION_THRESHOLD = 15.0

    @classmethod
    def extract_dominant_color(
        cls,
        roi_bgr: np.ndarray,
        filter_skin: bool = False,
        scene_is_ir: bool = False,
    ) -> Tuple[str, float, bool]:
        """Convenience method returning (dominant_color, confidence, is_illumination_uncertain)."""
        res = cls.extract_from_roi(roi_bgr, filter_skin=filter_skin, scene_is_ir=scene_is_ir)
        return res["dominant_color"], res["dominant_confidence"], res["is_illumination_uncertain"]

    @classmethod
    def extract_from_roi(
        cls,
        roi_bgr: np.ndarray,
        filter_skin: bool = False,
        scene_is_ir: bool = False,
        scene_is_underexposed: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Extract dominant color from a pre-cropped Region of Interest (ROI).

        Returns:
            Dict containing dominant_color, confidence, secondary_color, etc.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return {
                "dominant_color": "unknown",
                "dominant_confidence": 0.0,
                "secondary_color": None,
                "secondary_confidence": 0.0,
                "is_illumination_uncertain": False,
                "mean_luminance": 0.0,
                "mean_saturation": 0.0,
                "color_counts": {},
                "total_pixels": 0,
            }

        rh, rw = roi_bgr.shape[:2]
        if rh < 6 or rw < 6:
            return {
                "dominant_color": "unknown",
                "dominant_confidence": 0.0,
                "secondary_color": None,
                "secondary_confidence": 0.0,
                "is_illumination_uncertain": False,
                "mean_luminance": 0.0,
                "mean_saturation": 0.0,
                "color_counts": {},
                "total_pixels": rh * rw,
            }

        # Convert to HSV color space
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        mean_lum = float(np.mean(v))
        mean_sat = float(np.mean(s))

        # Check for illumination deficiency or monochromatic IR vision
        is_white_achromatic = (mean_lum >= 180.0 and mean_sat < 35.0)

        # Monochromatic IR or low illumination check (Step 4):
        # If scene is declared IR, reject color extraction.
        # If crop is dark (mean_lum < 30), mark uncertain unless daytime scene context confirms a black object.
        # If crop is desaturated (mean_sat < 15) and not white, mark uncertain.
        is_deficient = False
        reason = ""
        if scene_is_ir:
            is_deficient = True
            reason = "Scene classified as monochromatic IR"
        elif mean_lum < cls.MIN_LUMINANCE_THRESHOLD:
            if scene_is_underexposed is not False:
                is_deficient = True
                reason = "Crop mean luminance < 30 (underexposed)"
        elif mean_sat < cls.MIN_SATURATION_THRESHOLD and not is_white_achromatic:
            if scene_is_ir or scene_is_underexposed is not False:
                is_deficient = True
                reason = "Crop mean saturation < 15 (monochromatic)"

        if is_deficient:
            return {
                "dominant_color": "uncertain",
                "dominant_confidence": 0.30,
                "secondary_color": None,
                "secondary_confidence": 0.0,
                "is_illumination_uncertain": True,
                "mean_luminance": round(mean_lum, 2),
                "mean_saturation": round(mean_sat, 2),
                "color_counts": {},
                "total_pixels": rh * rw,
                "reason": reason,
            }

        # Optional skin-tone mask exclusion for neck/chest margins - restrict to top slice of ROI
        valid_mask = np.ones((rh, rw), dtype=bool)
        if filter_skin:
            neck_y = max(1, int(rh * 0.18))
            skin_h = ((h[:neck_y, :] <= 25) | (h[:neck_y, :] >= 168))
            skin_s = (s[:neck_y, :] >= 35) & (s[:neck_y, :] <= 150)
            skin_v = (v[:neck_y, :] >= 55)
            skin_mask_neck = skin_h & skin_s & skin_v
            valid_mask[:neck_y, :][skin_mask_neck] = False

        total_valid = int(np.count_nonzero(valid_mask))
        if total_valid < 16:
            return {
                "dominant_color": "unknown",
                "dominant_confidence": 0.0,
                "secondary_color": None,
                "secondary_confidence": 0.0,
                "is_illumination_uncertain": False,
                "mean_luminance": round(mean_lum, 2),
                "mean_saturation": round(mean_sat, 2),
                "color_counts": {},
                "total_pixels": total_valid,
            }

        counts: Dict[str, int] = {
            c: 0 for c in cls.COLOR_VOCABULARY if c not in ["unknown", "other", "uncertain"]
        }

        # 1. Achromatic: Black
        # In surveillance/CCTV footage, dark fabrics under cool fluorescent or low light
        # exhibit slight sensor/illumination cast with weak blue margin (< 12) or weak B-R (< 16).
        b_val, g_val, r_val = cv2.split(roi_bgr)
        b_r_diff = b_val.astype(int) - r_val.astype(int)
        blue_margin = b_val.astype(int) - np.maximum(r_val.astype(int), g_val.astype(int))

        cool_cast_achromatic = (v < 75) & (h >= 75) & (h <= 140) & ((blue_margin < 12) | (b_r_diff < 16) | (s < 45))

        black_mask = valid_mask & (((v < 38)) | ((v < 72) & (s < 38)) | (cool_cast_achromatic & (v < 72)))
        counts["black"] = int(np.count_nonzero(black_mask))

        # 2. Achromatic: White (S < 30 and V > 185)
        white_mask = valid_mask & (s < 30) & (v > 185)
        counts["white"] = int(np.count_nonzero(white_mask))

        # 3. Achromatic: Grey / Silver (S < 40 and 52 <= V <= 185, not white/black)
        grey_base = valid_mask & ~black_mask & ~white_mask & (((s < 40) & (v >= 52) & (v <= 185)) | (cool_cast_achromatic & (v >= 75)))
        silver_mask = grey_base & (s < 25) & (v >= 140)
        counts["silver"] = int(np.count_nonzero(silver_mask))
        counts["grey"] = int(np.count_nonzero(grey_base & ~silver_mask))

        # Chromatic mask: strictly pixels that are NOT achromatic, with sufficient saturation & value
        achromatic_combined = black_mask | white_mask | grey_base
        chromatic = valid_mask & ~achromatic_combined & (s >= 35) & (v >= 40)

        # Red: H in [0, 10] or [165, 180]
        red_mask = chromatic & ((h <= 10) | (h >= 165))
        counts["red"] = int(np.count_nonzero(red_mask))

        # Orange: H in (10, 24] and V >= 55 and S >= 55 and (R - B >= 30) (genuine saturated chromatic orange)
        orange_red_diff = r_val.astype(int) - b_val.astype(int)
        orange_mask = chromatic & (h > 10) & (h <= 24) & (v >= 55) & (s >= 55) & (orange_red_diff >= 30)
        counts["orange"] = int(np.count_nonzero(orange_mask))

        # Brown: H in (10, 24] and 35 <= V < 120 and S >= 35
        brown_mask = chromatic & (h > 10) & (h <= 24) & (v >= 35) & (v < 120) & ~orange_mask
        counts["brown"] = int(np.count_nonzero(brown_mask))

        # Yellow: H in (24, 38]
        yellow_mask = chromatic & (h > 24) & (h <= 38)
        counts["yellow"] = int(np.count_nonzero(yellow_mask))

        # Green: H in (38, 85]
        green_mask = chromatic & (h > 38) & (h <= 85)
        counts["green"] = int(np.count_nonzero(green_mask))

        # Blue: H in (85, 135]
        # True blue requires genuine chromatic presence:
        blue_mask = chromatic & (h > 85) & (h <= 135) & (blue_margin >= 10) & (s >= 40)
        counts["blue"] = int(np.count_nonzero(blue_mask))

        # Achromatic-First Reliability Model:
        achromatic_cnt = counts["black"] + counts["white"] + counts["grey"] + counts["silver"]
        chromatic_cnt = sum(counts[c] for c in cls.CHROMATIC_FAMILIES)
        achromatic_ratio = achromatic_cnt / float(total_valid)
        median_sat = float(np.median(s[valid_mask]))

        sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        top_color, top_cnt = sorted_counts[0]
        second_color, second_cnt = sorted_counts[1] if len(sorted_counts) > 1 else (None, 0)

        chrom_counts = [(c, counts[c]) for c in cls.CHROMATIC_FAMILIES if counts[c] > 0]
        best_chrom_color, best_chrom_cnt = max(chrom_counts, key=lambda kv: kv[1]) if chrom_counts else (None, 0)
        best_chrom_ratio = best_chrom_cnt / float(total_valid)

        achrom_counts = [(c, counts[c]) for c in ["black", "white", "grey", "silver"] if counts[c] > 0]
        top_achrom_color, top_achrom_cnt = max(achrom_counts, key=lambda kv: kv[1]) if achrom_counts else ("grey", 0)

        # Achromatic dominance rule:
        # A dominant achromatic garment remains achromatic unless chromatic presence is genuinely established.
        if top_color in ["black", "grey", "silver", "white"]:
            if best_chrom_cnt > 0 and best_chrom_ratio >= 0.22 and best_chrom_cnt >= 0.65 * top_achrom_cnt and median_sat >= 35.0:
                second_color, second_cnt = top_color, top_cnt
                top_color, top_cnt = best_chrom_color, best_chrom_cnt
        else:
            # If top color is chromatic, verify that chromatic is not completely eclipsed by achromatic
            if best_chrom_cnt <= 0.60 * achromatic_cnt and best_chrom_ratio < 0.20:
                second_color, second_cnt = top_color, top_cnt
                top_color, top_cnt = top_achrom_color, top_achrom_cnt

        dominant_ratio = top_cnt / float(total_valid)
        secondary_ratio = second_cnt / float(total_valid) if second_cnt > 0 else 0.0

        if dominant_ratio < 0.16 or top_cnt == 0:
            final_dominant = "unknown"
            final_conf = 0.20
        else:
            final_dominant = top_color
            if final_dominant in cls.CHROMATIC_FAMILIES:
                sat_scale = min(1.0, max(0.40, median_sat / 60.0))
                margin = max(0.0, (top_cnt - top_achrom_cnt) / float(total_valid))
                final_conf = min(0.95, max(0.35, dominant_ratio * sat_scale * (0.8 + 0.2 * margin)))
            else:
                final_conf = min(0.92, max(0.40, dominant_ratio))

        final_secondary = second_color if secondary_ratio >= 0.15 else None
        final_sec_conf = min(0.95, secondary_ratio) if final_secondary else None

        return {
            "dominant_color": final_dominant,
            "dominant_confidence": round(final_conf, 4),
            "secondary_color": final_secondary,
            "secondary_confidence": round(final_sec_conf, 4) if final_sec_conf is not None else None,
            "is_illumination_uncertain": False,
            "mean_luminance": round(mean_lum, 2),
            "mean_saturation": round(mean_sat, 2),
            "color_counts": {k: v for k, v in counts.items() if v > 0},
            "total_pixels": total_valid,
        }



class TemporalColorFilter:
    """
    Maintains and smooths color attributes across multi-frame anonymous tracks.
    Eliminates frame-to-frame color flipping and prevents isolated shadows/glare
    from corrupting an established track color.
    """

    def __init__(self, history_window: int = 12):
        self.history_window = max(2, int(history_window))
        # track_id -> list of (timestamp, color, confidence, is_uncertain)
        self._track_history: Dict[str, List[Tuple[float, str, float, bool]]] = {}

    def reset(self, video_id: str = "") -> None:
        """Clear color histories across video transitions."""
        self._track_history.clear()

    def update_track_color(
        self,
        track_id: str,
        color: str,
        confidence: float,
        timestamp: float,
        is_illumination_uncertain: bool = False,
    ) -> Tuple[str, float, int, bool]:
        """
        Record observation and return consensus smoothed color attribute.

        Returns:
            Tuple of (stable_color, smoothed_confidence, observation_count, is_uncertain)
        """
        if not track_id:
            return color, confidence, 1, is_illumination_uncertain

        if track_id not in self._track_history:
            self._track_history[track_id] = []

        history = self._track_history[track_id]

        # Record this observation
        history.append((timestamp, color, confidence, is_illumination_uncertain))
        if len(history) > self.history_window:
            history.pop(0)

        # Check existing confirmed history
        valid_obs = [obs for obs in history if not obs[3] and obs[1] not in ("unknown", "uncertain")]

        if not valid_obs:
            # All observations uncertain or unknown
            return "uncertain" if is_illumination_uncertain else "unknown", confidence, len(history), is_illumination_uncertain

        # If incoming observation is uncertain or unknown, preserve confirmed history
        if (is_illumination_uncertain or color in ("unknown", "uncertain")) and valid_obs:
            # Do NOT allow single bad frame to overwrite established track color
            best_past_color, best_past_conf = max(
                [(obs[1], obs[2]) for obs in valid_obs], key=lambda x: x[1]
            )
            return best_past_color, round(best_past_conf * 0.90, 4), len(history), False

        # Accumulate confidence-weighted votes across valid history
        color_weights: Dict[str, float] = {}
        color_counts: Dict[str, int] = {}
        for _, c, conf, _ in valid_obs:
            color_weights[c] = color_weights.get(c, 0.0) + conf
            color_counts[c] = color_counts.get(c, 0) + 1

        total_weight = sum(color_weights.values())
        top_color, top_weight = max(color_weights.items(), key=lambda kv: kv[1])
        top_count = color_counts[top_color]
        agreement_ratio = top_weight / total_weight if total_weight > 0 else 0.0

        # Consensus conditions
        if len(valid_obs) == 1:
            # Single observation: tentative
            return top_color, round(confidence * 0.85, 4), 1, False

        if agreement_ratio >= 0.55 and top_count >= 2:
            # Confirmed stable attribute
            smoothed_conf = min(0.98, max(0.50, agreement_ratio * 0.95))
            return top_color, round(smoothed_conf, 4), len(valid_obs), False
        elif agreement_ratio >= 0.45:
            # Mild consensus
            return top_color, round(confidence * 0.70, 4), len(valid_obs), False
        else:
            # High color variance across frames -> uncertain
            return "uncertain", 0.40, len(valid_obs), False

    def update(
        self,
        track_id: str,
        color: str,
        confidence: float,
        timestamp: float,
        is_illumination_uncertain: bool = False,
    ) -> Tuple[str, float, bool]:
        """Convenience method returning (stable_color, confidence, is_confirmed)."""
        col, conf, count, is_unc = self.update_track_color(
            track_id=track_id,
            color=color,
            confidence=confidence,
            timestamp=timestamp,
            is_illumination_uncertain=is_illumination_uncertain,
        )
        is_confirmed = (count >= 3 and not is_unc and conf >= 0.60)
        return col, conf, is_confirmed

    def get_stable_color(self, track_id: str) -> Tuple[str, float, bool]:
        """Query current consensus color for track."""
        history = self._track_history.get(track_id, [])
        if not history:
            return "unknown", 0.0, False
        last_obs = history[-1]
        col, conf, count, is_unc = self.update_track_color(
            track_id=track_id,
            color=last_obs[1],
            confidence=last_obs[2],
            timestamp=last_obs[0],
            is_illumination_uncertain=last_obs[3],
        )
        is_confirmed = (count >= 3 and not is_unc and conf >= 0.60)
        return col, conf, is_confirmed


class VehicleColorAnalyzer:
    """
    Extracts and classifies dominant vehicle body colors from image crops with
    region-aware core masking and temporal track voting.
    """

    COLOR_VOCABULARY = RobustColorExtractor.COLOR_VOCABULARY

    def __init__(self, min_confidence_threshold: float = 0.40):
        self.min_confidence_threshold = min_confidence_threshold
        self.temporal_filter = TemporalColorFilter()

    def reset(self, video_id: str = "") -> None:
        """Reset temporal state between videos."""
        self.temporal_filter.reset(video_id)

    def analyze(
        self,
        frame_bgr: np.ndarray,
        bbox: BoundingBox,
        timestamp: float,
        object_class: str = "car",
        track_id: Optional[str] = None,
        scene_report: Optional[Any] = None,
    ) -> VehicleAttribute:
        """
        Analyze vehicle crop from a BGR image frame and classify dominant and secondary color.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return VehicleAttribute(
                color="unknown",
                confidence=0.0,
                timestamp=timestamp,
                bounding_box=bbox,
                object_class=object_class,
                track_id=track_id,
            )

        h_frame, w_frame = frame_bgr.shape[:2]
        x1 = max(0, min(int(bbox.x1), w_frame - 1))
        y1 = max(0, min(int(bbox.y1), h_frame - 1))
        x2 = max(x1 + 1, min(int(bbox.x2), w_frame))
        y2 = max(y1 + 1, min(int(bbox.y2), h_frame))

        crop = frame_bgr[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]

        if ch < 12 or cw < 12:
            return VehicleAttribute(
                color="unknown",
                confidence=0.0,
                timestamp=timestamp,
                bounding_box=bbox,
                object_class=object_class,
                track_id=track_id,
                color_space_metrics={"error": "Crop too small for color analysis"},
            )

        # Vehicle body core: 22%–80% height, 15%–85% width
        # Eliminates windshield/roof reflection and bottom wheels/tires/asphalt
        cy1 = int(ch * 0.22)
        cy2 = int(ch * 0.80)
        cx1 = int(cw * 0.15)
        cx2 = int(cw * 0.85)

        core_crop = crop[cy1:cy2, cx1:cx2]
        if core_crop.size == 0:
            core_crop = crop

        is_ir = False
        if scene_report is not None:
            ill_type = getattr(scene_report, "illumination_type", None)
            if ill_type and hasattr(ill_type, "value"):
                ill_val = ill_type.value
            else:
                ill_val = str(ill_type or "")
            if "night_ir" in ill_val:
                is_ir = True

        frame_mean_lum = float(np.mean(frame_bgr))
        scene_underexposed = (frame_mean_lum < 30.0)

        raw_metrics = RobustColorExtractor.extract_from_roi(
            core_crop,
            filter_skin=False,
            scene_is_ir=is_ir,
            scene_is_underexposed=scene_underexposed,
        )

        raw_color = raw_metrics["dominant_color"]
        raw_conf = raw_metrics["dominant_confidence"]
        is_illum_unc = raw_metrics.get("is_illumination_uncertain", False)

        # Temporal aggregation over anonymous track ID
        if track_id:
            stable_color, stable_conf, obs_cnt, is_unc = self.temporal_filter.update_track_color(
                track_id=track_id,
                color=raw_color,
                confidence=raw_conf,
                timestamp=timestamp,
                is_illumination_uncertain=is_illum_unc,
            )
            final_color = stable_color
            final_conf = stable_conf
            final_is_unc = is_unc
        else:
            final_color = raw_color
            final_conf = raw_conf
            final_is_unc = is_illum_unc

        # Approximate orientation from bounding box aspect ratio
        aspect = float(cw) / float(max(1, ch))
        if aspect > 1.6:
            orient = "side_profile"
        elif aspect < 0.9:
            orient = "front_or_rear"
        else:
            orient = "angled"

        return VehicleAttribute(
            color=final_color,
            confidence=round(final_conf, 4),
            secondary_color=raw_metrics.get("secondary_color"),
            secondary_confidence=raw_metrics.get("secondary_confidence"),
            approximate_orientation=orient,
            is_occluded=False,
            is_illumination_uncertain=final_is_unc,
            timestamp=timestamp,
            bounding_box=bbox,
            object_class=object_class,
            track_id=track_id,
            color_space_metrics={
                "dominant_ratio": raw_metrics.get("dominant_confidence"),
                "counts": raw_metrics.get("color_counts", {}),
                "mean_v": raw_metrics.get("mean_luminance"),
                "mean_s": raw_metrics.get("mean_saturation"),
                "temporal_samples": obs_cnt if track_id else 1,
            },
        )
