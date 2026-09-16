"""
Vehicle Color Analysis Module for Sentinel Surveillance Intelligence

Performs visual color analysis on cropped vehicle bounding boxes using HSV color-space
segmentation and dominant color classification over a controlled security vocabulary.

CONTROLLED VOCABULARY:
['black', 'white', 'grey', 'silver', 'red', 'blue', 'green', 'yellow', 'orange', 'brown', 'other', 'unknown']

SAFETY CONSTRAINT:
Never guess vehicle color. If pixel confidence is below threshold, classifies as 'unknown'.
Gemini must never invent colors. Colors are derived solely from actual pixel data.
"""
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np

from ai.schemas import BoundingBox, VehicleAttribute


class VehicleColorAnalyzer:
    """
    Extracts and classifies dominant vehicle body colors from image crops.
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
    ]

    def __init__(self, min_confidence_threshold: float = 0.40):
        self.min_confidence_threshold = min_confidence_threshold

    def analyze(
        self,
        frame_bgr: np.ndarray,
        bbox: BoundingBox,
        timestamp: float,
        object_class: str = "car",
        track_id: Optional[str] = None,
    ) -> VehicleAttribute:
        """
        Analyze vehicle crop from a BGR image frame and classify dominant color.
        """
        h_frame, w_frame = frame_bgr.shape[:2]

        # Clamp bounding box to frame boundaries
        x1 = max(0, min(int(bbox.x1), w_frame - 1))
        y1 = max(0, min(int(bbox.y1), h_frame - 1))
        x2 = max(x1 + 1, min(int(bbox.x2), w_frame))
        y2 = max(y1 + 1, min(int(bbox.y2), h_frame))

        crop = frame_bgr[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]

        # If crop is too tiny to analyze meaningfully (< 12x12 px)
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

        # Focus on vehicle body core (avoiding top roof/sky and bottom tires/asphalt)
        cy1 = int(ch * 0.20)
        cy2 = int(ch * 0.85)
        cx1 = int(cw * 0.15)
        cx2 = int(cw * 0.85)
        body_crop = crop[cy1:cy2, cx1:cx2]

        if body_crop.size == 0:
            body_crop = crop

        # Convert to HSV color space
        hsv = cv2.cvtColor(body_crop, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        total_pixels = body_crop.shape[0] * body_crop.shape[1]
        if total_pixels == 0:
            return VehicleAttribute(
                color="unknown",
                confidence=0.0,
                timestamp=timestamp,
                bounding_box=bbox,
                object_class=object_class,
                track_id=track_id,
            )

        # Count pixels matching each color family
        counts: Dict[str, int] = {c: 0 for c in self.COLOR_VOCABULARY if c not in ["unknown", "other"]}

        # 1. Achromatic: Black (V < 50)
        black_mask = (v < 50)
        counts["black"] = int(np.count_nonzero(black_mask))

        # 2. Achromatic: White (S < 35 and V > 185)
        white_mask = (s < 35) & (v > 185)
        counts["white"] = int(np.count_nonzero(white_mask))

        # 3. Achromatic: Grey / Silver (S < 45 and 50 <= V <= 185)
        grey_mask = (s < 45) & (v >= 50) & (v <= 185)
        silver_mask = (s < 30) & (v >= 140) & (v <= 185)
        counts["silver"] = int(np.count_nonzero(silver_mask))
        counts["grey"] = int(np.count_nonzero(grey_mask & ~silver_mask))

        # Chromatic pixels (S >= 40 and V >= 50)
        chromatic = (s >= 40) & (v >= 50)

        # Red: H in [0, 10] or [165, 180]
        red_mask = chromatic & ((h <= 10) | (h >= 165))
        counts["red"] = int(np.count_nonzero(red_mask))

        # Orange: H in (10, 25]
        orange_mask = chromatic & (h > 10) & (h <= 25)
        counts["orange"] = int(np.count_nonzero(orange_mask))

        # Yellow: H in (25, 38]
        yellow_mask = chromatic & (h > 25) & (h <= 38)
        counts["yellow"] = int(np.count_nonzero(yellow_mask))

        # Green: H in (38, 85]
        green_mask = chromatic & (h > 38) & (h <= 85)
        counts["green"] = int(np.count_nonzero(green_mask))

        # Blue: H in (85, 135]
        blue_mask = chromatic & (h > 85) & (h <= 135)
        counts["blue"] = int(np.count_nonzero(blue_mask))

        # Brown: H in (10, 25], lower V (40 <= V < 110), moderate S
        brown_mask = (h > 10) & (h <= 25) & (v >= 40) & (v < 110) & (s >= 40)
        counts["brown"] = int(np.count_nonzero(brown_mask))

        # Determine dominant color
        best_color, best_count = max(counts.items(), key=lambda item: item[1])
        confidence = float(best_count) / float(total_pixels)

        # If highest confidence is below threshold, label as unknown
        if confidence < self.min_confidence_threshold or best_count == 0:
            final_color = "unknown"
            final_conf = max(0.1, round(confidence, 3))
        else:
            final_color = best_color
            final_conf = min(0.99, round(confidence, 3))

        return VehicleAttribute(
            color=final_color,
            confidence=final_conf,
            timestamp=timestamp,
            bounding_box=bbox,
            object_class=object_class,
            track_id=track_id,
            color_space_metrics={
                "dominant_ratio": round(confidence, 3),
                "counts": {k: v for k, v in counts.items() if v > 0},
                "mean_v": round(float(np.mean(v)), 1),
                "mean_s": round(float(np.mean(s)), 1),
            },
        )
