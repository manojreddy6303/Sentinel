"""
Potential Theft & Object Takeaway Detector (Phase 10)

Detects physical object-removal and takeaway sequences:
1. Person approaches detectable object.
2. Dwells in interaction proximity.
3. Observational disappearance or co-movement takeaway.
4. Person departs original coordinates.
5. Spatially grounds both person and object bounding coordinates.
"""
import math
from typing import List, Dict, Any, Optional, Tuple

from ai.schemas import BoundingBox, TrackedObject
from ai.incidents.base_detector import BaseIncidentDetector
from ai.incidents.schemas import (
    IncidentCandidate,
    IncidentContext,
    IncidentCategory,
    SupportingSignal,
    EvidenceCandidate,
    SpatialContext,
)
from ai.incidents.scoring import IncidentScorer
from ai.incidents.negative_evidence import NegativeEvidenceEngine

try:
    from app.core.config import settings
except ImportError:
    from backend.app.core.config import settings


class TheftAndTakeawayDetector(BaseIncidentDetector):
    """
    Evaluates interactions between person tracks and portable items for takeaway patterns.
    """

    detector_name: str = "theft_and_takeaway_detector"
    detector_version: str = "2.1.0"
    category: IncidentCategory = IncidentCategory.PROPERTY

    # Phase Z Contract Declarations
    required_signals: List[str] = ["Interaction Proximity", "Takeaway Pattern"]
    supporting_signals_declared: List[str] = ["Interaction Proximity", "Takeaway Pattern", "Object Grounding"]
    contradictory_signals_declared: List[str] = [
        "Negative: Object Remains Present",
        "Negative: Frame Boundary Exit",
    ]
    context_requirements: Dict[str, Any] = {"portable_objects": True}
    evidence_requirements: Dict[str, Any] = {"person_track": True, "object_track": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        min_interaction_seconds: Optional[float] = None,
        interaction_max_distance: Optional[float] = None,
        min_departure_distance: Optional[float] = None,
        object_loss_window: Optional[float] = None,
        enabled: bool = True,
    ):

        super().__init__(enabled=enabled)
        self.min_interaction_seconds = (
            min_interaction_seconds
            if min_interaction_seconds is not None
            else getattr(settings, "THEFT_MIN_INTERACTION_SECONDS", 2.0)
        )
        self.interaction_max_distance = (
            interaction_max_distance
            if interaction_max_distance is not None
            else getattr(settings, "THEFT_INTERACTION_MAX_DISTANCE", 120.0)
        )
        self.min_departure_distance = (
            min_departure_distance
            if min_departure_distance is not None
            else getattr(settings, "THEFT_MIN_DEPARTURE_DISTANCE", 45.0)
        )
        self.object_loss_window = (
            object_loss_window
            if object_loss_window is not None
            else getattr(settings, "THEFT_OBJECT_LOSS_WINDOW", 4.0)
        )

    @staticmethod
    def _find_closest_bbox_observation(
        track: TrackedObject,
        target_timestamp: float,
        max_tolerance: float = 3.0,
    ) -> Tuple[Optional[BoundingBox], Optional[float], bool]:
        if not track or not track.history_bboxes:
            return None, None, False

        closest = min(track.history_bboxes, key=lambda b: abs(b["timestamp"] - target_timestamp))
        obs_ts = closest["timestamp"]
        b_dict = closest["bbox"]
        bbox = BoundingBox(
            x1=float(b_dict["x1"]),
            y1=float(b_dict["y1"]),
            x2=float(b_dict["x2"]),
            y2=float(b_dict["y2"]),
        )

        time_diff = obs_ts - target_timestamp
        if abs(time_diff) <= max_tolerance:
            return bbox, obs_ts, False
        if time_diff < 0:
            return bbox, obs_ts, True
        return bbox, obs_ts, False

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        person_tracks = [t for t in context.tracks if t.object_class == "person"]

        target_classes = getattr(
            settings,
            "THEFT_TARGET_CLASSES",
            {
                "backpack", "handbag", "suitcase", "laptop", "cell phone", "bottle",
                "umbrella", "bicycle", "box", "package", "book", "cup", "vase",
                "scissors", "teddy bear", "clock", "remote", "merchandise", "general_object", "unknown_portable_object",
            },
        )
        non_person_tracks = [t for t in context.tracks if t.object_class in target_classes]

        scale_factor = getattr(context, "resolution_scale_factor", 1.0)
        effective_interaction_dist = self.interaction_max_distance * scale_factor
        effective_departure_dist = self.min_departure_distance * scale_factor

        for p_track in person_tracks:
            if not p_track.trajectory or len(p_track.trajectory) < 2:
                continue

            for o_track in non_person_tracks:
                if not o_track.trajectory:
                    continue

                orig_o_cx, orig_o_cy = o_track.trajectory[0][1], o_track.trajectory[0][2]

                # Check proximity with resolution-scaled distance during object presence window
                proximity_moments = []
                for p_pt in p_track.trajectory:
                    p_t, p_cx, p_cy = p_pt[0], p_pt[1], p_pt[2]
                    # Contemporaneous presence check: person can only interact with an object while it exists
                    if not (o_track.first_seen - 1.0 <= p_t <= o_track.last_seen + 1.0):
                        continue
                    dist_to_orig = math.hypot(p_cx - orig_o_cx, p_cy - orig_o_cy)
                    if dist_to_orig <= effective_interaction_dist:
                        proximity_moments.append((p_t, p_cx, p_cy))

                if not proximity_moments:
                    continue

                interaction_start = min(m[0] for m in proximity_moments)
                interaction_end = max(m[0] for m in proximity_moments)
                interaction_duration = round(interaction_end - interaction_start, 2)

                # Check rapid takeaway / shoplifting condition (grab-and-go in single 1 FPS frame)
                is_rapid_loss = (
                    o_track.last_seen <= interaction_end + self.object_loss_window
                    and p_track.last_seen > o_track.last_seen
                )

                if interaction_duration < self.min_interaction_seconds and len(proximity_moments) < 2 and not is_rapid_loss:
                    continue

                # Departure condition with resolution-scaled departure distance
                subsequent_person_pts = [
                    pt for pt in p_track.trajectory
                    if pt[0] >= interaction_start + self.min_interaction_seconds or pt[0] >= interaction_end
                ]
                if not subsequent_person_pts:
                    continue

                max_departure_disp = max(
                    math.hypot(pt[1] - orig_o_cx, pt[2] - orig_o_cy)
                    for pt in subsequent_person_pts
                )

                if max_departure_disp < effective_departure_dist:
                    continue

                # Require minimally stable object track to avoid flagging single-frame detector dropouts as theft,
                # unless rapid takeaway or verified interaction dwell occurred during person proximity
                if len(o_track.trajectory) < 2 and o_track.duration_seconds < 0.5:
                    if not is_rapid_loss and (len(proximity_moments) < 2 or interaction_duration < 1.0):
                        continue

                # Disappearance or co-movement
                is_disappeared = (
                    o_track.last_seen <= interaction_end + self.object_loss_window
                    and p_track.last_seen > o_track.last_seen
                )

                is_co_moving = False
                if not is_disappeared:
                    o_max_disp = max(
                        math.hypot(pt[1] - orig_o_cx, pt[2] - orig_o_cy)
                        for pt in o_track.trajectory
                    )
                    if o_max_disp >= effective_departure_dist:
                        is_co_moving = True

                if not (is_disappeared or is_co_moving):
                    continue

                pattern_type = "object_disappearance" if is_disappeared else "co_movement_takeaway"
                duration_total = round(p_track.last_seen - interaction_start, 2)
                takeaway_timestamp = round(o_track.last_seen if is_disappeared else interaction_end, 2)
                event_start = max(0.0, round(interaction_start, 2))
                event_end = round(min(p_track.last_seen, max(event_start + 2.0, takeaway_timestamp + 3.0)), 2)

                # Ground coordinates around the takeaway moment
                p_bbox, p_obs_ts, _ = self._find_closest_bbox_observation(p_track, takeaway_timestamp, max_tolerance=3.0)
                if not p_bbox:
                    p_bbox, p_obs_ts, _ = self._find_closest_bbox_observation(p_track, interaction_start, max_tolerance=3.0)

                o_max_tol = max(3.0, interaction_duration + self.object_loss_window + 5.0)
                o_bbox, o_obs_ts, is_prior_o = self._find_closest_bbox_observation(o_track, takeaway_timestamp, max_tolerance=o_max_tol)
                if not o_bbox:
                    o_bbox, o_obs_ts, is_prior_o = self._find_closest_bbox_observation(o_track, interaction_start, max_tolerance=o_max_tol)

                active_o_bbox = o_bbox if (o_bbox is not None and o_obs_ts is not None) else None

                signals = [
                    SupportingSignal(
                        signal_type="Interaction Proximity",
                        description=f"Person [{p_track.track_id}] approached {o_track.object_class} [{o_track.track_id}] (dwell: {interaction_duration:.1f}s)",
                        confidence=0.90,
                        timestamp=interaction_start,
                        track_id=p_track.track_id,
                    ),
                    SupportingSignal(
                        signal_type="Takeaway Pattern",
                        description=f"Pattern: {pattern_type} (person departure displacement: {max_departure_disp:.1f}px)",
                        confidence=0.88,
                        timestamp=takeaway_timestamp,
                        track_id=p_track.track_id,
                    ),
                ]

                if active_o_bbox:
                    signals.append(
                        SupportingSignal(
                            signal_type="Object Grounding",
                            description=f"Target {o_track.object_class} [{o_track.track_id}] grounded at {o_obs_ts:.1f}s",
                            confidence=o_track.confidence,
                            timestamp=o_obs_ts,
                            track_id=o_track.track_id,
                        )
                    )

                # Negative Evidence evaluation
                neg_signals = NegativeEvidenceEngine.evaluate_theft_negative_evidence(
                    p_track, o_track, context, interaction_end
                )
                critical_negatives = [
                    "Negative: Object Remains Present",
                    "Negative: Frame Boundary Exit",
                    "Negative: Object Ceased Prior to Proximity",
                ]
                if len(o_track.trajectory) < 2 and interaction_duration < 1.0:
                    critical_negatives.append("Negative: Transient Object Detection")

                if any(s.signal_type in critical_negatives for s in neg_signals):
                    # Object never moved and remained present, or exited frame boundary, or was transient flicker, or ceased before arrival -> non-theft
                    continue

                scoring = IncidentScorer.calculate_evidence_score(
                    base_confidence=0.75,
                    supporting_signals=signals,
                    tracks=[p_track, o_track],
                    duration_seconds=interaction_duration,
                    expected_duration_threshold=self.min_interaction_seconds,
                    contradictory_signals=neg_signals,
                    incident_type="POTENTIAL_THEFT",
                    validation_decision="REVIEW_REQUIRED",
                )

                spatial_ctx = SpatialContext(
                    centroid=p_bbox.centroid if p_bbox else None,
                    bounding_box=p_bbox,
                    inter_track_distances={o_track.track_id: round(max_departure_disp, 2)},
                    metadata={
                        "person_track_id": p_track.track_id,
                        "object_track_id": o_track.track_id,
                        "object_class": o_track.object_class,
                        "object_bounding_box": active_o_bbox.to_dict() if active_o_bbox else None,
                        "object_observation_timestamp": o_obs_ts if active_o_bbox else None,
                        "is_prior_object_observation": is_prior_o if active_o_bbox else False,
                    }
                )

                removal_desc = "disappeared from scene" if is_disappeared else "co-moved with departing person"
                cand = self.build_candidate(
                    video_id=context.video_id,
                    event_type="POTENTIAL_THEFT",
                    start_time=event_start,
                    end_time=event_end,
                    severity="HIGH",
                    confidence=scoring["score"],
                    explanation=(
                        f"Potential Theft Pattern ({scoring['verification_note']}): "
                        f"Person [{p_track.track_id}] approached {o_track.object_class} [{o_track.track_id}], "
                        f"remained in proximity for {interaction_duration:.1f}s, after which the object {removal_desc} "
                        f"while person departed ({max_departure_disp:.1f}px displacement)."
                    ),
                    track_ids=[p_track.track_id, o_track.track_id],
                    object_classes=["person", o_track.object_class],
                    supporting_signals=signals,
                    contradictory_signals=neg_signals,
                    spatial_context=spatial_ctx,
                    evidence_candidates=[
                        EvidenceCandidate(
                            timestamp=takeaway_timestamp,
                            bounding_box=p_bbox,
                            target_track_id=p_track.track_id,
                            reason=f"Suspected object takeaway ({pattern_type}) at {takeaway_timestamp:.1f}s",
                        )
                    ],
                    pattern_evidence_strength=scoring.get("pattern_strength", scoring["score"]),
                    assessment_score=scoring["score"],
                    prefix="THEFT",
                )
                cand.validation_decision = "REVIEW_REQUIRED"
                cand.human_verification_required = True
                candidates.append(cand)

        return candidates
