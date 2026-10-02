"""
Behavior & Security Anomaly Intelligence Module for Sentinel

Detects prolonged presence (loitering), abandoned object candidates, and aggregates
multi-signal observational security anomalies with explainable causal factors.

SAFETY CONSTRAINTS:
1. Observational language only ('Prolonged presence detected', 'Potential abandoned object candidate').
2. Zero criminal attribution or malicious intent claims.
3. Every anomaly explains which verified physical signals triggered the observation.
"""
import uuid
import math
from typing import List, Dict, Any, Optional, Tuple

from ai.schemas import BoundingBox, TrackedObject, SecurityEvent


try:
    from app.core.config import settings
except ImportError:
    from backend.app.core.config import settings


class BehaviorAnalyzer:
    """
    Evaluates temporal-spatial tracking patterns to detect security anomalies.
    """

    def __init__(
        self,
        loitering_threshold_seconds: float = 4.0,      # Configurable threshold for prolonged presence
        max_loitering_displacement: float = 120.0,      # Maximum centroid shift (pixels/normalized) to be considered in same area
        abandoned_stationary_seconds: float = 4.0,     # Time object must remain stationary
        abandoned_separation_distance: float = 120.0,  # Distance from departing person
        theft_min_interaction_seconds: Optional[float] = None,
        theft_interaction_max_distance: Optional[float] = None,
        theft_min_departure_distance: Optional[float] = None,
        theft_object_loss_window: Optional[float] = None,
    ):
        self.loitering_threshold_seconds = loitering_threshold_seconds
        self.max_loitering_displacement = max_loitering_displacement
        self.abandoned_stationary_seconds = abandoned_stationary_seconds
        self.abandoned_separation_distance = abandoned_separation_distance
        self.theft_min_interaction_seconds = (
            theft_min_interaction_seconds
            if theft_min_interaction_seconds is not None
            else getattr(settings, "THEFT_MIN_INTERACTION_SECONDS", 2.0)
        )
        self.theft_interaction_max_distance = (
            theft_interaction_max_distance
            if theft_interaction_max_distance is not None
            else getattr(settings, "THEFT_INTERACTION_MAX_DISTANCE", 120.0)
        )
        self.theft_min_departure_distance = (
            theft_min_departure_distance
            if theft_min_departure_distance is not None
            else getattr(settings, "THEFT_MIN_DEPARTURE_DISTANCE", 45.0)
        )
        self.theft_object_loss_window = (
            theft_object_loss_window
            if theft_object_loss_window is not None
            else getattr(settings, "THEFT_OBJECT_LOSS_WINDOW", 4.0)
        )

    @staticmethod
    def _find_closest_bbox_observation(
        track: TrackedObject,
        target_timestamp: float,
        max_tolerance: float = 3.0,
    ) -> Tuple[Optional[BoundingBox], Optional[float], bool]:
        """
        Locate the track's bounding box observation closest to target_timestamp.
        Returns:
            (bounding_box, observation_timestamp, is_prior_observation)
        """
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

    def detect_prolonged_presence(
        self,
        tracks: List[TrackedObject],
        threshold_seconds: Optional[float] = None,
    ) -> List[SecurityEvent]:
        """
        Identify tracked objects that remain within a localized region for longer than threshold.
        """
        thresh = threshold_seconds if threshold_seconds is not None else self.loitering_threshold_seconds
        events: List[SecurityEvent] = []

        for track in tracks:
            if track.duration_seconds < thresh or len(track.trajectory) < 2:
                continue

            # Calculate total spatial displacement across trajectory
            start_cx, start_cy = track.trajectory[0][1], track.trajectory[0][2]
            end_cx, end_cy = track.trajectory[-1][1], track.trajectory[-1][2]
            net_displacement = math.hypot(end_cx - start_cx, end_cy - start_cy)

            # Check if object remained localized within the spatial boundary
            if net_displacement <= self.max_loitering_displacement:
                p_start_bbox, _, _ = self._find_closest_bbox_observation(track, track.first_seen)
                events.append(
                    SecurityEvent(
                        event_id=f"EV-LOIT-{uuid.uuid4().hex[:8]}",
                        event_type="PROLONGED_PRESENCE",
                        severity="NORMAL" if track.duration_seconds < (thresh * 2) else "HIGH",
                        timestamp=track.first_seen,
                        duration_seconds=track.duration_seconds,
                        confidence=round(min(0.95, 0.60 + (track.duration_seconds / (thresh * 3))), 2),
                        description=(
                            f"Prolonged presence detected: {track.object_class} [{track.track_id}] "
                            f"remained stationary/localized for {track.duration_seconds:.1f}s "
                            f"(displacement: {net_displacement:.1f}px)."
                        ),
                        observable_signals=[
                            f"Track ID: {track.track_id}",
                            f"Object class: {track.object_class}",
                            f"Start time: {track.first_seen:.2f}s",
                            f"End time: {track.last_seen:.2f}s",
                            f"Total duration: {track.duration_seconds:.2f}s (threshold: {thresh:.1f}s)",
                            f"Localized net displacement: {net_displacement:.1f}px",
                        ],
                        track_id=track.track_id,
                        object_class=track.object_class,
                        bounding_box=p_start_bbox or track.current_bbox,
                    )
                )

        return events

    def detect_abandoned_objects(
        self,
        tracks: List[TrackedObject],
        threshold_seconds: Optional[float] = None,
    ) -> List[SecurityEvent]:
        """
        Identify unattended belongings (backpacks, suitcases, handbags) that remain stationary
        while nearby person tracks depart.
        """
        thresh = threshold_seconds if threshold_seconds is not None else self.abandoned_stationary_seconds
        belonging_classes = {"backpack", "suitcase", "handbag"}
        person_tracks = [t for t in tracks if t.object_class == "person"]
        belonging_tracks = [t for t in tracks if t.object_class in belonging_classes]

        events: List[SecurityEvent] = []

        for b_track in belonging_tracks:
            # Condition 1: Belonging must be observed for >= threshold
            if b_track.duration_seconds < thresh or len(b_track.trajectory) < 2:
                continue

            # Condition 2: Belonging must be stationary (small displacement)
            b_start_x, b_start_y = b_track.trajectory[0][1], b_track.trajectory[0][2]
            b_end_x, b_end_y = b_track.trajectory[-1][1], b_track.trajectory[-1][2]
            b_disp = math.hypot(b_end_x - b_start_x, b_end_y - b_start_y)
            if b_disp > 40.0:  # Moving object is not abandoned
                continue

            # Condition 3: Check proximity of person tracks over time
            # Was there a person nearby at the start?
            associated_person_id = None
            for p_track in person_tracks:
                if not p_track.trajectory:
                    continue
                p_early = [pt for pt in p_track.trajectory if pt[0] <= b_track.first_seen + 2.0]
                for pt in p_early:
                    if math.hypot(pt[1] - b_start_x, pt[2] - b_start_y) <= self.abandoned_separation_distance:
                        associated_person_id = p_track.track_id
                        break
                if associated_person_id:
                    break

            # Is any person currently nearby at the end of the belonging observation?
            person_nearby_at_end = False
            for p_track in person_tracks:
                if not p_track.trajectory:
                    continue
                p_late = [pt for pt in p_track.trajectory if pt[0] >= b_track.last_seen - 2.0]
                for pt in p_late:
                    if math.hypot(pt[1] - b_end_x, pt[2] - b_end_y) <= self.abandoned_separation_distance:
                        person_nearby_at_end = True
                        break
                if person_nearby_at_end:
                    break

            # If no person nearby at the end, flag as potential abandoned object candidate
            if not person_nearby_at_end:
                b_start_bbox, _, _ = self._find_closest_bbox_observation(b_track, b_track.last_seen)
                events.append(
                    SecurityEvent(
                        event_id=f"EV-ABAN-{uuid.uuid4().hex[:8]}",
                        event_type="POTENTIAL_ABANDONED_OBJECT",
                        severity="HIGH",
                        timestamp=b_track.last_seen,
                        duration_seconds=b_track.duration_seconds,
                        confidence=round(min(0.92, 0.65 + (b_track.duration_seconds / (thresh * 3))), 2),
                        description=(
                            f"Potential abandoned object candidate: stationary {b_track.object_class} "
                            f"[{b_track.track_id}] observed unattended for {b_track.duration_seconds:.1f}s."
                        ),
                        observable_signals=[
                            f"Object: {b_track.object_class} [{b_track.track_id}]",
                            f"Stationary duration: {b_track.duration_seconds:.2f}s (threshold: {thresh:.1f}s)",
                            f"Net movement: {b_disp:.1f}px (stationary)",
                            f"Associated initial person: {associated_person_id or 'none observed'}",
                            "Current status: No tracked person in immediate proximity",
                        ],
                        track_id=b_track.track_id,
                        object_class=b_track.object_class,
                        bounding_box=b_start_bbox or b_track.current_bbox,
                    )
                )

        return events

    def detect_theft_and_removal_patterns(
        self,
        tracks: List[TrackedObject],
    ) -> List[SecurityEvent]:
        """
        Deterministic behavioral analysis layer for detecting potential theft and object takeaway patterns.

        Observable physical sequence:
        1. Person track detected.
        2. Person approaches another detectable object (belonging/package/portable item or any non-person track).
        3. Person remains within interaction distance (<= theft_interaction_max_distance) for >= theft_min_interaction_seconds.
        4. Object is present before/during interaction.
        5. Object disappearance OR co-movement:
           a) Disappearance: Object is no longer detected after interaction, while person subsequently departs
              >= theft_min_departure_distance away from the object's original location.
           b) Co-movement: Object moves in tandem with the person away from original position
              >= theft_min_departure_distance.
        6. Spatially grounds the event to actual person & object coordinates at event timestamp.
        """
        events: List[SecurityEvent] = []
        person_tracks = [t for t in tracks if t.object_class == "person"]
        target_classes = getattr(
            settings,
            "THEFT_TARGET_CLASSES",
            {"backpack", "handbag", "suitcase", "laptop", "cell phone", "bottle", "umbrella", "bicycle", "box", "package"},
        )
        non_person_tracks = [t for t in tracks if t.object_class in target_classes]

        for p_track in person_tracks:
            if not p_track.trajectory or len(p_track.trajectory) < 2:
                continue

            for o_track in non_person_tracks:
                if not o_track.trajectory:
                    continue

                # 1. Determine original object location
                orig_o_cx, orig_o_cy = o_track.trajectory[0][1], o_track.trajectory[0][2]

                # 2. Check person proximity to the object's original location
                proximity_moments: List[Tuple[float, float, float]] = []  # (t, p_cx, p_cy)
                for p_pt in p_track.trajectory:
                    p_t, p_cx, p_cy = p_pt[0], p_pt[1], p_pt[2]
                    dist_to_orig = math.hypot(p_cx - orig_o_cx, p_cy - orig_o_cy)
                    if dist_to_orig <= self.theft_interaction_max_distance:
                        proximity_moments.append((p_t, p_cx, p_cy))

                if not proximity_moments:
                    continue

                interaction_start = min(m[0] for m in proximity_moments)
                interaction_end = max(m[0] for m in proximity_moments)
                interaction_duration = round(interaction_end - interaction_start, 2)

                # Require temporal persistence: dwell within interaction proximity for >= theft_min_interaction_seconds
                if interaction_duration < self.theft_min_interaction_seconds and len(proximity_moments) < 2:
                    continue

                # 3. Check takeaway / departure condition:
                subsequent_person_pts = [
                    pt for pt in p_track.trajectory
                    if pt[0] >= interaction_start + self.theft_min_interaction_seconds or pt[0] >= interaction_end
                ]
                if not subsequent_person_pts:
                    continue

                max_departure_disp = max(
                    math.hypot(pt[1] - orig_o_cx, pt[2] - orig_o_cy)
                    for pt in subsequent_person_pts
                )

                if max_departure_disp < self.theft_min_departure_distance:
                    continue

                # 4. Check object removal pattern:
                is_disappeared = (
                    o_track.last_seen <= interaction_end + self.theft_object_loss_window
                    and p_track.last_seen > o_track.last_seen
                )

                is_co_moving = False
                if not is_disappeared:
                    o_max_disp = max(
                        math.hypot(pt[1] - orig_o_cx, pt[2] - orig_o_cy)
                        for pt in o_track.trajectory
                    )
                    if o_max_disp >= self.theft_min_departure_distance:
                        is_co_moving = True

                if not (is_disappeared or is_co_moving):
                    continue

                pattern_type = "object_disappearance" if is_disappeared else "co_movement_takeaway"
                duration_total = round(p_track.last_seen - interaction_start, 2)
                confidence = round(
                    min(0.95, 0.70 + min(0.15, (interaction_duration / (self.theft_min_interaction_seconds * 2.0)) * 0.10) + min(0.10, (max_departure_disp / (self.theft_min_departure_distance * 2.0)) * 0.10)),
                    2
                )

                removal_desc = (
                    f"after which the object was no longer detected at its original position"
                    if is_disappeared
                    else f"after which the object moved in tandem with the person away from its original position"
                )

                # 5. Retrieve actual spatial bounding boxes at event timestamp.
                # For the person: use closest observation within 3s of interaction_start.
                p_bbox, p_obs_ts, _ = self._find_closest_bbox_observation(p_track, interaction_start, max_tolerance=3.0)

                # For the object: Use the closest observation with a generous tolerance that
                # spans the entire interaction + loss window. This handles the common case where
                # YOLO detects an object only once (at the end of a clip or during the disappearance
                # window) — that single observation is still valid forensic spatial grounding.
                o_max_tolerance = max(3.0, interaction_duration + self.theft_object_loss_window + 5.0)
                o_bbox, o_obs_ts, is_prior_o = self._find_closest_bbox_observation(
                    o_track, interaction_start, max_tolerance=o_max_tolerance
                )

                # Classify object grounding: did we observe the object at/near interaction start,
                # or only at a subsequent timestamp (prior obs before or later in sequence)?
                active_o_bbox = None
                subsequent_o_obs = None
                if o_bbox is not None and o_obs_ts is not None:
                    # Always use the object bbox when it exists — it proves the object was there
                    active_o_bbox = o_bbox
                    # Mark as prior/subsequent for the UI label
                    if o_obs_ts > interaction_start + 3.0:
                        is_prior_o = False  # Subsequent observation (e.g. the object appeared later)
                        subsequent_o_obs = o_obs_ts  # For fallback signal text if needed

                score_pct = int(confidence * 100)
                score_tier = "High Confidence" if score_pct >= 85 else ("Medium Confidence" if score_pct >= 60 else "Low Confidence")

                event_id = f"EV-THEFT-{uuid.uuid4().hex[:8]}"
                reason = (
                    f"Person [{p_track.track_id}] approached {o_track.object_class} [{o_track.track_id}], "
                    f"remained within interaction distance for {interaction_duration:.1f}s, "
                    f"{removal_desc} while the person departed ({max_departure_disp:.1f}px displacement)."
                )

                obs_signals = [
                    f"Signal 1: Person track [{p_track.track_id}] approached {o_track.object_class} [{o_track.track_id}]",
                    f"Signal 2: Interaction proximity dwell duration: {interaction_duration:.1f}s (threshold: {self.theft_min_interaction_seconds:.1f}s)",
                    f"Signal 3: Observational removal pattern: {pattern_type} (last detected at {o_track.last_seen:.2f}s)",
                    f"Signal 4: Person departure displacement: {max_departure_disp:.1f}px (threshold: {self.theft_min_departure_distance:.1f}px)",
                ]

                if p_bbox:
                    obs_signals.append(
                        f"Signal 5: Person [{p_track.track_id}] spatially grounded at [{p_bbox.x1:.1f}, {p_bbox.y1:.1f} -> {p_bbox.x2:.1f}, {p_bbox.y2:.1f}]"
                    )
                else:
                    obs_signals.append(f"Signal 5: Person [{p_track.track_id}] — no spatial bounding box available at {interaction_start:.2f}s")

                if active_o_bbox:
                    if is_prior_o:
                        o_tag = f"(Prior supporting observation at {o_obs_ts:.1f}s)"
                    elif subsequent_o_obs is not None:
                        o_tag = f"(Subsequent supporting observation at {o_obs_ts:.1f}s — object present during sequence)"
                    else:
                        o_tag = f"(Observed at {o_obs_ts:.1f}s)"
                    obs_signals.append(
                        f"Signal 6: Object [{o_track.object_class.upper()} • {o_track.track_id}] grounded at [{active_o_bbox.x1:.1f}, {active_o_bbox.y1:.1f} -> {active_o_bbox.x2:.1f}, {active_o_bbox.y2:.1f}] {o_tag}"
                    )
                else:
                    obs_signals.append(
                        f"Signal 6: Object [{o_track.object_class.upper()} • {o_track.track_id}] — no bounding box available (object not detected at interaction timestamp)"
                    )

                obs_signals.append(f"Pattern Evidence Strength: {score_pct}% ({score_tier}) — Human verification required")

                events.append(
                    SecurityEvent(
                        event_id=event_id,
                        event_type="POTENTIAL_THEFT",
                        severity="HIGH",
                        timestamp=interaction_start,
                        duration_seconds=duration_total,
                        confidence=confidence,
                        description=f"Potential Theft Pattern (Pattern Evidence Strength: {score_pct}% — Human verification required): {reason}",
                        observable_signals=obs_signals,
                        track_id=p_track.track_id,
                        object_class=o_track.object_class,
                        bounding_box=p_bbox,
                        person_track_id=p_track.track_id,
                        object_track_id=o_track.track_id,
                        object_bounding_box=active_o_bbox,
                        object_observation_timestamp=o_obs_ts if active_o_bbox else (subsequent_o_obs if subsequent_o_obs else None),
                        is_prior_object_observation=is_prior_o if active_o_bbox else False,
                    )
                )

        return events

    def detect_observational_anomalies(
        self,
        tracks: List[TrackedObject],
        security_events: List[SecurityEvent],
    ) -> List[SecurityEvent]:
        """
        Aggregate multi-signal observations into high-level explainable security anomalies.
        """
        anomalies: List[SecurityEvent] = []

        intrusions = [e for e in security_events if e.event_type == "POTENTIAL_INTRUSION"]
        loiterings = [e for e in security_events if e.event_type == "PROLONGED_PRESENCE"]
        abandoned = [e for e in security_events if e.event_type == "POTENTIAL_ABANDONED_OBJECT"]
        thefts = [e for e in security_events if e.event_type == "POTENTIAL_THEFT"]

        # Anomaly Pattern 1: Intrusion with prolonged presence (high priority)
        intrusion_tracks = {e.track_id: e for e in intrusions if e.track_id}
        for l_ev in loiterings:
            if l_ev.track_id in intrusion_tracks:
                int_ev = intrusion_tracks[l_ev.track_id]
                anomalies.append(
                    SecurityEvent(
                        event_id=f"EV-ANOM-{uuid.uuid4().hex[:8]}",
                        event_type="OBSERVATIONAL_ANOMALY",
                        severity="HIGH",
                        timestamp=l_ev.timestamp,
                        duration_seconds=l_ev.duration_seconds,
                        confidence=0.92,
                        description=(
                            f"Potentially anomalous pattern: {l_ev.object_class} [{l_ev.track_id}] "
                            f"entered restricted zone '{int_ev.zone_name}' and exhibited prolonged presence "
                            f"({l_ev.duration_seconds:.1f}s)."
                        ),
                        observable_signals=[
                            f"Signal 1: Restricted zone entry into '{int_ev.zone_name}' at {int_ev.timestamp:.2f}s",
                            f"Signal 2: Prolonged duration in localized zone area for {l_ev.duration_seconds:.2f}s",
                            f"Signal 3: Track ID {l_ev.track_id} ({l_ev.object_class})",
                        ],
                        track_id=l_ev.track_id,
                        object_class=l_ev.object_class,
                        zone_name=int_ev.zone_name,
                        bounding_box=l_ev.bounding_box,
                    )
                )

        # Anomaly Pattern 2: Abandoned object candidate
        for ab_ev in abandoned:
            anomalies.append(
                SecurityEvent(
                    event_id=f"EV-ANOM-{uuid.uuid4().hex[:8]}",
                    event_type="OBSERVATIONAL_ANOMALY",
                    severity="HIGH",
                    timestamp=ab_ev.timestamp,
                    duration_seconds=ab_ev.duration_seconds,
                    confidence=ab_ev.confidence,
                    description=(
                        f"Potentially anomalous pattern: unattended stationary {ab_ev.object_class} "
                        f"[{ab_ev.track_id}] without accompanying person presence."
                    ),
                    observable_signals=ab_ev.observable_signals,
                    track_id=ab_ev.track_id,
                    object_class=ab_ev.object_class,
                    bounding_box=ab_ev.bounding_box,
                )
            )

        # Anomaly Pattern 3: Potential theft / takeaway pattern
        for th_ev in thefts:
            anomalies.append(
                SecurityEvent(
                    event_id=f"EV-ANOM-{uuid.uuid4().hex[:8]}",
                    event_type="OBSERVATIONAL_ANOMALY",
                    severity="HIGH",
                    timestamp=th_ev.timestamp,
                    duration_seconds=th_ev.duration_seconds,
                    confidence=th_ev.confidence,
                    description=(
                        f"Potentially anomalous pattern: suspected object takeaway pattern (Human verification required) involving "
                        f"person [{th_ev.track_id}] and {th_ev.object_class}."
                    ),
                    observable_signals=th_ev.observable_signals,
                    track_id=th_ev.track_id,
                    object_class=th_ev.object_class,
                    bounding_box=th_ev.bounding_box,
                    person_track_id=th_ev.person_track_id,
                    object_track_id=th_ev.object_track_id,
                    object_bounding_box=th_ev.object_bounding_box,
                    object_observation_timestamp=th_ev.object_observation_timestamp,
                    is_prior_object_observation=th_ev.is_prior_object_observation,
                )
            )

        return anomalies
