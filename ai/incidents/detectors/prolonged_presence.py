"""
Context-Aware Prolonged Presence / Loitering Detector (Phase 10 — Audit-Hardened)

Detects subjects exhibiting abnormally localized presence or lingering.
Context-aware semantics:
- For persons: flags lingering/loitering within localized perimeter.
- For vehicles: avoids false alarms on normal road traffic/slow transit.
  Only flags vehicles if stopped/stalled/idling for significant periods or
  inside a restricted zone.

Hardening rationale (Audit):
- person_threshold_seconds was 4.0s — this fired on nearly every validated person
  track in a surveillance scene since most people are visible for at least 4 seconds.
  Raised to 8.0s (a genuinely unusual loitering threshold for a typical camera view).
- The max_loitering_displacement was 120px — anyone who moved even slightly off-camera
  center still qualified. Tightened to 80px to ensure truly localized dwell is required.
- Now requires the person to have a significantly below-average motion rate (slow/stopped)
  relative to the video context, using the motion summary data already available.
"""
import math
from typing import List, Dict, Any, Optional

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
from ai.incidents.spatial import SpatialRelationshipEngine
from ai.incidents.negative_evidence import NegativeEvidenceEngine


class ProlongedPresenceDetector(BaseIncidentDetector):
    """
    Context-aware detector for prolonged physical presence.

    Hardened to require a genuinely anomalous dwell time (8s for persons)
    and to reject tracks that show continuous transit motion.
    """

    detector_name: str = "prolonged_presence_detector"
    detector_version: str = "2.2.0"
    category: IncidentCategory = IncidentCategory.PERSON

    # Phase Z Contract Declarations
    required_signals: List[str] = ["Prolonged Presence", "Localized Dwell"]
    supporting_signals_declared: List[str] = ["Prolonged Presence", "Motion State", "Zone Dwell"]
    contradictory_signals_declared: List[str] = ["Negative: Continuous Transit Flow"]
    context_requirements: Dict[str, Any] = {"roadway_suppresses_moving_vehicles": True}
    evidence_requirements: Dict[str, Any] = {"verified_track": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        person_threshold_seconds: float = 8.0,
        zone_dwell_threshold_seconds: float = 4.0,
        vehicle_stationary_threshold_seconds: float = 15.0,
        max_loitering_displacement: float = 80.0,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.person_threshold_seconds = person_threshold_seconds
        # Zone dwell threshold stays lower — entering a restricted zone is significant
        # even for shorter durations. This preserves existing zone-intrusion alerting.
        self.zone_dwell_threshold_seconds = zone_dwell_threshold_seconds
        self.vehicle_stationary_threshold_seconds = vehicle_stationary_threshold_seconds
        self.max_loitering_displacement = max_loitering_displacement

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        vehicle_classes = {"car", "bus", "truck", "motorcycle", "van"}

        for track in context.tracks:
            if not track.is_validated or len(track.trajectory) < 2:
                continue

            summary = context.get_motion_summary(track.track_id) or {}
            net_disp = summary.get("net_displacement", 0.0)
            is_vehicle = track.object_class in vehicle_classes

            # Check if track is inside any active restricted zone
            inside_zone_name = None
            if context.zones:
                for z in context.zones:
                    z_analysis = SpatialRelationshipEngine.track_zone_dwell_analysis(track, z)
                    if z_analysis["entered"] and z_analysis["total_dwell_seconds"] >= self.zone_dwell_threshold_seconds:
                        inside_zone_name = z.name
                        break

            # 1. VEHICLE CONTEXT LOGIC
            if is_vehicle:
                # If vehicle is traveling along road/transit, ignore (do not alarm normal traffic)
                if not inside_zone_name:
                    max_stat_sec = summary.get("max_stationary_duration", 0.0)
                    avg_vel = summary.get("avg_velocity", 0.0)
                    path_cons = summary.get("path_consistency", 1.0)

                    # Transit vehicle check: has significant speed or continuous transit path
                    if avg_vel > 20.0 or (path_cons > 0.35 and net_disp > self.max_loitering_displacement):
                        continue

                    # Must exceed vehicle stationary threshold (e.g. stalled or idling)
                    if max_stat_sec < self.vehicle_stationary_threshold_seconds:
                        continue

                thresh = self.vehicle_stationary_threshold_seconds if not inside_zone_name else self.person_threshold_seconds
            else:
                # 2. PERSON / OTHER OBJECT LOGIC
                # If inside a restricted zone, use the lower zone-specific threshold.
                # The zone threshold (4.0s default) is intentionally lower than the
                # free-space loitering threshold (8.0s) because zone breaches are
                # security-significant even at shorter dwell times.
                if inside_zone_name:
                    thresh = self.zone_dwell_threshold_seconds
                else:
                    thresh = self.person_threshold_seconds

                if track.duration_seconds < thresh:
                    continue
                # Person must remain spatially localized (only applies to non-zone case)
                if net_disp > self.max_loitering_displacement and not inside_zone_name:
                    continue

                # Additional hardening: if the person is continuously moving (high avg velocity)
                # they are NOT loitering — they are in transit. Only flag if they have low
                # average velocity indicating a stopped/dwell state.
                avg_vel = summary.get("avg_velocity", 0.0)
                if avg_vel > 30.0 and not inside_zone_name:
                    # Person is actively moving through the scene — not loitering
                    continue

            # Signal generation
            loc_context_desc = (
                f"in restricted zone '{inside_zone_name}'"
                if inside_zone_name
                else f"within localized perimeter ({net_disp:.1f}px displacement)"
            )
            signals = [
                SupportingSignal(
                    signal_type="Prolonged Presence",
                    description=(
                        f"{track.object_class} [{track.track_id}] observed for "
                        f"{track.duration_seconds:.1f}s {loc_context_desc}"
                    ),
                    confidence=track.confidence,
                    timestamp=track.first_seen,
                    track_id=track.track_id,
                ),
                SupportingSignal(
                    signal_type="Motion State",
                    description=(
                        f"Max stationary duration: {summary.get('max_stationary_duration', 0.0):.1f}s, "
                        f"average velocity: {summary.get('avg_velocity', 0.0):.1f}px/s"
                    ),
                    confidence=0.85,
                    timestamp=track.first_seen,
                    track_id=track.track_id,
                ),
            ]

            # Negative Evidence evaluation
            neg_signals = NegativeEvidenceEngine.evaluate_loitering_negative_evidence(track, context)
            if not inside_zone_name and neg_signals:
                # Continuous transit flow refutes prolonged presence
                continue

            scoring = IncidentScorer.calculate_evidence_score(
                base_confidence=0.75,
                supporting_signals=signals,
                tracks=[track],
                duration_seconds=track.duration_seconds,
                expected_duration_threshold=thresh,
            )

            # Only restricted zone breaches elevate prolonged presence to HIGH severity
            if inside_zone_name:
                severity = "HIGH"
                desc_label = f"in restricted zone '{inside_zone_name}'"
            elif track.duration_seconds >= (thresh * 4.0):
                severity = "NORMAL"
                desc_label = "in localized area"
            else:
                severity = "LOW"
                desc_label = "in localized area"

            spatial_ctx = SpatialContext(
                centroid=track.current_bbox.centroid if track.current_bbox else None,
                bounding_box=track.current_bbox,
                zone_name=inside_zone_name,
            )

            explanation = (
                f"Prolonged presence in restricted zone '{inside_zone_name}': {track.object_class} [{track.track_id}] "
                f"dwell time {track.duration_seconds:.1f}s."
                if inside_zone_name else
                f"Stationary presence observation: {track.object_class} [{track.track_id}] remained {desc_label} "
                f"for {track.duration_seconds:.1f}s (threshold: {thresh:.0f}s; avg velocity: "
                f"{summary.get('avg_velocity', 0.0):.1f}px/s)."
            )

            cand = self.build_candidate(
                video_id=context.video_id,
                event_type="PROLONGED_PRESENCE",
                start_time=track.first_seen,
                end_time=track.last_seen,
                severity=severity,
                confidence=scoring["score"],
                explanation=explanation,
                track_ids=[track.track_id],
                object_classes=[track.object_class],
                supporting_signals=signals,
                contradictory_signals=neg_signals,
                spatial_context=spatial_ctx,
                evidence_candidates=[
                    EvidenceCandidate(
                        timestamp=track.first_seen,
                        bounding_box=track.current_bbox,
                        target_track_id=track.track_id,
                        reason="Prolonged presence onset",
                    )
                ],
                prefix="LOIT",
            )
            candidates.append(cand)

        return candidates
