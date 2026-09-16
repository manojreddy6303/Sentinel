"""
Potential Wrong-Way Vehicle Detector (Sentinel Phase 11)

Detects vehicles traveling in the direction opposing dominant traffic flow:
- Infers dominant roadway direction dynamically from visual trajectory clusters
- Evaluates persistent counter-flow travel
- Suppresses turns, intersection maneuvers, and lane changes
- Explicitly abstains when roadway flow is ambiguous or multidirectional
"""
import math
from typing import List, Dict, Any, Optional, Tuple

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


class WrongWayVehicleDetector(BaseIncidentDetector):
    """
    Evaluates vehicle heading vectors against scene-wide dominant traffic corridors.
    """

    detector_name: str = "wrong_way_detector"
    detector_version: str = "1.0.0"
    category: IncidentCategory = IncidentCategory.VEHICLE

    # Phase Z Contract Declarations
    required_signals: List[str] = ["Dominant Traffic Corridor", "Persistent Opposing Heading"]
    supporting_signals_declared: List[str] = [
        "Dominant Traffic Corridor",
        "Persistent Opposing Heading",
        "Sustained Counter-Flow Transit",
    ]
    contradictory_signals_declared: List[str] = [
        "Negative: Turning / Intersection Maneuver",
        "Negative: Multidirectional / Ambiguous Traffic Flow",
        "Negative: Low Trajectory Continuity",
    ]
    context_requirements: Dict[str, Any] = {"roadway_flow_min_confidence": 0.60}
    evidence_requirements: Dict[str, Any] = {"verified_track": True, "sustained_trajectory": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        min_opposing_angle_deg: float = 135.0,
        min_opposing_duration_seconds: float = 2.0,
        min_flow_sample_vehicles: int = 3,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.min_opposing_angle_deg = min_opposing_angle_deg
        self.min_opposing_duration_seconds = min_opposing_duration_seconds
        self.min_flow_sample_vehicles = min_flow_sample_vehicles
        self.vehicle_classes = {"car", "bus", "truck", "motorcycle", "van"}

    def _infer_dominant_flow(self, vehicles: List[Any]) -> Tuple[Optional[float], float]:
        """
        Calculates the dominant heading vector angle and the flow consistency ratio.
        Returns: (dominant_angle_deg, confidence_ratio)
        """
        headings = []
        for v in vehicles:
            pts = v.trajectory or []
            if len(pts) >= 2:
                dx = pts[-1][1] - pts[0][1]
                dy = pts[-1][2] - pts[0][2]
                if math.hypot(dx, dy) >= 30.0:
                    ang = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
                    headings.append(ang)

        if len(headings) < self.min_flow_sample_vehicles:
            return None, 0.0

        # Cluster headings into 8 sectors (45 degrees each)
        sectors = [0] * 8
        for h in headings:
            sec = int(((h + 22.5) % 360.0) // 45.0)
            sectors[sec] += 1

        max_count = max(sectors)
        dominant_sec = sectors.index(max_count)
        consistency = max_count / len(headings)

        # Dominant angle is center of sector
        dom_angle = (dominant_sec * 45.0) % 360.0
        return dom_angle, consistency

    def _is_turning_maneuver(self, track: Any) -> bool:
        """Check if vehicle trajectory represents a legitimate turn rather than wrong-way travel."""
        pts = track.trajectory or []
        if len(pts) < 4:
            return False

        # Calculate initial vs final heading
        dx_start = pts[1][1] - pts[0][1]
        dy_start = pts[1][2] - pts[0][2]
        dx_end = pts[-1][1] - pts[-2][1]
        dy_end = pts[-1][2] - pts[-2][2]

        ang_start = math.atan2(dy_start, dx_start)
        ang_end = math.atan2(dy_end, dx_end)

        sweep = abs(math.degrees(ang_start - ang_end)) % 360.0
        if sweep > 180.0:
            sweep = 360.0 - sweep

        # Large angular turn (e.g. 70 deg or more) indicates turning maneuver at intersection
        return sweep >= 70.0

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        vehicles = [t for t in context.tracks if t.object_class in self.vehicle_classes and t.is_validated]
        if len(vehicles) < self.min_flow_sample_vehicles:
            # Insufficient vehicles to establish roadway direction. ABSTAIN.
            return candidates

        # Infer dominant flow
        dom_flow, flow_conf = self._infer_dominant_flow(vehicles)
        if dom_flow is None or flow_conf < 0.60:
            # Flow is multidirectional, mixed, or ambiguous. ABSTAIN.
            return candidates

        for track in vehicles:
            pts = track.trajectory or []
            if len(pts) < 2 or track.duration_seconds < self.min_opposing_duration_seconds:
                continue

            # Calculate vehicle overall heading
            dx = pts[-1][1] - pts[0][1]
            dy = pts[-1][2] - pts[0][2]
            disp = math.hypot(dx, dy)
            if disp < 40.0:  # Localized / stationary vehicle is not a wrong-way transit
                continue

            v_angle = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
            angle_diff = abs(v_angle - dom_flow) % 360.0
            if angle_diff > 180.0:
                angle_diff = 360.0 - angle_diff

            # Must oppose dominant flow by at least 135 degrees
            if angle_diff >= self.min_opposing_angle_deg:
                # Negative evidence: check for turning maneuver
                if self._is_turning_maneuver(track):
                    continue

                signals = [
                    SupportingSignal(
                        signal_type="Dominant Traffic Corridor",
                        description=f"Roadway flow established at {dom_flow:.0f}° ({flow_conf * 100:.0f}% traffic consistency)",
                        confidence=flow_conf,
                        timestamp=track.first_seen,
                    ),
                    SupportingSignal(
                        signal_type="Persistent Opposing Heading",
                        description=f"{track.object_class} [{track.track_id}] heading ({v_angle:.0f}°) opposes dominant flow by {angle_diff:.0f}°",
                        confidence=0.88,
                        timestamp=track.first_seen,
                        track_id=track.track_id,
                    ),
                    SupportingSignal(
                        signal_type="Sustained Counter-Flow Transit",
                        description=f"Counter-flow displacement of {disp:.1f}px sustained over {track.duration_seconds:.1f}s",
                        confidence=0.85,
                        timestamp=track.last_seen,
                        track_id=track.track_id,
                    ),
                ]

                scoring = IncidentScorer.calculate_evidence_score(
                    base_confidence=0.75,
                    supporting_signals=signals,
                    tracks=[track],
                )

                c_box = track.current_bbox
                cand = self.build_candidate(
                    video_id=context.video_id,
                    event_type="POTENTIAL_WRONG_WAY_VEHICLE",
                    start_time=track.first_seen,
                    end_time=track.last_seen,
                    severity="HIGH",
                    confidence=scoring["score"],
                    explanation=(
                        f"Potential Wrong-Way Vehicle: {track.object_class} [{track.track_id}] traveled {disp:.1f}px "
                        f"in direction ({v_angle:.0f}°) opposing dominant traffic corridor ({dom_flow:.0f}°) by {angle_diff:.0f}°."
                    ),
                    track_ids=[track.track_id],
                    object_classes=[track.object_class],
                    supporting_signals=signals,
                    spatial_context=SpatialContext(
                        centroid=c_box.centroid if c_box else None,
                        bounding_box=c_box,
                        metadata={"dominant_flow_angle": dom_flow, "vehicle_heading": v_angle},
                    ),
                    evidence_candidates=[
                        EvidenceCandidate(
                            timestamp=track.first_seen,
                            bounding_box=c_box,
                            target_track_id=track.track_id,
                            reason="Onset of counter-flow vehicle trajectory",
                        )
                    ],
                    prefix="WRONG",
                )
                candidates.append(cand)

        return candidates
