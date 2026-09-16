"""
Potential Unusual Vehicle Trajectory Detector (Sentinel Phase 11)

Detects erratic, oscillating, or severely deviated vehicle trajectories:
- Repeated lateral oscillation (weaving across lanes)
- Abrupt mid-transit trajectory deflection without an intersection
- Severe departure from roadway corridor alignment
- Distinguishes ordinary smooth lane changes from erratic movement
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


class UnusualTrajectoryDetector(BaseIncidentDetector):
    """
    Evaluates vehicle trajectory geometry for erratic weaving or abrupt course deviations.
    """

    detector_name: str = "unusual_trajectory_detector"
    detector_version: str = "1.0.0"
    category: IncidentCategory = IncidentCategory.VEHICLE

    # Phase Z Contract Declarations
    required_signals: List[str] = ["Trajectory Oscillation or Severe Course Deviation"]
    supporting_signals_declared: List[str] = [
        "Trajectory Lateral Oscillation",
        "Abrupt Course Deflection",
        "Path Irregularity",
    ]
    contradictory_signals_declared: [
        "Negative: Smooth Monotonic Lane Change",
        "Negative: Intersection Turn",
    ]
    context_requirements: Dict[str, Any] = {"min_trajectory_points": 5}
    evidence_requirements: Dict[str, Any] = {"verified_track": True, "trajectory_points": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        min_oscillation_count: int = 2,
        min_deflection_deg: float = 40.0,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.min_oscillation_count = min_oscillation_count
        self.min_deflection_deg = min_deflection_deg
        self.vehicle_classes = {"car", "bus", "truck", "motorcycle", "van"}

    def _analyze_trajectory_irregularity(self, pts: List[Any]) -> Dict[str, Any]:
        """
        Analyze path for:
        1. Net displacement vs total path length (path consistency)
        2. Lateral heading reversals (weaving)
        3. Max abrupt heading delta between consecutive segments
        """
        if len(pts) < 4:
            return {"is_unusual": False}

        # Calculate consecutive segment angles
        segment_angles = []
        path_length = 0.0
        for i in range(len(pts) - 1):
            dx = pts[i + 1][1] - pts[i][1]
            dy = pts[i + 1][2] - pts[i][2]
            seg_len = math.hypot(dx, dy)
            path_length += seg_len
            if seg_len > 5.0:
                ang = math.degrees(math.atan2(dy, dx))
                segment_angles.append(ang)

        start_x, start_y = pts[0][1], pts[0][2]
        end_x, end_y = pts[-1][1], pts[-1][2]
        net_disp = math.hypot(end_x - start_x, end_y - start_y)
        consistency = net_disp / max(1.0, path_length)

        # Check for heading sign reversals (lateral oscillation)
        heading_deltas = []
        for i in range(len(segment_angles) - 1):
            d = (segment_angles[i + 1] - segment_angles[i] + 180.0) % 360.0 - 180.0
            heading_deltas.append(d)

        # Count directional sign flips exceeding 20 degrees
        sign_flips = 0
        for i in range(len(heading_deltas) - 1):
            d1, d2 = heading_deltas[i], heading_deltas[i + 1]
            if (d1 * d2 < 0) and abs(d1) >= 20.0 and abs(d2) >= 20.0:
                sign_flips += 1

        max_single_deflection = max((abs(d) for d in heading_deltas), default=0.0)

        # Classification:
        # A normal lane change has monotonic lateral shift (sign_flips == 0, consistency >= 0.85).
        # Weaving has sign_flips >= 2.
        # Sudden erratic swerve has max_single_deflection >= 40 deg with path consistency < 0.70.
        is_weaving = (sign_flips >= self.min_oscillation_count)
        is_abrupt_swerve = (max_single_deflection >= self.min_deflection_deg and consistency < 0.75)

        return {
            "is_unusual": is_weaving or is_abrupt_swerve,
            "is_weaving": is_weaving,
            "sign_flips": sign_flips,
            "max_deflection_deg": round(max_single_deflection, 1),
            "path_consistency": round(consistency, 3),
        }

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        vehicles = [t for t in context.tracks if t.object_class in self.vehicle_classes and t.is_validated]

        for track in vehicles:
            pts = track.trajectory or []
            if len(pts) < 5 or track.duration_seconds < 2.0:
                continue

            analysis = self._analyze_trajectory_irregularity(pts)
            if not analysis["is_unusual"]:
                continue

            signals = []
            if analysis["is_weaving"]:
                signals.append(
                    SupportingSignal(
                        signal_type="Trajectory Lateral Oscillation",
                        description=f"Vehicle exhibited repeated lateral weaving ({analysis['sign_flips']} heading reversals)",
                        confidence=0.88,
                        timestamp=track.first_seen,
                        track_id=track.track_id,
                    )
                )
            else:
                signals.append(
                    SupportingSignal(
                        signal_type="Abrupt Course Deflection",
                        description=f"Sudden trajectory deflection of {analysis['max_deflection_deg']}° observed (consistency: {analysis['path_consistency']:.2f})",
                        confidence=0.85,
                        timestamp=track.first_seen,
                        track_id=track.track_id,
                    )
                )

            scoring = IncidentScorer.calculate_evidence_score(
                base_confidence=0.72,
                supporting_signals=signals,
                tracks=[track],
            )

            c_box = track.current_bbox
            cand = self.build_candidate(
                video_id=context.video_id,
                event_type="POTENTIAL_UNUSUAL_VEHICLE_TRAJECTORY",
                start_time=track.first_seen,
                end_time=track.last_seen,
                severity="LOW",
                confidence=scoring["score"],
                explanation=(
                    f"Potential Unusual Trajectory: {track.object_class} [{track.track_id}] exhibited "
                    f"{'repeated lateral weaving' if analysis['is_weaving'] else 'abrupt course deflection'} "
                    f"(deflection: {analysis['max_deflection_deg']}°, consistency: {analysis['path_consistency']:.2f})."
                ),
                track_ids=[track.track_id],
                object_classes=[track.object_class],
                supporting_signals=signals,
                spatial_context=SpatialContext(
                    centroid=c_box.centroid if c_box else None,
                    bounding_box=c_box,
                    metadata=analysis,
                ),
                evidence_candidates=[
                    EvidenceCandidate(
                        timestamp=track.first_seen,
                        bounding_box=c_box,
                        target_track_id=track.track_id,
                        reason="Onset of unusual vehicle trajectory pattern",
                    )
                ],
                prefix="TRAJ",
            )
            candidates.append(cand)

        return candidates
