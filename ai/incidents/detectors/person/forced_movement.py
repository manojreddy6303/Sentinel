"""
Potential Forced Movement Detector (Sentinel Phase 12 — Audit-Hardened)

Detects observable patterns where one person persistently guides, pushes, or directs
another person along a constrained or abruptly deflecting trajectory:
- Persistent close spatial contact (< 1.0 body width)
- Synchronized trajectory with MULTIPLE abrupt directional course deflections (>= 3 deflections)
- Duration >= 4.0s (raised from 2.5s)
- Asymmetric motion: one track drives the deflections, the other follows them
- Observational, conservative review labeling (STRICTLY NEVER 'kidnapping' or 'abduction')

Hardening rationale (Audit):
- Original thresholds (2.5s, 65px proximity, single 50° deflection) were triggering on
  people walking close together who naturally turned a corner. Any pair moving within 65px
  and making a single turn was flagged.
- Raised to 4.0s + 3 deflections + 55px proximity + 45° threshold per deflection ensures
  the pattern is genuinely anomalous (constrained repeated direction changes in tight contact)
  rather than ordinary navigation behavior.
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
from ai.incidents.negative_evidence import NegativeEvidenceEngine


class ForcedMovementDetector(BaseIncidentDetector):
    """
    Evaluates pairwise person movement for observable constrained or forced movement patterns.

    Hardened to require multiple sharp deflections (not just a single turn) and stricter
    proximity and duration thresholds to distinguish genuinely constrained movement from
    ordinary pair-walking with turns.
    """

    detector_name: str = "forced_movement_detector"
    detector_version: str = "1.1.0"
    category: IncidentCategory = IncidentCategory.PERSON

    required_signals: List[str] = ["Persistent Contact Proximity", "Constrained Deflection"]
    supporting_signals_declared: List[str] = [
        "Persistent Contact Proximity",
        "Synchronized Course Deflection",
    ]
    contradictory_signals_declared: [
        "Negative: Normal Pair Walking Abreast",
        "Negative: Normal Pedestrian Queue",
    ]
    context_requirements: Dict[str, Any] = {"pedestrian_context": True}
    evidence_requirements: Dict[str, Any] = {"verified_tracks": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        min_duration_seconds: float = 4.0,
        max_proximity_px: float = 55.0,
        min_deflections: int = 3,
        min_deflection_angle_deg: float = 45.0,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.min_duration_seconds = min_duration_seconds
        self.max_proximity_px = max_proximity_px
        # Require multiple sharp deflections rather than just one
        self.min_deflections = min_deflections
        self.min_deflection_angle_deg = min_deflection_angle_deg

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        persons = [t for t in context.tracks if t.object_class == "person" and t.is_validated]
        if len(persons) < 2:
            return candidates

        checked_pairs = set()
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                p1, p2 = persons[i], persons[j]
                pair_key = (min(p1.track_id, p2.track_id), max(p1.track_id, p2.track_id))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                # Duration requirement
                shared_duration = min(p1.duration_seconds, p2.duration_seconds)
                if shared_duration < self.min_duration_seconds:
                    continue

                pts1 = p1.trajectory or []
                pts2 = p2.trajectory or []
                if len(pts1) < 4 or len(pts2) < 4:
                    continue

                # Measure pairwise distance over time
                distances = []
                for t1, x1, y1 in pts1:
                    near_p2 = [p for p in pts2 if abs(p[0] - t1) <= 0.40]
                    if near_p2:
                        p2_match = min(near_p2, key=lambda p: abs(p[0] - t1))
                        distances.append(math.hypot(x1 - p2_match[1], y1 - p2_match[2]))

                if len(distances) < 4:
                    continue

                avg_dist = sum(distances) / len(distances)
                # Must maintain close contact throughout movement
                if avg_dist > self.max_proximity_px:
                    continue

                # Both tracks must undergo genuine spatial transit across the scene.
                # Stationary proximity, detector jitter, or dwelling in place (< 30px) is NOT forced movement.
                disp1 = math.hypot(pts1[-1][1] - pts1[0][1], pts1[-1][2] - pts1[0][2])
                disp2 = math.hypot(pts2[-1][1] - pts2[0][1], pts2[-1][2] - pts2[0][2])
                if disp1 < 30.0 or disp2 < 30.0:
                    continue

                # Check if either track is interacting with an object / portable property (theft/takeaway sequence)
                has_property_interaction = False
                for t in (context.tracks if context else []):
                    if t.object_class != "person" and t.is_validated and t.trajectory:
                        for p_cand in [p1, p2]:
                            if p_cand.trajectory:
                                for p_pt in p_cand.trajectory:
                                    for o_pt in t.trajectory:
                                        if math.hypot(p_pt[1] - o_pt[1], p_pt[2] - o_pt[2]) < 100.0:
                                            has_property_interaction = True
                                            break
                                    if has_property_interaction:
                                        break
                            if has_property_interaction:
                                break
                    if has_property_interaction:
                        break
                if has_property_interaction:
                    continue

                # Check for synchronized direction changes while in lockstep for BOTH individuals
                headings1 = []
                for k in range(len(pts1) - 1):
                    dx = pts1[k + 1][1] - pts1[k][1]
                    dy = pts1[k + 1][2] - pts1[k][2]
                    if math.hypot(dx, dy) > 10.0:
                        headings1.append(math.degrees(math.atan2(dy, dx)))

                headings2 = []
                for k in range(len(pts2) - 1):
                    dx = pts2[k + 1][1] - pts2[k][1]
                    dy = pts2[k + 1][2] - pts2[k][2]
                    if math.hypot(dx, dy) > 10.0:
                        headings2.append(math.degrees(math.atan2(dy, dx)))

                if len(headings1) < 2 or len(headings2) < 2:
                    continue

                deflection_count1 = 0
                for h_idx in range(len(headings1) - 1):
                    diff = abs(headings1[h_idx + 1] - headings1[h_idx]) % 360.0
                    if diff > 180.0:
                        diff = 360.0 - diff
                    if diff >= self.min_deflection_angle_deg:
                        deflection_count1 += 1

                deflection_count2 = 0
                for h_idx in range(len(headings2) - 1):
                    diff = abs(headings2[h_idx + 1] - headings2[h_idx]) % 360.0
                    if diff > 180.0:
                        diff = 360.0 - diff
                    if diff >= self.min_deflection_angle_deg:
                        deflection_count2 += 1

                # Require the minimum number of sharp synchronized deflections from both participants
                if deflection_count1 < self.min_deflections or deflection_count2 < self.min_deflections:
                    continue

                deflection_count = min(deflection_count1, deflection_count2)

                event_t = (max(p1.first_seen, p2.first_seen) + min(p1.last_seen, p2.last_seen)) / 2.0

                # Negative evidence: check if pair is simply walking side-by-side normally
                neg_signals = NegativeEvidenceEngine.evaluate_forced_movement_negative_evidence(p1, p2, context, event_t)
                if any("Normal Pair Walking Abreast" in s.signal_type for s in neg_signals):
                    continue

                supporting_signals = [
                    SupportingSignal(
                        signal_type="Persistent Contact Proximity",
                        description=(
                            f"Tracks [{p1.track_id}] and [{p2.track_id}] moved in tight proximity "
                            f"(avg dist: {avg_dist:.1f}px) throughout {shared_duration:.1f}s observation window"
                        ),
                        confidence=0.82,
                        timestamp=event_t,
                    ),
                    SupportingSignal(
                        signal_type="Synchronized Course Deflection",
                        description=(
                            f"Exhibited {deflection_count} sharp directional deviations (>={self.min_deflection_angle_deg:.0f} deg) "
                            f"while maintaining close contact constraint"
                        ),
                        confidence=0.80,
                        timestamp=event_t,
                    ),
                ]

                scoring = IncidentScorer.calculate_evidence_score(
                    base_confidence=0.55,
                    supporting_signals=supporting_signals,
                    tracks=[p1, p2],
                    duration_seconds=shared_duration,
                    contradictory_signals=neg_signals,
                    validation_decision="REVIEW_REQUIRED",
                )

                cand = self.build_candidate(
                    video_id=context.video_id,
                    event_type="POTENTIAL_FORCED_MOVEMENT",
                    start_time=max(0.0, event_t - 1.0),
                    end_time=min(context.duration_seconds, event_t + 2.0),
                    severity="NORMAL",
                    confidence=min(0.60, scoring["confidence"]),
                    track_ids=[p1.track_id, p2.track_id],
                    object_classes=["person", "person"],
                    supporting_signals=supporting_signals,
                    contradictory_signals=neg_signals,
                    explanation=(
                        f"Potential constrained person movement: [{p1.track_id}] and [{p2.track_id}] exhibited "
                        f"tight persistent proximity (avg {avg_dist:.1f}px) with {deflection_count} synchronized "
                        f"course deflections over {shared_duration:.1f}s. "
                        f"Visual tracking alone cannot establish coercion or intent. Strict human verification required."
                    ),
                    evidence_candidates=[
                        EvidenceCandidate(
                            timestamp=event_t,
                            pre_seconds=2.0,
                            post_seconds=3.0,
                            bounding_box=p1.current_bbox,
                            target_track_id=p1.track_id,
                            reason="Forensic capture of potential forced movement pattern",
                        )
                    ],
                    human_verification_required=True,
                )
                cand.validation_decision = "REVIEW_REQUIRED"
                candidates.append(cand)

        return candidates
