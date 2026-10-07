"""
Potential Person Fall Detector (Sentinel Phase 12)

Detects observable person fall patterns over time:
- Substantial downward centroid displacement and downward velocity
- Transition from upright aspect ratio (< 0.65) to horizontal/low geometry (>= 0.85)
- Motion arrest / velocity reduction immediately following descent
- Multi-signal validation rejecting normal sitting, crouching, or brief occlusions
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
    TemporalContext,
)
from ai.incidents.scoring import IncidentScorer
from ai.incidents.negative_evidence import NegativeEvidenceEngine
from ai.incidents.detectors.person.motion_features import PersonMotionFeatureEngine
from ai.incidents.detectors.person.pose_features import PoseFeatureEngine


class PersonFallDetector(BaseIncidentDetector):
    """
    Evaluates tracked person dynamics for potential fall incidents.
    """

    detector_name: str = "person_fall_detector"
    detector_version: str = "1.0.0"
    category: IncidentCategory = IncidentCategory.PERSON

    required_signals: List[str] = ["Downward Descent", "Aspect Ratio Inversion"]
    supporting_signals_declared: List[str] = [
        "Rapid Downward Velocity",
        "Aspect Ratio Inversion (Upright to Horizontal)",
        "Post-Descent Motion Arrest",
        "Ground Contact Persistence",
    ]
    contradictory_signals_declared: List[str] = [
        "Negative: Resumed Upright Walking",
        "Negative: Seated Bench Context",
        "Negative: Upright Posture Maintained",
    ]
    context_requirements: Dict[str, Any] = {"pedestrian_context": True}
    evidence_requirements: Dict[str, Any] = {"verified_track": True, "bounding_box_grounding": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        min_downward_velocity_px_s: float = 30.0,
        min_aspect_ratio_delta: float = 0.30,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.min_downward_velocity_px_s = min_downward_velocity_px_s
        self.min_aspect_ratio_delta = min_aspect_ratio_delta

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        persons = [t for t in context.tracks if t.object_class == "person" and t.is_validated]

        fw = None
        fh = None
        if hasattr(context, "frame_dimensions") and context.frame_dimensions:
            fw, fh = context.frame_dimensions
        elif hasattr(context, "video_metadata") and context.video_metadata:
            fw = float(context.video_metadata.get("width") or 0.0) or None
            fh = float(context.video_metadata.get("height") or 0.0) or None

        for track in persons:
            if track.duration_seconds < 0.8:
                continue

            dynamics = PersonMotionFeatureEngine.compute_person_dynamics(
                track, context.track_motions, frame_width=fw, frame_height=fh
            )
            has_ar_trans = dynamics.get("has_aspect_ratio_transition", False)
            has_persistence = dynamics.get("has_posture_persistence", False)
            down_vel = dynamics.get("max_downward_velocity_px_s", 0.0)
            down_bl_s = dynamics.get("max_downward_velocity_bl_s", 0.0)
            event_t = dynamics.get("downward_event_time") or track.first_seen

            # 1. Camera edge clipping & exit suppression:
            if (
                dynamics.get("is_edge_clipped", False)
                or dynamics.get("is_transition_edge_clipped", False)
                or dynamics.get("is_track_terminating_at_boundary", False)
            ):
                continue

            # 2. Negative evidence check:
            neg_signals = NegativeEvidenceEngine.evaluate_fall_negative_evidence(
                track, context, event_t, frame_width=fw, frame_height=fh
            )
            neg_names = {s.signal_type for s in neg_signals}
            if any(k in s for s in neg_names for k in [
                "Resumed Upright Walking",
                "Frame Boundary Truncation",
                "Track Termination At Frame Exit",
                "Insufficient Post-Descent Persistence",
                "Upright Posture Maintained",
            ]):
                continue

            # 3. Temporal Posture Persistence:
            if not has_persistence:
                continue

            # 4. Scale-Normalized Downward Velocity (must drop >= 0.65 body-lengths/s and >= 20.0 px/s):
            has_downward_motion = down_bl_s >= 0.65 and down_vel >= 20.0

            # Fall candidate requires both verified aspect ratio transition and normalized descent
            is_fall_candidate = has_ar_trans and has_downward_motion

            if not is_fall_candidate:
                continue

            supporting_signals = [
                SupportingSignal(
                    signal_type="Rapid Downward Velocity",
                    description=f"Person exhibited downward descent rate of {down_vel:.1f}px/s ({down_bl_s:.1f} body-heights/s)",
                    confidence=0.82,
                    timestamp=event_t,
                    track_id=track.track_id,
                ),
                SupportingSignal(
                    signal_type="Aspect Ratio Inversion",
                    description=(
                        f"Aspect ratio transitioned from {dynamics['initial_aspect_ratio']:.2f} (upright) "
                        f"to {dynamics['max_aspect_ratio']:.2f} (horizontal/low)"
                    ),
                    confidence=0.85,
                    timestamp=event_t,
                    track_id=track.track_id,
                ),
            ]

            if dynamics.get("is_low_mobility", False):
                supporting_signals.append(
                    SupportingSignal(
                        signal_type="Post-Descent Motion Arrest",
                        description=f"Person remained at low mobility for {dynamics['horizontal_dwell_seconds']:.1f}s post-descent",
                        confidence=0.80,
                        timestamp=event_t,
                        track_id=track.track_id,
                    )
                )

            pose_status = PoseFeatureEngine.get_status()
            if not pose_status["is_available"]:
                supporting_signals.append(
                    SupportingSignal(
                        signal_type="Kinematic Bounding-Box Inference",
                        description="Pose estimation unavailable; finding derived from aspect-ratio and trajectory telemetry",
                        confidence=0.60,
                        timestamp=event_t,
                        track_id=track.track_id,
                    )
                )

            scoring = IncidentScorer.calculate_evidence_score(
                base_confidence=0.72,
                supporting_signals=supporting_signals,
                tracks=[track],
                duration_seconds=track.duration_seconds,
                contradictory_signals=neg_signals,
            )

            cand = self.build_candidate(
                video_id=context.video_id,
                event_type="POTENTIAL_PERSON_FALL",
                start_time=max(0.0, event_t - 0.5),
                end_time=min(context.duration_seconds, event_t + 2.5),
                severity="HIGH",
                confidence=scoring["confidence"],
                track_ids=[track.track_id],
                object_classes=["person"],
                supporting_signals=supporting_signals,
                contradictory_signals=neg_signals,
                explanation=(
                    f"Potential person fall: [{track.track_id}] exhibited rapid downward descent ({down_vel:.1f}px/s) "
                    f"and aspect-ratio inversion to horizontal geometry. Human verification required."
                ),
                evidence_candidates=[
                    EvidenceCandidate(
                        timestamp=event_t,
                        pre_seconds=2.0,
                        post_seconds=3.0,
                        bounding_box=track.current_bbox,
                        target_track_id=track.track_id,
                        reason="Forensic capture of potential person fall",
                    )
                ],
                human_verification_required=True,
            )
            cand.validation_decision = "REVIEW_REQUIRED"
            candidates.append(cand)

        return candidates
