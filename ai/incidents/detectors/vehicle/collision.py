"""
Potential Vehicle Collision Detector (Sentinel Phase 11)

Two-stage generalized vehicle collision detector:
Stage 1: Generates collision candidates only when meaningful motion and proximity signals exist.
Stage 2: Verifies physical contact geometry, motion discontinuity, deceleration, and post-event kinematics.
Evaluates physical counter-evidence via NegativeEvidenceEngine and abstains on normal traffic passage.
"""
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
from ai.incidents.detectors.vehicle.interaction import VehicleInteractionModel
from ai.incidents.detectors.vehicle.sampling import AdaptiveTemporalSamplingEngine


class VehicleCollisionDetector(BaseIncidentDetector):
    """
    Two-stage perspective-aware detector for potential vehicle collisions.
    Requires independent physical evidence:
    - Rapid convergence
    - Verified physical contact (bounding-box overlap or edge contact)
    - Motion discontinuity or sudden post-impact deceleration
    - Absence of strong negative evidence (lane following, continuous transit, clearance)
    """

    detector_name: str = "vehicle_collision_detector"
    detector_version: str = "2.2.0"
    category: IncidentCategory = IncidentCategory.VEHICLE

    # Phase Z Contract Declarations
    required_signals: List[str] = ["Rapid Vehicle Convergence", "Physical Contact or Motion Discontinuity"]
    supporting_signals_declared: List[str] = [
        "Rapid Vehicle Convergence",
        "Physical Contact / Bounding Box Overlap",
        "Abrupt Deceleration / Stoppage",
        "Trajectory Intersection",
        "Post-Impact Deflection",
        "Adaptive Burst Sampling Verified",
    ]
    contradictory_signals_declared: List[str] = [
        "Negative: Zero Physical Contact",
        "Negative: Continued Normal Transit",
        "Negative: Parallel Lane Following",
    ]
    context_requirements: Dict[str, Any] = {"scene_type_priors": {"roadway": "strong_negative_prior_without_stoppage"}}
    evidence_requirements: Dict[str, Any] = {"verified_tracks": True, "bounding_box_grounding": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        min_approach_rate: float = 35.0,
        proximity_threshold_px: float = 60.0,
        deceleration_threshold: float = 60.0,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.min_approach_rate = min_approach_rate
        self.proximity_threshold_px = proximity_threshold_px
        self.deceleration_threshold = deceleration_threshold
        self.vehicle_classes = {"car", "bus", "truck", "motorcycle", "van"}
        self.sampling_engine = AdaptiveTemporalSamplingEngine()

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        vehicles = [t for t in context.tracks if t.object_class in self.vehicle_classes and t.is_validated]
        if len(vehicles) < 2:
            return candidates

        checked_pairs = set()
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                v1, v2 = vehicles[i], vehicles[j]
                pair_key = (min(v1.track_id, v2.track_id), max(v1.track_id, v2.track_id))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                # Step 1: Pairwise Interaction Analysis
                interaction = VehicleInteractionModel.analyze_interaction(
                    v1, v2, context_motions=context.track_motions
                )
                if not interaction:
                    continue

                min_dist = interaction["min_distance_px"]
                max_approach = interaction["max_approach_rate_px_s"]
                event_t = interaction["event_time"]
                has_physical_contact = interaction["has_physical_contact"]
                max_iou = interaction["max_iou"]
                is_parallel = interaction["is_parallel_flow"]
                v1_drop = interaction["v1_velocity_drop"]
                v2_drop = interaction["v2_velocity_drop"]
                has_deceleration = (v1_drop >= self.deceleration_threshold) or (v2_drop >= self.deceleration_threshold)

                # Motion pre-filter: must have rapid convergence and close proximity
                if min_dist > self.proximity_threshold_px or max_approach < self.min_approach_rate:
                    continue

                # Step 2: Negative Evidence Evaluation (CRITICAL)
                neg_signals = NegativeEvidenceEngine.evaluate_collision_negative_evidence(
                    v1, v2, context, event_t
                )

                neg_signal_names = {s.signal_type for s in neg_signals}
                no_physical_contact = any("Zero Physical Contact" in s for s in neg_signal_names)
                continued_normal = any("Continued Normal Transit" in s for s in neg_signal_names)

                # UNIVERSAL ABSTENTION:
                # If clearance was maintained throughout AND vehicles continued transit at speed,
                # this is normal highway traffic. ABSTAIN.
                if no_physical_contact and continued_normal:
                    continue

                if is_parallel and no_physical_contact and not has_deceleration:
                    continue

                if not has_deceleration and no_physical_contact:
                    continue

                # Step 3: Adaptive Temporal Sufficiency Assessment
                temporal_check = self.sampling_engine.assess_temporal_sufficiency(
                    actual_sample_rate_fps=context.sample_rate_fps,
                    target_velocity=max_approach,
                )

                # Supporting Signals
                signals = [
                    SupportingSignal(
                        signal_type="Rapid Vehicle Convergence",
                        description=f"Vehicles [{v1.track_id}] and [{v2.track_id}] converged at {max_approach:.1f}px/s to proximity {min_dist:.1f}px",
                        confidence=0.85,
                        timestamp=event_t,
                    ),
                ]

                if has_physical_contact:
                    signals.append(
                        SupportingSignal(
                            signal_type="Physical Contact / Overlap",
                            description=f"Bounding-box contact verified (max IoU: {max_iou:.2f}) between {v1.object_class} and {v2.object_class}",
                            confidence=0.92,
                            timestamp=event_t,
                        )
                    )

                if has_deceleration:
                    signals.append(
                        SupportingSignal(
                            signal_type="Abrupt Deceleration / Stoppage",
                            description=f"Velocity discontinuity detected post-convergence (drop: v1={v1_drop:.1f}px/s, v2={v2_drop:.1f}px/s)",
                            confidence=0.88,
                            timestamp=event_t,
                        )
                    )

                if temporal_check["is_limited"]:
                    signals.append(
                        SupportingSignal(
                            signal_type="Temporal Resolution Notice",
                            description=temporal_check["limitation_reason"],
                            confidence=0.50,
                            timestamp=event_t,
                        )
                    )

                scoring = IncidentScorer.calculate_evidence_score(
                    base_confidence=0.70,
                    supporting_signals=signals,
                    tracks=[v1, v2],
                )

                cpa_box1 = interaction["cpa_box1"]
                cpa_box2 = interaction["cpa_box2"]

                severity = "HIGH" if (has_physical_contact and has_deceleration) else "NORMAL"

                cand = self.build_candidate(
                    video_id=context.video_id,
                    event_type="POTENTIAL_VEHICLE_COLLISION",
                    start_time=max(0.0, event_t - 1.5),
                    end_time=event_t + 2.5,
                    severity=severity,
                    confidence=scoring["score"],
                    explanation=(
                        f"Potential Vehicle Collision candidate: {v1.object_class} [{v1.track_id}] and "
                        f"{v2.object_class} [{v2.track_id}] converged ({max_approach:.1f}px/s, {min_dist:.1f}px) at {event_t:.1f}s."
                    ),
                    track_ids=[v1.track_id, v2.track_id],
                    object_classes=[v1.object_class, v2.object_class],
                    supporting_signals=signals,
                    contradictory_signals=neg_signals,
                    spatial_context=SpatialContext(
                        centroid=cpa_box1.centroid if cpa_box1 else None,
                        bounding_box=cpa_box1,
                        inter_track_distances={v2.track_id: round(min_dist, 2)},
                        metadata={
                            "v1_track_id": v1.track_id,
                            "v2_track_id": v2.track_id,
                            "temporal_evidence_tag": temporal_check["temporal_evidence_tag"],
                            "normalized_distance": interaction["min_normalized_distance"],
                        },
                    ),
                    evidence_candidates=[
                        EvidenceCandidate(
                            timestamp=event_t,
                            bounding_box=cpa_box1,
                            target_track_id=v1.track_id,
                            reason="Point of closest approach for potential vehicle collision",
                        )
                    ],
                    prefix="CRASH",
                )
                candidates.append(cand)

        return candidates
