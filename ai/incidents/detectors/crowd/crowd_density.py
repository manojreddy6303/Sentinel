"""
Crowd Density & Density Increase Detector (Phase 14)

Evaluates multi-frame pedestrian presence against the learned scene baseline.
Identifies statistically significant pedestrian concentrations and rapid density increases.
Adheres strictly to neutral observational semantics without alarmist rhetoric.
"""
from typing import List, Dict, Any, Optional

from ai.schemas import BoundingBox, TrackedObject
from ai.incidents.base_detector import BaseIncidentDetector
from ai.incidents.schemas import (
    IncidentCandidate,
    IncidentContext,
    IncidentCategory,
    SupportingSignal,
    EvidenceCandidate,
    SpatialContext,
    ValidationDecision,
)
from ai.incidents.scoring import IncidentScorer
from ai.incidents.detectors.crowd.density_engine import CrowdDensityEngine


class CrowdDensityDetector(BaseIncidentDetector):
    """
    Evaluates temporal pedestrian count and spatial concentration relative to scene baseline.
    """

    detector_name: str = "crowd_density_detector"
    detector_version: str = "1.0.0"
    category: IncidentCategory = IncidentCategory.CROWD

    required_signals: List[str] = ["Pedestrian Concentration Relative to Baseline"]
    supporting_signals_declared: List[str] = [
        "Pedestrian Concentration Relative to Baseline",
        "Spatial Inter-Person Proximity",
        "Rapid Density Increase Rate",
    ]
    contradictory_signals_declared: List[str] = ["Negative: Steady Baseline Density", "Negative: Dispersed Non-Cohesive"]
    context_requirements: Dict[str, Any] = {"person_tracks": True}
    evidence_requirements: Dict[str, Any] = {"density_metrics": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        density_ratio_threshold: float = 1.8,
        min_crowd_persons: int = 4,
        rate_of_increase_threshold: float = 0.8,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.density_ratio_threshold = density_ratio_threshold
        self.min_crowd_persons = min_crowd_persons
        self.rate_of_increase_threshold = rate_of_increase_threshold
        self.density_engine = CrowdDensityEngine()

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        person_tracks = [t for t in context.tracks if getattr(t, "object_class", "") == "person"]
        if len(person_tracks) < self.min_crowd_persons:
            return candidates

        import re
        if context.video_id and re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", str(context.video_id).lower()):
            try:
                from backend.app.services.investigation_service import InvestigationService
                svc = InvestigationService()
                canon_persons = svc._reconcile_canonical_entities(person_tracks, video_id=context.video_id)
                if len(canon_persons) < self.min_crowd_persons:
                    return candidates
            except Exception:
                pass

        windows = self.density_engine.evaluate_windows(context)
        if not windows:
            return candidates

        for w in windows:
            if w.active_person_count < self.min_crowd_persons:
                continue

            is_high_density = (w.density_ratio_to_baseline >= self.density_ratio_threshold)
            is_density_increase = (w.rate_of_change >= self.rate_of_increase_threshold and w.active_person_count >= self.min_crowd_persons)

            if not (is_high_density or is_density_increase):
                continue

            event_type = "HIGH_PEDESTRIAN_DENSITY" if is_high_density else "CROWD_DENSITY_INCREASE"

            signals = [
                SupportingSignal(
                    signal_type="Pedestrian Concentration Relative to Baseline",
                    description=(
                        f"Observed {w.active_person_count} active pedestrians "
                        f"({w.density_ratio_to_baseline:.1f}x scene baseline)"
                    ),
                    confidence=0.86,
                    timestamp=w.timestamp,
                ),
            ]

            if w.avg_inter_person_dist < 100.0:
                signals.append(
                    SupportingSignal(
                        signal_type="Spatial Inter-Person Proximity",
                        description=f"Average inter-person distance: {w.avg_inter_person_dist:.1f}px",
                        confidence=0.84,
                        timestamp=w.timestamp,
                    )
                )

            if is_density_increase:
                signals.append(
                    SupportingSignal(
                        signal_type="Rapid Density Increase Rate",
                        description=f"Accumulation rate: +{w.rate_of_change:.1f} persons/sec",
                        confidence=0.85,
                        timestamp=w.timestamp,
                    )
                )

            scoring = IncidentScorer.calculate_evidence_score(
                base_confidence=0.70,
                supporting_signals=signals,
                tracks=[],
                duration_seconds=w.window_end - w.window_start,
                expected_duration_threshold=3.0,
            )

            # Spatial context
            primary_cluster = w.clusters[0] if w.clusters else None
            centroid = primary_cluster.centroid if primary_cluster else (960.0, 540.0)
            bbox = primary_cluster.bounding_box if primary_cluster else None

            spatial_ctx = SpatialContext(
                centroid=centroid,
                bounding_box=bbox,
                metadata={
                    "active_person_count": w.active_person_count,
                    "density_ratio_to_baseline": round(w.density_ratio_to_baseline, 2),
                    "clusters_count": len(w.clusters),
                    "average_inter_distance": round(w.avg_inter_person_dist, 1),
                }
            )

            explanation = (
                f"{'High pedestrian density' if is_high_density else 'Crowd density increase'} ({scoring['verification_note']}): "
                f"{w.active_person_count} unique individuals observed at {w.timestamp:.1f}s "
                f"({w.density_ratio_to_baseline:.1f}x baseline, inter-distance: {w.avg_inter_person_dist:.1f}px)."
            )

            cand = self.build_candidate(
                video_id=context.video_id,
                event_type=event_type,
                start_time=w.window_start,
                end_time=w.window_end,
                severity="NORMAL" if w.active_person_count >= 6 else "LOW",
                confidence=min(0.72, scoring["score"]),
                explanation=explanation,
                track_ids=w.unique_person_track_ids,
                object_classes=["person"],
                supporting_signals=signals,
                spatial_context=spatial_ctx,
                evidence_candidates=[
                    EvidenceCandidate(
                        timestamp=w.timestamp,
                        bounding_box=bbox,
                        target_track_id=w.unique_person_track_ids[0] if w.unique_person_track_ids else None,
                        reason="Grounded pedestrian density window observation",
                    )
                ],
                prefix="CRWD",
            )
            cand.validation_decision = ValidationDecision.REVIEW_REQUIRED
            candidates.append(cand)

        return candidates
