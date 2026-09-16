"""
Activity Analysis Detector (Phase 10)

Detects temporal activity peaks, crowd surges, and sudden density fluctuations.
Context-aware: in dense scenes, high activity is classified observationally
without false threat attribution.
"""
from collections import defaultdict
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


class ActivityAnalysisDetector(BaseIncidentDetector):
    """
    Evaluates temporal activity distribution for significant density peaks.
    """

    detector_name: str = "activity_analysis_detector"
    detector_version: str = "1.1.0"
    category: IncidentCategory = IncidentCategory.CROWD

    # Phase Z Contract Declarations
    required_signals: List[str] = ["Activity Density Surge"]
    supporting_signals_declared: List[str] = ["Activity Density Surge", "Multi-Class Co-occurrence"]
    contradictory_signals_declared: List[str] = ["Negative: Steady Baseline Density"]
    context_requirements: Dict[str, Any] = {"temporal_window_seconds": 5.0}
    evidence_requirements: Dict[str, Any] = {"detection_clusters": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        window_seconds: float = 5.0,
        density_multiplier_threshold: float = 2.0,
        min_peak_detections: int = 15,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.window_seconds = window_seconds
        self.density_multiplier_threshold = density_multiplier_threshold
        self.min_peak_detections = min_peak_detections


    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        if not context.validated_detections:
            return candidates

        # Group by window
        window_counts = defaultdict(int)
        window_classes = defaultdict(set)
        window_dets = defaultdict(list)

        for det in context.validated_detections:
            t = float(det.get("timestamp") or det.get("timestamp_seconds") or 0.0)
            w_idx = int(t // self.window_seconds)
            window_counts[w_idx] += 1
            cls = det.get("object_class")
            if cls:
                window_classes[w_idx].add(cls)
            window_dets[w_idx].append(det)

        if not window_counts:
            return candidates

        counts = list(window_counts.values())
        avg_count = sum(counts) / len(counts)
        peak_threshold = max(float(self.min_peak_detections), avg_count * self.density_multiplier_threshold)

        for w_idx, count in window_counts.items():
            if count >= peak_threshold:
                w_start = round(w_idx * self.window_seconds, 2)
                w_end = round(w_start + self.window_seconds, 2)

                classes_list = sorted(list(window_classes[w_idx]))
                sample_det = window_dets[w_idx][0] if window_dets[w_idx] else None
                sample_bbox = None
                if sample_det and sample_det.get("bounding_box"):
                    bb = sample_det["bounding_box"]
                    from ai.schemas import BoundingBox
                    sample_bbox = BoundingBox(
                        x1=float(bb.get("x1", 0.0)),
                        y1=float(bb.get("y1", 0.0)),
                        x2=float(bb.get("x2", 0.0)),
                        y2=float(bb.get("y2", 0.0)),
                    )

                signals = [
                    SupportingSignal(
                        signal_type="Activity Peak",
                        description=f"{count} detections observed in {self.window_seconds:.0f}s window (baseline avg: {avg_count:.1f})",
                        confidence=0.88,
                        timestamp=w_start,
                    ),
                    SupportingSignal(
                        signal_type="Object Classes",
                        description=f"Active classes: {', '.join(classes_list[:5])}",
                        confidence=0.90,
                        timestamp=w_start,
                    ),
                ]

                scoring = IncidentScorer.calculate_evidence_score(
                    base_confidence=0.80,
                    supporting_signals=signals,
                    duration_seconds=self.window_seconds,
                )

                spatial_ctx = SpatialContext(
                    bounding_box=sample_bbox,
                    metadata={"peak_count": count, "baseline_avg": round(avg_count, 1)},
                )

                cand = self.build_candidate(
                    video_id=context.video_id,
                    event_type="HIGH_ACTIVITY_PERIOD",
                    start_time=w_start,
                    end_time=w_end,
                    severity="NORMAL",
                    confidence=scoring["score"],
                    explanation=(
                        f"High activity period detected: {count} detections concentrated between "
                        f"{w_start:.1f}s and {w_end:.1f}s (baseline average: {avg_count:.1f}/window)."
                    ),
                    object_classes=classes_list,
                    supporting_signals=signals,
                    spatial_context=spatial_ctx,
                    evidence_candidates=[
                        EvidenceCandidate(
                            timestamp=w_start + (self.window_seconds / 2.0),
                            bounding_box=sample_bbox,
                            reason=f"High activity peak ({count} detections)",
                        )
                    ],
                    prefix="ACTV",
                )
                candidates.append(cand)

        return candidates
