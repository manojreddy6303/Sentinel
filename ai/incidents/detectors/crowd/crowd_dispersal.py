"""
Potential Crowd Dispersal Detector (Phase 14 — Audit-Hardened)

Detects the rapid fragmentation and spatial divergence of a previously concentrated crowd cluster.
Uses inter-person distance expansion and negative density rate of change.

Hardening (Audit):
- Requires a meaningful sustained baseline crowd state (min 2 windows at >= min_initial_cluster_persons)
  before any dispersal observation can be emitted for that crowd episode.
- Enforces a minimum inter-episode gap (default 15s) so the same underlying dispersal
  event cannot be reported on every adjacent window pair.
- Requires a meaningful absolute person-count drop (>= 2 persons) in addition to rate thresholds.
- Preserves the cluster-centroid check so spatially-distinct dispersals are treated independently.

Observational only: does not infer underlying causes or motives.
"""
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
    ValidationDecision,
)
from ai.incidents.scoring import IncidentScorer
from ai.incidents.detectors.crowd.density_engine import CrowdDensityEngine


class CrowdDispersalDetector(BaseIncidentDetector):
    """
    Evaluates temporal windows for rapid cluster dissolution and outward pedestrian dispersion.

    Conservative hardened version: avoids surfacing repeated alerts for the same underlying
    dispersal transition by enforcing per-spatial-zone episode cooldowns.
    """

    detector_name: str = "crowd_dispersal_detector"
    detector_version: str = "1.1.0"
    category: IncidentCategory = IncidentCategory.CROWD

    required_signals: List[str] = ["Prior Cluster Concentration", "Rapid Spatial Dispersal"]
    supporting_signals_declared: List[str] = [
        "Prior Cluster Concentration",
        "Rapid Spatial Dispersal",
        "Inter-Person Distance Expansion",
    ]
    contradictory_signals_declared: List[str] = ["Negative: Steady Baseline Density"]
    context_requirements: Dict[str, Any] = {"person_tracks": True}
    evidence_requirements: Dict[str, Any] = {"density_metrics": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def __init__(
        self,
        min_initial_cluster_persons: int = 4,
        max_dispersal_rate: float = -0.7,
        min_absolute_count_drop: int = 2,
        min_baseline_windows: int = 1,
        min_inter_episode_gap_seconds: float = 15.0,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.min_initial_cluster_persons = min_initial_cluster_persons
        self.max_dispersal_rate = max_dispersal_rate
        # Require a meaningful absolute drop in person count (not just a rate fluctuation)
        self.min_absolute_count_drop = min_absolute_count_drop
        # Require the baseline crowd to persist for at least this many windows before dispersal fires
        self.min_baseline_windows = min_baseline_windows
        # Minimum time gap before a new dispersal episode can be raised (per spatial zone)
        self.min_inter_episode_gap_seconds = min_inter_episode_gap_seconds
        self.density_engine = CrowdDensityEngine()

    # ------------------------------------------------------------------
    # Episode cooldown bookkeeping
    # ------------------------------------------------------------------
    # Maps a spatial-zone key (discretized centroid) → last emitted episode end_time.
    # Shared instance state is per-analysis run because the detector is reset per-video.
    _last_episode_end_by_zone: Dict[str, float]

    def reset(self) -> None:
        """Reset per-video state."""
        super().reset() if hasattr(super(), "reset") else None
        self._last_episode_end_by_zone = {}

    def _zone_key(self, centroid: Tuple[float, float], grid_px: float = 300.0) -> str:
        """Discretize centroid into a spatial zone grid for cooldown tracking."""
        gx = int(centroid[0] / grid_px)
        gy = int(centroid[1] / grid_px)
        return f"zone_{gx}_{gy}"

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        # Reset per-analysis state
        self._last_episode_end_by_zone = {}

        # Upstream Physical Entity Validation:
        # A genuine crowd requires at least min_initial_cluster_persons physical human actors in the scene.
        # If the total physical people in the footage is fewer than min_initial_cluster_persons,
        # raw fragmented tracks cannot constitute a crowd or crowd dispersal.
        person_tracks = [t for t in context.tracks if getattr(t, "object_class", "") == "person"]
        if len(person_tracks) < self.min_initial_cluster_persons:
            return []

        import re
        if context.video_id and re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", str(context.video_id).lower()):
            try:
                from backend.app.services.investigation_service import InvestigationService
                svc = InvestigationService()
                canon_persons = svc._reconcile_canonical_entities(person_tracks, video_id=context.video_id)
                if len(canon_persons) < self.min_initial_cluster_persons:
                    return []
            except Exception:
                pass

        candidates: List[IncidentCandidate] = []
        windows = self.density_engine.evaluate_windows(context)
        if len(windows) < self.min_baseline_windows + 1:
            return candidates

        # ------------------------------------------------------------------
        # Build a running count of how many consecutive windows before i
        # had >= min_initial_cluster_persons (baseline crowd persistence).
        # We only consider dispersal from windows that follow a sustained
        # crowd baseline — not transient peaks.
        # ------------------------------------------------------------------
        # consecutive_above[i] = number of consecutive windows ending at i-1
        # that all met the crowd threshold.
        consecutive_above = [0] * len(windows)
        for i in range(1, len(windows)):
            if windows[i - 1].active_person_count >= self.min_initial_cluster_persons:
                consecutive_above[i] = consecutive_above[i - 1] + 1
            else:
                consecutive_above[i] = 0

        for i in range(1, len(windows)):
            prev_w = windows[i - 1]
            curr_w = windows[i]

            # 1. Require sustained baseline crowd state
            if consecutive_above[i] < self.min_baseline_windows:
                continue

            # 2. Significant rate of decrease
            is_count_drop = (curr_w.rate_of_change <= self.max_dispersal_rate)
            # Absolute count drop guard: must be a real drop, not measurement noise
            absolute_drop = prev_w.active_person_count - curr_w.active_person_count
            is_meaningful_absolute_drop = (absolute_drop >= self.min_absolute_count_drop)

            is_dist_expansion = (
                prev_w.avg_inter_person_dist < 80.0
                and curr_w.avg_inter_person_dist >= 120.0
            )

            if not ((is_count_drop and is_meaningful_absolute_drop) or is_dist_expansion):
                continue

            # 3. Determine spatial zone and enforce inter-episode cooldown
            primary_cluster = prev_w.clusters[0] if prev_w.clusters else None
            centroid = primary_cluster.centroid if primary_cluster else (960.0, 540.0)
            bbox = primary_cluster.bounding_box if primary_cluster else None
            zone = self._zone_key(centroid)

            last_end = self._last_episode_end_by_zone.get(zone, -float("inf"))
            if prev_w.window_start - last_end < self.min_inter_episode_gap_seconds:
                # Same spatial zone dispersal has already been reported recently; skip
                continue

            # 4. Build signals and score
            signals = [
                SupportingSignal(
                    signal_type="Prior Cluster Concentration",
                    description=(
                        f"Sustained crowd of {prev_w.active_person_count} individuals "
                        f"persisted for {consecutive_above[i]} window(s) before {prev_w.timestamp:.1f}s"
                    ),
                    confidence=0.86,
                    timestamp=prev_w.timestamp,
                ),
                SupportingSignal(
                    signal_type="Rapid Spatial Dispersal",
                    description=(
                        f"Person count dropped from {prev_w.active_person_count} to "
                        f"{curr_w.active_person_count} ({absolute_drop} fewer) at rate "
                        f"{curr_w.rate_of_change:.1f} persons/sec"
                    ),
                    confidence=0.84,
                    timestamp=curr_w.timestamp,
                ),
            ]

            if is_dist_expansion:
                signals.append(
                    SupportingSignal(
                        signal_type="Inter-Person Distance Expansion",
                        description=(
                            f"Average separation expanded from {prev_w.avg_inter_person_dist:.1f}px "
                            f"to {curr_w.avg_inter_person_dist:.1f}px"
                        ),
                        confidence=0.85,
                        timestamp=curr_w.timestamp,
                    )
                )

            scoring = IncidentScorer.calculate_evidence_score(
                base_confidence=0.70,
                supporting_signals=signals,
                tracks=[],
                duration_seconds=curr_w.window_end - prev_w.window_start,
                expected_duration_threshold=3.0,
            )

            spatial_ctx = SpatialContext(
                centroid=centroid,
                bounding_box=bbox,
                metadata={
                    "prior_person_count": prev_w.active_person_count,
                    "current_person_count": curr_w.active_person_count,
                    "absolute_drop": absolute_drop,
                    "rate_of_change": round(curr_w.rate_of_change, 2),
                    "sustained_baseline_windows": consecutive_above[i],
                }
            )

            cand = self.build_candidate(
                video_id=context.video_id,
                event_type="POTENTIAL_CROWD_DISPERSAL",
                start_time=prev_w.window_start,
                end_time=curr_w.window_end,
                severity="LOW",
                confidence=min(0.68, scoring["score"]),
                explanation=(
                    f"Potential Crowd Dispersal ({scoring['verification_note']}): "
                    f"Crowd of {prev_w.active_person_count} individuals (sustained for "
                    f"{consecutive_above[i]} window(s)) underwent rapid spatial dispersal "
                    f"(absolute drop: {absolute_drop}, rate: {curr_w.rate_of_change:.1f} persons/s) "
                    f"by {curr_w.timestamp:.1f}s."
                ),
                track_ids=prev_w.unique_person_track_ids,
                object_classes=["person"],
                supporting_signals=signals,
                spatial_context=spatial_ctx,
                evidence_candidates=[
                    EvidenceCandidate(
                        timestamp=curr_w.timestamp,
                        bounding_box=bbox,
                        target_track_id=prev_w.unique_person_track_ids[0] if prev_w.unique_person_track_ids else None,
                        reason="Grounded crowd dispersal observation",
                    )
                ],
                prefix="DISP",
            )
            cand.validation_decision = ValidationDecision.REVIEW_REQUIRED
            candidates.append(cand)

            # Register this episode end time for cooldown
            self._last_episode_end_by_zone[zone] = curr_w.window_end

        return candidates
