"""
Zone Intrusion Detector (Phase 10)

Detects boundary breaches and unauthorized presence within defined security zones.
Integrates SpatialRelationshipEngine to evaluate polygon entries.
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
    TemporalContext,
)
from ai.incidents.spatial import SpatialRelationshipEngine
from ai.incidents.scoring import IncidentScorer
from ai.incidents.negative_evidence import NegativeEvidenceEngine


class ZoneIntrusionDetector(BaseIncidentDetector):
    """
    Evaluates tracks against active ZoneDefinitions for perimeter violations.
    """

    detector_name: str = "zone_intrusion_detector"
    detector_version: str = "1.1.0"
    category: IncidentCategory = IncidentCategory.ZONE

    # Phase Z Contract Declarations
    required_signals: List[str] = ["Zone Breach"]
    supporting_signals_declared: List[str] = ["Zone Breach", "Zone Dwell"]
    contradictory_signals_declared: List[str] = ["Negative: Unconfigured Zone", "Negative: Unverified Track"]
    context_requirements: Dict[str, Any] = {"active_zones": True}
    evidence_requirements: Dict[str, Any] = {"zone_polygon": True, "entry_bounding_box": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        if not context.zones or not context.tracks:
            return candidates


        for zone in context.zones:
            if not zone.enabled:
                continue

            for track in context.tracks:
                if not track.is_validated:
                    continue

                # Filter target classes
                if zone.target_classes and track.object_class not in zone.target_classes:
                    continue

                dwell_info = SpatialRelationshipEngine.track_zone_dwell_analysis(track, zone)
                if not dwell_info["entered"]:
                    continue

                entry_t = dwell_info["entry_time"]
                dwell_sec = dwell_info["total_dwell_seconds"]

                # Observable signals
                signals = [
                    SupportingSignal(
                        signal_type="Zone Breach",
                        description=f"{track.object_class} [{track.track_id}] crossed boundary into '{zone.name}' at {entry_t:.2f}s",
                        confidence=track.confidence,
                        timestamp=entry_t,
                        track_id=track.track_id,
                        metadata={"zone_id": zone.zone_id, "zone_name": zone.name},
                    ),
                    SupportingSignal(
                        signal_type="Zone Dwell",
                        description=f"Dwell duration inside restricted zone: {dwell_sec:.1f}s ({dwell_info['inside_observations_count']} observations)",
                        confidence=min(0.95, 0.70 + (dwell_sec / 20.0)),
                        timestamp=entry_t,
                        track_id=track.track_id,
                    ),
                ]

                # Score
                scoring = IncidentScorer.calculate_evidence_score(
                    base_confidence=0.85,
                    supporting_signals=signals,
                    tracks=[track],
                    duration_seconds=dwell_sec,
                )

                # Nearest bbox at entry timestamp
                entry_bbox = track.current_bbox
                if track.history_bboxes:
                    closest = min(track.history_bboxes, key=lambda b: abs(b["timestamp"] - entry_t))
                    b_dict = closest["bbox"]
                    from ai.schemas import BoundingBox
                    entry_bbox = BoundingBox(
                        x1=float(b_dict["x1"]),
                        y1=float(b_dict["y1"]),
                        x2=float(b_dict["x2"]),
                        y2=float(b_dict["y2"]),
                    )

                spatial_ctx = SpatialContext(
                    centroid=entry_bbox.centroid if entry_bbox else None,
                    bounding_box=entry_bbox,
                    zone_name=zone.name,
                )

                # Negative Evidence evaluation
                neg_signals = NegativeEvidenceEngine.evaluate_intrusion_negative_evidence(track, zone, context)
                if any(s.signal_type == "Negative: Unconfigured Zone" for s in neg_signals):
                    continue

                cand = self.build_candidate(
                    video_id=context.video_id,
                    event_type="POTENTIAL_INTRUSION",
                    start_time=entry_t,
                    end_time=dwell_info["exit_time"] or track.last_seen,
                    severity="HIGH",
                    confidence=scoring["score"],
                    explanation=(
                        f"Potential Intrusion: {track.object_class} [{track.track_id}] breached restricted "
                        f"zone '{zone.name}' at {entry_t:.1f}s (dwell: {dwell_sec:.1f}s)."
                    ),
                    track_ids=[track.track_id],
                    object_classes=[track.object_class],
                    supporting_signals=signals,
                    contradictory_signals=neg_signals,
                    spatial_context=spatial_ctx,
                    evidence_candidates=[
                        EvidenceCandidate(
                            timestamp=entry_t,
                            bounding_box=entry_bbox,
                            target_track_id=track.track_id,
                            reason=f"Intrusion entry into {zone.name}",
                        )
                    ],
                    prefix="INTRU",
                )
                candidates.append(cand)

        return candidates
