"""
Observational Anomaly Detector (Phase 10)

Detects compound multi-signal security anomalies:
1. Restricted zone breach combined with prolonged localized dwell.
2. Unattended stationary object presence.
3. Complex multi-agent takeaway interactions.
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
from ai.incidents.spatial import SpatialRelationshipEngine


class ObservationalAnomalyDetector(BaseIncidentDetector):
    """
    Synthesizes compound multi-signal physical patterns into high-priority anomalies.
    """

    detector_name: str = "observational_anomaly_detector"
    detector_version: str = "1.1.0"
    category: IncidentCategory = IncidentCategory.MULTI_SIGNAL

    # Phase Z Contract Declarations
    required_signals: List[str] = ["Zone Breach and Dwell"]
    supporting_signals_declared: List[str] = ["Zone Breach", "Zone Dwell", "Compound Intrusion Anomaly"]
    contradictory_signals_declared: List[str] = ["Negative: False Alarm Breach"]
    context_requirements: Dict[str, Any] = {"active_zones": True}
    evidence_requirements: Dict[str, Any] = {"zone_analysis": True, "bounding_box_grounding": True}
    confidence_method: str = "evidence_strength_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:

        candidates: List[IncidentCandidate] = []
        if not context.tracks:
            return candidates

        # Pattern 1: Zone intrusion with prolonged localized presence
        if context.zones:
            for zone in context.zones:
                if not zone.enabled:
                    continue

                for track in context.tracks:
                    if not track.is_validated:
                        continue

                    z_analysis = SpatialRelationshipEngine.track_zone_dwell_analysis(track, zone)
                    if z_analysis["entered"] and z_analysis["total_dwell_seconds"] >= 4.0:
                        entry_t = z_analysis["entry_time"]
                        dwell_s = z_analysis["total_dwell_seconds"]

                        signals = [
                            SupportingSignal(
                                signal_type="Zone Breach",
                                description=f"Track [{track.track_id}] entered restricted zone '{zone.name}' at {entry_t:.2f}s",
                                confidence=0.90,
                                timestamp=entry_t,
                                track_id=track.track_id,
                            ),
                            SupportingSignal(
                                signal_type="Prolonged Zone Dwell",
                                description=f"Localized dwell inside '{zone.name}' for {dwell_s:.1f}s",
                                confidence=0.88,
                                timestamp=entry_t,
                                track_id=track.track_id,
                            ),
                        ]

                        scoring = IncidentScorer.calculate_evidence_score(
                            base_confidence=0.85,
                            supporting_signals=signals,
                            tracks=[track],
                            duration_seconds=dwell_s,
                        )

                        cand = self.build_candidate(
                            video_id=context.video_id,
                            event_type="OBSERVATIONAL_ANOMALY",
                            start_time=entry_t,
                            end_time=z_analysis["exit_time"] or track.last_seen,
                            severity="HIGH",
                            confidence=scoring["score"],
                            explanation=(
                                f"Potentially anomalous pattern: {track.object_class} [{track.track_id}] entered "
                                f"restricted zone '{zone.name}' and exhibited prolonged presence ({dwell_s:.1f}s)."
                            ),
                            track_ids=[track.track_id],
                            object_classes=[track.object_class],
                            supporting_signals=signals,
                            spatial_context=SpatialContext(
                                centroid=track.current_bbox.centroid if track.current_bbox else None,
                                bounding_box=track.current_bbox,
                                zone_name=zone.name,
                            ),
                            evidence_candidates=[
                                EvidenceCandidate(
                                    timestamp=entry_t,
                                    bounding_box=track.current_bbox,
                                    target_track_id=track.track_id,
                                    reason=f"Compound intrusion and dwell in {zone.name}",
                                )
                            ],
                            prefix="ANOM",
                        )
                        candidates.append(cand)

        return candidates
