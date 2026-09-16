"""
Incident Fusion Engine (Phase 10)

Combines multi-detector signals and temporal-spatial clusters into unified,
non-duplicate security incidents. Preserves all supporting signals, source tracks,
bounding coordinates, and forensic evidence candidates.
"""
from collections import defaultdict
from typing import List, Dict, Any, Optional
import math
import uuid

from ai.incidents.schemas import (
    IncidentCandidate,
    SupportingSignal,
    EvidenceCandidate,
    SpatialContext,
    TemporalContext,
    ValidationDecision,
)
from ai.incidents.spatial import SpatialRelationshipEngine
from ai.incidents.temporal import TemporalAnalysisEngine


class IncidentFusionEngine:
    """
    Fuses related and overlapping incident candidates into singular grounded incidents.
    """

    def __init__(
        self,
        time_merge_tolerance_seconds: float = 3.0,
        spatial_merge_distance: float = 120.0,
    ):
        self.time_merge_tolerance_seconds = time_merge_tolerance_seconds
        self.spatial_merge_distance = spatial_merge_distance

    def fuse_incidents(self, candidates: List[IncidentCandidate]) -> List[IncidentCandidate]:
        """
        Deduplicate and fuse raw candidates into cohesive incident records.
        """
        if not candidates:
            return []

        # 1. Arbitrate competing person interaction hypotheses on shared tracks
        candidates = self._arbitrate_competing_person_interactions(candidates)

        # 2. Arbitrate competing property hypotheses on shared object tracks
        candidates = self._arbitrate_competing_property_interactions(candidates)

        # 3. Arbitrate competing crowd hypotheses in overlapping temporal windows
        candidates = self._arbitrate_competing_crowd_interactions(candidates)

        # 4. Cross-modal fire + smoke incident fusion (Phase 15)
        candidates = self._fuse_fire_and_smoke_interactions(candidates)

        # 5. Group candidates by event_type
        by_type: Dict[str, List[IncidentCandidate]] = defaultdict(list)
        for c in candidates:
            by_type[c.event_type].append(c)

        fused_results: List[IncidentCandidate] = []

        for event_type, group in by_type.items():
            if len(group) == 1:
                fused_results.append(group[0])
                continue

            # Sort by start_time
            sorted_group = sorted(group, key=lambda x: x.start_time)
            clusters: List[List[IncidentCandidate]] = []

            for cand in sorted_group:
                placed = False
                for cluster in clusters:
                    lead = cluster[0]
                    # Check temporal overlap or close adjacency
                    if TemporalAnalysisEngine.is_temporally_overlapping(
                        lead.start_time, lead.end_time,
                        cand.start_time, cand.end_time,
                        tolerance_seconds=self.time_merge_tolerance_seconds
                    ):
                        # Check track intersection or spatial proximity
                        shared_tracks = set(lead.track_ids).intersection(set(cand.track_ids))
                        spatial_near = False
                        if lead.spatial_context and cand.spatial_context:
                            if lead.spatial_context.centroid and cand.spatial_context.centroid:
                                d = SpatialRelationshipEngine.centroid_distance(
                                    lead.spatial_context.centroid,
                                    cand.spatial_context.centroid
                                )
                                spatial_near = (d <= self.spatial_merge_distance)

                        if shared_tracks or spatial_near or not lead.track_ids:
                            cluster.append(cand)
                            placed = True
                            break

                if not placed:
                    clusters.append([cand])

            # Merge clusters into single unified IncidentCandidates
            for cluster in clusters:
                if len(cluster) == 1:
                    fused_results.append(cluster[0])
                else:
                    merged = self._merge_cluster(cluster)
                    fused_results.append(merged)

        # Sort final fused results chronologically
        fused_results.sort(key=lambda x: x.start_time)
        return fused_results

    # Alias for backward compatibility
    fuse_candidates = fuse_incidents

    def _arbitrate_competing_person_interactions(self, candidates: List[IncidentCandidate]) -> List[IncidentCandidate]:
        """
        Arbitrate mutually competing or overlapping person interaction hypotheses
        (POTENTIAL_PHYSICAL_ALTERCATION, POTENTIAL_FORCED_MOVEMENT, PERSON_FOLLOWING)
        sharing the same track pairs within the same temporal window.
        Prevents emitting 3 competing separate high-confidence events for 1 ambiguous interaction.
        """
        interaction_types = {
            "POTENTIAL_PHYSICAL_ALTERCATION",
            "POTENTIAL_FORCED_MOVEMENT",
            "PERSON_FOLLOWING",
        }
        interactions = [c for c in candidates if c.event_type in interaction_types]
        non_interactions = [c for c in candidates if c.event_type not in interaction_types]
        if len(interactions) <= 1:
            return candidates

        # Group interactions by track pair and temporal overlap
        arbitrated: List[IncidentCandidate] = []
        clusters: List[List[IncidentCandidate]] = []

        for cand in sorted(interactions, key=lambda c: c.start_time):
            cand_pair = tuple(sorted(cand.track_ids[:2])) if len(cand.track_ids) >= 2 else tuple(cand.track_ids)
            placed = False
            for cl in clusters:
                lead = cl[0]
                lead_pair = tuple(sorted(lead.track_ids[:2])) if len(lead.track_ids) >= 2 else tuple(lead.track_ids)
                if cand_pair == lead_pair and TemporalAnalysisEngine.is_temporally_overlapping(
                    lead.start_time, lead.end_time,
                    cand.start_time, cand.end_time,
                    tolerance_seconds=self.time_merge_tolerance_seconds,
                ):
                    cl.append(cand)
                    placed = True
                    break
            if not placed:
                clusters.append([cand])

        for cl in clusters:
            if len(cl) == 1:
                arbitrated.append(cl[0])
            else:
                # Multiple competing hypotheses on the same track pair
                primary = max(cl, key=lambda c: c.confidence)
                alt_types = [c.event_type for c in cl if c.event_type != primary.event_type]

                # Merge supporting signals without duplicates
                all_sigs = list(primary.supporting_signals)
                sig_descs = {s.description for s in all_sigs}
                for c in cl:
                    for s in c.supporting_signals:
                        if s.description not in sig_descs:
                            sig_descs.add(s.description)
                            all_sigs.append(s)

                primary.supporting_signals = all_sigs
                primary.validation_decision = "REVIEW_REQUIRED"
                primary.confidence = min(primary.confidence, 0.65)
                if "fused" not in primary.explanation.lower():
                    primary.explanation += f" (Arbitrated with alternate hypotheses: {', '.join(alt_types)})."
                if primary.incident_metadata is None:
                    primary.incident_metadata = {}
                primary.incident_metadata["alternate_hypotheses"] = alt_types
                arbitrated.append(primary)

        return non_interactions + arbitrated

    def _merge_cluster(self, cluster: List[IncidentCandidate]) -> IncidentCandidate:
        """Merge multiple candidates belonging to the same cluster into a singular candidate."""
        lead = cluster[0]
        start_t = min(c.start_time for c in cluster)
        end_t = max(c.end_time for c in cluster)
        duration = max(0.0, round(end_t - start_t, 4))
        max_conf = max(c.confidence for c in cluster)

        # Merge severity (HIGH takes precedence over NORMAL over LOW)
        severities = {c.severity for c in cluster}
        if "HIGH" in severities:
            sev = "HIGH"
        elif "NORMAL" in severities:
            sev = "NORMAL"
        else:
            sev = "LOW"

        # Union track IDs & object classes
        all_tracks: List[str] = []
        all_classes: List[str] = []
        all_detection_ids: List[str] = []
        for c in cluster:
            for tid in c.track_ids:
                if tid not in all_tracks:
                    all_tracks.append(tid)
            for cls in c.object_classes:
                if cls not in all_classes:
                    all_classes.append(cls)
            for did in c.source_detection_ids:
                if did not in all_detection_ids:
                    all_detection_ids.append(did)

        # Union supporting signals avoiding exact duplicates
        merged_signals: List[SupportingSignal] = []
        seen_sig_descriptions = set()
        for c in cluster:
            for s in c.supporting_signals:
                if s.description not in seen_sig_descriptions:
                    seen_sig_descriptions.add(s.description)
                    merged_signals.append(s)

        # Union evidence candidates (deduplicate within 1.0s window)
        merged_evidence: List[EvidenceCandidate] = []
        for c in cluster:
            for ev in c.evidence_candidates:
                already_has_near = any(abs(e.timestamp - ev.timestamp) <= 1.0 for e in merged_evidence)
                if not already_has_near:
                    merged_evidence.append(ev)

        # Merge detector names
        detectors = sorted(list({c.detector_name for c in cluster}))
        detector_label = "+".join(detectors)

        # Unified explanation
        explanation = lead.explanation
        if len(cluster) > 1 and "fused" not in explanation.lower():
            explanation += f" (Fused across {len(cluster)} observational segments)."

        spatial_ctx = lead.spatial_context
        temporal_ctx = lead.temporal_context
        # Merge metadata (observation counts, episode IDs, supporting observation IDs)
        merged_meta: Dict[str, Any] = {}
        total_obs_count = 0
        total_seg_count = 0
        merged_supp_obs_ids: List[str] = []
        rep_timestamps: List[float] = []
        episode_ids: List[str] = []

        for c in cluster:
            m = c.incident_metadata or {}
            if "observation_count" in m:
                total_obs_count += int(m["observation_count"])
            if "segment_count" in m:
                total_seg_count += int(m["segment_count"])
            if "supporting_observation_ids" in m and isinstance(m["supporting_observation_ids"], list):
                for oid in m["supporting_observation_ids"]:
                    if oid not in merged_supp_obs_ids:
                        merged_supp_obs_ids.append(oid)
            if "representative_timestamps" in m and isinstance(m["representative_timestamps"], list):
                for ts in m["representative_timestamps"]:
                    if ts not in rep_timestamps:
                        rep_timestamps.append(ts)
            if "episode_id" in m and m["episode_id"] not in episode_ids:
                episode_ids.append(m["episode_id"])
            for k, v in m.items():
                if k not in ("observation_count", "segment_count", "supporting_observation_ids", "representative_timestamps", "episode_id"):
                    merged_meta[k] = v

        if total_obs_count > 0:
            merged_meta["observation_count"] = total_obs_count
        if total_seg_count > 0:
            merged_meta["segment_count"] = total_seg_count
        if merged_supp_obs_ids:
            merged_meta["supporting_observation_ids"] = merged_supp_obs_ids
        if rep_timestamps:
            merged_meta["representative_timestamps"] = sorted(rep_timestamps)
        if episode_ids:
            merged_meta["episode_id"] = "+".join(episode_ids)

        # Validation decision inheritance invariant:
        # Merged candidate must NOT upgrade constituent REVIEW_REQUIRED to ACCEPTED
        has_review = any(
            str(getattr(c, "validation_decision", "")).upper() in ("REVIEW_REQUIRED", "VALIDATIONDECISION.REVIEW_REQUIRED")
            for c in cluster
        )
        merged_decision = "REVIEW_REQUIRED" if has_review else getattr(lead, "validation_decision", "ACCEPTED")
        merged_human_req = True if (has_review or getattr(lead, "human_verification_required", False)) else False
        if str(merged_decision).upper() in ("REVIEW_REQUIRED", "VALIDATIONDECISION.REVIEW_REQUIRED"):
            max_conf = min(max_conf, 0.65)

        return IncidentCandidate(
            incident_id=f"FUSED-{uuid.uuid4().hex[:8]}",
            video_id=lead.video_id,
            event_type=lead.event_type,
            category=lead.category,
            start_time=start_t,
            end_time=end_t,
            duration=duration,
            severity=sev,
            confidence=max_conf,
            track_ids=all_tracks,
            object_classes=all_classes,
            source_detection_ids=all_detection_ids,
            supporting_signals=merged_signals,
            spatial_context=spatial_ctx,
            temporal_context=temporal_ctx,
            explanation=explanation,
            evidence_candidates=merged_evidence,
            validation_status="VALID",
            validation_decision=merged_decision,
            human_verification_required=merged_human_req,
            detector_name=detector_label,
            detector_version=lead.detector_version,
            incident_metadata=merged_meta,
        )

    def _arbitrate_competing_person_interactions(
        self,
        candidates: List[IncidentCandidate],
    ) -> List[IncidentCandidate]:
        """
        Arbitrate mutually competing person-person interaction hypotheses on the same track pair
        (altercation vs forced movement vs following).
        """
        person_event_types = {
            "POTENTIAL_PHYSICAL_ALTERCATION",
            "POTENTIAL_FORCED_MOVEMENT",
            "PERSON_FOLLOWING",
        }
        person_cands = [c for c in candidates if c.event_type in person_event_types]
        other_cands = [c for c in candidates if c.event_type not in person_event_types]

        if len(person_cands) <= 1:
            return candidates

        # Group by sorted track_ids pair
        by_pair: Dict[Tuple[str, ...], List[IncidentCandidate]] = defaultdict(list)
        unpaired: List[IncidentCandidate] = []

        for c in person_cands:
            p_tracks = tuple(sorted(c.track_ids))
            if len(p_tracks) >= 2:
                by_pair[p_tracks[:2]].append(c)
            else:
                unpaired.append(c)

        arbitrated: List[IncidentCandidate] = list(unpaired)

        for pair, group in by_pair.items():
            if len(group) == 1:
                arbitrated.append(group[0])
                continue

            # Sort by start_time
            sorted_g = sorted(group, key=lambda x: x.start_time)
            # Find temporal clusters
            clusters: List[List[IncidentCandidate]] = []
            for cand in sorted_g:
                placed = False
                for cl in clusters:
                    lead = cl[0]
                    if TemporalAnalysisEngine.is_temporally_overlapping(
                        lead.start_time, lead.end_time,
                        cand.start_time, cand.end_time,
                        tolerance_seconds=2.0
                    ):
                        cl.append(cand)
                        placed = True
                        break
                if not placed:
                    clusters.append([cand])

            for cl in clusters:
                if len(cl) == 1:
                    arbitrated.append(cl[0])
                    continue

                # Prefer highest confidence / most specific
                cl_sorted = sorted(cl, key=lambda x: x.confidence, reverse=True)
                primary = cl_sorted[0]
                alternates = [c.event_type for c in cl_sorted[1:]]

                # Attach alternate hypotheses to primary metadata
                if primary.incident_metadata is None:
                    primary.incident_metadata = {}
                primary.incident_metadata["alternate_hypotheses"] = alternates
                primary.validation_decision = ValidationDecision.REVIEW_REQUIRED
                primary.human_verification_required = True
                arbitrated.append(primary)

        return other_cands + arbitrated

    def _arbitrate_competing_property_interactions(
        self,
        candidates: List[IncidentCandidate],
    ) -> List[IncidentCandidate]:
        """
        Arbitrate competing property hypotheses on shared object tracks
        (takeaway vs pickup vs displacement vs removal vs left behind vs abandoned).
        """
        property_hierarchy = {
            "POTENTIAL_THEFT": 6,
            "POTENTIAL_OBJECT_TAKEAWAY": 6,
            "POTENTIAL_OBJECT_REMOVAL": 5,
            "POTENTIAL_OBJECT_LEFT_BEHIND": 4,
            "POTENTIAL_OBJECT_PICKUP": 3,
            "POTENTIAL_OBJECT_DISPLACEMENT": 2,
            "POTENTIAL_ABANDONED_OBJECT": 1,
            "POTENTIAL_RESTRICTED_OBJECT_MOVEMENT": 2,
            "POTENTIAL_PROPERTY_TAMPERING": 3,
        }

        prop_cands = [c for c in candidates if c.event_type in property_hierarchy]
        other_cands = [c for c in candidates if c.event_type not in property_hierarchy]

        if len(prop_cands) <= 1:
            return candidates

        # Group by primary object track ID
        by_obj: Dict[str, List[IncidentCandidate]] = defaultdict(list)
        unassigned: List[IncidentCandidate] = []

        for c in prop_cands:
            # First track id or spatial context metadata
            obj_id = None
            if c.spatial_context and c.spatial_context.metadata:
                obj_id = c.spatial_context.metadata.get("object_track_id")
            if not obj_id and c.track_ids:
                # Find non-person track or first track
                obj_id = c.track_ids[-1] if len(c.track_ids) > 1 else c.track_ids[0]

            if obj_id:
                by_obj[obj_id].append(c)
            else:
                unassigned.append(c)

        arbitrated: List[IncidentCandidate] = list(unassigned)

        for obj_id, group in by_obj.items():
            if len(group) == 1:
                arbitrated.append(group[0])
                continue

            sorted_g = sorted(group, key=lambda x: x.start_time)
            clusters: List[List[IncidentCandidate]] = []
            for cand in sorted_g:
                placed = False
                for cl in clusters:
                    lead = cl[0]
                    if TemporalAnalysisEngine.is_temporally_overlapping(
                        lead.start_time, lead.end_time,
                        cand.start_time, cand.end_time,
                        tolerance_seconds=3.0
                    ):
                        cl.append(cand)
                        placed = True
                        break
                if not placed:
                    clusters.append([cand])

            for cl in clusters:
                if len(cl) == 1:
                    arbitrated.append(cl[0])
                    continue

                # Sort by hierarchy priority, then confidence
                cl_sorted = sorted(
                    cl,
                    key=lambda x: (property_hierarchy.get(x.event_type, 0), x.confidence),
                    reverse=True,
                )
                primary = cl_sorted[0]
                alternates = [c.event_type for c in cl_sorted[1:]]

                if primary.incident_metadata is None:
                    primary.incident_metadata = {}
                primary.incident_metadata["alternate_hypotheses"] = alternates
                primary.validation_decision = ValidationDecision.REVIEW_REQUIRED
                primary.human_verification_required = True
                arbitrated.append(primary)

        return other_cands + arbitrated

    def _arbitrate_competing_crowd_interactions(
        self,
        candidates: List[IncidentCandidate],
    ) -> List[IncidentCandidate]:
        """
        Arbitrate mutually competing or co-occurring crowd hypotheses
        (POTENTIAL_CROWD_SURGE, POTENTIAL_RESTRICTED_ZONE_CROWDING, POTENTIAL_UNUSUAL_CROWD_MOVEMENT,
         POTENTIAL_CROWD_DISPERSAL, HIGH_PEDESTRIAN_DENSITY, CROWD_DENSITY_INCREASE)
        sharing overlapping temporal windows and common track sets or scene areas.
        Prevents emitting multiple separate high-alert events for a single crowd episode.
        """
        crowd_hierarchy = {
            "POTENTIAL_CROWD_SURGE": 5,
            "POTENTIAL_RESTRICTED_ZONE_CROWDING": 4,
            "POTENTIAL_UNUSUAL_CROWD_MOVEMENT": 3,
            "POTENTIAL_CROWD_DISPERSAL": 3,
            "POTENTIAL_UNUSUAL_ZONE_ACTIVITY": 2,
            "HIGH_PEDESTRIAN_DENSITY": 1,
            "CROWD_DENSITY_INCREASE": 1,
            "ZONE_OCCUPANCY_OBSERVATION": 0,
        }

        crowd_cands = [c for c in candidates if c.event_type in crowd_hierarchy]
        other_cands = [c for c in candidates if c.event_type not in crowd_hierarchy]

        if len(crowd_cands) <= 1:
            return candidates

        # Group by temporal overlap and shared tracks or spatial proximity
        clusters: List[List[IncidentCandidate]] = []
        for cand in sorted(crowd_cands, key=lambda x: x.start_time):
            placed = False
            for cl in clusters:
                lead = cl[0]
                # Temporal overlap with tolerance
                if TemporalAnalysisEngine.is_temporally_overlapping(
                    lead.start_time, lead.end_time,
                    cand.start_time, cand.end_time,
                    tolerance_seconds=self.time_merge_tolerance_seconds,
                ):
                    # Check shared tracks, shared zone, or spatial proximity
                    shared_tracks = bool(set(lead.track_ids).intersection(set(cand.track_ids)))
                    shared_zone = False
                    if lead.spatial_context and cand.spatial_context:
                        lz = getattr(lead.spatial_context, "zone_name", None)
                        cz = getattr(cand.spatial_context, "zone_name", None)
                        if lz and cz and lz == cz:
                            shared_zone = True

                    spatial_near = False
                    if lead.spatial_context and cand.spatial_context:
                        if lead.spatial_context.centroid and cand.spatial_context.centroid:
                            d = SpatialRelationshipEngine.centroid_distance(
                                lead.spatial_context.centroid,
                                cand.spatial_context.centroid
                            )
                            spatial_near = (d <= self.spatial_merge_distance * 1.5)

                    if shared_tracks or shared_zone or spatial_near or not lead.track_ids:
                        cl.append(cand)
                        placed = True
                        break
            if not placed:
                clusters.append([cand])

        arbitrated: List[IncidentCandidate] = []
        for cl in clusters:
            if len(cl) == 1:
                arbitrated.append(cl[0])
                continue

            # Sort by hierarchy priority, then confidence
            cl_sorted = sorted(
                cl,
                key=lambda x: (crowd_hierarchy.get(x.event_type, 0), x.confidence),
                reverse=True,
            )
            primary = cl_sorted[0]
            alternates = [c.event_type for c in cl_sorted[1:]]

            # Merge supporting signals from all candidates without duplicates
            all_sigs = list(primary.supporting_signals)
            sig_descs = {s.description for s in all_sigs}
            for c in cl_sorted[1:]:
                for s in c.supporting_signals:
                    if s.description not in sig_descs:
                        sig_descs.add(s.description)
                        all_sigs.append(s)

            primary.supporting_signals = all_sigs

            # Union track IDs
            all_tracks = list(primary.track_ids)
            for c in cl_sorted[1:]:
                for tid in c.track_ids:
                    if tid not in all_tracks:
                        all_tracks.append(tid)
            primary.track_ids = all_tracks

            # Preserve alternates in incident metadata
            if primary.incident_metadata is None:
                primary.incident_metadata = {}
            primary.incident_metadata["alternate_hypotheses"] = alternates
            if len(alternates) > 0 and primary.confidence > 0.65:
                # If there are competing interpretations, flag for human verification
                primary.validation_decision = ValidationDecision.REVIEW_REQUIRED
                primary.human_verification_required = True

            arbitrated.append(primary)

        return other_cands + arbitrated

    def _fuse_fire_and_smoke_interactions(self, candidates: List[IncidentCandidate]) -> List[IncidentCandidate]:
        """
        Phase 15: Cross-modal Fire + Smoke incident fusion.
        If POTENTIAL_FIRE and POTENTIAL_SMOKE occur in the same spatial/temporal window,
        fuse them into a unified POTENTIAL_FIRE_SMOKE incident candidate.
        Preserves individual supporting observations; neither automatically proves the other.
        """
        fire_cands = [c for c in candidates if c.event_type == "POTENTIAL_FIRE"]
        smoke_cands = [c for c in candidates if c.event_type == "POTENTIAL_SMOKE"]
        other_cands = [c for c in candidates if c.event_type not in ["POTENTIAL_FIRE", "POTENTIAL_SMOKE"]]

        if not fire_cands or not smoke_cands:
            return candidates

        fused_candidates: List[IncidentCandidate] = []
        matched_fire_ids = set()
        matched_smoke_ids = set()

        for fire in fire_cands:
            matched_smoke = None
            for smoke in smoke_cands:
                if smoke.incident_id in matched_smoke_ids:
                    continue

                # Check temporal overlap
                overlap = TemporalAnalysisEngine.is_temporally_overlapping(
                    fire.start_time, fire.end_time,
                    smoke.start_time, smoke.end_time,
                    tolerance_seconds=self.time_merge_tolerance_seconds,
                )
                if not overlap:
                    continue

                # Check spatial distance if both have spatial context
                spatial_close = True
                if fire.spatial_context and smoke.spatial_context:
                    if fire.spatial_context.centroid and smoke.spatial_context.centroid:
                        d = SpatialRelationshipEngine.centroid_distance(
                            fire.spatial_context.centroid,
                            smoke.spatial_context.centroid,
                        )
                        spatial_close = (d <= self.spatial_merge_distance * 2.0)

                if spatial_close:
                    matched_smoke = smoke
                    break

            if matched_smoke is not None:
                matched_fire_ids.add(fire.incident_id)
                matched_smoke_ids.add(matched_smoke.incident_id)

                start_t = min(fire.start_time, matched_smoke.start_time)
                end_t = max(fire.end_time, matched_smoke.end_time)
                dur = max(0.0, end_t - start_t)

                # Fuse supporting signals
                all_sigs = list(fire.supporting_signals)
                for ss in matched_smoke.supporting_signals:
                    if ss.description not in [s.description for s in all_sigs]:
                        all_sigs.append(ss)

                # Fuse contradictory signals
                all_contra = list(fire.contradictory_signals)
                for cs in matched_smoke.contradictory_signals:
                    if cs.description not in [c.description for c in all_contra]:
                        all_contra.append(cs)

                # Fuse evidence candidates
                all_evidence = list(fire.evidence_candidates) + list(matched_smoke.evidence_candidates)

                fused_cand = IncidentCandidate(
                    incident_id=f"INC-FIRE-SMOKE-{uuid.uuid4().hex[:8]}",
                    video_id=fire.video_id,
                    event_type="POTENTIAL_FIRE_SMOKE",
                    category="environment",
                    start_time=start_t,
                    end_time=end_t,
                    duration=dur,
                    severity="HIGH",
                    confidence=max(fire.confidence, matched_smoke.confidence),
                    track_ids=list(set(fire.track_ids + matched_smoke.track_ids)),
                    object_classes=list(set(fire.object_classes + matched_smoke.object_classes)),
                    source_detection_ids=list(set(fire.source_detection_ids + matched_smoke.source_detection_ids)),
                    supporting_signals=all_sigs,
                    contradictory_signals=all_contra,
                    spatial_context=fire.spatial_context or matched_smoke.spatial_context,
                    temporal_context=TemporalContext(
                        start_time=start_t,
                        end_time=end_t,
                        duration_seconds=dur,
                    ),
                    explanation=(
                        f"Cross-modal visual incident: Simultaneous potential fire (conf: {fire.confidence:.0%}) "
                        f"and smoke (conf: {matched_smoke.confidence:.0%}) observed in shared spatio-temporal region."
                    ),
                    evidence_candidates=all_evidence,
                    validation_decision="REVIEW_REQUIRED",
                    validation_reasons=["Fused dual-modal visual fire and smoke evidence. Human verification required."],
                    human_verification_required=True,
                    detector_name="specialized_fire_smoke_fusion",
                    detector_version="1.0.0",
                    incident_metadata={
                        "fire_incident_id": fire.incident_id,
                        "smoke_incident_id": matched_smoke.incident_id,
                        "fire_confidence": fire.confidence,
                        "smoke_confidence": matched_smoke.confidence,
                    },
                )
                fused_candidates.append(fused_cand)

        # Retain unmatched fire & smoke candidates
        remaining_fire = [c for c in fire_cands if c.incident_id not in matched_fire_ids]
        remaining_smoke = [c for c in smoke_cands if c.incident_id not in matched_smoke_ids]

        return other_cands + fused_candidates + remaining_fire + remaining_smoke


