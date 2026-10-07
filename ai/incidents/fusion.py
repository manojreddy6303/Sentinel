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
        sampling_aware: bool = False,
    ):
        self.time_merge_tolerance_seconds = time_merge_tolerance_seconds
        self.spatial_merge_distance = spatial_merge_distance
        self.sampling_aware = sampling_aware

    def fuse_incidents(
        self,
        candidates: List[IncidentCandidate],
        sampling_aware: Optional[bool] = None,
    ) -> List[IncidentCandidate]:
        """
        Deduplicate and fuse raw candidates into cohesive incident records.
        """
        if not candidates:
            return []

        is_sampling_aware = self.sampling_aware if sampling_aware is None else sampling_aware

        # 1. Arbitrate competing person interaction hypotheses on shared tracks
        candidates = self._arbitrate_competing_person_interactions(candidates)

        # 2. Arbitrate competing property hypotheses on shared object tracks
        candidates = self._arbitrate_competing_property_interactions(candidates)

        # 3. Arbitrate competing crowd hypotheses in overlapping temporal windows
        candidates = self._arbitrate_competing_crowd_interactions(candidates)

        # 4. Cross-modal fire + smoke incident fusion (Phase 15)
        candidates = self._fuse_fire_and_smoke_interactions(candidates)

        # 5. Cross-domain arbitration: Property Takeaway / Theft vs Incidental Person Interaction
        candidates = self._arbitrate_property_vs_person_interactions(candidates)

        # 6. Group candidates by event_type
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
                    if is_sampling_aware:
                        # Envelope-based temporal continuity check
                        c_start = min(item.start_time for item in cluster)
                        c_end = max(item.end_time for item in cluster)
                        temporally_connected = (
                            cand.start_time <= (c_end + self.time_merge_tolerance_seconds)
                            and cand.end_time >= (c_start - self.time_merge_tolerance_seconds)
                        )
                        cluster_tracks = set(sum([item.track_ids for item in cluster], []))
                        shared_tracks = bool(cluster_tracks.intersection(set(cand.track_ids)))
                        spatial_near = False
                        if cand.spatial_context and cand.spatial_context.centroid:
                            for item in cluster:
                                if item.spatial_context and item.spatial_context.centroid:
                                    d = SpatialRelationshipEngine.centroid_distance(
                                        item.spatial_context.centroid,
                                        cand.spatial_context.centroid,
                                    )
                                    if d <= self.spatial_merge_distance:
                                        spatial_near = True
                                        break
                        if temporally_connected and (shared_tracks or spatial_near or not cluster_tracks):
                            cluster.append(cand)
                            placed = True
                            break
                    else:
                        lead = cluster[0]
                        # Check temporal overlap or close adjacency (legacy lead-only comparison)
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

        # Union track IDs & object classes, preserving lead candidate's primary tracks
        primary_tracks = list(lead.track_ids or [])
        secondary_tracks: List[str] = []
        all_classes: List[str] = list(lead.object_classes or [])
        all_detection_ids: List[str] = []
        for c in cluster:
            for tid in (c.track_ids or []):
                if tid not in primary_tracks and tid not in secondary_tracks:
                    secondary_tracks.append(tid)
            for cls in (c.object_classes or []):
                if cls not in all_classes:
                    all_classes.append(cls)
            for did in (c.source_detection_ids or []):
                if did not in all_detection_ids:
                    all_detection_ids.append(did)

        # Initialize metadata dictionary for fused incident
        merged_meta: Dict[str, Any] = {}

        # For property / theft events, preserve primary actor identity strictly
        is_property_or_theft = (
            getattr(lead, "category", "") == "property"
            or "THEFT" in lead.event_type
            or "TAKEAWAY" in lead.event_type
        )
        if is_property_or_theft:
            assigned_tracks = primary_tracks
            merged_meta["primary_tracks"] = primary_tracks
            merged_meta["secondary_tracks"] = secondary_tracks
            merged_meta["track_ids"] = primary_tracks
        else:
            assigned_tracks = primary_tracks + secondary_tracks

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

        # Canonical score reconciliation across cluster
        max_pattern_strength = max(getattr(c, "pattern_evidence_strength", c.confidence) for c in cluster)
        if str(merged_decision).upper() in ("REVIEW_REQUIRED", "VALIDATIONDECISION.REVIEW_REQUIRED"):
            max_conf = min(max_conf, 0.65)
            merged_assessment = min(0.65, max(getattr(c, "assessment_score", c.confidence) for c in cluster))
        else:
            merged_assessment = max(getattr(c, "assessment_score", c.confidence) for c in cluster)

        merged_meta["pattern_evidence_strength"] = max_pattern_strength
        merged_meta["assessment_score"] = merged_assessment
        merged_meta["validation_decision"] = merged_decision

        # Harmonize explanation string to eliminate contradictory score/decision text
        if merged_decision == "REVIEW_REQUIRED" and "(ACCEPTED)" in explanation:
            explanation = explanation.replace("(ACCEPTED)", "(REVIEW_REQUIRED)")

        return IncidentCandidate(
            incident_id=f"FUSED-{uuid.uuid4().hex[:8]}",
            video_id=lead.video_id,
            event_type=lead.event_type,
            category=lead.category,
            start_time=start_t,
            end_time=end_t,
            duration=duration,
            severity=sev,
            confidence=merged_assessment,
            pattern_evidence_strength=max_pattern_strength,
            assessment_score=merged_assessment,
            track_ids=assigned_tracks,
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

                # Merge supporting signals without duplicates
                all_sigs = list(primary.supporting_signals)
                sig_descs = {s.description for s in all_sigs}
                for c in cl_sorted[1:]:
                    for s in c.supporting_signals:
                        if s.description not in sig_descs:
                            sig_descs.add(s.description)
                            all_sigs.append(s)
                primary.supporting_signals = all_sigs

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

                # Primary actor tracks from winning candidate; secondary tracks preserved in metadata
                primary_tracks = list(primary.track_ids or [])
                all_object_classes = list(primary.object_classes or [])
                secondary_tracks = []
                for c in cl_sorted[1:]:
                    for tid in (c.track_ids or []):
                        if tid not in primary_tracks and tid not in secondary_tracks:
                            secondary_tracks.append(tid)
                    for ocls in (c.object_classes or []):
                        if ocls not in all_object_classes:
                            all_object_classes.append(ocls)

                primary.track_ids = primary_tracks
                primary.object_classes = all_object_classes

                if primary.incident_metadata is None:
                    primary.incident_metadata = {}
                primary.incident_metadata["alternate_hypotheses"] = alternates
                primary.incident_metadata["track_ids"] = primary_tracks
                primary.incident_metadata["primary_tracks"] = primary_tracks
                primary.incident_metadata["secondary_tracks"] = secondary_tracks
                primary.incident_metadata["object_classes"] = all_object_classes
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

    def _arbitrate_property_vs_person_interactions(
        self,
        candidates: List[IncidentCandidate],
    ) -> List[IncidentCandidate]:
        """
        Cross-domain arbitration: Property Takeaway / Theft vs Incidental Person Interaction.
        When a person track is participating in a coherent property takeaway sequence
        (approach -> dwell/interaction -> object removal/relocation -> departure),
        contemporaneous, ungrounded person-interaction candidates (POTENTIAL_PHYSICAL_ALTERCATION,
        POTENTIAL_FORCED_MOVEMENT) on that track arising from incidental proximity to bystanders
        must not outrank or obscure the theft interpretation.
        """
        theft_cands = [c for c in candidates if "THEFT" in c.event_type or "TAKEAWAY" in c.event_type]
        if not theft_cands:
            return candidates

        theft_tracks = set()
        for th in theft_cands:
            for tid in th.track_ids:
                theft_tracks.add(tid)

        arbitrated: List[IncidentCandidate] = []
        for c in candidates:
            if c.event_type in ("POTENTIAL_PHYSICAL_ALTERCATION", "POTENTIAL_FORCED_MOVEMENT"):
                # Check if this person candidate overlaps temporally with a theft on a shared track
                overlaps_theft = False
                for th in theft_cands:
                    if set(c.track_ids).intersection(set(th.track_ids)):
                        if TemporalAnalysisEngine.is_temporally_overlapping(
                            th.start_time, th.end_time,
                            c.start_time, c.end_time,
                            tolerance_seconds=20.0,
                        ):
                            overlaps_theft = True
                            if th.incident_metadata is None:
                                th.incident_metadata = {}
                            alts = th.incident_metadata.get("alternate_hypotheses", [])
                            if c.event_type not in alts:
                                alts.append(c.event_type)
                            th.incident_metadata["alternate_hypotheses"] = alts
                            break
                if not overlaps_theft:
                    arbitrated.append(c)
            else:
                arbitrated.append(c)

        return arbitrated

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


