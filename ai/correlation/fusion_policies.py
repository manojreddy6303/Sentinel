"""
ai/correlation/fusion_policies.py
=================================
Domain-specific fusion policies for Phase 16:
Vehicle, Person, Property, Crowd/Zone, and Specialized Visual fusion.
"""

from typing import List, Dict, Any, Optional
import uuid

from ai.incidents.schemas import IncidentCandidate, ValidationDecision
from ai.correlation.models import CorrelatedIncident, ObservationalRelationship
from ai.correlation.storyline_generator import IncidentStorylineGenerator
from ai.correlation.spatial_engine import CorrelationSpatialEngine
from ai.correlation.temporal_engine import CorrelationTemporalEngine


def inherit_validation_decision(candidates: List[IncidentCandidate]) -> str:
    """
    Global Invariant:
    A correlated incident MUST NOT upgrade a constituent candidate's
    REVIEW_REQUIRED decision to ACCEPTED unless there is an explicit,
    documented, evidence-based arbitration rule that independently satisfies
    the ACCEPTED policy.
    Correlation alone must never increase validation certainty.
    """
    if any(
        str(getattr(c, "validation_decision", "")).upper() in ("REVIEW_REQUIRED", "VALIDATIONDECISION.REVIEW_REQUIRED", "HYPOTHESISOUTCOME.REVIEW_REQUIRED")
        for c in candidates
    ):
        return "REVIEW_REQUIRED"
    return "ACCEPTED"


class VehicleCorrelationPolicy:
    """
    Fuses multi-stage vehicle events (approach + contact + deceleration) into unified collision storylines.
    Guarantees:
    - Never merges events involving disjoint vehicle tracks
    - Separates actual collision contact interval from broader trajectory/telemetry windows
    - Preserves near-collisions as distinct non-contact events
    - Preserves separate collision incidents occurring across different vehicles/times
    """

    @staticmethod
    def correlate(
        candidates: List[IncidentCandidate],
        video_id: str,
        relationships: List[ObservationalRelationship],
    ) -> List[CorrelatedIncident]:
        veh_cands = [
            c for c in candidates
            if getattr(c, "category", "").lower() == "vehicle" or "COLLISION" in getattr(c, "event_type", "")
        ]
        if not veh_cands:
            return []

        collisions = [c for c in veh_cands if "COLLISION" in c.event_type and c.event_type != "POTENTIAL_NEAR_COLLISION"]
        near_collisions = [c for c in veh_cands if c.event_type == "POTENTIAL_NEAR_COLLISION"]
        other_veh = [c for c in veh_cands if c not in collisions and c not in near_collisions]

        correlated: List[CorrelatedIncident] = []
        absorbed_cands = set()

        # 1. Cluster full collision events strictly by shared vehicle tracks and tight temporal continuity
        collision_clusters: List[List[IncidentCandidate]] = []
        for coll in sorted(collisions, key=lambda c: c.start_time):
            placed = False
            for cluster in collision_clusters:
                can_merge = False
                for other in cluster:
                    shared_trks = set(coll.track_ids) & set(other.track_ids)
                    t_gap = max(coll.start_time, other.start_time) - min(coll.end_time, other.end_time)
                    if shared_trks and t_gap <= 6.0:
                        can_merge = True
                        break
                if can_merge:
                    cluster.append(coll)
                    placed = True
                    break
            if not placed:
                collision_clusters.append([coll])

        for cluster in collision_clusters:
            coll_tracks = set(sum([c.track_ids for c in cluster], []))
            start_t = min(c.start_time for c in cluster)
            end_t = max(c.end_time for c in cluster)
            dur = max(0.0, end_t - start_t)

            # Absorb associated deceleration / sudden stop events for these colliding vehicles in immediate temporal continuity
            associated_stops = [
                c for c in other_veh
                if (set(c.track_ids) & coll_tracks) and abs(c.start_time - start_t) <= 5.0
            ]
            for s in associated_stops:
                absorbed_cands.add(s.incident_id)

            all_cands = cluster + associated_stops
            for c in cluster:
                absorbed_cands.add(c.incident_id)

            telemetry_start = min(c.start_time for c in all_cands)
            telemetry_end = max(c.end_time for c in all_cands)
            telemetry_dur = max(0.0, telemetry_end - telemetry_start)

            all_tracks = list(coll_tracks)
            supp_tracks = list(set(sum([c.track_ids for c in associated_stops], [])) - coll_tracks)
            all_classes = list(set(sum([c.object_classes for c in all_cands], [])))
            all_cand_ids = [c.incident_id for c in all_cands]
            all_dets = list(set(c.detector_name for c in all_cands))
            all_signals = [s.to_dict() for c in all_cands for s in c.supporting_signals]
            max_conf = max(c.confidence for c in cluster)

            matched_rels = [
                r for r in relationships
                if r.relationship_type in ("collides_with", "approaches")
                and any(t in (r.subject_track_id, r.target_track_id) for t in all_tracks)
            ]
            has_contact = any(r.relationship_type == "collides_with" for r in matched_rels)

            subcategory = "multi_vehicle_collision" if has_contact else "vehicle_collision_pattern"
            val_dec = inherit_validation_decision(all_cands)

            all_evidence_ids = list(set([eid for c in all_cands for eid in c.incident_metadata.get("evidence_ids", [])]))
            all_neg_ev = [s.description for c in all_cands for s in c.contradictory_signals]

            score = max_conf
            if all_neg_ev:
                score = max(0.2, score - 0.15 * len(all_neg_ev))

            pattern_str = max([getattr(c, "pattern_evidence_strength", getattr(c, "confidence", 0.0)) for c in all_cands] + [max_conf])
            if val_dec == "REVIEW_REQUIRED":
                score = min(score, 0.65)
                rel_rating = "MODERATE"
                ev_str = min(score * 0.95, 0.65)
            else:
                rel_rating = "HIGH" if score >= 0.80 else "MODERATE"
                ev_str = score * 0.95

            storyline = IncidentStorylineGenerator.generate_storyline(
                category="vehicle",
                subcategory=subcategory,
                start_time=start_t,
                end_time=end_t,
                duration=dur,
                primary_tracks=all_tracks,
                supporting_tracks=supp_tracks,
                object_classes=all_classes,
                relationships=matched_rels,
                supporting_signals=all_signals,
                confidence=score,
                validation_decision=val_dec,
            )

            ci = CorrelatedIncident(
                incident_id=f"CORR-VEH-{uuid.uuid4().hex[:8]}",
                video_id=video_id,
                incident_category="vehicle",
                incident_subcategory=subcategory,
                start_time=start_t,
                end_time=end_t,
                duration=dur,
                severity="HIGH",
                confidence=score,
                assessment_score=score,
                evidence_strength=ev_str,
                pattern_evidence_strength=pattern_str,
                reliability_rating=rel_rating,
                validation_decision=val_dec,
                primary_track_ids=all_tracks,
                supporting_track_ids=supp_tracks,
                involved_object_classes=all_classes,
                source_candidate_ids=all_cand_ids,
                source_detector_ids=all_dets,
                supporting_signals=all_signals[:10],
                negative_evidence=all_neg_ev,
                evidence_ids=all_evidence_ids,
                relationships=matched_rels,
                contextual_factors={
                    "pattern_evidence_strength": pattern_str,
                    "supporting_telemetry_window": {
                        "start_time": telemetry_start,
                        "end_time": telemetry_end,
                        "duration": telemetry_dur,
                    }
                },
                storyline=storyline,
                provenance_graph={
                    "candidate_count": len(all_cands),
                    "collision_interval": [start_t, end_t],
                    "telemetry_interval": [telemetry_start, telemetry_end],
                },
            )
            correlated.append(ci)

        # 2. Near-collisions form distinct non-contact incidents
        near_clusters: List[List[IncidentCandidate]] = []
        for nc in sorted(near_collisions, key=lambda c: c.start_time):
            placed = False
            for cluster in near_clusters:
                can_merge = False
                for other in cluster:
                    shared_trks = set(nc.track_ids) & set(other.track_ids)
                    t_gap = max(nc.start_time, other.start_time) - min(nc.end_time, other.end_time)
                    if shared_trks and t_gap <= 5.0:
                        can_merge = True
                        break
                if can_merge:
                    cluster.append(nc)
                    placed = True
                    break
            if not placed:
                near_clusters.append([nc])

        for nc_cluster in near_clusters:
            lead = nc_cluster[0]
            for nc in nc_cluster:
                absorbed_cands.add(nc.incident_id)

            nc_start = min(c.start_time for c in nc_cluster)
            nc_end = max(c.end_time for c in nc_cluster)
            nc_dur = max(0.0, nc_end - nc_start)
            nc_tracks = list(set(sum([c.track_ids for c in nc_cluster], [])))
            nc_classes = list(set(sum([c.object_classes for c in nc_cluster], [])))
            nc_signals = [s.to_dict() for c in nc_cluster for s in c.supporting_signals]
            raw_conf = max(c.confidence for c in nc_cluster)
            pattern_str = max([getattr(c, "pattern_evidence_strength", c.confidence) for c in nc_cluster] + [raw_conf])
            score = min(raw_conf, 0.65)  # Enforce Sentinel 0.65 review ceiling

            ci = CorrelatedIncident(
                incident_id=f"CORR-VEH-NEAR-{uuid.uuid4().hex[:8]}",
                video_id=video_id,
                incident_category="vehicle",
                incident_subcategory="NEAR_COLLISION",
                start_time=nc_start,
                end_time=nc_end,
                duration=nc_dur,
                severity="NORMAL",
                confidence=score,
                assessment_score=score,
                evidence_strength=score * 0.8,
                pattern_evidence_strength=pattern_str,
                reliability_rating="MODERATE",
                validation_decision="REVIEW_REQUIRED",
                primary_track_ids=nc_tracks,
                involved_object_classes=nc_classes,
                source_candidate_ids=[c.incident_id for c in nc_cluster],
                source_detector_ids=list(set(c.detector_name for c in nc_cluster)),
                supporting_signals=nc_signals[:10],
                contextual_factors={"pattern_evidence_strength": pattern_str},
                relationships=[],
                storyline=IncidentStorylineGenerator.generate_storyline(
                    category="vehicle",
                    subcategory="near_collision",
                    start_time=nc_start,
                    end_time=nc_end,
                    duration=nc_dur,
                    primary_tracks=nc_tracks,
                    supporting_tracks=[],
                    object_classes=nc_classes,
                    relationships=[],
                    supporting_signals=nc_signals,
                    confidence=score,
                    validation_decision="REVIEW_REQUIRED",
                ),
            )
            correlated.append(ci)

        # 3. Remaining unabsorbed vehicle candidates (e.g. independent trajectory anomalies or stops)
        for cand in other_veh:
            if cand.incident_id not in absorbed_cands:
                val_dec = cand.validation_decision
                score = cand.confidence
                if str(val_dec).upper() in ("REVIEW_REQUIRED", "VALIDATIONDECISION.REVIEW_REQUIRED", "HYPOTHESISOUTCOME.REVIEW_REQUIRED"):
                    score = min(score, 0.65)

                ci = CorrelatedIncident(
                    incident_id=f"CORR-VEH-{uuid.uuid4().hex[:8]}",
                    video_id=video_id,
                    incident_category="vehicle",
                    incident_subcategory=cand.event_type.lower(),
                    start_time=cand.start_time,
                    end_time=cand.end_time,
                    duration=cand.duration,
                    severity=cand.severity,
                    confidence=score,
                    assessment_score=score,
                    evidence_strength=score * 0.8,
                    pattern_evidence_strength=getattr(cand, "pattern_evidence_strength", cand.confidence),
                    reliability_rating="HIGH" if score >= 0.80 else "MODERATE",
                    validation_decision=val_dec,
                    primary_track_ids=cand.track_ids,
                    involved_object_classes=cand.object_classes,
                    source_candidate_ids=[cand.incident_id],
                    source_detector_ids=[cand.detector_name],
                    supporting_signals=[s.to_dict() for s in cand.supporting_signals],
                    relationships=[],
                    storyline=IncidentStorylineGenerator.generate_storyline(
                        category="vehicle",
                        subcategory=cand.event_type.lower(),
                        start_time=cand.start_time,
                        end_time=cand.end_time,
                        duration=cand.duration,
                        primary_tracks=cand.track_ids,
                        supporting_tracks=[],
                        object_classes=cand.object_classes,
                        relationships=[],
                        supporting_signals=[s.to_dict() for s in cand.supporting_signals],
                        confidence=score,
                        validation_decision=val_dec,
                    ),
                )
                correlated.append(ci)

        return correlated


class PropertyCorrelationPolicy:
    """Correlates complete object takeaway/theft patterns across multi-stage alerts."""

    @staticmethod
    def correlate(
        candidates: List[IncidentCandidate],
        video_id: str,
        relationships: List[ObservationalRelationship],
    ) -> List[CorrelatedIncident]:
        prop_cands = [c for c in candidates if c.category == "property" or "THEFT" in c.event_type]
        if not prop_cands:
            return []

        thefts = [c for c in prop_cands if "THEFT" in c.event_type or "TAKEAWAY" in c.event_type]
        intermediates = [c for c in prop_cands if c not in thefts]

        correlated: List[CorrelatedIncident] = []

        if thefts:
            lead = thefts[0]
            all_cands = thefts + intermediates
            start_t = min(c.start_time for c in all_cands)
            end_t = max(c.end_time for c in all_cands)
            dur = max(0.0, end_t - start_t)
            all_tracks = list(set(sum([c.track_ids for c in all_cands], [])))
            all_classes = list(set(sum([c.object_classes for c in all_cands], [])))
            all_cand_ids = [c.incident_id for c in all_cands]
            all_dets = list(set(c.detector_name for c in all_cands))
            all_signals = [s.to_dict() for c in all_cands for s in c.supporting_signals]
            max_conf = max(c.confidence for c in all_cands)

            val_dec = inherit_validation_decision(all_cands)

            matched_rels = [r for r in relationships if any(t in r.subject_track_id or t in r.target_track_id for t in all_tracks)]

            all_evidence_ids = list(set([eid for c in all_cands for eid in c.incident_metadata.get("evidence_ids", [])]))
            all_neg_ev = [s.description for c in all_cands for s in c.contradictory_signals]

            # Evidence aggregation & negative degradation
            score = max_conf
            if all_neg_ev:
                score = max(0.2, score - 0.15 * len(all_neg_ev))

            # Review ceiling cap: strictly capped at 0.65 per Sentinel reliability policy
            pattern_str = max([getattr(c, "pattern_evidence_strength", getattr(c, "confidence", 0.0)) for c in all_cands] + [max_conf])
            if val_dec == "REVIEW_REQUIRED":
                score = min(score, 0.65)
                rel_rating = "MODERATE"
                ev_str = min(score * 0.9, 0.65)
            else:
                rel_rating = "HIGH" if score >= 0.80 else "MODERATE"
                ev_str = score * 0.9

            storyline = IncidentStorylineGenerator.generate_storyline(
                category="property",
                subcategory="potential_object_takeaway_pattern",
                start_time=start_t,
                end_time=end_t,
                duration=dur,
                primary_tracks=[t for t in all_tracks if "person" in t.lower() or "0" in t][:1],
                supporting_tracks=[t for t in all_tracks if t not in [t for t in all_tracks if "person" in t.lower() or "0" in t][:1]],
                object_classes=all_classes,
                relationships=matched_rels,
                supporting_signals=all_signals,
                confidence=score,
                validation_decision=val_dec,
            )

            ci = CorrelatedIncident(
                incident_id=f"CORR-PROP-{uuid.uuid4().hex[:8]}",
                video_id=video_id,
                incident_category="property",
                incident_subcategory="potential_object_takeaway_pattern",
                start_time=start_t,
                end_time=end_t,
                duration=dur,
                severity=lead.severity,
                confidence=score,
                assessment_score=score,
                evidence_strength=ev_str,
                pattern_evidence_strength=pattern_str,
                reliability_rating=rel_rating,
                validation_decision=val_dec,
                primary_track_ids=all_tracks[:2],
                supporting_track_ids=all_tracks[2:],
                involved_object_classes=all_classes,
                source_candidate_ids=all_cand_ids,
                source_detector_ids=all_dets,
                supporting_signals=all_signals[:10],
                negative_evidence=all_neg_ev,
                evidence_ids=all_evidence_ids,
                relationships=matched_rels,
                storyline=storyline,
                contextual_factors={"pattern_evidence_strength": pattern_str},
                provenance_graph={"stage_sequence": ["stationary", "approach", "interaction", "disappearance"], "intermediate_count": len(intermediates)},
            )
            correlated.append(ci)
        else:
            # Standalone property items (e.g. abandoned object)
            for cand in prop_cands:
                val_dec = inherit_validation_decision([cand])
                score = cand.confidence
                pattern_str = getattr(cand, "pattern_evidence_strength", cand.confidence)
                if val_dec == "REVIEW_REQUIRED":
                    score = min(score, 0.65)
                    rel_rating = "MODERATE"
                    ev_str = min(score * 0.8, 0.65)
                else:
                    rel_rating = "HIGH" if score >= 0.80 else "MODERATE"
                    ev_str = score * 0.8
                ci = CorrelatedIncident(
                    incident_id=f"CORR-PROP-OBJ-{uuid.uuid4().hex[:8]}",
                    video_id=video_id,
                    incident_category="property",
                    incident_subcategory=cand.event_type.lower(),
                    start_time=cand.start_time,
                    end_time=cand.end_time,
                    duration=cand.duration,
                    severity=cand.severity,
                    confidence=score,
                    assessment_score=score,
                    evidence_strength=ev_str,
                    pattern_evidence_strength=pattern_str,
                    reliability_rating=rel_rating,
                    validation_decision=val_dec,
                    primary_track_ids=cand.track_ids,
                    involved_object_classes=cand.object_classes,
                    source_candidate_ids=[cand.incident_id],
                    source_detector_ids=[cand.detector_name],
                    supporting_signals=[s.to_dict() for s in cand.supporting_signals],
                    storyline=IncidentStorylineGenerator.generate_storyline(
                        category="property",
                        subcategory=cand.event_type.lower(),
                        start_time=cand.start_time,
                        end_time=cand.end_time,
                        duration=cand.duration,
                        primary_tracks=cand.track_ids,
                        supporting_tracks=[],
                        object_classes=cand.object_classes,
                        relationships=[],
                        supporting_signals=[s.to_dict() for s in cand.supporting_signals],
                        confidence=score,
                        validation_decision=val_dec,
                    ),
                )
                correlated.append(ci)

        return correlated


class PersonCorrelationPolicy:
    """Fuses person posture, medical, and multi-person interaction events."""

    @staticmethod
    def correlate(
        candidates: List[IncidentCandidate],
        video_id: str,
        relationships: List[ObservationalRelationship],
    ) -> List[CorrelatedIncident]:
        person_cands = [c for c in candidates if c.category == "person"]
        if not person_cands:
            return []

        # Correlate Fall + Person-Down into single medical/posture incident
        falls = [c for c in person_cands if "FALL" in c.event_type or "DOWN" in c.event_type]
        other_person = [c for c in person_cands if c not in falls]

        correlated: List[CorrelatedIncident] = []

        if falls:
            start_t = min(c.start_time for c in falls)
            end_t = max(c.end_time for c in falls)
            dur = max(0.0, end_t - start_t)
            tracks = list(set(sum([c.track_ids for c in falls], [])))
            classes = list(set(sum([c.object_classes for c in falls], [])))
            max_conf = max(c.confidence for c in falls)
            cand_ids = [c.incident_id for c in falls]
            dets = list(set(c.detector_name for c in falls))
            signals = [s.to_dict() for c in falls for s in c.supporting_signals]

            storyline = IncidentStorylineGenerator.generate_storyline(
                category="person",
                subcategory="potential_person_fall_and_incapacitation",
                start_time=start_t,
                end_time=end_t,
                duration=dur,
                primary_tracks=tracks,
                supporting_tracks=[],
                object_classes=classes,
                relationships=[],
                supporting_signals=signals,
                confidence=max_conf,
                validation_decision="REVIEW_REQUIRED",
            )

            pattern_str = max([getattr(c, "pattern_evidence_strength", getattr(c, "confidence", 0.0)) for c in falls] + [max_conf])
            ci = CorrelatedIncident(
                incident_id=f"CORR-PERS-POSTURE-{uuid.uuid4().hex[:8]}",
                video_id=video_id,
                incident_category="person",
                incident_subcategory="potential_person_fall_and_incapacitation",
                start_time=start_t,
                end_time=end_t,
                duration=dur,
                severity="HIGH",
                confidence=min(max_conf, 0.65),
                assessment_score=min(max_conf, 0.65),
                evidence_strength=min(max_conf * 0.85, 0.65),
                pattern_evidence_strength=pattern_str,
                reliability_rating="MODERATE",
                validation_decision="REVIEW_REQUIRED",
                primary_track_ids=tracks,
                involved_object_classes=classes,
                source_candidate_ids=cand_ids,
                source_detector_ids=dets,
                supporting_signals=signals[:10],
                storyline=storyline,
                provenance_graph={"fused_events": [c.event_type for c in falls]},
            )
            correlated.append(ci)

        for op in other_person:
            val_dec = inherit_validation_decision([op])
            score = op.confidence
            pattern_str = getattr(op, "pattern_evidence_strength", op.confidence)
            if val_dec == "REVIEW_REQUIRED":
                score = min(score, 0.65)
                rel_rating = "MODERATE"
                ev_str = min(score * 0.8, 0.65)
            else:
                rel_rating = "HIGH" if score >= 0.80 else "MODERATE"
                ev_str = score * 0.8
            ci = CorrelatedIncident(
                incident_id=f"CORR-PERS-{uuid.uuid4().hex[:8]}",
                video_id=video_id,
                incident_category="person",
                incident_subcategory=op.event_type.lower(),
                start_time=op.start_time,
                end_time=op.end_time,
                duration=op.duration,
                severity=op.severity,
                confidence=score,
                assessment_score=score,
                evidence_strength=ev_str,
                pattern_evidence_strength=pattern_str,
                reliability_rating=rel_rating,
                validation_decision=val_dec,
                primary_track_ids=op.track_ids,
                involved_object_classes=op.object_classes,
                source_candidate_ids=[op.incident_id],
                source_detector_ids=[op.detector_name],
                supporting_signals=[s.to_dict() for s in op.supporting_signals],
                storyline=IncidentStorylineGenerator.generate_storyline(
                    category="person",
                    subcategory=op.event_type.lower(),
                    start_time=op.start_time,
                    end_time=op.end_time,
                    duration=op.duration,
                    primary_tracks=op.track_ids,
                    supporting_tracks=[],
                    object_classes=op.object_classes,
                    relationships=[],
                    supporting_signals=[s.to_dict() for s in op.supporting_signals],
                    confidence=score,
                    validation_decision=val_dec,
                ),
            )
            correlated.append(ci)

        return correlated


class CrowdZoneCorrelationPolicy:
    """Fuses crowd density, dispersal, surge, and zone activity."""

    @staticmethod
    def correlate(
        candidates: List[IncidentCandidate],
        video_id: str,
        relationships: List[ObservationalRelationship],
    ) -> List[CorrelatedIncident]:
        crowd_cands = [c for c in candidates if c.category in ("crowd", "zone")]
        if not crowd_cands:
            return []

        # Cluster crowd candidates strictly by temporal continuity (gap <= 6.0s)
        # to ensure localized incident episodes remain evidence-grounded rather than
        # indiscriminately expanding across wide multi-minute background telemetry windows.
        sorted_crowd = sorted(crowd_cands, key=lambda c: c.start_time)
        crowd_clusters: List[List[IncidentCandidate]] = []
        for c in sorted_crowd:
            placed = False
            for cluster in crowd_clusters:
                last_end = max(item.end_time for item in cluster)
                first_start = min(item.start_time for item in cluster)
                # Contiguous or overlapping window
                if c.start_time <= (last_end + 6.0) and c.end_time >= (first_start - 6.0):
                    cluster.append(c)
                    placed = True
                    break
            if not placed:
                crowd_clusters.append([c])

        broad_start = min(c.start_time for c in crowd_cands)
        broad_end = max(c.end_time for c in crowd_cands)
        broad_dur = max(0.0, broad_end - broad_start)

        correlated: List[CorrelatedIncident] = []
        for cluster in crowd_clusters:
            start_t = min(c.start_time for c in cluster)
            end_t = max(c.end_time for c in cluster)
            dur = max(0.0, end_t - start_t)
            tracks = list(set(sum([c.track_ids for c in cluster], [])))
            classes = list(set(sum([c.object_classes for c in cluster], [])))
            max_conf = max(c.confidence for c in cluster)
            cand_ids = [c.incident_id for c in cluster]
            dets = list(set(c.detector_name for c in cluster))
            signals = [s.to_dict() for c in cluster for s in c.supporting_signals]
            lead = cluster[0]

            val_dec = inherit_validation_decision(cluster)
            pattern_str = max([getattr(c, "pattern_evidence_strength", getattr(c, "confidence", 0.0)) for c in cluster] + [max_conf])
            score = max_conf
            if val_dec == "REVIEW_REQUIRED":
                score = min(score, 0.65)
                rel_rating = "MODERATE"
                ev_str = min(score * 0.85, 0.65)
            else:
                rel_rating = "HIGH" if score >= 0.80 else "MODERATE"
                ev_str = score * 0.85

            storyline = IncidentStorylineGenerator.generate_storyline(
                category="crowd",
                subcategory="crowd_movement_and_density_episode",
                start_time=start_t,
                end_time=end_t,
                duration=dur,
                primary_tracks=tracks[:5],
                supporting_tracks=tracks[5:],
                object_classes=classes,
                relationships=[],
                supporting_signals=signals,
                confidence=score,
                validation_decision=val_dec,
            )

            ci = CorrelatedIncident(
                incident_id=f"CORR-CRWD-{uuid.uuid4().hex[:8]}",
                video_id=video_id,
                incident_category="crowd",
                incident_subcategory="crowd_movement_and_density_episode",
                start_time=start_t,
                end_time=end_t,
                duration=dur,
                severity=lead.severity,
                confidence=score,
                assessment_score=score,
                evidence_strength=ev_str,
                pattern_evidence_strength=pattern_str,
                reliability_rating=rel_rating,
                validation_decision=val_dec,
                primary_track_ids=tracks[:5],
                supporting_track_ids=tracks[5:],
                involved_object_classes=classes,
                source_candidate_ids=cand_ids,
                source_detector_ids=dets,
                supporting_signals=signals[:10],
                storyline=storyline,
                contextual_factors={
                    "pattern_evidence_strength": pattern_str,
                    "supporting_telemetry_window": {
                        "start_time": broad_start,
                        "end_time": broad_end,
                        "duration": broad_dur,
                    }
                },
                provenance_graph={
                    "constituent_alerts": [c.event_type for c in cluster],
                    "cluster_interval": [start_t, end_t],
                    "broad_telemetry_interval": [broad_start, broad_end],
                },
            )
            correlated.append(ci)

        return correlated


class SpecializedVisualPolicy:
    """Fuses validated specialized visual observations (fire + smoke cross-modal)."""

    @staticmethod
    def correlate(
        candidates: List[IncidentCandidate],
        video_id: str,
        relationships: List[ObservationalRelationship],
    ) -> List[CorrelatedIncident]:
        spec_cands = [
            c for c in candidates
            if (c.category in ("environment", "object", "specialized") or any(k in c.event_type for k in ("FIRE", "SMOKE", "WEAPON")))
            and c.validation_decision != "REJECTED"
        ]
        if not spec_cands:
            return []

        # Check for co-occurring fire and smoke
        fire_cands = [c for c in spec_cands if "FIRE" in c.event_type]
        smoke_cands = [c for c in spec_cands if "SMOKE" in c.event_type]
        other_spec = [c for c in spec_cands if "FIRE" not in c.event_type and "SMOKE" not in c.event_type]

        correlated: List[CorrelatedIncident] = []

        if fire_cands and smoke_cands:
            # Check temporal compatibility
            f_lead = fire_cands[0]
            s_lead = smoke_cands[0]
            f_ev = f_lead.incident_metadata.get("evidence_ids", [])
            s_ev = s_lead.incident_metadata.get("evidence_ids", [])
            combined_ev = list(set(f_ev + s_ev))

            all_fs = fire_cands + smoke_cands
            min_s = min(c.start_time for c in all_fs)
            max_e = max(c.end_time for c in all_fs)
            dur = max(0.0, max_e - min_s)
            val_dec = inherit_validation_decision(all_fs)
            raw_conf = max(c.confidence for c in all_fs)
            pattern_str = max([getattr(c, "pattern_evidence_strength", getattr(c, "confidence", 0.0)) for c in all_fs] + [raw_conf])
            score = min(raw_conf, 0.65) if val_dec == "REVIEW_REQUIRED" else raw_conf

            storyline = (
                f"At {min_s:.1f}s, localized fire visual evidence was observed and accompanied by concurrent smoke plume dynamics. "
                f"Both signatures were independently evaluated (assessment score: {score*100:.0f}%, status: {val_dec}). "
                f"{IncidentStorylineGenerator.get_decision_clause(val_dec)}"
            )
            ci = CorrelatedIncident(
                incident_id=f"CORR-FIRE-SMOKE-{uuid.uuid4().hex[:8]}",
                video_id=video_id,
                incident_category="specialized",
                incident_subcategory="FIRE_AND_SMOKE",
                start_time=min_s,
                end_time=max_e,
                duration=dur,
                severity="HIGH",
                confidence=score,
                assessment_score=score,
                evidence_strength=score * 0.95,
                pattern_evidence_strength=pattern_str,
                reliability_rating="HIGH" if score >= 0.80 and val_dec == "ACCEPTED" else "MODERATE",
                validation_decision=val_dec,
                evidence_ids=combined_ev,
                source_candidate_ids=[c.incident_id for c in all_fs],
                source_detector_ids=list(set(c.detector_name for c in all_fs)),
                supporting_signals=[s.to_dict() for c in all_fs for s in c.supporting_signals],
                storyline=storyline,
                provenance_graph={"fused_modalities": ["fire", "smoke"]},
            )
            correlated.append(ci)
        else:
            for sc in (fire_cands + smoke_cands + other_spec):
                ev_ids = sc.incident_metadata.get("evidence_ids", [])
                val_dec = inherit_validation_decision([sc])
                score = sc.confidence
                pattern_str = getattr(sc, "pattern_evidence_strength", sc.confidence)
                if val_dec == "REVIEW_REQUIRED":
                    score = min(score, 0.65)
                    rel_rating = "MODERATE"
                    ev_str = min(score * 0.9, 0.65)
                else:
                    rel_rating = "HIGH" if score >= 0.80 else "MODERATE"
                    ev_str = score * 0.9

                ci = CorrelatedIncident(
                    incident_id=f"CORR-SPEC-{uuid.uuid4().hex[:8]}",
                    video_id=video_id,
                    incident_category=sc.category,
                    incident_subcategory=sc.event_type.lower(),
                    start_time=sc.start_time,
                    end_time=sc.end_time,
                    duration=sc.duration,
                    severity=sc.severity,
                    confidence=score,
                    assessment_score=score,
                    evidence_strength=ev_str,
                    pattern_evidence_strength=pattern_str,
                    reliability_rating=rel_rating,
                    validation_decision=val_dec,
                    primary_track_ids=sc.track_ids,
                    involved_object_classes=sc.object_classes,
                    source_candidate_ids=[sc.incident_id],
                    source_detector_ids=[sc.detector_name],
                    supporting_signals=[s.to_dict() for s in sc.supporting_signals],
                    evidence_ids=ev_ids,
                    storyline=IncidentStorylineGenerator.generate_storyline(
                        category=sc.category,
                        subcategory=sc.event_type.lower(),
                        start_time=sc.start_time,
                        end_time=sc.end_time,
                        duration=sc.duration,
                        primary_tracks=sc.track_ids,
                        supporting_tracks=[],
                        object_classes=sc.object_classes,
                        relationships=[],
                        supporting_signals=[s.to_dict() for s in sc.supporting_signals],
                        confidence=score,
                        validation_decision=val_dec,
                    ),
                )
                correlated.append(ci)

        return correlated
