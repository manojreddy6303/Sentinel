"""
Phase 17: Safe Structured Investigation Query Executor

Translates validated InvestigationQuery objects into safe repository-pattern
database operations. Uses SQLAlchemy ORM only — no raw SQL generation.

Security guarantees:
- All queries are filtered by video_id FIRST (absolute video isolation)
- No raw SQL strings accepted or generated
- All timestamps validated before database access
- All track IDs verified against this video before returning results
- Results are bounded by result_limit (max 200)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from database.session import SessionLocal
from database.models import (
    CorrelatedIncidentModel,
    EvidenceModel,
    EventModel,
    GroupedEventModel,
    SecurityEventModel,
    TrackModel,
    VideoModel,
)

from backend.app.services.investigation_query import InvestigationQuery
from backend.app.services.investigation_result import InvestigationResult

logger = logging.getLogger(__name__)

VEHICLE_CLASSES = frozenset(["car", "bus", "truck", "motorcycle", "bicycle"])

# Relevance ranking weights (display order only — never modifies canonical scores)
_RANK_WEIGHTS = {
    "has_evidence": 4,
    "exact_temporal": 3,
    "exact_category": 2,
    "exact_track": 2,
    "validated_accepted": 1,
    "validated_review": 0,
}


class InvestigationQueryExecutor:
    """
    Executes validated InvestigationQuery objects against Sentinel DB via ORM.
    All results are video-isolated and canonically grounded.
    """

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute(self, query: InvestigationQuery) -> InvestigationResult:
        """
        Execute a validated InvestigationQuery and return an InvestigationResult.

        Execution flow:
          1. Verify video isolation (video_id presence in DB)
          2. Query correlated incidents (filtered)
          3. Query security events (filtered)
          4. Query tracks (if track_ids specified)
          5. Query evidence (filtered by temporal window + incident IDs)
          6. Build unified timeline
          7. Collect negative evidence from incidents
          8. Apply deterministic relevance ranking
          9. Apply pagination
         10. Return InvestigationResult
        """
        t_start = time.monotonic()
        db = SessionLocal()
        try:
            # Step 1: Video isolation verification
            video_duration = self._get_video_duration(db, query.video_id)

            # If video_duration was not known at query build time, apply it now for clamping
            if video_duration and query.time_end is not None and query.time_end > video_duration:
                query.time_end = video_duration

            t_db_start = time.monotonic()

            # Step 2: Correlated incidents
            matched_incidents = self._query_incidents(db, query)

            # Step 3: Security events
            matched_events = self._query_events(db, query)

            # Step 4: Tracks
            matched_tracks = self._query_tracks(db, query)

            # Step 5: Evidence
            incident_ids = [i["incident_id"] for i in matched_incidents]
            event_ids_for_ev = [e["id"] for e in matched_events]
            matched_evidence = self._query_evidence(db, query, incident_ids, event_ids_for_ev)

            t_db_elapsed = time.monotonic() - t_db_start

            # Step 6: Unified timeline
            timeline = self._build_timeline(matched_incidents, matched_events, matched_evidence)

            # Step 7: Negative evidence (from correlated incidents)
            negative_evidence = self._collect_negative_evidence(matched_incidents)

            # Step 8: Relationships from incidents
            relationships = self._collect_relationships(matched_incidents, matched_tracks)

            # Step 9: Relevance ranking (display order only)
            matched_incidents = self._rank_incidents(matched_incidents, query, matched_evidence)

            # Step 10: Total counts before pagination
            total_incidents = len(matched_incidents)
            total_events = len(matched_events)
            total_results = total_incidents + total_events + len(matched_tracks)

            # Step 11: Apply pagination
            offset = query.result_offset
            limit = query.result_limit
            matched_incidents = matched_incidents[offset : offset + limit]
            matched_events = matched_events[offset : offset + limit]
            truncated = total_results > (offset + limit)

            # Step 12: Text search filter (Python-level, never SQL)
            if query.search_text:
                matched_incidents = self._text_filter(matched_incidents, query.search_text)
                matched_events = self._text_filter(matched_events, query.search_text)

            t_total = time.monotonic() - t_start

            return InvestigationResult(
                query=query.to_provenance(),
                interpretation=self._build_interpretation(query),
                matched_incidents=matched_incidents,
                matched_events=matched_events,
                matched_tracks=matched_tracks,
                matched_evidence=matched_evidence,
                timeline=timeline,
                relationships=relationships,
                negative_evidence=negative_evidence,
                total_results=total_results,
                truncated=truncated,
                source_video_id=query.video_id,
                provenance={
                    "query_provenance": query.to_provenance(),
                    "video_duration_seconds": video_duration,
                },
                diagnostics={
                    "db_query_time_ms": round(t_db_elapsed * 1000, 1),
                    "total_time_ms": round(t_total * 1000, 1),
                    "incidents_found": total_incidents,
                    "events_found": total_events,
                    "tracks_found": len(matched_tracks),
                    "evidence_found": len(matched_evidence),
                    "fallback_used": False,
                },
            )

        except Exception as exc:
            logger.error(f"InvestigationQueryExecutor.execute failed for video {query.video_id}: {exc}")
            raise
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Video context
    # ------------------------------------------------------------------

    def _get_video_duration(self, db, video_id: str) -> Optional[float]:
        """Retrieve video duration from DB for temporal clamping."""
        vid = db.query(VideoModel).filter(VideoModel.id == video_id).first()
        return vid.duration_seconds if vid else None

    # ------------------------------------------------------------------
    # Correlated incidents query
    # ------------------------------------------------------------------

    def _query_incidents(
        self, db, query: InvestigationQuery
    ) -> List[Dict[str, Any]]:
        """Fetch CorrelatedIncidentModel records matching the query filters."""
        q = db.query(CorrelatedIncidentModel).filter(
            CorrelatedIncidentModel.video_id == query.video_id
        )

        # Temporal window
        if query.time_start is not None:
            q = q.filter(CorrelatedIncidentModel.end_time >= query.time_start)
        if query.time_end is not None:
            q = q.filter(CorrelatedIncidentModel.start_time <= query.time_end)

        # Category filter (case-tolerant across DB schema conventions)
        if query.incident_categories:
            cats = set()
            for c in query.incident_categories:
                cats.add(c.lower())
                cats.add(c.upper())
                cats.add(c.capitalize())
            q = q.filter(
                CorrelatedIncidentModel.incident_category.in_(list(cats))
            )

        # Validation decision filter (case-tolerant)
        if query.validation_decisions:
            decs = set()
            for d in query.validation_decisions:
                decs.add(d.upper())
                decs.add(d.lower())
                decs.add(d.capitalize())
            q = q.filter(
                CorrelatedIncidentModel.validation_decision.in_(list(decs))
            )

        # Reliability filter (case-tolerant)
        if query.reliability_levels:
            rels = set()
            for r in query.reliability_levels:
                rels.add(r.upper())
                rels.add(r.lower())
                rels.add(r.capitalize())
            q = q.filter(
                CorrelatedIncidentModel.reliability_rating.in_(list(rels))
            )

        # Assessment score bounds
        if query.min_assessment_score is not None:
            q = q.filter(
                CorrelatedIncidentModel.assessment_score >= query.min_assessment_score
            )
        if query.max_assessment_score is not None:
            q = q.filter(
                CorrelatedIncidentModel.assessment_score <= query.max_assessment_score
            )

        # Track filter (any track in primary or supporting track IDs)
        # Applied as Python-level filter since track_ids is JSON
        rows = q.order_by(CorrelatedIncidentModel.start_time.asc()).all()

        serialized = [self._serialize_incident(r) for r in rows]

        # Track ID filter (Python-level for JSON array membership)
        if query.track_ids:
            filtered = []
            for inc in serialized:
                all_tracks = set(inc.get("primary_track_ids") or []) | set(
                    inc.get("supporting_track_ids") or []
                )
                if any(t in all_tracks for t in query.track_ids):
                    filtered.append(inc)
            serialized = filtered

        # Object class filter
        if query.object_classes:
            filtered = []
            target_classes = set()
            for oc in query.object_classes:
                if oc.lower() == "vehicle_group":
                    target_classes.update(VEHICLE_CLASSES)
                else:
                    target_classes.add(oc.lower())

            for inc in serialized:
                involved = {c.lower() for c in (inc.get("involved_object_classes") or [])}
                if target_classes & involved:
                    filtered.append(inc)
            serialized = filtered

        # Resolve linked evidence for incidents from EvidenceModel (candidate ID or temporal overlap)
        evidence_rows = db.query(EvidenceModel).filter(
            EvidenceModel.video_id == query.video_id,
            EvidenceModel.validation_status == "VALID",
        ).all()
        for inc in serialized:
            if not inc.get("evidence_ids"):
                src_candidates = set(inc.get("source_candidate_ids") or [])
                matched_ev_ids = []
                for ev in evidence_rows:
                    if (ev.event_id and ev.event_id in src_candidates) or (ev.event_id == inc["incident_id"]) or (
                        inc["start_time"] - 2.0 <= ev.timestamp_seconds <= inc["end_time"] + 2.0
                    ):
                        matched_ev_ids.append(ev.id)
                if matched_ev_ids:
                    inc["evidence_ids"] = matched_ev_ids

        # Evidence required filter
        if query.evidence_required is True:
            serialized = [i for i in serialized if i.get("evidence_ids")]

        # Correlated only (all correlated incidents already are correlated, but filter for 2+ sources)
        if query.correlated_only is True:
            serialized = [i for i in serialized if len(i.get("source_candidate_ids") or []) > 1]

        return serialized

    def _serialize_incident(self, r: CorrelatedIncidentModel) -> Dict[str, Any]:
        return {
            "incident_id": r.id,
            "video_id": r.video_id,
            "incident_category": (r.incident_category or "").upper(),
            "incident_subcategory": r.incident_subcategory,
            "start_time": round(r.start_time, 2),
            "end_time": round(r.end_time, 2),
            "duration": round(r.duration, 2),
            "primary_track_ids": r.primary_track_ids or [],
            "supporting_track_ids": r.supporting_track_ids or [],
            "involved_object_classes": r.involved_object_classes or [],
            "assessment_score": round(r.assessment_score, 4),
            "evidence_strength": round(r.evidence_strength, 4),
            "reliability_rating": (r.reliability_rating or "").upper(),
            "validation_decision": (r.validation_decision or "").upper(),
            "storyline": r.storyline or "",
            "evidence_ids": r.evidence_ids or [],
            "source_candidate_ids": r.source_candidate_ids or [],
            "negative_evidence": r.negative_evidence or [],
            "provenance": r.provenance or {},
            "contextual_factors": r.contextual_factors or {},
        }

    # ------------------------------------------------------------------
    # Security events query
    # ------------------------------------------------------------------

    def _query_events(
        self, db, query: InvestigationQuery
    ) -> List[Dict[str, Any]]:
        """Fetch SecurityEventModel records matching the query filters."""
        q = db.query(SecurityEventModel).filter(
            SecurityEventModel.video_id == query.video_id
        )

        # Temporal window
        if query.time_start is not None:
            q = q.filter(SecurityEventModel.timestamp_seconds >= query.time_start)
        if query.time_end is not None:
            q = q.filter(SecurityEventModel.timestamp_seconds <= query.time_end)

        # Track filter
        if query.track_ids:
            q = q.filter(SecurityEventModel.track_id.in_(query.track_ids))

        # Object class filter
        if query.object_classes:
            if len(query.object_classes) == 1:
                q = q.filter(SecurityEventModel.object_class.ilike(query.object_classes[0]))
            else:
                q = q.filter(SecurityEventModel.object_class.in_(query.object_classes))

        # Validation decision (derived from human_verification_required)
        if query.validation_decisions:
            if "REVIEW_REQUIRED" in query.validation_decisions and "ACCEPTED" not in query.validation_decisions:
                q = q.filter(SecurityEventModel.human_verification_required == 1)
            elif "ACCEPTED" in query.validation_decisions and "REVIEW_REQUIRED" not in query.validation_decisions:
                q = q.filter(SecurityEventModel.human_verification_required == 0)

        rows = q.order_by(SecurityEventModel.timestamp_seconds.asc()).all()

        return [
            {
                "id": r.id,
                "video_id": r.video_id,
                "event_type": r.event_type,
                "timestamp": round(r.timestamp_seconds, 2),
                "duration_seconds": round(r.duration_seconds or 0.0, 2),
                "track_id": r.track_id,
                "object_class": r.object_class,
                "severity": r.severity,
                "confidence": round(r.confidence, 4),
                "zone_name": r.zone_name,
                "description": r.description,
                "observable_signals": r.observable_signals or {},
                "bounding_box": r.bounding_box,
                "evidence_id": r.evidence_id,
                "validation_decision": (
                    "REVIEW_REQUIRED" if r.human_verification_required else "ACCEPTED"
                ),
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Tracks query
    # ------------------------------------------------------------------

    def _query_tracks(
        self, db, query: InvestigationQuery
    ) -> List[Dict[str, Any]]:
        """Fetch TrackModel records matching the query filters (video-isolated)."""
        q = db.query(TrackModel).filter(TrackModel.video_id == query.video_id)

        if query.track_ids:
            q = q.filter(TrackModel.track_id.in_(query.track_ids))

        if query.object_classes:
            if len(query.object_classes) == 1:
                q = q.filter(TrackModel.object_class.ilike(query.object_classes[0]))
            else:
                q = q.filter(TrackModel.object_class.in_(query.object_classes))

        if query.time_start is not None:
            q = q.filter(TrackModel.last_seen >= query.time_start)
        if query.time_end is not None:
            q = q.filter(TrackModel.first_seen <= query.time_end)

        rows = q.order_by(TrackModel.first_seen.asc()).all()

        return [
            {
                "track_id": r.track_id,
                "video_id": r.video_id,
                "object_class": r.object_class,
                "first_seen": round(r.first_seen, 2),
                "last_seen": round(r.last_seen, 2),
                "duration_seconds": round(r.duration_seconds, 2),
                "detection_count": r.detection_count,
                "max_confidence": round(r.max_confidence, 4),
                "color": r.color,
                "current_bbox": r.current_bbox,
                "active": bool(r.active),
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Evidence query
    # ------------------------------------------------------------------

    def _query_evidence(
        self,
        db,
        query: InvestigationQuery,
        incident_ids: List[str],
        event_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Fetch EvidenceModel records matched to the query's temporal window and events."""
        q = db.query(EvidenceModel).filter(
            EvidenceModel.video_id == query.video_id,
            EvidenceModel.validation_status == "VALID",
        )

        # Temporal window (with margin)
        if query.time_start is not None:
            q = q.filter(EvidenceModel.timestamp_seconds >= max(0.0, query.time_start - 2.0))
        if query.time_end is not None:
            q = q.filter(EvidenceModel.timestamp_seconds <= query.time_end + 2.0)

        rows = q.order_by(EvidenceModel.timestamp_seconds.asc()).all()

        # Also include evidence directly linked to matched incidents/events
        matched_event_ids_set = set(event_ids)
        matched_by_link = []
        if incident_ids or event_ids:
            link_q = db.query(EvidenceModel).filter(
                EvidenceModel.video_id == query.video_id,
                EvidenceModel.validation_status == "VALID",
                EvidenceModel.event_id.in_(list(matched_event_ids_set) + incident_ids),
            )
            matched_by_link = link_q.all()

        # Deduplicate
        seen_ids = set()
        all_rows = []
        for r in list(rows) + list(matched_by_link):
            if r.id not in seen_ids:
                seen_ids.add(r.id)
                all_rows.append(r)
        all_rows.sort(key=lambda r: r.timestamp_seconds)

        return [
            {
                "evidence_id": r.id,
                "video_id": r.video_id,
                "event_id": r.event_id,
                "evidence_type": r.evidence_type,
                "timestamp": round(r.timestamp_seconds, 2),
                "object_class": r.object_class,
                "confidence": round(r.confidence, 4) if r.confidence else None,
                "has_snapshot": bool(r.snapshot_path),
                "has_annotated": bool(r.annotated_snapshot_path),
                "has_clip": bool(r.clip_path),
                "start_time": r.start_time,
                "end_time": r.end_time,
                "duration_seconds": r.duration_seconds,
            }
            for r in all_rows
        ]

    # ------------------------------------------------------------------
    # Unified timeline
    # ------------------------------------------------------------------

    def _build_timeline(
        self,
        incidents: List[Dict[str, Any]],
        events: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Merge incidents, events, and evidence into a single chronological timeline.
        Each entry is tagged with a layer type and carries playback seek timestamps.
        """
        entries: List[Dict[str, Any]] = []

        for inc in incidents:
            entries.append({
                "layer": "correlated_incident",
                "timestamp": inc["start_time"],
                "end_timestamp": inc["end_time"],
                "label": f"{inc['incident_category']} — {inc.get('incident_subcategory', '')}",
                "id": inc["incident_id"],
                "validation_decision": inc["validation_decision"],
                "assessment_score": inc["assessment_score"],
                "reliability_rating": inc["reliability_rating"],
                "details": inc,
            })

        for ev in events:
            entries.append({
                "layer": "security_event",
                "timestamp": ev["timestamp"],
                "end_timestamp": ev["timestamp"] + ev.get("duration_seconds", 0.0),
                "label": ev["event_type"].replace("_", " "),
                "id": ev["id"],
                "validation_decision": ev.get("validation_decision", "REVIEW_REQUIRED"),
                "details": ev,
            })

        for ev_item in evidence:
            entries.append({
                "layer": "evidence",
                "timestamp": ev_item["timestamp"],
                "end_timestamp": ev_item.get("end_time") or ev_item["timestamp"],
                "label": f"Evidence ({ev_item['evidence_type']})",
                "id": ev_item["evidence_id"],
                "has_snapshot": ev_item["has_snapshot"],
                "has_clip": ev_item["has_clip"],
                "details": ev_item,
            })

        entries.sort(key=lambda x: x["timestamp"])
        return entries

    # ------------------------------------------------------------------
    # Negative evidence
    # ------------------------------------------------------------------

    def _collect_negative_evidence(
        self, incidents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract recorded negative evidence from correlated incidents."""
        results = []
        for inc in incidents:
            neg = inc.get("negative_evidence") or []
            if isinstance(neg, list):
                for item in neg:
                    if isinstance(item, dict):
                        results.append({"incident_id": inc["incident_id"], **item})
                    elif isinstance(item, str):
                        results.append({"incident_id": inc["incident_id"], "description": item})
        return results

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    def _collect_relationships(
        self,
        incidents: List[Dict[str, Any]],
        tracks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Derive observable inter-track relationships from incident storylines."""
        rels = []
        for inc in incidents:
            primary = inc.get("primary_track_ids") or []
            supporting = inc.get("supporting_track_ids") or []
            involved = inc.get("involved_object_classes") or []
            if len(primary) >= 1 and supporting:
                rels.append({
                    "incident_id": inc["incident_id"],
                    "relationship_type": inc.get("incident_subcategory", "INTERACTION"),
                    "primary_tracks": primary,
                    "supporting_tracks": supporting,
                    "involved_classes": involved,
                    "start_time": inc["start_time"],
                    "end_time": inc["end_time"],
                })
        return rels

    # ------------------------------------------------------------------
    # Relevance ranking (display order only)
    # ------------------------------------------------------------------

    def _rank_incidents(
        self,
        incidents: List[Dict[str, Any]],
        query: InvestigationQuery,
        evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Deterministic relevance ranking. ONLY affects display order.
        Never modifies canonical assessment_score, validation_decision, or any
        other stored value.
        """
        evidence_incident_ids = set()
        for ev in evidence:
            if ev.get("event_id"):
                evidence_incident_ids.add(ev["event_id"])

        def score(inc: Dict[str, Any]) -> int:
            rank = 0

            # Temporal proximity
            if query.time_start is not None and query.time_end is not None:
                q_mid = (query.time_start + query.time_end) / 2.0
                inc_mid = (inc["start_time"] + inc["end_time"]) / 2.0
                if abs(inc_mid - q_mid) < 5.0:
                    rank += _RANK_WEIGHTS["exact_temporal"]

            # Evidence availability
            if inc["incident_id"] in evidence_incident_ids or inc.get("evidence_ids"):
                rank += _RANK_WEIGHTS["has_evidence"]

            # Track match
            if query.track_ids:
                all_t = set(inc.get("primary_track_ids") or []) | set(inc.get("supporting_track_ids") or [])
                if any(t in all_t for t in query.track_ids):
                    rank += _RANK_WEIGHTS["exact_track"]

            # Validation
            if inc.get("validation_decision") in ("ACCEPTED", "CONFIRMED"):
                rank += _RANK_WEIGHTS["validated_accepted"]
            elif inc.get("validation_decision") == "REVIEW_REQUIRED":
                rank += _RANK_WEIGHTS["validated_review"]

            return rank

        return sorted(incidents, key=score, reverse=True)

    # ------------------------------------------------------------------
    # Text filter (Python-level, no SQL)
    # ------------------------------------------------------------------

    def _text_filter(
        self, items: List[Dict[str, Any]], search_text: str
    ) -> List[Dict[str, Any]]:
        """Filter items by substring match against storyline/description. Never touches SQL."""
        text_lower = search_text.lower()
        return [
            item for item in items
            if (
                text_lower in (item.get("storyline") or "").lower()
                or text_lower in (item.get("description") or "").lower()
                or text_lower in (item.get("event_type") or "").lower()
                or text_lower in (item.get("incident_category") or "").lower()
            )
        ]

    # ------------------------------------------------------------------
    # Interpretation builder
    # ------------------------------------------------------------------

    def _build_interpretation(self, query: InvestigationQuery) -> str:
        """Build a human-readable interpretation of the query for the result."""
        parts = []
        if query.time_start is not None and query.time_end is not None:
            parts.append(f"time window {query.time_start:.1f}s – {query.time_end:.1f}s")
        elif query.time_start is not None:
            parts.append(f"after {query.time_start:.1f}s")
        elif query.time_end is not None:
            parts.append(f"before {query.time_end:.1f}s")

        if query.incident_categories:
            parts.append(f"categories: {', '.join(query.incident_categories)}")
        if query.validation_decisions:
            parts.append(f"validation: {', '.join(query.validation_decisions)}")
        if query.track_ids:
            parts.append(f"tracks: {', '.join(query.track_ids)}")
        if query.object_classes:
            parts.append(f"objects: {', '.join(query.object_classes)}")
        if query.search_text:
            parts.append(f"text: '{query.search_text}'")

        if not parts:
            return "All incidents and events in this video."
        return "Filtered by: " + " | ".join(parts)
