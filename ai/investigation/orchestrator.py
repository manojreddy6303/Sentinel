"""
Sentinel Investigation Orchestrator

Coordinates the full AI investigation lifecycle:
1. Translates natural language inquiries into validated structured queries via LLM Provider
2. Enforces ethical guardrails (identity, facial recognition, criminal labeling)
3. Queries Sentinel database via existing InvestigationService (Ground Truth)
4. Correlates preserved evidence records from EvidenceModel
5. Performs observational activity analysis and video summarization
6. Generates strictly grounded responses citing timestamps, detections, and evidence
7. Provides seamless deterministic fallback if LLM provider is unavailable
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy import func

from backend.app.core.config import settings
from backend.app.services.investigation_service import InvestigationService
from ai.investigation.provider import LLMProvider, get_llm_provider, LLMProviderError
from ai.investigation.validator import StructuredIntentValidator, ValidationError
from database.session import SessionLocal
from database.models import VideoModel, EventModel, GroupedEventModel, EvidenceModel

logger = logging.getLogger(__name__)

STANDARD_LIMITATION = (
    "Sentinel analysis describes observable video detections and does not "
    "establish identity, intent, or criminal culpability."
)


class InvestigationOrchestrator:
    """Orchestrates LLM-assisted, evidence-grounded investigations."""

    def __init__(self, provider: Optional[LLMProvider] = None):
        self.provider = provider or get_llm_provider()
        self.investigation_service = InvestigationService()

    def process_investigation(
        self,
        video_id: str,
        user_query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point for AI investigation inquiries.
        """
        cleaned_query = (user_query or "").strip()
        if not cleaned_query:
            return {
                "video_id": video_id,
                "query": user_query,
                "mode": "deterministic_fallback",
                "is_supported": False,
                "answer": "Query cannot be empty. Please enter an investigation question.",
                "structured_query": {},
                "sources": {"detections": [], "events": [], "evidence": []},
                "confidence": None,
                "limitations": [STANDARD_LIMITATION],
            }

        # Step 1: Check ethical guardrails before any external LLM calls
        guardrail_result = StructuredIntentValidator.check_guardrails(cleaned_query)
        if guardrail_result:
            return {
                "video_id": video_id,
                "query": cleaned_query,
                "mode": "guardrail_enforced",
                "is_supported": False,
                "answer": guardrail_result["message"],
                "structured_query": {},
                "sources": {"detections": [], "events": [], "evidence": []},
                "confidence": "1.0",
                "limitations": [STANDARD_LIMITATION],
            }

        # Step 2: Determine if LLM provider is available
        llm_available = self.provider.is_available()

        # Step 3: Extract structured intent
        structured_intent: Dict[str, Any] = {}
        used_mode = "ai_assisted" if llm_available else "deterministic_fallback"

        if llm_available:
            try:
                raw_intent = self.provider.generate_structured_intent(cleaned_query, history)
                structured_intent = StructuredIntentValidator.validate_and_sanitize(
                    raw_intent, cleaned_query
                )
                if not structured_intent.get("is_supported", True):
                    return {
                        "video_id": video_id,
                        "query": cleaned_query,
                        "mode": "guardrail_enforced",
                        "is_supported": False,
                        "answer": structured_intent.get("message", "Inquiry rejected by safety guardrails."),
                        "structured_query": {},
                        "sources": {"detections": [], "events": [], "evidence": []},
                        "confidence": None,
                        "limitations": [STANDARD_LIMITATION],
                    }
            except (LLMProviderError, ValidationError) as err:
                logger.warning(f"LLM intent generation/validation failed: {err}. Falling back to deterministic analysis.")
                used_mode = "deterministic_fallback"
                structured_intent = self._deterministic_intent_fallback(cleaned_query)
        else:
            structured_intent = self._deterministic_intent_fallback(cleaned_query)

        # Step 4: Branch on intent type
        if structured_intent.get("is_summary_request"):
            return self._handle_video_summary(video_id, cleaned_query, used_mode)

        if structured_intent.get("is_activity_request"):
            return self._handle_activity_analysis(video_id, cleaned_query, structured_intent, used_mode)

        return self._handle_filtered_investigation(video_id, cleaned_query, structured_intent, history, used_mode)

    def _deterministic_intent_fallback(self, query: str) -> Dict[str, Any]:
        """Use Phase 5A parser to extract structured intent when LLM is unavailable."""
        q = query.lower()
        if any(w in q for w in ["summary", "summarize", "overview", "what happened in this video"]):
            return {
                "intent": "summarize",
                "is_supported": True,
                "is_summary_request": True,
                "is_activity_request": False,
                "object_class": None,
                "start_time": None,
                "end_time": None,
                "min_confidence": None,
                "result_type": "events",
            }
        if any(w in q for w in ["theft", "stealing", "takeaway", "burglary", "stolen"]):
            return {
                "intent": "investigate",
                "is_supported": True,
                "is_summary_request": False,
                "is_activity_request": False,
                "object_class": None,
                "start_time": None,
                "end_time": None,
                "min_confidence": None,
                "event_type": "POTENTIAL_THEFT",
                "result_type": "security_events",
            }

        if any(w in q for w in ["suspicious", "unusual", "activity", "noteworthy", "security event", "security events", "review"]):
            return {
                "intent": "activity_analysis",
                "is_supported": True,
                "is_summary_request": False,
                "is_activity_request": True,
                "object_class": None,
                "start_time": None,
                "end_time": None,
                "min_confidence": None,
                "result_type": "events",
            }

        parsed = self.investigation_service.parser.parse_query(query)
        filters = parsed.get("interpreted_filters", {})
        return {
            "intent": "investigate",
            "is_supported": parsed.get("is_supported", False),
            "is_summary_request": False,
            "is_activity_request": False,
            "object_class": filters.get("object_class"),
            "start_time": filters.get("start_time"),
            "end_time": filters.get("end_time"),
            "min_confidence": filters.get("min_confidence"),
            "event_type": filters.get("event_type"),
            "category": filters.get("category"),
            "track_id": filters.get("track_id"),
            "color": filters.get("color"),
            "result_type": parsed.get("result_type", "detections"),
            "fallback_message": parsed.get("message"),
        }

    def _handle_video_summary(self, video_id: str, query: str, mode: str) -> Dict[str, Any]:
        """Aggregate ground truth summary from database records."""
        db = SessionLocal()
        try:
            vid = db.query(VideoModel).filter(VideoModel.id == video_id).first()
            duration_seconds: Optional[float] = None

            if vid:
                duration_seconds = vid.duration_seconds
            else:
                # Fallback: check if detection events exist in DB or metadata sidecar exists on disk
                events_count = db.query(EventModel).filter(EventModel.video_id == video_id).count()
                meta_path = settings.STORAGE_UPLOADS_DIR / f"{video_id}.json"
                if events_count == 0 and not meta_path.exists():
                    return {
                        "video_id": video_id,
                        "query": query,
                        "mode": mode,
                        "is_supported": True,
                        "answer": f"Video with ID '{video_id}' has not been processed or does not exist.",
                        "structured_query": {"is_summary_request": True},
                        "sources": {"detections": [], "events": [], "evidence": []},
                        "limitations": [STANDARD_LIMITATION],
                    }
                if meta_path.exists():
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f_meta:
                            meta_data = json.load(f_meta)
                            duration_seconds = meta_data.get("duration_seconds")
                    except Exception:
                        pass

            # Breakdown by object class (validated records only)
            class_counts = (
                db.query(EventModel.object_class, func.count(EventModel.id))
                .filter(EventModel.video_id == video_id, EventModel.validation_status == "VALID")
                .group_by(EventModel.object_class)
                .all()
            )
            detected_classes = {cls: count for cls, count in class_counts}
            total_detections = sum(detected_classes.values())

            # Query unique anonymous tracks
            from database.models import TrackModel
            track_counts = (
                db.query(TrackModel.object_class, func.count(TrackModel.id))
                .filter(TrackModel.video_id == video_id)
                .group_by(TrackModel.object_class)
                .all()
            )
            detected_tracks = {cls: count for cls, count in track_counts}
            total_tracks = sum(detected_tracks.values())

            # Timeline events count
            grouped_events = (
                db.query(GroupedEventModel)
                .filter(GroupedEventModel.video_id == video_id)
                .order_by(GroupedEventModel.start_time.asc())
                .all()
            )

            # Preserved evidence items (validated only)
            evidence_items = (
                db.query(EvidenceModel)
                .filter(EvidenceModel.video_id == video_id, EvidenceModel.validation_status == "VALID")
                .order_by(EvidenceModel.timestamp_seconds.asc())
                .all()
            )

            # Query security events
            from database.models import SecurityEventModel
            sec_events = (
                db.query(SecurityEventModel)
                .filter(SecurityEventModel.video_id == video_id)
                .order_by(SecurityEventModel.timestamp_seconds.asc())
                .all()
            )
            theft_sec = [e for e in sec_events if e.event_type in ("POTENTIAL_THEFT", "POTENTIAL_OBJECT_TAKEAWAY")]

            # Find peak activity window
            peak_window = None
            if grouped_events:
                max_event = max(grouped_events, key=lambda e: e.total_detections)
                peak_window = f"{max_event.start_time:.2f}s – {max_event.end_time:.2f}s ({max_event.total_detections} detections)"

            duration_str = f"{duration_seconds:.2f} seconds" if duration_seconds is not None else "Unknown"

            # Format grounded response with strict detection observation vs track semantics
            class_lines_list = []
            for c, cnt in sorted(detected_classes.items()):
                t_cnt = detected_tracks.get(c, 0)
                if t_cnt > 0:
                    class_lines_list.append(f"{c.capitalize()} ({cnt} observations, {t_cnt} anonymous tracks)")
                else:
                    class_lines_list.append(f"{c.capitalize()} ({cnt} observations)")
            class_lines = ", ".join(class_lines_list) if class_lines_list else "None"

            answer_parts = [
                f"### VIDEO ACTIVITY SUMMARY",
                f"- **Duration:** {duration_str}",
                f"- **Validated Detection Observations:** {total_detections}",
                f"- **Anonymous Tracks:** {total_tracks}",
                f"- **Detected Object Classes:** {class_lines}",
                f"- **Timeline Events:** {len(grouped_events)}",
            ]
            if peak_window:
                answer_parts.append(f"- **Peak Activity Window:** {peak_window}")
            if sec_events:
                answer_parts.append(f"- **Security Events:** {len(sec_events)}")
                if theft_sec:
                    th = theft_sec[0]
                    answer_parts.append(f"- **Security Findings:** Potential object-takeaway pattern observed around {th.timestamp_seconds:.1f}s.")
            answer_parts.append(f"- **Preserved Evidence Items:** {len(evidence_items)}")

            formatted_evidence = [
                {
                    "evidence_id": ev.id,
                    "timestamp": ev.timestamp_seconds,
                    "evidence_type": ev.evidence_type,
                    "has_snapshot": bool(ev.snapshot_path),
                    "has_annotated": bool(ev.annotated_snapshot_path),
                    "has_clip": bool(ev.clip_path),
                }
                for ev in evidence_items
            ]

            formatted_events = [
                {
                    "event_id": ge.id,
                    "event_type": ge.event_type,
                    "start_time": ge.start_time,
                    "end_time": ge.end_time,
                    "total_detections": ge.total_detections,
                    "priority": ge.priority,
                }
                for ge in grouped_events[:8]
            ]

            return {
                "video_id": video_id,
                "query": query,
                "mode": mode,
                "is_supported": True,
                "answer": "\n".join(answer_parts),
                "structured_query": {"intent": "summarize", "is_summary_request": True},
                "summary": {
                    "duration_seconds": duration_seconds,
                    "total_detections": total_detections,
                    "detected_classes": detected_classes,
                    "total_events": len(grouped_events),
                    "peak_window": peak_window,
                    "total_evidence": len(evidence_items),
                },
                "sources": {
                    "detections": [],
                    "events": formatted_events,
                    "evidence": formatted_evidence,
                },
                "limitations": [STANDARD_LIMITATION],
            }
        finally:
            db.close()

    def _handle_activity_analysis(
        self, video_id: str, query: str, structured_intent: Dict[str, Any], mode: str
    ) -> Dict[str, Any]:
        """Analyze observable detection density and temporal clustering."""
        db = SessionLocal()
        try:
            from database.models import SecurityEventModel
            sec_events = (
                db.query(SecurityEventModel)
                .filter(SecurityEventModel.video_id == video_id)
                .order_by(SecurityEventModel.timestamp_seconds.asc())
                .all()
            )
            theft_events = [e for e in sec_events if e.event_type == "POTENTIAL_THEFT"]
            is_theft_query = any(w in query.lower() for w in ["theft", "stealing", "takeaway", "burglary", "stolen"])

            if theft_events:
                th = theft_events[0]
                th_end = th.timestamp_seconds + (th.duration_seconds or 0.0)
                answer = (
                    f"Sentinel identified a potential object-takeaway pattern around [{th.timestamp_seconds:.1f}s – {th_end:.1f}s]. "
                    f"{th.description} Review the linked evidence."
                )
                evidence_records = (
                    db.query(EvidenceModel)
                    .filter(
                        EvidenceModel.video_id == video_id,
                        ((EvidenceModel.event_id == th.id) |
                         ((EvidenceModel.timestamp_seconds >= th.timestamp_seconds - 2.0) &
                          (EvidenceModel.timestamp_seconds <= th_end + 2.0)))
                    )
                    .all()
                )
                formatted_evidence = [
                    {
                        "evidence_id": ev.id,
                        "timestamp": ev.timestamp_seconds,
                        "evidence_type": ev.evidence_type,
                        "has_snapshot": bool(ev.snapshot_path),
                        "has_annotated": bool(ev.annotated_snapshot_path),
                        "has_clip": bool(ev.clip_path),
                    }
                    for ev in evidence_records
                ]
                formatted_events = [
                    {
                        "event_id": e.id,
                        "event_type": e.event_type,
                        "start_time": e.timestamp_seconds,
                        "end_time": e.timestamp_seconds + (e.duration_seconds or 0.0),
                        "total_detections": 1,
                        "priority": e.severity,
                    }
                    for e in theft_events
                ]
                return {
                    "video_id": video_id,
                    "query": query,
                    "mode": mode,
                    "is_supported": True,
                    "answer": answer,
                    "structured_query": structured_intent,
                    "sources": {"detections": [], "events": formatted_events, "evidence": formatted_evidence},
                    "limitations": [STANDARD_LIMITATION],
                }

            if is_theft_query and not theft_events:
                return {
                    "video_id": video_id,
                    "query": query,
                    "mode": mode,
                    "is_supported": True,
                    "answer": "No potential theft pattern was detected in the available visual evidence.",
                    "structured_query": structured_intent,
                    "sources": {"detections": [], "events": [], "evidence": []},
                    "limitations": [STANDARD_LIMITATION],
                }

            # Query grouped events
            grouped_events = (
                db.query(GroupedEventModel)
                .filter(GroupedEventModel.video_id == video_id)
                .order_by(GroupedEventModel.total_detections.desc())
                .all()
            )

            if not grouped_events:
                return {
                    "video_id": video_id,
                    "query": query,
                    "mode": mode,
                    "is_supported": True,
                    "answer": "No notable detection clustering or elevated activity was observed in this video.",
                    "structured_query": structured_intent,
                    "sources": {"detections": [], "events": [], "evidence": []},
                    "limitations": [STANDARD_LIMITATION],
                }

            # Top activity events
            top_events = grouped_events[:3]
            event_summaries = []
            for ev in top_events:
                event_summaries.append(
                    f"• At [{ev.start_time:.2f}s – {ev.end_time:.2f}s], {ev.total_detections} detections "
                    f"were clustered ({ev.event_type}) with max confidence of {round(ev.max_confidence * 100)}% [EVENT-{ev.id[:8]}]."
                )

            # Check evidence in top window
            top_event = top_events[0]
            evidence_records = (
                db.query(EvidenceModel)
                .filter(
                    EvidenceModel.video_id == video_id,
                    EvidenceModel.timestamp_seconds >= top_event.start_time - 1.0,
                    EvidenceModel.timestamp_seconds <= top_event.end_time + 1.0,
                )
                .all()
            )

            ev_status = (
                f"Preserved evidence is available for this period ({len(evidence_records)} item(s))."
                if evidence_records
                else "No preserved evidence currently exists for this specific window."
            )

            answer = (
                f"Potentially noteworthy activity was observed around [{top_event.start_time:.2f}s – {top_event.end_time:.2f}s] "
                f"because detection density was elevated relative to other periods in this surveillance footage.\n\n"
                f"**Key Observations:**\n" + "\n".join(event_summaries) + "\n\n"
                f"**Evidence Status:** {ev_status}\n\n"
                f"*Note: This is an automated observational finding based on detection density and does not "
                f"establish criminal or malicious behavior.*"
            )

            formatted_evidence = [
                {
                    "evidence_id": ev.id,
                    "timestamp": ev.timestamp_seconds,
                    "evidence_type": ev.evidence_type,
                    "has_snapshot": bool(ev.snapshot_path),
                    "has_annotated": bool(ev.annotated_snapshot_path),
                    "has_clip": bool(ev.clip_path),
                }
                for ev in evidence_records
            ]

            formatted_events = [
                {
                    "event_id": ge.id,
                    "event_type": ge.event_type,
                    "start_time": ge.start_time,
                    "end_time": ge.end_time,
                    "total_detections": ge.total_detections,
                    "priority": ge.priority,
                }
                for ge in top_events
            ]

            return {
                "video_id": video_id,
                "query": query,
                "mode": mode,
                "is_supported": True,
                "answer": answer,
                "structured_query": structured_intent,
                "sources": {
                    "detections": [],
                    "events": formatted_events,
                    "evidence": formatted_evidence,
                },
                "limitations": [STANDARD_LIMITATION],
            }
        finally:
            db.close()

    def _handle_filtered_investigation(
        self,
        video_id: str,
        query: str,
        structured_intent: Dict[str, Any],
        history: Optional[List[Dict[str, str]]],
        mode: str,
    ) -> Dict[str, Any]:
        """Execute filtered investigation query and synthesize grounded answer."""
        if not structured_intent.get("is_supported", True):
            return {
                "video_id": video_id,
                "query": query,
                "mode": mode,
                "is_supported": False,
                "answer": structured_intent.get(
                    "fallback_message",
                    "This inquiry could not be parsed into a supported Sentinel filter.",
                ),
                "structured_query": structured_intent,
                "sources": {"detections": [], "events": [], "evidence": []},
                "limitations": [STANDARD_LIMITATION],
            }

        filters = {
            "object_class": structured_intent.get("object_class"),
            "start_time": structured_intent.get("start_time"),
            "end_time": structured_intent.get("end_time"),
            "min_confidence": structured_intent.get("min_confidence"),
            "event_type": structured_intent.get("event_type"),
            "category": structured_intent.get("category"),
            "track_id": structured_intent.get("track_id"),
            "color": structured_intent.get("color"),
        }
        result_type = structured_intent.get("result_type", "detections")

        # Query ground truth via InvestigationService
        db_res = self.investigation_service.investigate_filters(
            video_id=video_id,
            filters=filters,
            result_type=result_type,
            query_text=query,
        )

        results = db_res.get("results", [])
        total_count = db_res.get("count", len(results))

        # Check for matching evidence records
        matched_evidence = self._find_relevant_evidence(video_id, filters, results)

        # Package ground truth payload
        retrieval_payload = {
            "query": query,
            "filters": filters,
            "result_type": result_type,
            "count": total_count,
            "results": results[:20],  # Bound context payload
            "evidence": matched_evidence,
        }

        # Synthesize answer
        answer = ""
        if mode == "ai_assisted":
            try:
                answer = self.provider.generate_grounded_response(
                    user_query=query,
                    retrieved_data=retrieval_payload,
                    history=history,
                    context_notes=f"Found {total_count} total {result_type} records in database.",
                )
            except Exception as e:
                logger.warning(f"LLM grounded response synthesis failed: {e}. Falling back to template.")
                answer = self._format_deterministic_grounded_response(retrieval_payload)
        else:
            answer = self._format_deterministic_grounded_response(retrieval_payload)

        # Format sources for front-end clickable citations
        detections_sources = [
            {
                "event_id": r.get("event_id") or r.get("id"),
                "timestamp": r.get("timestamp") if r.get("timestamp") is not None else r.get("start_time"),
                "object_class": r.get("object_class"),
                "event_type": r.get("event_type"),
                "confidence": r.get("confidence") if r.get("confidence") is not None else r.get("max_confidence", 1.0),
                "bounding_box": r.get("bounding_box"),
            }
            for r in results[:15]
            if "bounding_box" in r or "timestamp" in r or "start_time" in r
        ]

        event_sources = [
            {
                "event_id": r.get("event_id") or r.get("id"),
                "event_type": r.get("event_type"),
                "start_time": r.get("start_time") if r.get("start_time") is not None else r.get("timestamp"),
                "end_time": r.get("end_time") if r.get("end_time") is not None else r.get("timestamp"),
                "total_detections": r.get("total_detections", 1),
            }
            for r in results[:10]
            if "event_type" in r
        ]

        return {
            "video_id": video_id,
            "query": query,
            "mode": mode,
            "is_supported": True,
            "answer": answer,
            "structured_query": structured_intent,
            "count": total_count,
            "sources": {
                "detections": detections_sources,
                "events": event_sources,
                "evidence": matched_evidence,
            },
            "limitations": [STANDARD_LIMITATION],
        }

    def _find_relevant_evidence(
        self, video_id: str, filters: Dict[str, Any], results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find preserved evidence records correlating to the queried window or events."""
        db = SessionLocal()
        try:
            q = db.query(EvidenceModel).filter(EvidenceModel.video_id == video_id)

            start_t = filters.get("start_time")
            end_t = filters.get("end_time")

            if start_t is not None and end_t is not None:
                q = q.filter(
                    EvidenceModel.timestamp_seconds >= max(0.0, start_t - 1.5),
                    EvidenceModel.timestamp_seconds <= end_t + 1.5,
                )
            elif start_t is not None:
                q = q.filter(EvidenceModel.timestamp_seconds >= max(0.0, start_t - 1.5))
            elif end_t is not None:
                q = q.filter(EvidenceModel.timestamp_seconds <= end_t + 1.5)

            ev_rows = q.order_by(EvidenceModel.timestamp_seconds.asc()).all()

            return [
                {
                    "evidence_id": ev.id,
                    "event_id": ev.event_id,
                    "timestamp": round(ev.timestamp_seconds, 2),
                    "evidence_type": ev.evidence_type,
                    "object_class": ev.object_class,
                    "confidence": round(ev.confidence, 4) if ev.confidence else None,
                    "has_snapshot": bool(ev.snapshot_path),
                    "has_annotated": bool(ev.annotated_snapshot_path),
                    "has_clip": bool(ev.clip_path),
                    "duration_seconds": ev.duration_seconds,
                    "start_time": ev.start_time,
                    "end_time": ev.end_time,
                }
                for ev in ev_rows
            ]
        finally:
            db.close()

    def _format_deterministic_grounded_response(self, payload: Dict[str, Any]) -> str:
        """Deterministic grounded synthesizer when LLM is offline or in fallback mode."""
        count = payload.get("count", 0)
        results = payload.get("results", [])
        evidence = payload.get("evidence", [])
        filters = payload.get("filters", {})

        target_category = filters.get("category")
        target_type = filters.get("event_type")
        is_vehicle_query = (
            target_category == "vehicle"
            or (target_type and (
                "VEHICLE" in target_type
                or target_type in ("POTENTIAL_NEAR_COLLISION", "POTENTIAL_SUDDEN_VEHICLE_STOP", "POTENTIAL_UNUSUAL_VEHICLE_TRAJECTORY")
            ))
        )

        PERSON_EVENT_TYPES = [
            "POTENTIAL_PERSON_FALL",
            "POTENTIAL_PERSON_DOWN",
            "POTENTIAL_PANIC_RUNNING",
            "UNUSUAL_RAPID_PERSON_MOVEMENT",
            "POTENTIAL_PHYSICAL_ALTERCATION",
            "POTENTIAL_FORCED_MOVEMENT",
            "PERSON_FOLLOWING",
            "COORDINATED_PERSON_MOVEMENT",
        ]
        is_person_query = (
            target_category == "person"
            or (target_type and target_type in PERSON_EVENT_TYPES)
        )

        if is_person_query:
            if count > 0 and results:
                first = results[0]
                ts = first.get("timestamp", 0.0)
                desc = first.get("description", "")
                ev_type = first.get("event_type", "PERSON_INCIDENT")
                return (
                    f"Sentinel identified {count} verified person incident event(s). "
                    f"Earliest observation ({ev_type.replace('_', ' ')}) around {ts:.1f}s. {desc}"
                )
            return "No reliable person incident was detected in the available Sentinel data."

        if is_vehicle_query:
            if count > 0 and results:
                first = results[0]
                ts = first.get("timestamp", 0.0)
                desc = first.get("description", "")
                ev_type = first.get("event_type", "VEHICLE_INCIDENT")
                return (
                    f"Sentinel identified {count} verified vehicle incident event(s). "
                    f"Earliest observation ({ev_type.replace('_', ' ')}) around {ts:.1f}s. {desc}"
                )
            return "No reliable vehicle incident was detected in the available Sentinel data."

        if filters.get("event_type") == "POTENTIAL_THEFT":
            if count > 0 and results:
                first = results[0]
                ts = first.get("timestamp", 0.0)
                desc = first.get("description", "")
                return (
                    f"Sentinel identified a Potential Theft Pattern (potential object-takeaway pattern) around {ts:.1f}s "
                    f"(Human verification required). {desc} Review the linked forensic evidence."
                )
            return "No potential theft pattern was detected in the available visual evidence."

        if count == 0:
            return "No matching Sentinel data was found for this question."

        obj_class = filters.get("object_class") or "detected object"
        time_desc = ""
        if filters.get("start_time") is not None and filters.get("end_time") is not None:
            time_desc = f" between {filters['start_time']}s and {filters['end_time']}s"
        elif filters.get("start_time") is not None:
            time_desc = f" after {filters['start_time']}s"
        elif filters.get("end_time") is not None:
            time_desc = f" before {filters['end_time']}s"

        lines = [f"Found {count} matching records{time_desc}."]

        if results:
            first = results[0]
            ts = first.get("timestamp") if first.get("timestamp") is not None else first.get("start_time")
            cls = first.get("object_class") or first.get("event_type", obj_class)
            conf = first.get("confidence") or first.get("max_confidence")
            conf_val = conf if conf is not None else 0.50

            # Distinguish weak isolated single-frame raw prediction from validated multi-frame activity
            if count == 1 and conf_val < 0.50:
                ts_str = f"at {ts:.1f}s" if ts is not None else ""
                lines.append(
                    f"One low-confidence model prediction for {cls.upper()} was observed {ts_str} "
                    f"({round(conf_val * 100)}% confidence). It did not persist across frames and was "
                    f"not confirmed as persistent visual intelligence."
                )
            elif count > 1:
                valid_ts = [
                    (r.get("timestamp") if r.get("timestamp") is not None else r.get("start_time"))
                    for r in results
                    if (r.get("timestamp") is not None or r.get("start_time") is not None)
                ]
                if valid_ts:
                    min_ts = min(valid_ts)
                    max_ts = max(valid_ts)
                    if max_ts > min_ts:
                        lines.append(f"{cls.upper()} activity was observed around [{min_ts:.1f}s – {max_ts:.1f}s].")
                    else:
                        lines.append(f"{cls.upper()} activity was observed around [{min_ts:.1f}s].")
                else:
                    lines.append(f"{cls.upper()} activity was observed across multiple frames.")
            else:
                ts_str = f"around [{ts:.1f}s]" if ts is not None else ""
                lines.append(f"{cls.upper()} activity was observed {ts_str} with {round(conf_val * 100)}% confidence.")

        if evidence:
            ev = evidence[0]
            lines.append(f"Preserved evidence is available: snapshot ({'yes' if ev.get('has_snapshot') else 'no'}), clip ({'yes' if ev.get('has_clip') else 'no'}).")
        else:
            lines.append("No preserved evidence currently exists for this event.")

        return " ".join(lines)
