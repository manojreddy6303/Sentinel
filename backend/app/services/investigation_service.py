"""
Investigation Service Module for Sentinel

Handles structured execution of parsed queries against the relational database:
- Translates parsed filters into database operations using SQLAlchemy models
- Supports detections listing, grouped events listing, and exact record counts
- Ensures results are chronologically ordered and grounded exclusively in actual DB records
- Rejects hallucinations and invalid queries
"""

import logging
from typing import Dict, Any, List, Optional
from database.session import SessionLocal
from database.models import (
    EventModel,
    GroupedEventModel,
    VideoModel,
    TrackModel,
    VehicleAttributeModel,
    FaceDetectionModel,
    SecurityEventModel,
    SpecializedObservationModel,
    CorrelatedIncidentModel,
    EvidenceModel,
)
from backend.app.services.investigation_parser import InvestigationParser

logger = logging.getLogger(__name__)

VEHICLE_CLASSES = ["car", "bus", "truck", "motorcycle", "bicycle"]


class InvestigationService:
    """Executes natural-language queries against Sentinel's detection and event database."""

    def __init__(self):
        self.parser = InvestigationParser()

    def investigate(self, video_id: str, query_text: str) -> Dict[str, Any]:
        """
        Parse and execute an investigation query for a given video.

        Returns structured response with count, interpreted filters, and records.
        """
        if not query_text or not query_text.strip():
            return {
                "query": query_text,
                "is_supported": False,
                "message": "Query cannot be empty. Please enter an investigation question.",
                "interpreted_filters": {},
                "result_type": "error",
                "count": 0,
                "results": [],
            }

        parsed = self.parser.parse_query(query_text)
        if not parsed["is_supported"]:
            return {
                "query": query_text,
                "is_supported": False,
                "message": parsed.get("message", "This investigation query is not currently supported."),
                "interpreted_filters": {},
                "result_type": "unsupported",
                "count": 0,
                "results": [],
            }

        filters = parsed["interpreted_filters"]
        result_type = parsed["result_type"]
        return self.investigate_filters(video_id, filters, result_type, query_text)

    def investigate_filters(
        self,
        video_id: str,
        filters: Dict[str, Any],
        result_type: str = "detections",
        query_text: str = "",
    ) -> Dict[str, Any]:
        """
        Directly execute structured filters against Sentinel database records.
        Preserves deterministic database execution for LLM orchestrator queries.
        """
        db = SessionLocal()
        try:
            # 1. Check if video exists
            video_exists = db.query(VideoModel).filter(VideoModel.id == video_id).first()
            # If not in DB table, verify if events exist with this video_id
            if not video_exists:
                events_count = db.query(EventModel).filter(EventModel.video_id == video_id, EventModel.validation_status == "VALID").count()
                if events_count == 0:
                    return {
                        "query": query_text,
                        "is_supported": True,
                        "message": f"Video with ID '{video_id}' has not been processed or does not exist.",
                        "interpreted_filters": filters,
                        "result_type": result_type,
                        "count": 0,
                        "results": [],
                    }

            # 2. Execute query based on result_type
            if result_type == "events":
                return self._query_grouped_events(db, video_id, query_text, filters)
            elif result_type == "count":
                return self._query_counts(db, video_id, query_text, filters)
            elif result_type == "vehicle_attributes":
                return self._query_vehicle_attributes(db, video_id, query_text, filters)
            elif result_type == "tracks":
                return self._query_tracks(db, video_id, query_text, filters)
            elif result_type == "faces":
                return self._query_faces(db, video_id, query_text, filters)
            elif result_type == "security_events":
                return self._query_security_events(db, video_id, query_text, filters)
            elif result_type == "specialized":
                return self._query_specialized_observations(db, video_id, query_text, filters)
            elif result_type == "correlated_incidents":
                return self._query_correlated_incidents(db, video_id, query_text, filters)
            else:
                return self._query_detections(db, video_id, query_text, filters)
        except Exception as exc:
            logger.error(f"Investigation query failed for video {video_id}: {exc}")
            raise
        finally:
            db.close()

    def _query_detections(
        self, db, video_id: str, query_text: str, filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fetch matching validated detections from DB."""
        q = db.query(EventModel).filter(EventModel.video_id == video_id, EventModel.validation_status == "VALID")

        obj_class = filters.get("object_class")
        if obj_class == "vehicle_group":
            q = q.filter(EventModel.object_class.in_(VEHICLE_CLASSES))
        elif obj_class:
            q = q.filter(EventModel.object_class.ilike(obj_class))

        if filters.get("start_time") is not None:
            q = q.filter(EventModel.timestamp_seconds >= filters["start_time"])
        if filters.get("end_time") is not None:
            q = q.filter(EventModel.timestamp_seconds <= filters["end_time"])
        if filters.get("min_confidence") is not None:
            q = q.filter(EventModel.confidence >= filters["min_confidence"])
        if filters.get("max_confidence") is not None:
            q = q.filter(EventModel.confidence <= filters["max_confidence"])

        rows = q.order_by(EventModel.timestamp_seconds.asc()).all()

        results = [
            {
                "event_id": r.id,
                "video_id": r.video_id,
                "timestamp": round(r.timestamp_seconds, 2),
                "object_class": r.object_class,
                "confidence": round(r.confidence, 4),
                "frame_number": r.frame_number,
                "bounding_box": {
                    "x1": round(r.bbox_x1, 1),
                    "y1": round(r.bbox_y1, 1),
                    "x2": round(r.bbox_x2, 1),
                    "y2": round(r.bbox_y2, 1),
                },
            }
            for r in rows
        ]

        if results:
            msg = f"Found {len(results)} matching detections."
        elif obj_class and obj_class != "vehicle_group":
            msg = f"No validated {obj_class} detections were found. No matching detections were found."
        else:
            msg = "No matching detections were found."

        return {
            "query": query_text,
            "is_supported": True,
            "result_type": "detections",
            "interpreted_filters": filters,
            "count": len(results),
            "message": msg,
            "results": results,
        }

    def _query_grouped_events(
        self, db, video_id: str, query_text: str, filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fetch matching grouped timeline events from DB."""
        q = db.query(GroupedEventModel).filter(GroupedEventModel.video_id == video_id)

        if filters.get("start_time") is not None:
            q = q.filter(GroupedEventModel.end_time >= filters["start_time"])
        if filters.get("end_time") is not None:
            q = q.filter(GroupedEventModel.start_time <= filters["end_time"])
        if filters.get("min_confidence") is not None:
            q = q.filter(GroupedEventModel.max_confidence >= filters["min_confidence"])

        rows = q.order_by(GroupedEventModel.start_time.asc()).all()

        results = []
        obj_class = filters.get("object_class")
        for r in rows:
            objs = r.objects_summary or []
            if obj_class == "vehicle_group":
                if not any(o.get("class", "").lower() in VEHICLE_CLASSES for o in objs):
                    continue
            elif obj_class:
                if not any(o.get("class", "").lower() == obj_class.lower() for o in objs):
                    continue

            results.append(
                {
                    "event_id": r.id,
                    "video_id": r.video_id,
                    "event_type": r.event_type,
                    "start_time": round(r.start_time, 2),
                    "end_time": round(r.end_time, 2),
                    "duration_seconds": r.duration_seconds,
                    "objects": objs,
                    "total_detections": r.total_detections,
                    "max_confidence": round(r.max_confidence, 4),
                    "priority": r.priority,
                }
            )

        msg = f"Found {len(results)} timeline events." if results else "No matching events were found."
        return {
            "query": query_text,
            "is_supported": True,
            "result_type": "events",
            "interpreted_filters": filters,
            "count": len(results),
            "message": msg,
            "results": results,
        }

    def _query_counts(
        self, db, video_id: str, query_text: str, filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate exact detection observations and unique track counts with strict semantic distinction."""
        detection_res = self._query_detections(db, video_id, query_text, filters)
        det_count = detection_res["count"]
        obj_name = filters.get("object_class") or "object"
        is_unique = filters.get("is_unique", False)

        # Query unique anonymous tracks for this class
        track_query = db.query(TrackModel).filter(TrackModel.video_id == video_id)
        if obj_name == "vehicle_group":
            track_query = track_query.filter(TrackModel.object_class.in_(VEHICLE_CLASSES))
        elif filters.get("object_class"):
            track_query = track_query.filter(TrackModel.object_class.ilike(obj_name))
        track_count = track_query.count()

        disclaimer = " (Note: Sentinel reports detection observations and does not attribute unique personal identities)."
        if is_unique:
            msg = f"{track_count} anonymous {obj_name} tracks.{disclaimer}"
            final_count = track_count
        else:
            if track_count > 0:
                msg = f"Detected {det_count} total {obj_name} detection records ({det_count} validated {obj_name} detection observations across {track_count} anonymous tracks).{disclaimer}"
            else:
                msg = f"Detected {det_count} total {obj_name} detection records ({det_count} validated {obj_name} detection observations; unique track count unavailable).{disclaimer}"
            final_count = det_count

        return {
            "query": query_text,
            "is_supported": True,
            "result_type": "count",
            "interpreted_filters": filters,
            "count": final_count,
            "detection_observations": det_count,
            "track_count": track_count,
            "message": msg,
            "results": detection_res["results"][:10],  # preview top 10
        }

    def _query_vehicle_attributes(
        self, db, video_id: str, query_text: str, filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fetch matching vehicle visual color attribute records from DB."""
        # Color queries are only supported when actual vehicle color-analysis records exist
        total_attrs = db.query(VehicleAttributeModel).filter(VehicleAttributeModel.video_id == video_id).count()
        if total_attrs == 0:
            return {
                "query": query_text,
                "is_supported": False,
                "message": "Vehicle color analysis records do not exist for this video. Color attributes have not been analyzed or are unavailable.",
                "interpreted_filters": filters,
                "result_type": "unsupported",
                "count": 0,
                "results": [],
            }

        q = db.query(VehicleAttributeModel).filter(VehicleAttributeModel.video_id == video_id)

        target_color = filters.get("color")
        if target_color:
            q = q.filter(VehicleAttributeModel.color.ilike(target_color))

        obj_class = filters.get("object_class")
        if obj_class and obj_class != "vehicle_group":
            q = q.filter(VehicleAttributeModel.object_class.ilike(obj_class))

        if filters.get("start_time") is not None:
            q = q.filter(VehicleAttributeModel.timestamp_seconds >= filters["start_time"])
        if filters.get("end_time") is not None:
            q = q.filter(VehicleAttributeModel.timestamp_seconds <= filters["end_time"])

        rows = q.order_by(VehicleAttributeModel.timestamp_seconds.asc()).all()
        results = [
            {
                "id": r.id,
                "video_id": r.video_id,
                "track_id": r.track_id,
                "object_class": r.object_class,
                "color": r.color,
                "confidence": round(r.confidence, 4),
                "timestamp": round(r.timestamp_seconds, 2),
                "bounding_box": r.bounding_box,
            }
            for r in rows
        ]

        if results:
            msg = f"Found {len(results)} verified vehicle records matching visual color '{target_color}'."
        else:
            msg = f"No vehicles with verified visual color '{target_color}' were found in database records."

        return {
            "query": query_text,
            "is_supported": True,
            "result_type": "vehicle_attributes",
            "interpreted_filters": filters,
            "count": len(results),
            "message": msg,
            "results": results,
        }

    def _query_tracks(
        self, db, video_id: str, query_text: str, filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fetch matching object tracking records from DB."""
        q = db.query(TrackModel).filter(TrackModel.video_id == video_id)

        target_track_id = filters.get("track_id")
        if target_track_id:
            q = q.filter(TrackModel.track_id.ilike(target_track_id))

        obj_class = filters.get("object_class")
        if obj_class:
            q = q.filter(TrackModel.object_class.ilike(obj_class))

        target_color = filters.get("color")
        if target_color:
            q = q.filter(TrackModel.color.ilike(target_color))

        rows = q.order_by(TrackModel.first_seen.asc()).all()
        results = [
            {
                "track_id": r.track_id,
                "object_class": r.object_class,
                "first_seen": round(r.first_seen, 2),
                "last_seen": round(r.last_seen, 2),
                "duration_seconds": round(r.duration_seconds, 2),
                "detection_count": r.detection_count,
                "max_confidence": round(r.max_confidence, 4),
                "color": r.color,
                "color_confidence": round(r.color_confidence, 4) if r.color_confidence else None,
                "current_bbox": r.current_bbox,
                "active": bool(r.active),
            }
            for r in rows
        ]

        if target_color and obj_class:
            msg = (
                f"Found {len(results)} multi-frame tracked {obj_class} objects matching visual color '{target_color}'. "
                "(Note: Track IDs represent consistent visual objects in this video only; zero personal identity attribution)."
            )
        else:
            msg = (
                f"Found {len(results)} multi-frame tracked objects. "
                "(Note: Track IDs represent consistent visual objects in this video only; zero personal identity attribution)."
            )

        return {
            "query": query_text,
            "is_supported": True,
            "result_type": "tracks",
            "interpreted_filters": filters,
            "count": len(results),
            "message": msg,
            "results": results,
        }

    def _query_faces(
        self, db, video_id: str, query_text: str, filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fetch anonymous face visual region detections from DB.
        STRICT SAFETY: Zero facial recognition, zero identity mapping.
        """
        q = db.query(FaceDetectionModel).filter(FaceDetectionModel.video_id == video_id)

        if filters.get("start_time") is not None:
            q = q.filter(FaceDetectionModel.timestamp_seconds >= filters["start_time"])
        if filters.get("end_time") is not None:
            q = q.filter(FaceDetectionModel.timestamp_seconds <= filters["end_time"])

        rows = q.order_by(FaceDetectionModel.timestamp_seconds.asc()).all()
        results = [
            {
                "id": r.id,
                "video_id": r.video_id,
                "track_id": r.track_id,
                "timestamp": round(r.timestamp_seconds, 2),
                "confidence": round(r.confidence, 4),
                "bounding_box": {
                    "x1": round(r.bbox_x1, 1),
                    "y1": round(r.bbox_y1, 1),
                    "x2": round(r.bbox_x2, 1),
                    "y2": round(r.bbox_y2, 1),
                },
            }
            for r in rows
        ]

        msg = (
            f"Found {len(results)} face visual region detections. "
            "STRICT OBSERVATIONAL NOTICE: Visual regions only; facial recognition, biometric identity inference, and database matching are strictly disabled."
        )

        return {
            "query": query_text,
            "is_supported": True,
            "result_type": "faces",
            "interpreted_filters": filters,
            "count": len(results),
            "message": msg,
            "results": results,
        }

    def _query_security_events(
        self, db, video_id: str, query_text: str, filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fetch structured security intelligence events from DB."""
        q = db.query(SecurityEventModel).filter(SecurityEventModel.video_id == video_id)

        target_type = filters.get("event_type")
        target_category = filters.get("category")

        VEHICLE_EVENT_TYPES = [
            "POTENTIAL_VEHICLE_COLLISION",
            "POTENTIAL_NEAR_COLLISION",
            "POTENTIAL_SUDDEN_VEHICLE_STOP",
            "POTENTIAL_WRONG_WAY_VEHICLE",
            "POTENTIAL_UNUSUAL_VEHICLE_TRAJECTORY",
            "POTENTIAL_STATIONARY_VEHICLE",
        ]

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

        PROPERTY_EVENT_TYPES = [
            "POTENTIAL_ABANDONED_OBJECT",
            "POTENTIAL_OBJECT_LEFT_BEHIND",
            "POTENTIAL_OBJECT_PICKUP",
            "POTENTIAL_THEFT",
            "POTENTIAL_OBJECT_TAKEAWAY",
            "POTENTIAL_OBJECT_DISPLACEMENT",
            "POTENTIAL_PROPERTY_TAMPERING",
            "POTENTIAL_RESTRICTED_OBJECT_MOVEMENT",
            "POTENTIAL_OBJECT_REMOVAL",
        ]

        CROWD_EVENT_TYPES = [
            "HIGH_PEDESTRIAN_DENSITY",
            "CROWD_DENSITY_INCREASE",
            "POTENTIAL_CROWD_SURGE",
            "POTENTIAL_CROWD_DISPERSAL",
            "POTENTIAL_UNUSUAL_CROWD_MOVEMENT",
            "POTENTIAL_RESTRICTED_ZONE_CROWDING",
            "POTENTIAL_UNUSUAL_ZONE_ACTIVITY",
            "ZONE_OCCUPANCY_OBSERVATION",
        ]

        SPECIALIZED_EVENT_TYPES = [
            "POTENTIAL_FIRE",
            "POTENTIAL_SMOKE",
            "POTENTIAL_FIRE_SMOKE",
            "POTENTIAL_WEAPON_VISUAL",
        ]

        if target_type:
            q = q.filter(SecurityEventModel.event_type == target_type)
        elif target_category == "vehicle":
            q = q.filter(SecurityEventModel.event_type.in_(VEHICLE_EVENT_TYPES))
        elif target_category == "person":
            q = q.filter(SecurityEventModel.event_type.in_(PERSON_EVENT_TYPES))
        elif target_category == "property":
            q = q.filter(SecurityEventModel.event_type.in_(PROPERTY_EVENT_TYPES))
        elif target_category in ("crowd", "zone"):
            q = q.filter(SecurityEventModel.event_type.in_(CROWD_EVENT_TYPES))
        elif target_category in ("environment", "specialized"):
            q = q.filter(SecurityEventModel.event_type.in_(SPECIALIZED_EVENT_TYPES))
        elif target_category == "object":
            q = q.filter(SecurityEventModel.event_type.in_(PROPERTY_EVENT_TYPES + ["POTENTIAL_WEAPON_VISUAL"]))

        if filters.get("start_time") is not None:
            q = q.filter(SecurityEventModel.timestamp_seconds >= filters["start_time"])
        if filters.get("end_time") is not None:
            q = q.filter(SecurityEventModel.timestamp_seconds <= filters["end_time"])
        if filters.get("min_confidence") is not None:
            q = q.filter(SecurityEventModel.confidence >= filters["min_confidence"])
        if filters.get("max_confidence") is not None:
            q = q.filter(SecurityEventModel.confidence <= filters["max_confidence"])

        rows = q.order_by(SecurityEventModel.timestamp_seconds.asc()).all()

        if not rows and target_type in SPECIALIZED_EVENT_TYPES:
            spec_cls = target_type.replace("POTENTIAL_", "").lower()
            return self._query_specialized_observations(db, video_id, query_text, {
                "class_name": spec_cls,
                "start_time": filters.get("start_time"),
                "end_time": filters.get("end_time"),
                "min_confidence": filters.get("min_confidence"),
            })

        results = [
            {
                "id": r.id,
                "video_id": r.video_id,
                "event_type": r.event_type,
                "timestamp": round(r.timestamp_seconds, 2),
                "duration_seconds": round(r.duration_seconds, 2) if r.duration_seconds else 0.0,
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
                    (r.incident_metadata or {}).get("validation_decision")
                    if isinstance(r.incident_metadata, dict) and (r.incident_metadata or {}).get("validation_decision")
                    else ("REVIEW_REQUIRED" if r.human_verification_required else "ACCEPTED")
                ),
                "human_verification_required": bool(r.human_verification_required),
            }
            for r in rows
        ]

        is_specialized_query = (
            target_category in ("environment", "specialized")
            or (target_type and target_type in SPECIALIZED_EVENT_TYPES)
        )

        is_vehicle_query = (
            target_category == "vehicle"
            or (target_type and (
                "VEHICLE" in target_type
                or target_type in ("POTENTIAL_NEAR_COLLISION", "POTENTIAL_SUDDEN_VEHICLE_STOP", "POTENTIAL_UNUSUAL_VEHICLE_TRAJECTORY")
            ))
        )

        is_person_query = (
            target_category == "person"
            or (target_type and target_type in PERSON_EVENT_TYPES)
        )

        is_property_query = (
            target_category in ("property", "object")
            or (target_type and target_type in PROPERTY_EVENT_TYPES and target_type != "POTENTIAL_THEFT")
        )

        is_crowd_query = (
            target_category in ("crowd", "zone")
            or (target_type and target_type in CROWD_EVENT_TYPES)
        )

        if is_specialized_query:
            if results:
                msg = (
                    f"Found {len(results)} specialized visual incident observation(s) matching '{query_text}'. "
                    "Observational machine evidence only; mandatory human verification required."
                )
            else:
                msg = f"No specialized visual events were detected matching '{query_text}' in available evidence."
        elif is_crowd_query:
            if results:
                msg = (
                    f"Found {len(results)} crowd & zone intelligence observation(s) in Sentinel database records. "
                    "(Observational density & spatial analytics only; zero inference of threat, intent, or criminality)."
                )
            else:
                msg = "No crowd, density, or zone activity events were detected in the available visual evidence."
        elif is_person_query:
            if results:
                msg = f"Found {len(results)} verified person incident event(s) in Sentinel database records."
            else:
                msg = "No reliable person incident was detected in the available Sentinel data."
        elif is_vehicle_query:
            if results:
                msg = f"Found {len(results)} verified vehicle incident event(s) in Sentinel database records."
            else:
                msg = "No reliable vehicle incident was detected in the available Sentinel data."
        elif is_property_query:
            if results:
                msg = f"Found {len(results)} verified property / object incident event(s) in Sentinel database records."
            else:
                msg = "No reliable property or object incidents were detected in the available Sentinel data."
        elif target_type == "POTENTIAL_THEFT":
            if results:
                first_ev = results[0]
                msg = (
                    f"Sentinel identified a potential object-takeaway pattern around {first_ev['timestamp']:.1f}s. "
                    f"{first_ev['description']} Review the linked evidence."
                )
            else:
                msg = "No potential theft pattern was detected in the available visual evidence."
        elif target_type:
            ev_label = target_type.replace("_", " ").lower()
            if results:
                msg = f"Found {len(results)} verified {ev_label} events in database records."
            else:
                msg = f"No {ev_label} events were detected in the available visual evidence."
        else:
            if results:
                msg = f"Found {len(results)} security intelligence events available for review."
            else:
                msg = "No security events or anomalies were detected in the available visual evidence."

        return {
            "query": query_text,
            "is_supported": True,
            "message": msg,
            "interpreted_filters": filters,
            "result_type": "security_events",
            "count": len(results),
            "results": results,
        }

    def _query_specialized_observations(
        self, db, video_id: str, query_text: str, filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fetch matching specialized visual observations from SpecializedObservationModel."""
        q = db.query(SpecializedObservationModel).filter(SpecializedObservationModel.video_id == video_id)

        cls_name = filters.get("object_class") or filters.get("class_name")
        if cls_name:
            q = q.filter(SpecializedObservationModel.class_name.ilike(f"%{cls_name}%"))

        if filters.get("start_time") is not None:
            q = q.filter(SpecializedObservationModel.timestamp_seconds >= filters["start_time"])
        if filters.get("end_time") is not None:
            q = q.filter(SpecializedObservationModel.timestamp_seconds <= filters["end_time"])
        if filters.get("min_confidence") is not None:
            q = q.filter(SpecializedObservationModel.confidence >= filters["min_confidence"])

        rows = q.order_by(SpecializedObservationModel.timestamp_seconds.asc()).all()
        results = [
            {
                "id": r.id,
                "video_id": r.video_id,
                "event_id": r.event_id,
                "detector_name": r.detector_name,
                "detector_version": r.detector_version,
                "class_name": r.class_name,
                "object_class": r.class_name,
                "timestamp": round(r.timestamp_seconds, 2),
                "confidence": round(r.confidence, 4),
                "evidence_strength": round(r.evidence_strength, 4),
                "validation_status": r.validation_status,
                "bounding_box": r.bounding_box,
                "metrics": r.metrics,
            }
            for r in rows
        ]
        target_name = cls_name or "specialized"
        msg = f"Identified {len(results)} specialized visual observation(s). Potential {target_name} visual evidence was detected in surveillance footage; mandatory human verification required."
        return {
            "query": query_text,
            "is_supported": True,
            "message": msg,
            "interpreted_filters": filters,
            "result_type": "specialized",
            "count": len(results),
            "results": results,
        }

    def _query_correlated_incidents(
        self, db, video_id: str, query_text: str, filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fetch matching correlated incidents from CorrelatedIncidentModel."""
        q = db.query(CorrelatedIncidentModel).filter(CorrelatedIncidentModel.video_id == video_id)

        category = filters.get("category")
        if category:
            q = q.filter(CorrelatedIncidentModel.incident_category == category)

        val_decision = filters.get("validation_decision")
        if val_decision:
            q = q.filter(CorrelatedIncidentModel.validation_decision == val_decision)

        rows = q.order_by(CorrelatedIncidentModel.start_time.asc()).all()

        object_pair = filters.get("object_pair")
        filtered_rows = []
        for r in rows:
            classes = [c.lower() for c in (r.involved_object_classes or [])]
            if object_pair:
                p1, p2 = object_pair[0].lower(), object_pair[1].lower()
                has_p1 = any(p1 in c for c in classes)
                has_p2 = any(p2 in c for c in classes) or (p2 == "object" and any(c not in ["person", "car", "bus", "truck", "motorcycle"] for c in classes))
                if p2 == "vehicle":
                    has_p2 = any(c in VEHICLE_CLASSES or "vehicle" in c for c in classes)
                if not (has_p1 and has_p2):
                    continue
            filtered_rows.append(r)

        results = [
            {
                "incident_id": r.id,
                "video_id": r.video_id,
                "incident_category": r.incident_category,
                "incident_subcategory": r.incident_subcategory,
                "start_time": round(r.start_time, 2),
                "end_time": round(r.end_time, 2),
                "duration": round(r.duration, 2),
                "primary_track_ids": r.primary_track_ids or [],
                "supporting_track_ids": r.supporting_track_ids or [],
                "involved_object_classes": r.involved_object_classes or [],
                "assessment_score": round(r.assessment_score, 4),
                "evidence_strength": round(r.evidence_strength, 4),
                "reliability_rating": r.reliability_rating,
                "validation_decision": r.validation_decision,
                "storyline": r.storyline or "",
                "evidence_ids": r.evidence_ids or [],
                "source_candidate_ids": r.source_candidate_ids or [],
                "provenance": r.provenance or {},
            }
            for r in filtered_rows
        ]

        query_focus = filters.get("query_focus")
        if query_focus == "pre_collision_sequence":
            msg = f"Retrieved {len(results)} collision sequence storyline(s) showing trajectory convergence and contact."
        elif query_focus == "theft_storyline":
            msg = f"Retrieved {len(results)} coherent object takeaway storyline(s) with supporting approach, dwell, and departure provenance."
        elif query_focus == "evidence_support":
            msg = f"Identified {len(results)} correlated incident(s) with canonical supporting evidence links and track provenance."
        elif query_focus == "duplicate_fusion":
            msg = f"Identified {len(results)} correlated incident(s) formed by fusing multiple overlapping or adjacent candidate signals."
        elif object_pair:
            msg = f"Identified {len(results)} correlated incident(s) involving {' and '.join(object_pair)} interaction."
        elif val_decision:
            msg = f"Found {len(results)} incident(s) categorized as {val_decision}."
        else:
            msg = f"Identified {len(results)} canonical correlated incident(s) in video intelligence records."

        return {
            "query": query_text,
            "is_supported": True,
            "message": msg,
            "interpreted_filters": filters,
            "result_type": "correlated_incidents",
            "count": len(results),
            "results": results,
        }

    # -----------------------------------------------------------------------
    # Phase 17: Structured query entry point
    # -----------------------------------------------------------------------

    def investigate_structured(
        self,
        video_id: str,
        query_text: str = "",
        video_duration_seconds: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Phase 17: Parse a natural-language query into an InvestigationQuery,
        then execute it via InvestigationQueryExecutor, returning a full
        InvestigationResult serialized to dict.

        Falls back to investigate_filters if executor fails.
        """
        from backend.app.services.investigation_query import InvestigationQuery
        from backend.app.services.investigation_query_executor import InvestigationQueryExecutor

        # Build typed query via extended parser
        try:
            iq = InvestigationParser.parse_investigation_query(
                user_query=query_text,
                video_id=video_id,
                video_duration_seconds=video_duration_seconds,
            )
        except Exception as parse_err:
            logger.warning(f"Phase 17 parser failed: {parse_err}. Falling back to Phase 5A.")
            return self.investigate(video_id=video_id, query_text=query_text)

        try:
            executor = InvestigationQueryExecutor()
            result = executor.execute(iq)
            result_dict = result.to_dict()
            result_dict["query_text"] = query_text
            return result_dict
        except Exception as exec_err:
            logger.warning(f"Phase 17 executor failed: {exec_err}. Falling back to Phase 5A.")
            return self.investigate(video_id=video_id, query_text=query_text)

    def investigate_track(
        self,
        video_id: str,
        track_id: str,
    ) -> Dict[str, Any]:
        """
        Phase 17: Full investigation of a single anonymous track.

        Returns:
        - Track metadata (first_seen, last_seen, object_class, color)
        - Associated security events
        - Associated correlated incidents
        - Linked evidence
        - Object lifecycle summary (DETECTED → appearances → LAST_SEEN)
        """
        db = SessionLocal()
        try:
            # Verify track belongs to this video (critical isolation check)
            track = (
                db.query(TrackModel)
                .filter(
                    TrackModel.video_id == video_id,
                    TrackModel.track_id == track_id,
                )
                .first()
            )
            if not track:
                return {
                    "video_id": video_id,
                    "track_id": track_id,
                    "is_supported": False,
                    "message": f"Track '{track_id}' not found in video '{video_id}'.",
                    "result_type": "track_investigation",
                    "results": {},
                }

            # Security events for this track
            sec_events = (
                db.query(SecurityEventModel)
                .filter(
                    SecurityEventModel.video_id == video_id,
                    SecurityEventModel.track_id == track_id,
                )
                .order_by(SecurityEventModel.timestamp_seconds.asc())
                .all()
            )

            # Correlated incidents referencing this track
            all_incidents = (
                db.query(CorrelatedIncidentModel)
                .filter(CorrelatedIncidentModel.video_id == video_id)
                .all()
            )
            related_incidents = [
                r for r in all_incidents
                if track_id in (r.primary_track_ids or [])
                or track_id in (r.supporting_track_ids or [])
            ]

            # Evidence correlated to this track's time window
            evidence = (
                db.query(EvidenceModel)
                .filter(
                    EvidenceModel.video_id == video_id,
                    EvidenceModel.validation_status == "VALID",
                    EvidenceModel.timestamp_seconds >= track.first_seen - 1.0,
                    EvidenceModel.timestamp_seconds <= track.last_seen + 1.0,
                )
                .order_by(EvidenceModel.timestamp_seconds.asc())
                .all()
            )

            # Object lifecycle: summarize appearance from trajectory
            lifecycle = self._build_object_lifecycle(track)

            track_data = {
                "track_id": track.track_id,
                "video_id": track.video_id,
                "object_class": track.object_class,
                "first_seen": round(track.first_seen, 2),
                "last_seen": round(track.last_seen, 2),
                "duration_seconds": round(track.duration_seconds, 2),
                "detection_count": track.detection_count,
                "max_confidence": round(track.max_confidence, 4),
                "color": track.color,
                "active": bool(track.active),
            }

            return {
                "video_id": video_id,
                "track_id": track_id,
                "is_supported": True,
                "result_type": "track_investigation",
                "message": (
                    f"Track {track_id} ({track.object_class}) was observed from "
                    f"{track.first_seen:.1f}s to {track.last_seen:.1f}s "
                    f"across {track.detection_count} detections."
                    " (Anonymous track — no identity inference.)"
                ),
                "track": track_data,
                "lifecycle": lifecycle,
                "security_events": [
                    {
                        "id": e.id,
                        "event_type": e.event_type,
                        "timestamp": round(e.timestamp_seconds, 2),
                        "description": e.description,
                        "severity": e.severity,
                    }
                    for e in sec_events
                ],
                "related_incidents": [
                    {
                        "incident_id": inc.id,
                        "incident_category": inc.incident_category,
                        "incident_subcategory": inc.incident_subcategory,
                        "start_time": round(inc.start_time, 2),
                        "end_time": round(inc.end_time, 2),
                        "validation_decision": inc.validation_decision,
                        "assessment_score": round(inc.assessment_score, 4),
                        "storyline": inc.storyline or "",
                    }
                    for inc in related_incidents
                ],
                "evidence": [
                    {
                        "evidence_id": ev.id,
                        "timestamp": round(ev.timestamp_seconds, 2),
                        "evidence_type": ev.evidence_type,
                        "has_snapshot": bool(ev.snapshot_path),
                        "has_clip": bool(ev.clip_path),
                    }
                    for ev in evidence
                ],
            }
        finally:
            db.close()

    def _build_object_lifecycle(self, track: TrackModel) -> Dict[str, Any]:
        """
        Build an observational object lifecycle summary for a track.
        Uses observational language — does NOT infer intent.
        """
        phases = []

        phases.append({
            "phase": "DETECTED",
            "timestamp": round(track.first_seen, 2),
            "note": f"First detected at {track.first_seen:.1f}s.",
        })

        # Check trajectory for stationary periods if available
        if track.trajectory and isinstance(track.trajectory, list) and len(track.trajectory) > 2:
            try:
                positions = [(t[1], t[2]) for t in track.trajectory if len(t) >= 3]
                if positions:
                    # Check if object was mostly stationary (low movement variance)
                    xs = [p[0] for p in positions]
                    ys = [p[1] for p in positions]
                    x_range = max(xs) - min(xs) if xs else 0
                    y_range = max(ys) - min(ys) if ys else 0
                    if x_range < 30 and y_range < 30 and len(positions) > 3:
                        phases.append({
                            "phase": "STATIONARY_PERIOD",
                            "note": "Object remained approximately stationary over multiple frames.",
                        })
            except (TypeError, IndexError):
                pass

        if track.duration_seconds > 1.0:
            phases.append({
                "phase": "PRESENT",
                "timestamp_start": round(track.first_seen, 2),
                "timestamp_end": round(track.last_seen, 2),
                "note": f"Present for {track.duration_seconds:.1f}s across {track.detection_count} detection frames.",
            })

        phases.append({
            "phase": "LAST_SEEN",
            "timestamp": round(track.last_seen, 2),
            "note": f"Last detected at {track.last_seen:.1f}s. Active: {bool(track.active)}.",
        })

        return {
            "track_id": track.track_id,
            "object_class": track.object_class,
            "phases": phases,
            "observational_note": (
                "This lifecycle is an observational summary of detection activity only. "
                "No identity, intent, or behavioral conclusion is inferred."
            ),
        }

    def investigate_zone(
        self,
        video_id: str,
        zone_name: str,
    ) -> Dict[str, Any]:
        """
        Phase 17: Zone-focused investigation.
        Returns all security events and incidents associated with a specific zone name.
        Video-isolated; zone_name matched case-insensitively.
        """
        db = SessionLocal()
        try:
            # Security events in this zone
            sec_events = (
                db.query(SecurityEventModel)
                .filter(
                    SecurityEventModel.video_id == video_id,
                    SecurityEventModel.zone_name.ilike(f"%{zone_name}%"),
                )
                .order_by(SecurityEventModel.timestamp_seconds.asc())
                .all()
            )

            # Correlated incidents mentioning this zone
            all_incidents = (
                db.query(CorrelatedIncidentModel)
                .filter(CorrelatedIncidentModel.video_id == video_id)
                .all()
            )
            zone_incidents = [
                inc for inc in all_incidents
                if zone_name.lower() in (inc.storyline or "").lower()
                or any(
                    zone_name.lower() in str(z).lower()
                    for z in (inc.zone_ids or [])
                )
            ]

            track_ids_in_zone = list({e.track_id for e in sec_events if e.track_id})

            events_serialized = [
                {
                    "id": e.id,
                    "event_type": e.event_type,
                    "timestamp": round(e.timestamp_seconds, 2),
                    "duration_seconds": round(e.duration_seconds or 0.0, 2),
                    "track_id": e.track_id,
                    "object_class": e.object_class,
                    "description": e.description,
                    "severity": e.severity,
                    "validation_decision": (
                        "REVIEW_REQUIRED" if e.human_verification_required else "ACCEPTED"
                    ),
                }
                for e in sec_events
            ]

            incidents_serialized = [
                {
                    "incident_id": inc.id,
                    "incident_category": inc.incident_category,
                    "start_time": round(inc.start_time, 2),
                    "end_time": round(inc.end_time, 2),
                    "validation_decision": inc.validation_decision,
                    "storyline": inc.storyline or "",
                }
                for inc in zone_incidents
            ]

            if sec_events or zone_incidents:
                msg = (
                    f"Found {len(sec_events)} security event(s) and {len(zone_incidents)} "
                    f"correlated incident(s) in zone '{zone_name}'."
                )
            else:
                msg = f"No activity detected in zone '{zone_name}' for this video."

            return {
                "video_id": video_id,
                "zone_name": zone_name,
                "is_supported": True,
                "result_type": "zone_investigation",
                "message": msg,
                "security_events": events_serialized,
                "correlated_incidents": incidents_serialized,
                "track_ids_observed": track_ids_in_zone,
            }
        finally:
            db.close()

    def get_unified_timeline(
        self,
        video_id: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        include_rejected: bool = False,
    ) -> Dict[str, Any]:
        """
        Phase 17: Unified forensic timeline.

        Merges all Sentinel intelligence layers into a single chronological sequence:
        - Grouped events (detection clusters)
        - Security events
        - Correlated incidents
        - Evidence markers

        Each entry carries layer type, playback timestamp, and detail.
        Results are bounded to [start_time, end_time] if provided.
        """
        db = SessionLocal()
        try:
            timeline_entries = []

            # Layer 1: Grouped events (detection timeline)
            ge_q = db.query(GroupedEventModel).filter(GroupedEventModel.video_id == video_id)
            if start_time is not None:
                ge_q = ge_q.filter(GroupedEventModel.end_time >= start_time)
            if end_time is not None:
                ge_q = ge_q.filter(GroupedEventModel.start_time <= end_time)
            grouped_events = ge_q.order_by(GroupedEventModel.start_time.asc()).all()
            for ge in grouped_events:
                timeline_entries.append({
                    "layer": "detection_event",
                    "timestamp": round(ge.start_time, 2),
                    "end_timestamp": round(ge.end_time, 2),
                    "id": ge.id,
                    "label": ge.event_type,
                    "total_detections": ge.total_detections,
                    "priority": ge.priority,
                })

            # Layer 2: Security events
            se_q = db.query(SecurityEventModel).filter(SecurityEventModel.video_id == video_id)
            if start_time is not None:
                se_q = se_q.filter(SecurityEventModel.timestamp_seconds >= start_time)
            if end_time is not None:
                se_q = se_q.filter(SecurityEventModel.timestamp_seconds <= end_time)
            sec_events = se_q.order_by(SecurityEventModel.timestamp_seconds.asc()).all()
            for se in sec_events:
                timeline_entries.append({
                    "layer": "security_event",
                    "timestamp": round(se.timestamp_seconds, 2),
                    "end_timestamp": round(se.timestamp_seconds + (se.duration_seconds or 0.0), 2),
                    "id": se.id,
                    "label": se.event_type.replace("_", " "),
                    "severity": se.severity,
                    "track_id": se.track_id,
                    "description": se.description,
                    "validation_decision": "REVIEW_REQUIRED" if se.human_verification_required else "ACCEPTED",
                })

            # Layer 3: Correlated incidents
            ci_q = db.query(CorrelatedIncidentModel).filter(CorrelatedIncidentModel.video_id == video_id)
            if start_time is not None:
                ci_q = ci_q.filter(CorrelatedIncidentModel.end_time >= start_time)
            if end_time is not None:
                ci_q = ci_q.filter(CorrelatedIncidentModel.start_time <= end_time)
            if not include_rejected:
                ci_q = ci_q.filter(CorrelatedIncidentModel.validation_decision != "REJECTED")
            incidents = ci_q.order_by(CorrelatedIncidentModel.start_time.asc()).all()
            for inc in incidents:
                timeline_entries.append({
                    "layer": "correlated_incident",
                    "timestamp": round(inc.start_time, 2),
                    "end_timestamp": round(inc.end_time, 2),
                    "id": inc.id,
                    "label": f"{inc.incident_category} — {inc.incident_subcategory or ''}",
                    "assessment_score": round(inc.assessment_score, 4),
                    "reliability_rating": inc.reliability_rating,
                    "validation_decision": inc.validation_decision,
                    "primary_track_ids": inc.primary_track_ids or [],
                })

            # Layer 4: Evidence markers
            ev_q = db.query(EvidenceModel).filter(
                EvidenceModel.video_id == video_id,
                EvidenceModel.validation_status == "VALID",
            )
            if start_time is not None:
                ev_q = ev_q.filter(EvidenceModel.timestamp_seconds >= start_time)
            if end_time is not None:
                ev_q = ev_q.filter(EvidenceModel.timestamp_seconds <= end_time)
            evidence = ev_q.order_by(EvidenceModel.timestamp_seconds.asc()).all()
            for ev in evidence:
                timeline_entries.append({
                    "layer": "evidence",
                    "timestamp": round(ev.timestamp_seconds, 2),
                    "end_timestamp": round(ev.end_time or ev.timestamp_seconds, 2),
                    "id": ev.id,
                    "label": f"Evidence ({ev.evidence_type})",
                    "has_snapshot": bool(ev.snapshot_path),
                    "has_clip": bool(ev.clip_path),
                    "event_id": ev.event_id,
                })

            # Sort by timestamp
            timeline_entries.sort(key=lambda x: x["timestamp"])

            return {
                "video_id": video_id,
                "is_supported": True,
                "result_type": "unified_timeline",
                "start_time_filter": start_time,
                "end_time_filter": end_time,
                "total_entries": len(timeline_entries),
                "timeline": timeline_entries,
                "layer_counts": {
                    "detection_events": len(grouped_events),
                    "security_events": len(sec_events),
                    "correlated_incidents": len(incidents),
                    "evidence": len(evidence),
                },
            }
        finally:
            db.close()


