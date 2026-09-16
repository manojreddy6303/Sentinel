"""
Event Repository Interface and Implementations (Database & JSON)
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from database.session import SessionLocal, init_db
from database.models import VideoModel, EventModel, GroupedEventModel

logger = logging.getLogger(__name__)


class EventRepository:
    """Abstract interface for detection event persistence."""

    def save_events(
        self,
        video_id: str,
        events: List[Dict[str, Any]],
        grouped_events: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Persist raw and grouped detection events for a video."""
        raise NotImplementedError

    def get_events(
        self,
        video_id: str,
        object_class: Optional[str] = None,
        min_confidence: Optional[float] = None,
        validation_status: Optional[str] = "VALID",
    ) -> List[Dict[str, Any]]:
        """Retrieve detection events with optional filters (defaults to validated only)."""
        raise NotImplementedError

    def get_grouped_events(
        self,
        video_id: str,
        object_class: Optional[str] = None,
        min_confidence: Optional[float] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve grouped timeline events with optional filters."""
        raise NotImplementedError

    def events_exist(self, video_id: str) -> bool:
        """Return True if events exist for the given video."""
        raise NotImplementedError


def ensure_video_events_validated(video_id: str, db=None) -> None:
    """
    Ensure all EventModel records for video_id have been evaluated by DetectionValidator.
    If any events have validation_reason IS NULL (legacy unvalidated records),
    evaluates them and commits their validation_status and validation_reason.
    """
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    try:
        unvalidated_count = (
            db.query(EventModel.id)
            .filter(EventModel.video_id == video_id, EventModel.validation_reason == None)
            .count()
        )
        if unvalidated_count == 0:
            return

        all_events = (
            db.query(EventModel)
            .filter(EventModel.video_id == video_id)
            .order_by(EventModel.timestamp_seconds.asc())
            .all()
        )
        if not all_events:
            return

        frame_w = None
        frame_h = None
        meta_path = Path(f"storage/uploads/{video_id}.json")
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    frame_w = meta.get("width")
                    frame_h = meta.get("height")
            except Exception:
                pass

        event_dicts = []
        for r in all_events:
            event_dicts.append({
                "id": r.id,
                "object_class": r.object_class,
                "confidence": r.confidence,
                "timestamp": r.timestamp_seconds,
                "bounding_box": {
                    "x1": r.bbox_x1,
                    "y1": r.bbox_y1,
                    "x2": r.bbox_x2,
                    "y2": r.bbox_y2,
                },
            })

        from ai.validation import DetectionValidator
        validator = DetectionValidator()
        validator.validate_events_list(event_dicts, frame_width=frame_w, frame_height=frame_h)

        id_to_model = {r.id: r for r in all_events}
        for ed in event_dicts:
            rec = id_to_model.get(ed["id"])
            if rec:
                rec.validation_status = ed.get("validation_status", "VALID")
                rec.validation_reason = ed.get("validation_reason")

        db.commit()
        logger.info(f"Synchronized legacy validation statuses for video {video_id}.")
    except Exception as exc:
        db.rollback()
        logger.warning(f"Error validating legacy events for {video_id}: {exc}")
    finally:
        if close_db:
            db.close()


class DatabaseEventRepository(EventRepository):
    """SQLAlchemy-backed repository (PostgreSQL / SQLite)."""

    def __init__(self):
        # Ensure tables exist
        init_db()

    def save_events(
        self,
        video_id: str,
        events: List[Dict[str, Any]],
        grouped_events: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        db = SessionLocal()
        try:
            # Ensure parent video record exists to satisfy foreign key constraints
            if not db.query(VideoModel.id).filter(VideoModel.id == video_id).first():
                meta_p = Path(f"storage/uploads/{video_id}.json")
                orig_fn = f"{video_id}.mp4"
                stor_p = f"storage/uploads/{video_id}.mp4"
                sz = 0
                if meta_p.exists():
                    try:
                        with open(meta_p, "r", encoding="utf-8") as mf:
                            m = json.load(mf)
                            orig_fn = m.get("filename") or orig_fn
                            stor_p = m.get("storage_path") or stor_p
                            sz = m.get("file_size_bytes", 0)
                    except Exception:
                        pass
                db.add(
                    VideoModel(
                        id=video_id,
                        original_filename=orig_fn,
                        storage_path=stor_p,
                        file_size_bytes=sz,
                        status="processed",
                    )
                )
                db.flush()

            # Clear previous events for reprocessing idempotency
            db.query(EventModel).filter(EventModel.video_id == video_id).delete()
            db.query(GroupedEventModel).filter(GroupedEventModel.video_id == video_id).delete()

            # Insert raw events
            raw_models = []
            for ev in events:
                bbox = ev.get("bounding_box", {})
                raw_models.append(
                    EventModel(
                        id=ev.get("event_id"),
                        video_id=video_id,
                        event_type=ev.get("event_type", "object_detected"),
                        object_class=ev.get("object_class"),
                        class_id=ev.get("class_id"),
                        timestamp_seconds=ev.get("timestamp"),
                        confidence=ev.get("confidence"),
                        bbox_x1=bbox.get("x1", 0.0),
                        bbox_y1=bbox.get("y1", 0.0),
                        bbox_x2=bbox.get("x2", 0.0),
                        bbox_y2=bbox.get("y2", 0.0),
                        frame_number=ev.get("frame_number", 0),
                        validation_status=ev.get("validation_status", "VALID"),
                        validation_reason=ev.get("validation_reason"),
                    )
                )
            db.bulk_save_objects(raw_models)

            # Insert grouped events
            if grouped_events:
                grp_models = []
                for grp in grouped_events:
                    grp_models.append(
                        GroupedEventModel(
                            id=grp.get("event_id"),
                            video_id=video_id,
                            event_type=grp.get("event_type"),
                            start_time=grp.get("start_time"),
                            end_time=grp.get("end_time"),
                            duration_seconds=grp.get("duration_seconds"),
                            objects_summary=grp.get("objects", []),
                            total_detections=grp.get("total_detections", 1),
                            max_confidence=grp.get("max_confidence"),
                            priority=grp.get("priority", "NORMAL"),
                        )
                    )
                db.bulk_save_objects(grp_models)

            db.commit()
            logger.info(
                f"Saved {len(events)} raw events and {len(grouped_events or [])} grouped events to DB for video {video_id}."
            )
        except Exception as exc:
            db.rollback()
            logger.error(f"Failed to save events to DB for {video_id}: {exc}")
            raise
        finally:
            db.close()

    def get_events(
        self,
        video_id: str,
        object_class: Optional[str] = None,
        min_confidence: Optional[float] = None,
        validation_status: Optional[str] = "VALID",
    ) -> List[Dict[str, Any]]:
        ensure_video_events_validated(video_id)
        db = SessionLocal()
        try:
            query = db.query(EventModel).filter(EventModel.video_id == video_id)
            if object_class:
                query = query.filter(EventModel.object_class.ilike(object_class))
            if min_confidence is not None:
                query = query.filter(EventModel.confidence >= min_confidence)
            if validation_status and validation_status.upper() != "ALL":
                query = query.filter(EventModel.validation_status == validation_status.upper())
            
            rows = query.order_by(EventModel.timestamp_seconds.asc()).all()
            return [
                {
                    "event_id": r.id,
                    "video_id": r.video_id,
                    "event_type": r.event_type,
                    "object_class": r.object_class,
                    "class_id": r.class_id,
                    "timestamp": r.timestamp_seconds,
                    "confidence": r.confidence,
                    "bounding_box": {
                        "x1": r.bbox_x1,
                        "y1": r.bbox_y1,
                        "x2": r.bbox_x2,
                        "y2": r.bbox_y2,
                    },
                    "frame_number": r.frame_number,
                    "validation_status": r.validation_status,
                    "validation_reason": r.validation_reason,
                }
                for r in rows
            ]
        finally:
            db.close()

    def get_grouped_events(
        self,
        video_id: str,
        object_class: Optional[str] = None,
        min_confidence: Optional[float] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        db = SessionLocal()
        try:
            query = db.query(GroupedEventModel).filter(GroupedEventModel.video_id == video_id)
            if min_confidence is not None:
                query = query.filter(GroupedEventModel.max_confidence >= min_confidence)
            if start_time is not None:
                query = query.filter(GroupedEventModel.end_time >= start_time)
            if end_time is not None:
                query = query.filter(GroupedEventModel.start_time <= end_time)

            rows = query.order_by(GroupedEventModel.start_time.asc()).all()

            results = []
            for r in rows:
                objs = r.objects_summary or []
                # Filter by object_class if requested
                if object_class:
                    if not any(o.get("class", "").lower() == object_class.lower() for o in objs):
                        continue
                results.append(
                    {
                        "event_id": r.id,
                        "video_id": r.video_id,
                        "event_type": r.event_type,
                        "start_time": r.start_time,
                        "end_time": r.end_time,
                        "duration_seconds": r.duration_seconds,
                        "objects": objs,
                        "total_detections": r.total_detections,
                        "max_confidence": r.max_confidence,
                        "priority": r.priority,
                    }
                )
            return results
        finally:
            db.close()

    def events_exist(self, video_id: str) -> bool:
        db = SessionLocal()
        try:
            count = db.query(EventModel).filter(EventModel.video_id == video_id).count()
            return count > 0
        finally:
            db.close()


class JSONEventRepository(EventRepository):
    """Local JSON file-backed fallback event repository."""

    def __init__(self, events_dir: str):
        self.events_dir = Path(events_dir)
        self.events_dir.mkdir(parents=True, exist_ok=True)

    def _event_file(self, video_id: str) -> Path:
        return self.events_dir / f"{video_id}_events.json"

    def save_events(
        self,
        video_id: str,
        events: List[Dict[str, Any]],
        grouped_events: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        payload = {"video_id": video_id, "events": events, "grouped_events": grouped_events or []}
        with open(self._event_file(video_id), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def get_events(
        self,
        video_id: str,
        object_class: Optional[str] = None,
        min_confidence: Optional[float] = None,
        validation_status: Optional[str] = "VALID",
    ) -> List[Dict[str, Any]]:
        ef = self._event_file(video_id)
        if not ef.exists():
            return []
        with open(ef, "r", encoding="utf-8") as f:
            payload = json.load(f)
        events = payload.get("events", [])
        if object_class:
            events = [e for e in events if e.get("object_class", "").lower() == object_class.lower()]
        if min_confidence is not None:
            events = [e for e in events if e.get("confidence", 0.0) >= min_confidence]
        if validation_status and validation_status.upper() != "ALL":
            events = [e for e in events if e.get("validation_status", "VALID") == validation_status.upper()]
        events.sort(key=lambda e: e.get("timestamp", 0.0))
        return events

    def get_grouped_events(
        self,
        video_id: str,
        object_class: Optional[str] = None,
        min_confidence: Optional[float] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        ef = self._event_file(video_id)
        if not ef.exists():
            return []
        with open(ef, "r", encoding="utf-8") as f:
            payload = json.load(f)
        grouped = payload.get("grouped_events", [])
        if min_confidence is not None:
            grouped = [g for g in grouped if g.get("max_confidence", 0.0) >= min_confidence]
        if start_time is not None:
            grouped = [g for g in grouped if g.get("end_time", 0.0) >= start_time]
        if end_time is not None:
            grouped = [g for g in grouped if g.get("start_time", 0.0) <= end_time]
        if object_class:
            grouped = [
                g for g in grouped
                if any(o.get("class", "").lower() == object_class.lower() for o in g.get("objects", []))
            ]
        grouped.sort(key=lambda g: g.get("start_time", 0.0))
        return grouped

    def events_exist(self, video_id: str) -> bool:
        return self._event_file(video_id).exists()


def get_event_repository() -> EventRepository:
    """Factory returning DatabaseEventRepository with fallback capability."""
    try:
        return DatabaseEventRepository()
    except Exception as exc:
        logger.warning(f"Database unavailable for event repository ({exc}), falling back to JSON repository.")
        from backend.app.core.config import settings
        return JSONEventRepository(events_dir=str(settings.STORAGE_EVENTS_DIR))
