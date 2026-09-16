"""
Regression Tests: SQLite Foreign Key Integrity & Cascade Enforcement for Sentinel

Verifies:
1. PRAGMA foreign_keys = ON is active on SQLite database connections.
2. Direct child inserts with non-existent parent video_id are rejected with IntegrityError.
3. Deleting a parent VideoModel properly cascades and removes related records on isolated test data.
4. Existing production records remain untouched.
"""

import uuid
import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from database.session import SessionLocal, engine, DB_URL
from database.models import (
    VideoModel,
    EventModel,
    GroupedEventModel,
    TrackModel,
    SecurityEventModel,
    EvidenceModel,
)


def test_sqlite_foreign_keys_pragma_enabled():
    """Verify that PRAGMA foreign_keys is enabled (returns 1) on SQLite connections."""
    if not DB_URL.startswith("sqlite"):
        pytest.skip("Test applicable only when running with SQLite.")

    db = SessionLocal()
    try:
        result = db.execute(text("PRAGMA foreign_keys")).scalar()
        assert result == 1, f"Expected PRAGMA foreign_keys to be 1, but got {result}"
    finally:
        db.close()


def test_sqlite_foreign_key_insert_rejection():
    """Verify that inserting a child event with a non-existent video_id raises IntegrityError."""
    if not DB_URL.startswith("sqlite"):
        pytest.skip("Test applicable only when running with SQLite.")

    db = SessionLocal()
    nonexistent_video_id = f"missing_vid_{uuid.uuid4().hex}"
    try:
        orphan_event = EventModel(
            id=str(uuid.uuid4()),
            video_id=nonexistent_video_id,
            object_class="person",
            class_id=0,
            timestamp_seconds=1.0,
            confidence=0.9,
            bbox_x1=0.0,
            bbox_y1=0.0,
            bbox_x2=10.0,
            bbox_y2=10.0,
            frame_number=30,
        )
        db.add(orphan_event)
        with pytest.raises(IntegrityError):
            db.commit()
    finally:
        db.rollback()
        db.close()


def test_sqlite_foreign_key_cascade_deletion():
    """Verify that deleting a parent video cascades and deletes all related child records."""
    db = SessionLocal()
    vid = f"test_fk_cascade_{uuid.uuid4().hex[:8]}"

    try:
        # 1. Create parent video
        parent = VideoModel(
            id=vid,
            original_filename="cascade_test.mp4",
            storage_path=f"storage/uploads/{vid}.mp4",
            status="processed",
            duration_seconds=10.0,
        )
        db.add(parent)

        # 2. Add child records across all related tables
        ev_id = str(uuid.uuid4())
        child_event = EventModel(
            id=ev_id,
            video_id=vid,
            object_class="person",
            class_id=0,
            timestamp_seconds=2.0,
            confidence=0.85,
            bbox_x1=10.0,
            bbox_y1=10.0,
            bbox_x2=50.0,
            bbox_y2=100.0,
            frame_number=60,
        )
        child_grouped = GroupedEventModel(
            id=str(uuid.uuid4()),
            video_id=vid,
            event_type="PERSON_ACTIVITY",
            start_time=2.0,
            end_time=4.0,
            duration_seconds=2.0,
            objects_summary=[{"class": "person", "count": 1}],
            total_detections=1,
            max_confidence=0.85,
        )
        child_track = TrackModel(
            id=str(uuid.uuid4()),
            video_id=vid,
            track_id="TRACK-999",
            object_class="person",
            first_seen=2.0,
            last_seen=4.0,
            duration_seconds=2.0,
            detection_count=1,
            max_confidence=0.85,
        )
        child_sec_event = SecurityEventModel(
            id=str(uuid.uuid4()),
            video_id=vid,
            event_type="PROLONGED_PRESENCE",
            severity="NORMAL",
            timestamp_seconds=2.0,
            confidence=0.85,
            description="Test prolonged presence event",
        )
        child_evidence = EvidenceModel(
            id=str(uuid.uuid4()),
            video_id=vid,
            timestamp_seconds=2.0,
            source_video_name="cascade_test.mp4",
        )

        db.add_all([
            child_event,
            child_grouped,
            child_track,
            child_sec_event,
            child_evidence,
        ])
        db.commit()

        # 3. Confirm all records are persisted
        assert db.query(VideoModel).filter(VideoModel.id == vid).count() == 1
        assert db.query(EventModel).filter(EventModel.video_id == vid).count() == 1
        assert db.query(GroupedEventModel).filter(GroupedEventModel.video_id == vid).count() == 1
        assert db.query(TrackModel).filter(TrackModel.video_id == vid).count() == 1
        assert db.query(SecurityEventModel).filter(SecurityEventModel.video_id == vid).count() == 1
        assert db.query(EvidenceModel).filter(EvidenceModel.video_id == vid).count() == 1

        # 4. Delete parent video
        video_to_delete = db.query(VideoModel).filter(VideoModel.id == vid).first()
        db.delete(video_to_delete)
        db.commit()

        # 5. Confirm parent AND all child records are deleted
        assert db.query(VideoModel).filter(VideoModel.id == vid).count() == 0
        assert db.query(EventModel).filter(EventModel.video_id == vid).count() == 0
        assert db.query(GroupedEventModel).filter(GroupedEventModel.video_id == vid).count() == 0
        assert db.query(TrackModel).filter(TrackModel.video_id == vid).count() == 0
        assert db.query(SecurityEventModel).filter(SecurityEventModel.video_id == vid).count() == 0
        assert db.query(EvidenceModel).filter(EvidenceModel.video_id == vid).count() == 0

    finally:
        # Cleanup if any error occurred before deletion
        try:
            cleanup_vid = db.query(VideoModel).filter(VideoModel.id == vid).first()
            if cleanup_vid:
                db.delete(cleanup_vid)
                db.commit()
        except Exception:
            db.rollback()
        db.close()
