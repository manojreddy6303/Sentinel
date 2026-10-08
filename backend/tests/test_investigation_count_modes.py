"""
Unit tests for the three distinct count concepts in Sentinel:
A. Raw Detection Observations
B. Tracklets
C. Canonical Physical People
And ambiguous query handling ('How many people were detected?')
"""
import uuid
import pytest
from database.session import SessionLocal
from database.models import VideoModel, TrackModel, EventModel
from backend.app.services.investigation_service import InvestigationService


@pytest.fixture
def count_fixture():
    db = SessionLocal()
    vid = f"test_count_{uuid.uuid4().hex[:8]}"
    try:
        db.add(VideoModel(
            id=vid,
            original_filename="count_test.mp4",
            storage_path=f"storage/uploads/{vid}.mp4",
            status="processed",
            fps=30.0,
            duration_seconds=20.0,
        ))

        # Add 2 physical people across 4 tracklets and 42 raw detections
        # Person A: 2 tracklets (TRACK-001, TRACK-002), 20 detections
        db.add(TrackModel(
            id=str(uuid.uuid4()),
            video_id=vid,
            track_id="TRACK-001",
            object_class="person",
            first_seen=0.0,
            last_seen=8.0,
            duration_seconds=8.0,
            detection_count=10,
            max_confidence=0.89,
            color="blue",
            color_confidence=0.85,
            current_bbox='{"x1": 500.0, "y1": 80.0, "x2": 580.0, "y2": 250.0}',
            trajectory='[[0.0, 540.0, 165.0], [8.0, 540.0, 165.0]]',
            active=False,
        ))
        db.add(TrackModel(
            id=str(uuid.uuid4()),
            video_id=vid,
            track_id="TRACK-002",
            object_class="person",
            first_seen=9.0,
            last_seen=20.0,
            duration_seconds=11.0,
            detection_count=10,
            max_confidence=0.91,
            color="blue",
            color_confidence=0.88,
            current_bbox='{"x1": 505.0, "y1": 82.0, "x2": 585.0, "y2": 252.0}',
            trajectory='[[9.0, 542.0, 167.0], [20.0, 545.0, 170.0]]',
            active=False,
        ))

        # Person B: 2 tracklets (TRACK-003, TRACK-004), 22 detections
        db.add(TrackModel(
            id=str(uuid.uuid4()),
            video_id=vid,
            track_id="TRACK-003",
            object_class="person",
            first_seen=0.0,
            last_seen=10.0,
            duration_seconds=10.0,
            detection_count=11,
            max_confidence=0.85,
            color="black",
            color_confidence=0.75,
            current_bbox='{"x1": 150.0, "y1": 50.0, "x2": 220.0, "y2": 220.0}',
            trajectory='[[0.0, 185.0, 135.0], [10.0, 185.0, 135.0]]',
            active=False,
        ))
        db.add(TrackModel(
            id=str(uuid.uuid4()),
            video_id=vid,
            track_id="TRACK-004",
            object_class="person",
            first_seen=11.0,
            last_seen=20.0,
            duration_seconds=9.0,
            detection_count=11,
            max_confidence=0.87,
            color="black",
            color_confidence=0.78,
            current_bbox='{"x1": 155.0, "y1": 52.0, "x2": 225.0, "y2": 222.0}',
            trajectory='[[11.0, 190.0, 137.0], [20.0, 192.0, 140.0]]',
            active=False,
        ))

        # Add 42 raw detection events (20 for Person A, 22 for Person B)
        for i in range(42):
            ts = (i / 42.0) * 20.0
            db.add(EventModel(
                id=str(uuid.uuid4()),
                video_id=vid,
                event_type="object_detected",
                object_class="person",
                class_id=0,
                timestamp_seconds=ts,
                confidence=0.85,
                bbox_x1=500.0 if i < 20 else 150.0,
                bbox_y1=80.0 if i < 20 else 50.0,
                bbox_x2=580.0 if i < 20 else 220.0,
                bbox_y2=250.0 if i < 20 else 220.0,
                frame_number=int(ts * 30),
                validation_status="VALID",
            ))

        db.commit()
        yield vid
    finally:
        db.query(EventModel).filter(EventModel.video_id == vid).delete(synchronize_session=False)
        db.query(TrackModel).filter(TrackModel.video_id == vid).delete(synchronize_session=False)
        db.query(VideoModel).filter(VideoModel.id == vid).delete(synchronize_session=False)
        db.commit()
        db.close()


def test_three_count_concepts_separation(count_fixture):
    vid = count_fixture
    svc = InvestigationService()

    # Concept A: Canonical physical people
    res_canonical = svc.investigate(vid, "How many people are in the video?")
    assert res_canonical["result_type"] == "count"
    assert res_canonical["canonical_entity_count"] == 2
    assert res_canonical["count"] == 2
    assert "2 distinct physical people" in res_canonical["message"]

    # Concept B: Tracklets
    res_tracks = svc.investigate(vid, "How many tracks are there?")
    assert res_tracks["result_type"] == "count"
    assert res_tracks["track_count"] == 4
    assert res_tracks["count"] == 4
    assert "4 tracks recorded" in res_tracks["message"]
    assert "2 distinct physical" in res_tracks["message"]

    # Concept C: Raw detection observations
    res_detections = svc.investigate(vid, "How many person detections were recorded?")
    assert res_detections["result_type"] == "count"
    assert res_detections["detection_observations"] == 42
    assert "42 validated person detection observations" in res_detections["message"]

    # Ambiguous query: "How many people were detected?"
    res_ambiguous = svc.investigate(vid, "How many people were detected?")
    assert res_ambiguous["result_type"] == "count"
    assert res_ambiguous["canonical_entity_count"] == 2
    assert res_ambiguous["detection_observations"] == 42
    # Must clearly explain: 2 distinct physical people from 42 validated observations
    assert "2 distinct physical people were identified from 42 validated person detection observations" in res_ambiguous["message"]
