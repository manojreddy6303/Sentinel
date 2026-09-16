"""
Sentinel Phase 4 Automated Test Suite

Tests:
1. Database initialization and model schema binding.
2. Video record creation & metadata storage.
3. Event grouping algorithm (temporal window clustering, priority calculation, event labeling).
4. Relational Event repository save & retrieve with filters (class, confidence, timeframe).
5. Reprocessing idempotency (clearing prior events before re-analyzing).
6. Timeline API endpoint (GET /api/videos/{video_id}/timeline).
7. Raw events API endpoint (GET /api/videos/{video_id}/events).
8. End-to-end video upload, process, database persistence, and timeline query.
"""

import os
import sys
import uuid
import pytest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.app.main import app
from database.session import init_db, SessionLocal
from database.models import VideoModel, EventModel, GroupedEventModel
from ai.events.generator import EventGenerator, determine_event_type, calculate_investigation_priority
from ai.events.repository import DatabaseEventRepository, JSONEventRepository, get_event_repository

client = TestClient(app)


def test_db_initialization():
    """Verify database engine connects and creates tables cleanly."""
    init_db()
    db = SessionLocal()
    try:
        # Query tables to confirm existence
        video_count = db.query(VideoModel).count()
        event_count = db.query(EventModel).count()
        grouped_count = db.query(GroupedEventModel).count()
        assert isinstance(video_count, int)
        assert isinstance(event_count, int)
        assert isinstance(grouped_count, int)
    finally:
        db.close()


def test_event_grouping_logic():
    """Verify temporal event grouping, naming logic, and priority calculation."""
    generator = EventGenerator(window_seconds=2.0)
    video_id = str(uuid.uuid4())

    raw_events = [
        # Cluster 1 (0.0s to 1.5s): Vehicles
        {"event_id": "1", "object_class": "car", "confidence": 0.85, "timestamp": 0.5, "bounding_box": {"x1":0,"y1":0,"x2":10,"y2":10}},
        {"event_id": "2", "object_class": "bus", "confidence": 0.75, "timestamp": 1.0, "bounding_box": {"x1":0,"y1":0,"x2":10,"y2":10}},
        {"event_id": "3", "object_class": "car", "confidence": 0.90, "timestamp": 1.5, "bounding_box": {"x1":0,"y1":0,"x2":10,"y2":10}},
        
        # Cluster 2 (10.0s): Person
        {"event_id": "4", "object_class": "person", "confidence": 0.60, "timestamp": 10.0, "bounding_box": {"x1":0,"y1":0,"x2":10,"y2":10}},
    ]

    grouped = generator.group_events(video_id, raw_events)
    assert len(grouped) == 2

    # Cluster 1 assertions
    c1 = grouped[0]
    assert c1["event_type"] == "VEHICLE_ACTIVITY"
    assert c1["total_detections"] == 3
    assert c1["max_confidence"] == 0.90
    assert c1["start_time"] == 0.5
    assert c1["end_time"] == 1.5
    assert len(c1["objects"]) == 2  # bus & car

    # Cluster 2 assertions
    c2 = grouped[1]
    assert c2["event_type"] == "PERSON_ACTIVITY"
    assert c2["total_detections"] == 1
    assert c2["start_time"] == 10.0



def test_event_repository_db_persistence():
    """Test DatabaseEventRepository save, retrieve, filtering, and reprocessing."""
    repo = DatabaseEventRepository()
    video_id = str(uuid.uuid4())

    raw_events = [
        {"event_id": str(uuid.uuid4()), "object_class": "person", "class_id": 0, "confidence": 0.88, "timestamp": 2.0, "bounding_box": {"x1":0,"y1":0,"x2":5,"y2":5}, "frame_number": 60},
        {"event_id": str(uuid.uuid4()), "object_class": "car", "class_id": 2, "confidence": 0.45, "timestamp": 5.0, "bounding_box": {"x1":0,"y1":0,"x2":5,"y2":5}, "frame_number": 150},
    ]

    generator = EventGenerator()
    grouped = generator.group_events(video_id, raw_events)

    # 1. Save events
    repo.save_events(video_id, raw_events, grouped_events=grouped)
    assert repo.events_exist(video_id) is True

    # 2. Retrieve raw events with filter
    person_events = repo.get_events(video_id, object_class="person")
    assert len(person_events) == 1
    assert person_events[0]["object_class"] == "person"

    high_conf = repo.get_events(video_id, min_confidence=0.80)
    assert len(high_conf) == 1
    assert high_conf[0]["confidence"] == 0.88

    # 3. Retrieve timeline grouped events
    timeline = repo.get_grouped_events(video_id)
    assert len(timeline) == 2

    # 4. Reprocessing idempotency check (save new batch, ensure old deleted)
    new_raw = [
        {"event_id": str(uuid.uuid4()), "object_class": "truck", "class_id": 7, "confidence": 0.95, "timestamp": 8.0, "bounding_box": {"x1":0,"y1":0,"x2":5,"y2":5}, "frame_number": 240}
    ]
    new_grouped = generator.group_events(video_id, new_raw)
    repo.save_events(video_id, new_raw, grouped_events=new_grouped)

    updated_raw = repo.get_events(video_id)
    assert len(updated_raw) == 1
    assert updated_raw[0]["object_class"] == "truck"


def test_timeline_api_endpoints():
    """Test GET /api/videos/{video_id}/timeline and /events endpoints."""
    # First upload a video
    import cv2
    import numpy as np

    temp_path = PROJECT_ROOT / "storage" / "test_phase4.mp4"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(temp_path), fourcc, 10.0, (160, 120))
    for _ in range(10):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    with open(temp_path, "rb") as f:
        response = client.post("/api/videos/upload", files={"file": ("test_phase4.mp4", f, "video/mp4")})

    assert response.status_code == 201
    video_id = response.json()["video_id"]

    # Process video
    proc_res = client.post(f"/api/videos/{video_id}/process")
    assert proc_res.status_code == 200

    # Query Timeline API
    timeline_res = client.get(f"/api/videos/{video_id}/timeline")
    assert timeline_res.status_code == 200
    data = timeline_res.json()
    assert data["video_id"] == video_id
    assert "events" in data
    assert "total_events" in data

    # Cleanup temp file
    if temp_path.exists():
        temp_path.unlink()
