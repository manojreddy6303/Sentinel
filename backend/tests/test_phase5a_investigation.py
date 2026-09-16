"""
Sentinel Phase 5A Automated Test Suite
Natural-Language Investigation Layer ("Ask Sentinel")

Covers:
1. Parse "show all people detected"
2. Parse "show cars between 8 and 12 seconds"
3. Parse confidence above 70%
4. Parse count query
5. Parse event query
6. Query database with object class filter
7. Query database with time filter
8. Query database with confidence filter
9. Combined filters
10. Zero-result query
11. Unsupported identity question (ethical guardrail)
12. Invalid / empty query
13. API success response (POST /api/videos/{video_id}/investigate)
14. API validation failure (400 bad video ID, 404 missing, 422 empty body)
15. Results sorted chronologically
16. Verification of existing pipeline tests
"""

import os
import sys
import uuid
import json
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
from backend.app.services.investigation_parser import InvestigationParser
from backend.app.services.investigation_service import InvestigationService
from ai.events.repository import DatabaseEventRepository
from ai.events.generator import EventGenerator
from backend.app.core.config import settings

client = TestClient(app)


# ---------------------------------------------------------------------------
# Test 1 to 5: Investigation Query Parser
# ---------------------------------------------------------------------------

def test_1_parse_show_all_people_detected():
    """1. Parse 'show all people detected' -> class 'person', result_type 'detections'."""
    parsed = InvestigationParser.parse_query("Show all people detected")
    assert parsed["is_supported"] is True
    assert parsed["result_type"] == "detections"
    assert parsed["interpreted_filters"]["object_class"] == "person"
    assert parsed["interpreted_filters"]["start_time"] is None
    assert parsed["interpreted_filters"]["end_time"] is None


def test_2_parse_show_cars_between_8_and_12_seconds():
    """2. Parse 'show cars between 8 and 12 seconds' -> class 'car', start 8, end 12."""
    parsed = InvestigationParser.parse_query("Show cars between 8 and 12 seconds")
    assert parsed["is_supported"] is True
    assert parsed["result_type"] == "detections"
    assert parsed["interpreted_filters"]["object_class"] == "car"
    assert parsed["interpreted_filters"]["start_time"] == 8.0
    assert parsed["interpreted_filters"]["end_time"] == 12.0


def test_3_parse_confidence_above_70():
    """3. Parse 'show detections with confidence above 70%' -> min_confidence 0.70."""
    parsed = InvestigationParser.parse_query("Show detections with confidence above 70%")
    assert parsed["is_supported"] is True
    assert parsed["interpreted_filters"]["min_confidence"] == 0.70

    # Also check decimal format: "confidence greater than 0.8"
    parsed_dec = InvestigationParser.parse_query("Show detections confidence greater than 0.8")
    assert parsed_dec["interpreted_filters"]["min_confidence"] == 0.80


def test_4_parse_count_query():
    """4. Parse 'How many people were detected?' -> result_type 'count', class 'person'."""
    parsed = InvestigationParser.parse_query("How many people were detected?")
    assert parsed["is_supported"] is True
    assert parsed["result_type"] == "count"
    assert parsed["interpreted_filters"]["object_class"] == "person"

    parsed_cars = InvestigationParser.parse_query("How many cars were detected?")
    assert parsed_cars["result_type"] == "count"
    assert parsed_cars["interpreted_filters"]["object_class"] == "car"


def test_5_parse_event_query():
    """5. Parse 'Show events between 10 and 15 seconds' -> result_type 'events'."""
    parsed = InvestigationParser.parse_query("Show events between 10 and 15 seconds")
    assert parsed["is_supported"] is True
    assert parsed["result_type"] == "events"
    assert parsed["interpreted_filters"]["start_time"] == 10.0
    assert parsed["interpreted_filters"]["end_time"] == 15.0


# ---------------------------------------------------------------------------
# Test Fixture for Database Ground-Truth Testing
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def seeded_video_id():
    """Seeds a test video with known detections and grouped events."""
    init_db()
    video_id = f"test_inv_{uuid.uuid4().hex[:10]}"

    # Persist video metadata sidecar so API checks succeed
    meta_path = settings.STORAGE_UPLOADS_DIR / f"{video_id}.json"
    meta = {
        "video_id": video_id,
        "filename": "surveillance_test.mp4",
        "saved_filename": f"{video_id}_surveillance_test.mp4",
        "file_size_bytes": 1024,
        "status": "processed",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    raw_detections = [
        {"event_id": str(uuid.uuid4()), "object_class": "person", "class_id": 0, "confidence": 0.88, "timestamp": 2.50, "frame_number": 75, "bounding_box": {"x1": 10, "y1": 10, "x2": 50, "y2": 100}},
        {"event_id": str(uuid.uuid4()), "object_class": "person", "class_id": 0, "confidence": 0.65, "timestamp": 5.20, "frame_number": 156, "bounding_box": {"x1": 20, "y1": 20, "x2": 60, "y2": 110}},
        {"event_id": str(uuid.uuid4()), "object_class": "car", "class_id": 2, "confidence": 0.81, "timestamp": 8.01, "frame_number": 240, "bounding_box": {"x1": 100, "y1": 200, "x2": 300, "y2": 400}},
        {"event_id": str(uuid.uuid4()), "object_class": "car", "class_id": 2, "confidence": 0.76, "timestamp": 9.02, "frame_number": 270, "bounding_box": {"x1": 110, "y1": 205, "x2": 310, "y2": 405}},
        {"event_id": str(uuid.uuid4()), "object_class": "car", "class_id": 2, "confidence": 0.55, "timestamp": 11.50, "frame_number": 345, "bounding_box": {"x1": 120, "y1": 210, "x2": 320, "y2": 410}},
        {"event_id": str(uuid.uuid4()), "object_class": "dog", "class_id": 16, "confidence": 0.92, "timestamp": 14.00, "frame_number": 420, "bounding_box": {"x1": 30, "y1": 300, "x2": 80, "y2": 350}},
    ]

    generator = EventGenerator(window_seconds=2.0)
    grouped = generator.group_events(video_id, raw_detections)

    repo = DatabaseEventRepository()
    repo.save_events(video_id, raw_detections, grouped)

    yield video_id

    # Teardown
    if meta_path.exists():
        try:
            meta_path.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Test 6 to 10: Database Ground-Truth Queries
# ---------------------------------------------------------------------------

def test_6_query_database_with_object_class_filter(seeded_video_id):
    """6. Query database with object class filter 'person'."""
    service = InvestigationService()
    res = service.investigate(seeded_video_id, "Show all people detected")
    assert res["is_supported"] is True
    assert res["count"] == 2
    for r in res["results"]:
        assert r["object_class"] == "person"


def test_7_query_database_with_time_filter(seeded_video_id):
    """7. Query database with time filter 'between 8 and 12 seconds'."""
    service = InvestigationService()
    res = service.investigate(seeded_video_id, "Show cars between 8 and 12 seconds")
    assert res["is_supported"] is True
    assert res["count"] == 3  # 8.01s, 9.02s, 11.50s
    for r in res["results"]:
        assert 8.0 <= r["timestamp"] <= 12.0
        assert r["object_class"] == "car"


def test_8_query_database_with_confidence_filter(seeded_video_id):
    """8. Query database with confidence filter 'confidence above 70%'."""
    service = InvestigationService()
    res = service.investigate(seeded_video_id, "Show detections with confidence above 70%")
    assert res["is_supported"] is True
    # In seeded data: 0.88, 0.81, 0.76, 0.92 = 4 detections
    assert res["count"] == 4
    for r in res["results"]:
        assert r["confidence"] >= 0.70


def test_9_combined_filters(seeded_video_id):
    """9. Combined filters: 'Show cars between 8 and 10 seconds with confidence above 75%'."""
    service = InvestigationService()
    res = service.investigate(seeded_video_id, "Show cars between 8 and 10 seconds confidence above 75%")
    assert res["is_supported"] is True
    # 8.01s (0.81) and 9.02s (0.76) match
    assert res["count"] == 2
    for r in res["results"]:
        assert r["object_class"] == "car"
        assert 8.0 <= r["timestamp"] <= 10.0
        assert r["confidence"] >= 0.75


def test_10_zero_result_query(seeded_video_id):
    """10. Zero-result query: returns clean response with count 0 and message."""
    service = InvestigationService()
    res = service.investigate(seeded_video_id, "Show airplanes between 1 and 3 seconds")
    assert res["is_supported"] is True
    assert res["count"] == 0
    assert len(res["results"]) == 0
    assert "No matching detections were found." in res["message"]


# ---------------------------------------------------------------------------
# Test 11 to 12: Ethical Guardrails & Validation
# ---------------------------------------------------------------------------

def test_11_unsupported_identity_question(seeded_video_id):
    """11. Unsupported identity/criminal queries must produce safety refusal, not hallucinations."""
    service = InvestigationService()
    unsupported_queries = [
        "Who is this person?",
        "Is this person a criminal?",
        "What is the person's name?",
        "Identify this person",
        "Is this definitely a theft?",
        "Show red car detected",  # color filter unsupported
    ]

    for q in unsupported_queries:
        res = service.investigate(seeded_video_id, q)
        assert res["is_supported"] is False
        assert res["result_type"] == "unsupported"
        assert res["count"] == 0
        assert len(res["results"]) == 0
        assert len(res["message"]) > 10


def test_12_invalid_empty_query(seeded_video_id):
    """12. Invalid/empty query returns clear error message."""
    service = InvestigationService()
    res = service.investigate(seeded_video_id, "   ")
    assert res["is_supported"] is False
    assert res["result_type"] == "error"
    assert "empty" in res["message"].lower()


# ---------------------------------------------------------------------------
# Test 13 to 15: API Integration & Chronological Ordering
# ---------------------------------------------------------------------------

def test_13_api_success_response(seeded_video_id):
    """13. API endpoint POST /api/videos/{video_id}/investigate returns HTTP 200 with structured data."""
    response = client.post(
        f"/api/videos/{seeded_video_id}/investigate",
        json={"query": "Show all people detected"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["video_id"] == seeded_video_id
    assert data["query"] == "Show all people detected"
    assert data["is_supported"] is True
    assert data["count"] == 2
    assert len(data["results"]) == 2
    assert "bounding_box" in data["results"][0]


def test_14_api_validation_failure():
    """14. API returns proper HTTP error codes for invalid inputs."""
    # Invalid video ID with path traversal characters -> 400 Bad Request
    res_bad_id = client.post(
        "/api/videos/../../etc/passwd/investigate",
        json={"query": "Show all cars"},
    )
    assert res_bad_id.status_code in [400, 404]

    # Non-existent valid video ID -> 404 Not Found
    res_not_found = client.post(
        "/api/videos/non_existent_vid_12345/investigate",
        json={"query": "Show all cars"},
    )
    assert res_not_found.status_code == 404

    # Empty payload or missing query -> 422 Unprocessable Entity
    res_invalid_body = client.post(
        "/api/videos/valid_id_format/investigate",
        json={"query": ""},
    )
    assert res_invalid_body.status_code == 422


def test_15_results_sorted_chronologically(seeded_video_id):
    """15. Detection results must be strictly ordered chronologically by timestamp."""
    service = InvestigationService()
    res = service.investigate(seeded_video_id, "Show all detections")
    assert res["is_supported"] is True
    assert res["count"] == 6
    timestamps = [r["timestamp"] for r in res["results"]]
    assert timestamps == sorted(timestamps)
    assert timestamps[0] == 2.50
    assert timestamps[-1] == 14.00


def test_16_database_count_query(seeded_video_id):
    """16. Exact count query on actual database records with personal identity disclaimer."""
    service = InvestigationService()
    res = service.investigate(seeded_video_id, "How many cars were detected?")
    assert res["is_supported"] is True
    assert res["result_type"] == "count"
    assert res["count"] == 3
    assert "Detected 3 total car detection records" in res["message"]
    assert "does not attribute unique personal identities" in res["message"]


def test_17_database_events_query(seeded_video_id):
    """17. Event query against grouped timeline events."""
    service = InvestigationService()
    res = service.investigate(seeded_video_id, "Show events between 8 and 12 seconds")
    assert res["is_supported"] is True
    assert res["result_type"] == "events"
    assert res["count"] >= 1
    for r in res["results"]:
        assert "event_type" in r
        assert "start_time" in r
        assert "end_time" in r
        assert "objects" in r

