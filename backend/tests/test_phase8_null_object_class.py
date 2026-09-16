"""
Phase 8 Regression Tests: Null Object Class in Investigation Results

Verifies:
1. Event with object_class string (e.g. "person", "car").
2. Event with object_class null (e.g. PROLONGED_PRESENCE, HIGH_ACTIVITY_PERIOD).
3. Event with object_class missing/undefined.
4. Event with event_type but no object class displays/processes without crash.
5. "Which events should I review?" response containing mixed events (both typed and null-class).
6. Preserves real object classes without inventing artificial ones.
"""

import uuid
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from database.session import SessionLocal
from database.models import VideoModel, SecurityEventModel
from ai.investigation.orchestrator import InvestigationOrchestrator
from ai.investigation.provider import MockLLMProvider
from backend.app.services.investigation_service import InvestigationService

client = TestClient(app)


@pytest.fixture(autouse=True)
def deterministic_llm_provider(monkeypatch):
    """Ensure investigation tests run deterministically using MockLLMProvider without live Gemini calls."""
    monkeypatch.setattr(
        "ai.investigation.orchestrator.get_llm_provider",
        lambda *args, **kwargs: MockLLMProvider(),
    )


@pytest.fixture
def seeded_mixed_events_video():
    """Create a video record with mixed events: some with object_class string, some with null."""
    import json
    from backend.app.core.config import settings

    db = SessionLocal()
    vid = f"test_null_obj_{uuid.uuid4().hex[:8]}"
    meta_path = settings.STORAGE_UPLOADS_DIR / f"{vid}.json"

    try:
        # Write sidecar metadata
        meta = {
            "video_id": vid,
            "filename": "mixed_test.mp4",
            "saved_filename": f"{vid}_mixed_test.mp4",
            "storage_path": str(settings.STORAGE_UPLOADS_DIR / f"{vid}_mixed_test.mp4"),
            "file_size_bytes": 1024,
            "duration_seconds": 30.0,
            "fps": 30.0,
            "frames_processed": 900,
            "detections_count": 10,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        # Create video
        video_rec = VideoModel(
            id=vid,
            original_filename="mixed_test.mp4",
            storage_path=f"storage/uploads/{vid}.mp4",
            file_size_bytes=1024,
            duration_seconds=30.0,
            fps=30.0,
            frame_count=900,
            status="PROCESSED",
        )
        db.add(video_rec)

        # 1. Event with object_class string
        ev_string = SecurityEventModel(
            id=f"sev_{uuid.uuid4().hex[:8]}",
            video_id=vid,
            event_type="POTENTIAL_INTRUSION",
            timestamp_seconds=4.5,
            duration_seconds=2.0,
            track_id="TRK-001",
            object_class="person",
            severity="HIGH",
            confidence=0.92,
            zone_name="Perimeter",
            description="Person entered secured zone.",
            bounding_box={"x1": 100, "y1": 150, "x2": 250, "y2": 450},
        )
        db.add(ev_string)

        # 2. Event with object_class null (PROLONGED_PRESENCE)
        ev_null = SecurityEventModel(
            id=f"sev_{uuid.uuid4().hex[:8]}",
            video_id=vid,
            event_type="PROLONGED_PRESENCE",
            timestamp_seconds=10.0,
            duration_seconds=5.0,
            track_id="TRK-002",
            object_class=None,
            severity="NORMAL",
            confidence=0.88,
            zone_name="Main Hall",
            description="Prolonged presence detected in visual field.",
            bounding_box={"x1": 300, "y1": 200, "x2": 450, "y2": 500},
        )
        db.add(ev_null)

        # 3. Event with event_type but no object class (HIGH_ACTIVITY_PERIOD)
        ev_activity = SecurityEventModel(
            id=f"sev_{uuid.uuid4().hex[:8]}",
            video_id=vid,
            event_type="HIGH_ACTIVITY_PERIOD",
            timestamp_seconds=18.0,
            duration_seconds=4.0,
            track_id=None,
            object_class=None,
            severity="NORMAL",
            confidence=0.85,
            zone_name=None,
            description="Elevated cluster density observed.",
            bounding_box=None,
        )
        db.add(ev_activity)

        # 4. Event with object_class string (suitcase takeaway)
        ev_theft = SecurityEventModel(
            id=f"sev_{uuid.uuid4().hex[:8]}",
            video_id=vid,
            event_type="POTENTIAL_THEFT",
            timestamp_seconds=24.0,
            duration_seconds=3.0,
            track_id="TRK-003",
            object_class="suitcase",
            severity="HIGH",
            confidence=0.94,
            zone_name=None,
            description="Object removed from scene after interaction.",
            bounding_box={"x1": 500, "y1": 300, "x2": 600, "y2": 450},
        )
        db.add(ev_theft)

        db.commit()
        yield vid
    finally:
        if meta_path.exists():
            meta_path.unlink()
        db.query(SecurityEventModel).filter(SecurityEventModel.video_id == vid).delete()
        db.query(VideoModel).filter(VideoModel.id == vid).delete()
        db.commit()
        db.close()


def test_01_event_with_object_class_string(seeded_mixed_events_video):
    """Verify security event with string object_class is accurately retrieved."""
    service = InvestigationService()
    res = service.investigate_filters(
        video_id=seeded_mixed_events_video,
        filters={"event_type": "POTENTIAL_INTRUSION"},
        result_type="security_events",
        query_text="Show intrusion events",
    )
    assert res["count"] == 1
    item = res["results"][0]
    assert item["object_class"] == "person"
    assert item["event_type"] == "POTENTIAL_INTRUSION"


def test_02_event_with_object_class_null(seeded_mixed_events_video):
    """Verify security event with null object_class is retrieved without inventing data."""
    service = InvestigationService()
    res = service.investigate_filters(
        video_id=seeded_mixed_events_video,
        filters={"event_type": "PROLONGED_PRESENCE"},
        result_type="security_events",
        query_text="Show prolonged presence",
    )
    assert res["count"] == 1
    item = res["results"][0]
    assert item["object_class"] is None  # Must NOT invent artificial object class
    assert item["event_type"] == "PROLONGED_PRESENCE"


def test_03_event_with_event_type_no_object_class(seeded_mixed_events_video):
    """Verify event with event_type but no object class is handled safely."""
    service = InvestigationService()
    res = service.investigate_filters(
        video_id=seeded_mixed_events_video,
        filters={"event_type": "HIGH_ACTIVITY_PERIOD"},
        result_type="security_events",
        query_text="Show high activity",
    )
    assert res["count"] == 1
    item = res["results"][0]
    assert item["object_class"] is None
    assert item["event_type"] == "HIGH_ACTIVITY_PERIOD"


def test_04_which_events_should_i_review_mixed_events(seeded_mixed_events_video):
    """Verify 'Which events should I review?' returns mixed events and packaging succeeds."""
    orchestrator = InvestigationOrchestrator()
    res = orchestrator.process_investigation(
        video_id=seeded_mixed_events_video,
        user_query="Which events should I review?",
    )

    assert res["is_supported"] is True
    assert res["count"] >= 4
    assert "sources" in res
    detections = res["sources"]["detections"]

    # Verify sources contain both null and non-null object classes
    has_null_class = False
    has_string_class = False
    has_event_type = False

    for det in detections:
        if det["object_class"] is None:
            has_null_class = True
        elif isinstance(det["object_class"], str):
            has_string_class = True
        if det.get("event_type") is not None:
            has_event_type = True

    assert has_null_class, "Expected at least one detection source with null object_class"
    assert has_string_class, "Expected at least one detection source with string object_class"
    assert has_event_type, "Expected detection sources to include event_type for fallback rendering"


def test_05_api_investigate_endpoint_mixed_events(seeded_mixed_events_video):
    """Verify POST /api/videos/{id}/investigate returns HTTP 200 with nullable object_class."""
    resp = client.post(
        f"/api/videos/{seeded_mixed_events_video}/investigate",
        json={"query": "Which events should I review?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_supported"] is True
    assert len(data["results"]) >= 4

    # Ensure items with null object_class are present and valid
    null_items = [r for r in data["results"] if r.get("object_class") is None]
    assert len(null_items) >= 2
    for item in null_items:
        assert item.get("event_type") in ("PROLONGED_PRESENCE", "HIGH_ACTIVITY_PERIOD")


def test_06_api_ai_investigate_endpoint_mixed_events(seeded_mixed_events_video):
    """Verify POST /api/videos/{id}/ai-investigate returns HTTP 200 with nullable sources."""
    resp = client.post(
        f"/api/videos/{seeded_mixed_events_video}/ai-investigate",
        json={"query": "Which events should I review?", "mode": "deterministic"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_supported"] is True
    assert "sources" in data
    assert "detections" in data["sources"]

    # Check sources.detections contains null object_class without crashing backend
    det_sources = data["sources"]["detections"]
    assert any(d.get("object_class") is None for d in det_sources)
    assert any(d.get("object_class") == "person" for d in det_sources)
