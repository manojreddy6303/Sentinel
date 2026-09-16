"""
Phase 19.1 UX, Workflow & Workspace Integration Hardening Tests
Validates all newly added backend endpoints, invariant parity checks,
entity linking, and database isolation.
"""

import os
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from database.session import engine, SessionLocal, init_db
from database.models import (
    CaseModel,
    VideoModel,
    CameraSourceModel,
    CorrelatedIncidentModel,
    EventModel,
    EvidenceModel,
)

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    init_db()


def test_database_isolation_environment():
    """Verify that tests run against test database and not the primary database."""
    db_url = str(engine.url)
    assert "test" in db_url.lower() or "sqlite:///:memory:" in db_url.lower(), (
        f"Database must be an isolated test database, but got {db_url}"
    )


def test_get_videos_endpoint():
    """Test GET /api/videos with filtering and pagination."""
    response = client.get("/api/videos?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "videos" in data
    assert isinstance(data["videos"], list)
    assert data["limit"] == 10
    assert data["offset"] == 0

    # Search filter test
    response_search = client.get("/api/videos?search=nonexistent_xyz_query")
    assert response_search.status_code == 200
    assert response_search.json()["total"] == 0


def test_get_evidence_endpoint():
    """Test GET /api/evidence with filtering."""
    response = client.get("/api/evidence?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "evidence" in data
    assert isinstance(data["evidence"], list)

    # Class filter test
    response_filtered = client.get("/api/evidence?object_class=person")
    assert response_filtered.status_code == 200
    assert isinstance(response_filtered.json()["evidence"], list)


def test_get_incidents_endpoint():
    """Test GET /api/incidents with category and review filters."""
    response = client.get("/api/incidents?limit=20")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "incidents" in data
    assert isinstance(data["incidents"], list)

    # If any incidents exist, verify schema contains critical fields
    for inc in data["incidents"]:
        assert "incident_id" in inc
        assert "category" in inc
        assert "assessment_score" in inc
        assert "evidence_strength" in inc
        assert "review_required" in inc
        # Canonical rule: assessment_score <= 0.65 means review_required is True
        if inc["assessment_score"] <= 0.65:
            assert inc["review_required"] is True, (
                f"Incident {inc['incident_id']} has score <= 0.65 but review_required is False"
            )


def test_cameras_endpoint_and_registration():
    """Test GET /api/cameras and POST /api/cameras."""
    # List cameras
    get_res = client.get("/api/cameras")
    assert get_res.status_code == 200
    cameras_data = get_res.json()
    assert "total" in cameras_data
    assert "cameras" in cameras_data

    # Create dummy video to associate camera
    db = SessionLocal()
    dummy_video = VideoModel(
        id="test_cam_vid_001",
        original_filename="cam_test.mp4",
        storage_path="/tmp/cam_test.mp4",
        status="processed",
    )
    db.merge(dummy_video)
    db.commit()
    db.close()

    # Register camera
    post_res = client.post(
        "/api/cameras",
        json={
            "camera_label": "Gate 1 North East",
            "video_id": "test_cam_vid_001",
            "position_hint": "North Gate Exterior",
            "field_of_view_hint": "Parking lot and entry gate",
            "adjacency_hints": ["Gate 2"],
        },
    )
    assert post_res.status_code in [200, 201]
    created = post_res.json()
    assert created["camera_label"] == "Gate 1 North East"
    assert created["video_id"] == "test_cam_vid_001"
    assert "camera_id" in created

    # Verify camera appears in listing
    list_res = client.get("/api/cameras?search=Gate 1")
    assert list_res.status_code == 200
    assert list_res.json()["total"] >= 1


def test_analytics_summary_endpoint():
    """Test GET /api/analytics returns accurate metrics and invariant consistency."""
    response = client.get("/api/analytics")
    assert response.status_code == 200
    data = response.json()

    assert "total_videos" in data
    assert "processed_videos" in data
    assert "raw_observations_count" in data
    assert "validated_detections_count" in data
    assert "rejected_detections_count" in data
    assert "parity_consistent" in data
    assert "total_tracks" in data
    assert "total_security_events" in data
    assert "total_correlated_incidents" in data
    assert "total_evidence" in data
    assert "review_ceiling" in data
    assert data["review_ceiling"] == 0.65
    assert "detector_health" in data
    assert isinstance(data["detector_health"], list)

    # Invariant: parity_consistent must match RAW = VALIDATED + REJECTED
    expected_parity = data["raw_observations_count"] == (
        data["validated_detections_count"] + data["rejected_detections_count"]
    )
    assert data["parity_consistent"] == expected_parity


def test_reports_endpoint():
    """Test GET /api/reports listing all generated dossiers."""
    response = client.get("/api/reports")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "count" in data
    assert "reports" in data
    assert isinstance(data["reports"], list)


def test_case_incident_linking():
    """Test linking and unlinking an incident to a security case without manual UUID entry."""
    db = SessionLocal()
    # Create test case
    test_case = CaseModel(
        id="case-inc-test-01",
        case_number="CASE-INC-001",
        title="Incident Linking Verification Case",
        description="Testing incident association",
        status="OPEN",
        priority="HIGH",
    )
    db.merge(test_case)

    # Create test incident
    test_incident = CorrelatedIncidentModel(
        id="inc-link-test-999",
        video_id="test_cam_vid_001",
        incident_category="UNAUTHORIZED_ACCESS",
        start_time=10.0,
        end_time=15.0,
        duration=5.0,
        assessment_score=0.82,
        evidence_strength=0.90,
        reliability_rating="HIGH",
        validation_decision="FLAGGED",
        storyline="Test incident for case linking",
    )
    db.merge(test_incident)
    db.commit()
    db.close()

    # Link incident to case
    link_res = client.post(
        "/api/cases/case-inc-test-01/incidents",
        json={"incident_id": "inc-link-test-999", "notes": "Primary suspicious entry"},
    )
    assert link_res.status_code in [200, 201]
    link_data = link_res.json()
    assert link_data["case_id"] == "case-inc-test-01"
    assert link_data["incident_id"] == "inc-link-test-999"

    # Unlink incident from case
    unlink_res = client.delete("/api/cases/case-inc-test-01/incidents/inc-link-test-999")
    assert unlink_res.status_code in [200, 204]
