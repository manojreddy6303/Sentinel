"""
Unit and integration tests for the Sentinel Cybersecurity Extension.
Validates:
1. Database Model & Schema Integrity (CyberSecurityEventModel)
2. Cybersecurity Service & Query Filters
3. FastAPI Endpoints (/api/cyber/...)
4. Cyber-Physical Correlation Engine
5. Investigation Agent Multi-Step Workflow & Controlled Tools
6. Safety Safeguards: Provenance Truthfulness, Anonymous Tracking, Reliability Ceiling
"""

import pytest
from datetime import datetime, timezone
from contextlib import contextmanager
from fastapi.testclient import TestClient

from database.session import SessionLocal, init_db
from database.models import (
    CyberSecurityEventModel,
    CameraSourceModel,
    VideoModel,
    CorrelatedIncidentModel,
)
from backend.app.main import app
from backend.app.services.cyber_service import (
    create_cyber_event,
    get_cyber_event,
    list_cyber_events,
    list_security_assets,
    get_security_asset_events,
    correlate_cyber_physical,
    seed_demo_telemetry,
    VALID_EVENT_TYPES,
    VALID_PROVENANCE_TYPES,
)
from ai.investigation.orchestrator import InvestigationOrchestrator

client = TestClient(app)


@contextmanager
def get_test_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()


# ─────────────────────────────────────────────────────────────
# 1. DATABASE MODEL & REPOSITORY TESTS
# ─────────────────────────────────────────────────────────────

def test_cyber_event_creation_and_fields():
    """Verify CyberSecurityEventModel can be created with required and optional fields."""
    with get_test_db() as db:
        event = create_cyber_event(
            db=db,
            asset_label="CAM-NORTH-01",
            event_type="CONFIGURATION_CHANGE",
            severity="HIGH",
            status="ACTIVE",
            description="Configuration checksum mismatch detected",
            provenance="REPLAYED_TELEMETRY",
            structured_metadata={"source_ip": "10.0.4.12", "parameter": "stream_framerate"},
            video_offset_seconds=120.5,
        )
        assert event.id is not None
        assert event.event_type == "CONFIGURATION_CHANGE"
        assert event.severity == "HIGH"
        assert event.status == "ACTIVE"
        assert event.provenance == "REPLAYED_TELEMETRY"
        assert event.structured_metadata["source_ip"] == "10.0.4.12"
        assert event.video_offset_seconds == 120.5
        assert event.asset_label == "CAM-NORTH-01"

        # Fetch back
        fetched = get_cyber_event(db, event.id)
        assert fetched is not None
        assert fetched.id == event.id
        assert fetched.asset_label == "CAM-NORTH-01"


def test_invalid_event_type_rejection():
    """Verify invalid event types raise ValueError."""
    with get_test_db() as db:
        with pytest.raises(ValueError, match="Invalid event_type"):
            create_cyber_event(
                db=db,
                asset_label="CAM-NORTH-01",
                event_type="UNAUTHORIZED_EXPLOIT_PAYLOAD",  # Invalid type
                severity="HIGH",
                description="Invalid event test",
            )


def test_invalid_provenance_rejection():
    """Verify invalid provenance types raise ValueError."""
    with get_test_db() as db:
        with pytest.raises(ValueError, match="Invalid provenance"):
            create_cyber_event(
                db=db,
                asset_label="CAM-NORTH-01",
                event_type="AUTHENTICATION_ANOMALY",
                severity="LOW",
                provenance="UNVERIFIED_HACKER_FEED",  # Invalid provenance
                description="Invalid provenance test",
            )


def test_event_filtering_by_asset_severity_status():
    """Verify list_cyber_events filters correctly."""
    with get_test_db() as db:
        # Create distinct test events
        create_cyber_event(
            db=db,
            asset_label="CAM-FILTER-01",
            event_type="AUTHENTICATION_ANOMALY",
            severity="CRITICAL",
            status="INVESTIGATING",
            description="Brute-force auth anomaly",
        )
        create_cyber_event(
            db=db,
            asset_label="CAM-FILTER-01",
            event_type="CONNECTION_ANOMALY",
            severity="LOW",
            status="RESOLVED",
            description="Transient heartbeat drop",
        )
        create_cyber_event(
            db=db,
            asset_label="CAM-FILTER-02",
            event_type="DIGITAL_INTEGRITY_ANOMALY",
            severity="MEDIUM",
            status="ACTIVE",
            description="Hash verification discrepancy",
        )

        # Filter by asset
        asset1_events = list_cyber_events(db, asset_label="CAM-FILTER-01")
        assert len(asset1_events) >= 2
        assert all(e.asset_label == "CAM-FILTER-01" for e in asset1_events)

        # Filter by severity
        crit_events = list_cyber_events(db, severity="CRITICAL")
        assert any(e.severity == "CRITICAL" and e.asset_label == "CAM-FILTER-01" for e in crit_events)

        # Filter by status
        resolved = list_cyber_events(db, status="RESOLVED")
        assert any(e.status == "RESOLVED" and e.asset_label == "CAM-FILTER-01" for e in resolved)


# ─────────────────────────────────────────────────────────────
# 2. FASTAPI ENDPOINT TESTS
# ─────────────────────────────────────────────────────────────

def test_api_list_and_create_cyber_events():
    """POST /api/cyber/events and GET /api/cyber/events."""
    payload = {
        "asset_label": "CAM-TEST-API",
        "event_type": "SECURITY_POLICY_VIOLATION",
        "severity": "HIGH",
        "status": "ACTIVE",
        "description": "TLS 1.0 handshake attempted from legacy gateway",
        "provenance": "REPLAYED_TELEMETRY",
        "structured_metadata": {"ciphersuite": "TLS_RSA_WITH_AES_128_CBC_SHA"},
        "timestamp_seconds": 45.0,
    }

    # Create
    post_res = client.post("/api/cyber/events", json=payload)
    assert post_res.status_code == 201
    data = post_res.json()
    event_id = data["id"]
    assert data["asset_label"] == "CAM-TEST-API"
    assert data["event_type"] == "SECURITY_POLICY_VIOLATION"
    assert data["severity"] == "HIGH"
    assert data["provenance"] == "REPLAYED_TELEMETRY"

    # Fetch list
    get_res = client.get("/api/cyber/events", params={"asset_label": "CAM-TEST-API"})
    assert get_res.status_code == 200
    res_json = get_res.json()
    events = res_json.get("events", [])
    assert any(e["id"] == event_id for e in events)

    # Fetch single
    single_res = client.get(f"/api/cyber/events/{event_id}")
    assert single_res.status_code == 200
    assert single_res.json()["id"] == event_id


def test_api_invalid_event_creation():
    """POST /api/cyber/events with invalid payload returns 400 or 422."""
    bad_payload = {
        "asset_label": "CAM-TEST-BAD",
        "event_type": "UNKNOWN_SUPER_HACK",
        "severity": "HIGH",
        "description": "Bad event",
    }
    res = client.post("/api/cyber/events", json=bad_payload)
    assert res.status_code in (400, 422)


def test_api_security_assets_endpoint():
    """GET /api/cyber/assets returns registered security assets."""
    res = client.get("/api/cyber/assets")
    assert res.status_code == 200
    data = res.json()
    assert "assets" in data
    assert isinstance(data["assets"], list)


def test_api_seed_demo_telemetry():
    """POST /api/cyber/seed-demo seeds deterministic prototype telemetry."""
    res = client.post("/api/cyber/seed-demo")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert body["provenance"] == "REPLAYED_TELEMETRY"


# ─────────────────────────────────────────────────────────────
# 3. CYBER-PHYSICAL CORRELATION ENGINE TESTS
# ─────────────────────────────────────────────────────────────

def test_cyber_physical_correlation_temporal_and_asset():
    """Verify cyber events correlate with physical incidents on same asset within time window."""
    with get_test_db() as db:
        # Create a video first
        vid_id = "vid-corr-test-01"
        existing_vid = db.query(VideoModel).filter(VideoModel.id == vid_id).first()
        if not existing_vid:
            vid = VideoModel(
                id=vid_id,
                original_filename="dock_footage.mp4",
                duration_seconds=300.0,
                storage_path=f"storage/{vid_id}.mp4",
                status="processed",
            )
            db.add(vid)
            db.commit()

        # Create camera source linked to video
        cam_label = "CAM-CORR-01"
        existing_cam = db.query(CameraSourceModel).filter(CameraSourceModel.camera_label == cam_label).first()
        if not existing_cam:
            cam = CameraSourceModel(
                id="cam-source-corr-01",
                camera_label=cam_label,
                video_id=vid_id,
                position_hint="Loading Dock West",
                field_of_view_hint="Facing Bay 4",
                status="ACTIVE",
            )
            db.add(cam)
            db.commit()
        else:
            cam = existing_cam

        # Create a physical incident in that video at T=50s
        inc_id = "CORR-PHYS-TEST-01"
        existing_inc = db.query(CorrelatedIncidentModel).filter(CorrelatedIncidentModel.id == inc_id).first()
        if not existing_inc:
            phys_inc = CorrelatedIncidentModel(
                id=inc_id,
                video_id=vid_id,
                incident_category="unauthorized_access",
                start_time=48.0,
                end_time=55.0,
                duration=7.0,
                assessment_score=0.60,
                evidence_strength=0.58,
                reliability_rating="MEDIUM",
                validation_decision="REVIEW_REQUIRED",
                storyline="Person observed entering dock door outside operating hours",
            )
            db.add(phys_inc)
            db.commit()

        # Create a cyber event on CAM-CORR-01 at T=42s (within window of 50s)
        cyber_ev = create_cyber_event(
            db=db,
            asset_label=cam_label,
            event_type="CONFIGURATION_CHANGE",
            severity="HIGH",
            video_id=vid_id,
            video_offset_seconds=42.0,
            description="Door sensor override parameter enabled via console",
            provenance="REPLAYED_TELEMETRY",
        )

        # Run correlation with 60-second window
        correlations = correlate_cyber_physical(
            db=db,
            asset_label_or_id=cam_label,
            time_window_seconds=60.0,
        )

        assert len(correlations) >= 1
        match = next((c for c in correlations if c["cyber_event"]["id"] == cyber_ev.id), None)
        assert match is not None
        assert match["physical_incident"]["id"] == inc_id
        # Score must strictly respect review ceiling
        assert match["confidence_score"] <= 0.65
        assert match["human_verification_required"] is True


def test_cyber_physical_correlation_score_ceiling():
    """Verify correlation confidence never exceeds 0.65 (human review required)."""
    with get_test_db() as db:
        correlations = correlate_cyber_physical(db=db)
        for c in correlations:
            assert c["confidence_score"] <= 0.65
            assert c["human_verification_required"] is True


# ─────────────────────────────────────────────────────────────
# 4. INVESTIGATION AGENT & CONTROLLED TOOLS TESTS
# ─────────────────────────────────────────────────────────────

def test_agent_controlled_tools():
    """Verify InvestigationOrchestrator exposes controlled application tools."""
    orchestrator = InvestigationOrchestrator()
    # Tool 1: search_cyber_events
    events = orchestrator.tool_search_cyber_events(asset_label="CAM-NORTH-01")
    assert isinstance(events, list)

    # Tool 2: get_security_asset
    asset = orchestrator.tool_get_security_asset("CAM-NORTH-01")
    # Returns asset info if registered, or None
    if asset:
        assert "label" in asset

    # Tool 4: correlate_cyber_physical_events
    corrs = orchestrator.tool_correlate_cyber_physical_events(asset_label="CAM-NORTH-01")
    assert isinstance(corrs, dict)
    assert "correlations" in corrs


def test_agent_12_step_investigation_workflow():
    """Verify process_cyber_investigation executes the complete 12-step agentic audit trail."""
    orchestrator = InvestigationOrchestrator()
    result = orchestrator.process_cyber_investigation(
        user_query="Investigate security anomalies associated with CAM-NORTH-01 between 22:00 and 22:30",
        asset_label="CAM-NORTH-01",
    )

    assert result["status"] == "REVIEW_REQUIRED"
    assert result["security_asset"] == "CAM-NORTH-01"

    # Verify safe auditable action history (12 steps)
    action_history = result["actions_taken"]
    assert len(action_history) == 12
    steps = [a["step"] for a in action_history]
    assert 1 in steps
    assert 12 in steps

    # Verify findings structure
    assert "findings" in result
    assert isinstance(result["findings"], str)
    assert len(result["findings"]) > 0

    # Verify human review is strictly required
    assert result["human_verification_required"] is True
    assert result["assessment_score"] <= 0.65

    # Verify truthful provenance is explicitly stated
    prov = result["provenance"]
    assert "REPLAYED_TELEMETRY" in prov

    # Verify no private chain-of-thought leaked
    assert "chain_of_thought" not in result
    for a in action_history:
        assert "private_reasoning" not in a
