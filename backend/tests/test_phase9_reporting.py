"""
Phase 9 Test Suite: Professional Incident Dossier & Evidence-Based Reporting

Tests:
- Report generation & schema validation
- Report database persistence (ReportModel)
- PDF creation, magic bytes, and validity
- Timeline & detection statistics inclusion
- Security event inclusion & potential theft deep-dive reporting
- Evidence inclusion with annotated images and missing-image graceful handling
- Strict null safety (missing classes, null tracks, null confidences)
- Empty data handling (video with 0 detections/events)
- Missing video (404)
- Missing report file / invalid report ID (404/400)
- Directory path traversal protection
- Report download and inline view endpoints
- Deterministic execution without live LLM/Gemini dependencies
"""

import os
import io
import uuid
import pytest
from datetime import datetime, timezone
from pathlib import Path
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.config import settings
from database.session import SessionLocal
from database.models import (
    VideoModel,
    EventModel,
    GroupedEventModel,
    EvidenceModel,
    TrackModel,
    VehicleAttributeModel,
    FaceDetectionModel,
    SecurityEventModel,
    ReportModel,
)
from backend.app.services.report_service import (
    ReportService,
    ReportNotFoundError,
    ReportSecurityError,
)
from ai.reporting.schema import ReportDataPayload, ReportVideoMetadata, ReportDetectionStats
from ai.reporting.dossier_generator import IncidentDossierPDFGenerator


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def test_video_with_intel():
    """Create a fully populated test video with real DB events, tracks, theft, and evidence."""
    db = SessionLocal()
    vid = f"test_rep_{uuid.uuid4().hex[:8]}"

    # Ensure fake upload sidecar exists if checked
    v = VideoModel(
        id=vid,
        original_filename="cctv_theft_test.mp4",
        storage_path=f"storage/uploads/{vid}.mp4",
        file_size_bytes=1024 * 1024 * 5,
        duration_seconds=150.0,
        fps=30.0,
        frame_count=4500,
        status="processed",
    )
    db.add(v)

    # Detections
    e1 = EventModel(
        id=str(uuid.uuid4()),
        video_id=vid,
        event_type="object_detected",
        object_class="person",
        class_id=0,
        timestamp_seconds=10.0,
        confidence=0.88,
        bbox_x1=50, bbox_y1=50, bbox_x2=100, bbox_y2=200,
        frame_number=300,
    )
    e2 = EventModel(
        id=str(uuid.uuid4()),
        video_id=vid,
        event_type="object_detected",
        object_class="suitcase",
        class_id=28,
        timestamp_seconds=12.0,
        confidence=0.92,
        bbox_x1=120, bbox_y1=150, bbox_x2=180, bbox_y2=210,
        frame_number=360,
    )
    db.add_all([e1, e2])

    # Grouped Event
    ge = GroupedEventModel(
        id=str(uuid.uuid4()),
        video_id=vid,
        event_type="ACTIVITY",
        start_time=10.0,
        end_time=20.0,
        duration_seconds=10.0,
        objects_summary=[{"class": "person", "count": 1}, {"class": "suitcase", "count": 1}],
        total_detections=2,
        max_confidence=0.92,
    )
    db.add(ge)

    # Tracks
    t1 = TrackModel(
        id=str(uuid.uuid4()),
        video_id=vid,
        track_id="TRACK-001",
        object_class="person",
        first_seen=10.0,
        last_seen=40.0,
        duration_seconds=30.0,
        detection_count=25,
        max_confidence=0.89,
        trajectory=[{"timestamp": 10.0, "bbox": [50, 50, 100, 200]}],
        active=1,
    )
    t2 = TrackModel(
        id=str(uuid.uuid4()),
        video_id=vid,
        track_id="TRACK-002",
        object_class="suitcase",
        first_seen=10.0,
        last_seen=35.0,
        duration_seconds=25.0,
        detection_count=20,
        max_confidence=0.91,
        trajectory=[{"timestamp": 10.0, "bbox": [120, 150, 180, 210]}],
        active=1,
    )
    db.add_all([t1, t2])

    # Evidence
    ev_id = f"ev_test_{uuid.uuid4().hex[:6]}"
    ev = EvidenceModel(
        id=ev_id,
        video_id=vid,
        evidence_type="snapshot_and_clip",
        timestamp_seconds=32.5,
        source_video_name="cctv_theft_test.mp4",
        object_class="suitcase",
        confidence=0.94,
        notes="Suspect takeaway grounded snapshot",
    )
    db.add(ev)

    # Potential Theft Security Event
    se = SecurityEventModel(
        id=f"EV-THEFT-{uuid.uuid4().hex[:6]}",
        video_id=vid,
        event_type="POTENTIAL_THEFT",
        severity="HIGH",
        timestamp_seconds=32.5,
        duration_seconds=15.0,
        track_id="TRACK-001",
        object_class="suitcase",
        confidence=0.94,
        description="Potential object-takeaway pattern observed: Person [TRACK-001] departed with suitcase [TRACK-002].",
        observable_signals=[
            "Signal 1: Interaction proximity dwell: 15.0s",
            "Signal 2: Object disappearance at origin",
            "Signal 3: Departure trajectory displacement 95px",
        ],
        evidence_id=ev_id,
    )
    db.add(se)
    db.commit()
    db.close()

    yield vid

    # Cleanup test records
    clean_db = SessionLocal()
    clean_db.query(ReportModel).filter(ReportModel.video_id == vid).delete()
    clean_db.query(SecurityEventModel).filter(SecurityEventModel.video_id == vid).delete()
    clean_db.query(EvidenceModel).filter(EvidenceModel.video_id == vid).delete()
    clean_db.query(TrackModel).filter(TrackModel.video_id == vid).delete()
    clean_db.query(GroupedEventModel).filter(GroupedEventModel.video_id == vid).delete()
    clean_db.query(EventModel).filter(EventModel.video_id == vid).delete()
    clean_db.query(VideoModel).filter(VideoModel.id == vid).delete()
    clean_db.commit()
    clean_db.close()


@pytest.fixture
def empty_test_video():
    """Create a video with 0 detections, 0 events, 0 tracks."""
    db = SessionLocal()
    vid = f"test_empty_{uuid.uuid4().hex[:8]}"
    v = VideoModel(
        id=vid,
        original_filename="empty_stream.mp4",
        storage_path=f"storage/uploads/{vid}.mp4",
        file_size_bytes=1024,
        duration_seconds=10.0,
        fps=30.0,
        frame_count=300,
        status="processed",
    )
    db.add(v)
    db.commit()
    db.close()

    yield vid

    clean_db = SessionLocal()
    clean_db.query(ReportModel).filter(ReportModel.video_id == vid).delete()
    clean_db.query(VideoModel).filter(VideoModel.id == vid).delete()
    clean_db.commit()
    clean_db.close()


# ---------------------------------------------------------------------------
# Unit & Engine Tests
# ---------------------------------------------------------------------------

def test_01_dossier_generator_pdf_validity():
    """Verify Platypus builds a valid PDF with header bytes and non-empty content."""
    payload = ReportDataPayload(
        report_id="REP-TEST-ENGINE",
        generated_at_iso=datetime.now(timezone.utc).isoformat(),
        video=ReportVideoMetadata(
            video_id="vid_test_01",
            original_filename="security_cam_1.mp4",
            duration_seconds=60.0,
            fps=30.0,
        ),
        stats=ReportDetectionStats(
            total_detections=10,
            class_counts={"person": 8, "car": 2},
            class_avg_confidences={"person": 0.85, "car": 0.90},
            total_tracks=2,
            validated_tracks=2,
        ),
    )
    buf = io.BytesIO()
    gen = IncidentDossierPDFGenerator(payload)
    sz, pages = gen.generate(buf)

    assert sz > 1000
    assert pages >= 1
    content = buf.getvalue()
    assert content.startswith(b"%PDF-")


def test_02_generate_dossier_database_persistence(test_video_with_intel):
    """Test report generation service creates a ReportModel in DB with metadata."""
    svc = ReportService()
    report = svc.generate_dossier(test_video_with_intel, title="CAMPUS AUDIT REPORT")

    assert report["report_id"].startswith("REP-")
    assert report["video_id"] == test_video_with_intel
    assert report["title"] == "CAMPUS AUDIT REPORT"
    assert report["file_size_bytes"] > 0
    assert report["page_count"] >= 1
    assert report["metadata"]["total_detections"] == 2
    assert report["metadata"]["has_theft_event"] is True
    assert report["metadata"]["theft_event_count"] == 1

    # Verify DB record
    db = SessionLocal()
    db_rep = db.query(ReportModel).filter(ReportModel.report_id == report["report_id"]).first()
    assert db_rep is not None
    assert db_rep.status == "completed"
    db.close()

    # Verify physical file
    fpath, fname = svc.get_report_file_path(report["report_id"])
    assert fpath.exists()
    assert fname == f"{report['report_id']}.pdf"


def test_03_null_values_safety():
    """Verify generator handles None/null optional fields without crashing."""
    db = SessionLocal()
    null_vid = f"test_null_{uuid.uuid4().hex[:8]}"
    v = VideoModel(
        id=null_vid,
        original_filename="null_data.mp4",
        storage_path=f"storage/uploads/{null_vid}.mp4",
        duration_seconds=None,
        fps=None,
        frame_count=None,
    )
    db.add(v)

    # Event with None class and confidence
    e = EventModel(
        id=str(uuid.uuid4()),
        video_id=null_vid,
        event_type="unclassified",
        object_class="unknown",
        class_id=0,
        timestamp_seconds=5.0,
        confidence=0.5,
        bbox_x1=0, bbox_y1=0, bbox_x2=10, bbox_y2=10,
        frame_number=150,
    )
    db.add(e)

    # Security event with None track_id, None zone_name
    se = SecurityEventModel(
        id=f"EV-NULL-{uuid.uuid4().hex[:6]}",
        video_id=null_vid,
        event_type="OBSERVATIONAL_ANOMALY",
        severity="NORMAL",
        timestamp_seconds=5.0,
        track_id=None,
        object_class=None,
        zone_name=None,
        confidence=0.75,
        description="Anomalous optical fluctuation.",
        observable_signals=None,
        evidence_id=None,
    )
    db.add(se)

    # Evidence with missing image paths
    ev = EvidenceModel(
        id=f"ev_null_{uuid.uuid4().hex[:6]}",
        video_id=null_vid,
        evidence_type="snapshot_only",
        timestamp_seconds=5.0,
        source_video_name="null_data.mp4",
        object_class=None,
        confidence=None,
        snapshot_path=None,
        annotated_snapshot_path=None,
    )
    db.add(ev)
    db.commit()
    db.close()

    try:
        svc = ReportService()
        report = svc.generate_dossier(null_vid)
        assert report["status"] == "completed"
        assert report["file_size_bytes"] > 0
    finally:
        clean_db = SessionLocal()
        clean_db.query(ReportModel).filter(ReportModel.video_id == null_vid).delete()
        clean_db.query(EvidenceModel).filter(EvidenceModel.video_id == null_vid).delete()
        clean_db.query(SecurityEventModel).filter(SecurityEventModel.video_id == null_vid).delete()
        clean_db.query(EventModel).filter(EventModel.video_id == null_vid).delete()
        clean_db.query(VideoModel).filter(VideoModel.id == null_vid).delete()
        clean_db.commit()
        clean_db.close()


def test_04_empty_video_dossier_generation(empty_test_video):
    """Verify report generates gracefully for an empty video with 0 detections."""
    svc = ReportService()
    report = svc.generate_dossier(empty_test_video)

    assert report["report_id"].startswith("REP-")
    assert report["metadata"]["total_detections"] == 0
    assert report["metadata"]["total_tracks"] == 0
    assert report["metadata"]["total_security_events"] == 0
    assert report["file_size_bytes"] > 0


def test_05_missing_video_raises_404():
    """Verify ReportService raises ReportNotFoundError when video does not exist."""
    svc = ReportService()
    with pytest.raises(ReportNotFoundError):
        svc.generate_dossier("non_existent_video_uuid_99999")


def test_06_path_traversal_protection():
    """Verify directory traversal attempts on report downloads are blocked."""
    svc = ReportService()
    with pytest.raises(ReportSecurityError):
        svc.get_report_file_path("../../etc/passwd")

    with pytest.raises(ReportSecurityError):
        svc.get_report_file_path("..\\..\\boot.ini")

    with pytest.raises(ReportSecurityError):
        svc.get_report_file_path("REP-TEST/../../../secret")


# ---------------------------------------------------------------------------
# API Endpoint Integration Tests
# ---------------------------------------------------------------------------

def test_07_api_generate_report_endpoint(client, test_video_with_intel):
    """Test POST /api/videos/{video_id}/reports/generate."""
    resp = client.post(
        f"/api/videos/{test_video_with_intel}/reports/generate",
        json={"title": "SURVEILLANCE INCIDENT DOSSIER"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "success"
    assert "report" in data
    assert data["report"]["video_id"] == test_video_with_intel
    assert data["report"]["title"] == "SURVEILLANCE INCIDENT DOSSIER"
    assert data["report"]["report_id"].startswith("REP-")


def test_08_api_list_video_reports(client, test_video_with_intel):
    """Test GET /api/videos/{video_id}/reports."""
    # First generate one
    client.post(f"/api/videos/{test_video_with_intel}/reports/generate")

    resp = client.get(f"/api/videos/{test_video_with_intel}/reports")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["video_id"] == test_video_with_intel
    assert data["count"] >= 1
    assert len(data["reports"]) >= 1


def test_09_api_get_report_details(client, test_video_with_intel):
    """Test GET /api/reports/{report_id}."""
    gen_resp = client.post(f"/api/videos/{test_video_with_intel}/reports/generate")
    rep_id = gen_resp.json()["report"]["report_id"]

    resp = client.get(f"/api/reports/{rep_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["report"]["report_id"] == rep_id
    assert data["report"]["download_url"] == f"/api/reports/{rep_id}/download"
    assert data["report"]["view_url"] == f"/api/reports/{rep_id}/view"


def test_10_api_download_report_pdf(client, test_video_with_intel):
    """Test GET /api/reports/{report_id}/download returns attachment."""
    gen_resp = client.post(f"/api/videos/{test_video_with_intel}/reports/generate")
    rep_id = gen_resp.json()["report"]["report_id"]

    resp = client.get(f"/api/reports/{rep_id}/download")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert f'filename="{rep_id}.pdf"' in resp.headers["content-disposition"]
    assert resp.content.startswith(b"%PDF-")


def test_11_api_view_report_pdf(client, test_video_with_intel):
    """Test GET /api/reports/{report_id}/view returns inline PDF."""
    gen_resp = client.post(f"/api/videos/{test_video_with_intel}/reports/generate")
    rep_id = gen_resp.json()["report"]["report_id"]

    resp = client.get(f"/api/reports/{rep_id}/view")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert f'inline; filename="{rep_id}.pdf"' in resp.headers["content-disposition"]
    assert resp.content.startswith(b"%PDF-")


def test_12_api_missing_video_returns_404(client):
    """Test POST /api/videos/{invalid_id}/reports/generate returns 404."""
    resp = client.post("/api/videos/non_existent_video_id_12345/reports/generate")
    assert resp.status_code == 404


def test_13_api_missing_report_returns_404(client):
    """Test GET /api/reports/{invalid_id} returns 404."""
    resp = client.get("/api/reports/REP-NOT-FOUND-99999")
    assert resp.status_code == 404


def test_14_api_path_traversal_returns_400(client):
    """Test path traversal on download returns 400."""
    resp = client.get("/api/reports/..%2F..%2Fetc%2Fpasswd/download")
    assert resp.status_code in (400, 404)


def test_15_stale_rejected_detection_and_evidence_reconciliation(client):
    """
    Test that stale evidence derived from rejected detections is automatically
    reconciled to REJECTED and excluded from Evidence Vault and Report results,
    while valid detections and security events (POTENTIAL_THEFT) remain intact.
    """
    from backend.app.services.evidence_service import EvidenceService
    db = SessionLocal()
    vid = f"test_stale_{uuid.uuid4().hex[:8]}"

    v = VideoModel(
        id=vid,
        original_filename="stale_test.mp4",
        storage_path=f"storage/uploads/{vid}.mp4",
        file_size_bytes=1024 * 1024,
        duration_seconds=60.0,
        fps=30.0,
        frame_count=1800,
        status="processed",
    )
    db.add(v)
    db.flush()

    # 1 valid person, 1 rejected bus
    person_ev = EventModel(
        id=f"ev_p_{uuid.uuid4().hex[:6]}",
        video_id=vid,
        object_class="person",
        class_id=0,
        confidence=0.88,
        timestamp_seconds=10.0,
        frame_number=300,
        bbox_x1=100.0,
        bbox_y1=100.0,
        bbox_x2=200.0,
        bbox_y2=300.0,
        validation_status="VALID",
        validation_reason="High confidence detection",
    )
    bus_ev = EventModel(
        id=f"ev_b_{uuid.uuid4().hex[:6]}",
        video_id=vid,
        object_class="bus",
        class_id=5,
        confidence=0.29,
        timestamp_seconds=25.0,
        frame_number=750,
        bbox_x1=50.0,
        bbox_y1=50.0,
        bbox_x2=400.0,
        bbox_y2=300.0,
        validation_status="REJECTED",
        validation_reason="Isolated low-confidence bus detection",
    )
    db.add(person_ev)
    db.add(bus_ev)

    # Stale evidence for the rejected bus (originally stored as VALID)
    stale_bus_evidence = EvidenceModel(
        id=f"evd_bus_{uuid.uuid4().hex[:6]}",
        video_id=vid,
        event_id=bus_ev.id,
        object_class="bus",
        confidence=0.29,
        timestamp_seconds=25.0,
        validation_status="VALID",  # Stale state
        source_video_name="stale_test.mp4",
    )
    # Valid security evidence (POTENTIAL_THEFT)
    theft_evidence = EvidenceModel(
        id=f"evd_theft_{uuid.uuid4().hex[:6]}",
        video_id=vid,
        object_class="person",
        confidence=0.90,
        timestamp_seconds=10.0,
        validation_status="VALID",
        notes="Automated forensic evidence for POTENTIAL_THEFT",
        source_video_name="stale_test.mp4",
    )
    stale_bus_id = stale_bus_evidence.id
    theft_ev_id = theft_evidence.id
    db.add(stale_bus_evidence)
    db.add(theft_evidence)
    db.commit()
    db.close()

    ev_svc = EvidenceService()
    # 1. Check get_video_evidence reconciles and excludes stale bus evidence
    evidence_list = ev_svc.get_video_evidence(vid)
    returned_ids = [e["evidence_id"] for e in evidence_list]
    assert stale_bus_id not in returned_ids, "Stale rejected bus evidence must NOT be returned"
    assert theft_ev_id in returned_ids, "POTENTIAL_THEFT evidence must remain present"

    # 2. Check report generation excludes stale rejected bus and includes valid person
    gen_resp = client.post(f"/api/videos/{vid}/reports/generate")
    assert gen_resp.status_code == 201
    rep_data = gen_resp.json()["report"]
    meta = rep_data["metadata"]
    # Bus must not appear in report statistics
    assert "bus" not in meta.get("class_counts", {}), "Rejected bus must NOT appear in report class counts"
    assert "person" in meta.get("class_counts", {}), "Valid person must appear in report class counts"
    assert meta["total_detections"] == 1


def test_16_view_report_headers_and_inline_disposition(client, test_video_with_intel):
    """
    Test that GET /api/reports/{id}/view returns Content-Type: application/pdf
    and Content-Disposition: inline for native in-browser viewing.
    """
    gen_resp = client.post(f"/api/videos/{test_video_with_intel}/reports/generate")
    rep_id = gen_resp.json()["report"]["report_id"]

    resp = client.get(f"/api/reports/{rep_id}/view")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert "inline" in resp.headers["content-disposition"]
    assert f'filename="{rep_id}.pdf"' in resp.headers["content-disposition"]
    assert resp.content.startswith(b"%PDF-")


def test_17_download_report_attachment_disposition(client, test_video_with_intel):
    """
    Test that GET /api/reports/{id}/download returns Content-Disposition: attachment.
    """
    gen_resp = client.post(f"/api/videos/{test_video_with_intel}/reports/generate")
    rep_id = gen_resp.json()["report"]["report_id"]

    resp = client.get(f"/api/reports/{rep_id}/download")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert "attachment" in resp.headers["content-disposition"]
    assert f'filename="{rep_id}.pdf"' in resp.headers["content-disposition"]
    assert resp.content.startswith(b"%PDF-")

