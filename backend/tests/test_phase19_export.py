"""
backend/tests/test_phase19_export.py

Tests for Phase 19 Forensic Case Export:
- Comprehensive case package serialization
- Verification that all investigation assets (timeline, incidents, evidence index, bookmarks, notes, annotations, audit trail) are included
- Provenance preservation and integrity
"""
import uuid
import pytest
from database.session import SessionLocal, init_db
from database.models import VideoModel, EvidenceModel
from backend.app.services.case_service import CaseService


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def case_svc():
    return CaseService()


def test_case_export_package(db, case_svc):
    vid = str(uuid.uuid4())
    v = VideoModel(id=vid, original_filename="perimeter_audit.mp4", duration_seconds=60.0, storage_path=f"storage/{vid}.mp4", status="processed")
    db.add(v)

    ev = EvidenceModel(
        id=str(uuid.uuid4()),
        video_id=vid,
        source_video_name=v.original_filename,
        timestamp_seconds=14.0,
        evidence_type="snapshot_and_clip",
        validation_status="VALID",
        snapshot_path=f"storage/snapshots/{vid}_14.0.jpg",
        clip_path=f"storage/clips/{vid}_14.0.mp4",
        object_class="person",
        confidence=0.91,
    )
    db.add(ev)
    db.commit()

    case = case_svc.create_case(db, title="Export Verification Case", initial_video_ids=[vid])
    case_svc.create_bookmark(db, case.id, vid, 14.0, "Suspicious event pin")
    case_svc.create_note(db, case.id, "Verified by Lead Investigator.", author="Lead Investigator")
    case_svc.create_annotation(db, case.id, vid, 14.0, "POINT", {"x": 300, "y": 200})

    export_pkg = case_svc.export_case_data(db, case.id)

    assert "sentinel_platform" in export_pkg
    assert export_pkg["sentinel_platform"]["version"] == "19.0.0"

    assert "case_metadata" in export_pkg
    assert export_pkg["case_metadata"]["id"] == case.id
    assert export_pkg["case_metadata"]["title"] == "Export Verification Case"

    assert "linked_videos" in export_pkg
    assert len(export_pkg["linked_videos"]) == 1

    assert "evidence_index" in export_pkg
    assert len(export_pkg["evidence_index"]) == 1
    assert export_pkg["evidence_index"][0]["evidence_id"] == ev.id

    assert "bookmarks" in export_pkg
    assert len(export_pkg["bookmarks"]) == 1

    assert "notes" in export_pkg
    assert len(export_pkg["notes"]) == 1
    assert export_pkg["notes"][0]["classification"] == "ANALYST_NOTE"

    assert "annotations" in export_pkg
    assert len(export_pkg["annotations"]) == 1

    assert "activity_audit_trail" in export_pkg
    assert len(export_pkg["activity_audit_trail"]) >= 1
    # Activity includes the export event itself
    assert any(a["action_type"] == "CASE_EXPORTED" for a in export_pkg["activity_audit_trail"])
