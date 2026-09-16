"""
backend/tests/test_phase19_case_management.py

Unit and integration tests for Phase 19 Forensic Case Management:
- Case creation, retrieval, listing, update, deletion
- Case lifecycle transitions (OPEN, INVESTIGATING, REVIEW, CLOSED)
- Case priority handling (LOW, MEDIUM, HIGH, CRITICAL)
- Video, camera, and incident linking/unlinking
- Input validation and error handling
"""
import uuid
import pytest
from datetime import datetime, timezone
from database.session import SessionLocal, init_db
from database.models import (
    CaseModel,
    VideoModel,
    CameraSourceModel,
    SurveillanceSessionModel,
    CorrelatedIncidentModel,
    SecurityEventModel,
)
from backend.app.services.case_service import (
    CaseService,
    CaseNotFoundError,
    CaseEntityNotFoundError,
    CaseValidationError,
)


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


@pytest.fixture
def dummy_video(db):
    vid = str(uuid.uuid4())
    v = VideoModel(
        id=vid,
        original_filename="cctv_entrance_test.mp4",
        storage_path=f"storage/{vid}.mp4",
        duration_seconds=60.0,
        fps=30.0,
        status="processed",
    )
    db.add(v)
    db.commit()
    db.refresh(v)
    return v


@pytest.fixture
def dummy_camera(db, dummy_video):
    sess_id = str(uuid.uuid4())
    sess = SurveillanceSessionModel(id=sess_id, name="Test Surveillance Site")
    db.add(sess)
    cam_id = str(uuid.uuid4())
    cam = CameraSourceModel(
        id=cam_id,
        session_id=sess_id,
        video_id=dummy_video.id,
        camera_label="Entrance Gate 01",
        position_hint="North perimeter",
    )
    db.add(cam)
    db.commit()
    db.refresh(cam)
    return cam


@pytest.fixture
def dummy_incident(db, dummy_video):
    inc_id = f"CORR-{uuid.uuid4().hex[:8]}"
    inc = CorrelatedIncidentModel(
        id=inc_id,
        video_id=dummy_video.id,
        incident_category="intrusion",
        incident_subcategory="perimeter_crossing",
        start_time=10.0,
        end_time=18.0,
        duration=8.0,
        assessment_score=0.65,
        evidence_strength=0.90,
        reliability_rating="MEDIUM",
        validation_decision="REVIEW_REQUIRED",
        storyline="Subject crossed perimeter fence and moved toward facility.",
    )
    db.add(inc)
    db.commit()
    db.refresh(inc)
    return inc


def test_create_and_get_case(db, case_svc):
    case = case_svc.create_case(
        db,
        title="Warehouse Perimeter Breach Investigation",
        description="Investigation into unauthorized gate access incident.",
        priority="HIGH",
        assigned_investigator="Senior Analyst Jane",
        tags=["Intrusion", "Perimeter", "Night Shift"],
        summary="Initial case created based on automated detection.",
    )
    assert case.id is not None
    assert case.case_number.startswith("CASE-")
    assert case.title == "Warehouse Perimeter Breach Investigation"
    assert case.status == "OPEN"
    assert case.priority == "HIGH"
    assert case.assigned_investigator == "Senior Analyst Jane"
    assert len(case.tags) == 3

    fetched = case_svc.get_case(db, case.id)
    assert fetched.id == case.id
    assert fetched.case_number == case.case_number


def test_case_validation_errors(db, case_svc):
    # Empty title should raise CaseValidationError
    with pytest.raises(CaseValidationError):
        case_svc.create_case(db, title="")

    # Invalid priority should raise CaseValidationError
    with pytest.raises(CaseValidationError):
        case_svc.create_case(db, title="Valid Title", priority="EXTREME")


def test_list_cases_filtering(db, case_svc):
    c1 = case_svc.create_case(db, title="Vehicle Theft Case", priority="CRITICAL", tags=["Vehicle", "Theft"])
    c2 = case_svc.create_case(db, title="Crowd Loitering Review", priority="LOW", tags=["Crowd"])

    all_cases = case_svc.list_cases(db)
    assert len(all_cases) >= 2

    crit_cases = case_svc.list_cases(db, priority="CRITICAL")
    assert any(c.id == c1.id for c in crit_cases)
    assert not any(c.id == c2.id for c in crit_cases)

    vehicle_cases = case_svc.list_cases(db, tag="Vehicle")
    assert any(c.id == c1.id for c in vehicle_cases)

    search_cases = case_svc.list_cases(db, search="Theft")
    assert any(c.id == c1.id for c in search_cases)


def test_case_lifecycle_transitions(db, case_svc):
    case = case_svc.create_case(db, title="Lifecycle Audit Case", priority="MEDIUM")
    assert case.status == "OPEN"

    case = case_svc.update_case(db, case.id, status="INVESTIGATING")
    assert case.status == "INVESTIGATING"

    case = case_svc.update_case(db, case.id, status="REVIEW")
    assert case.status == "REVIEW"

    case = case_svc.update_case(db, case.id, status="CLOSED")
    assert case.status == "CLOSED"

    # Invalid status should fail
    with pytest.raises(CaseValidationError):
        case_svc.update_case(db, case.id, status="ARCHIVED")


def test_case_deletion_cascade(db, case_svc, dummy_video):
    case = case_svc.create_case(db, title="Temporary Deletion Case", initial_video_ids=[dummy_video.id])
    case_id = case.id

    bm = case_svc.create_bookmark(db, case_id, dummy_video.id, 12.5, "Suspicious movement")
    note = case_svc.create_note(db, case_id, "Initial note on suspect vehicle.")

    case_svc.delete_case(db, case_id)

    with pytest.raises(CaseNotFoundError):
        case_svc.get_case(db, case_id)


def test_video_linking_and_unlinking(db, case_svc, dummy_video):
    case = case_svc.create_case(db, title="Video Linking Case")

    # Link video
    link = case_svc.link_video(db, case.id, dummy_video.id, notes="Primary gate camera recording")
    assert link.video_id == dummy_video.id

    linked = case_svc.get_linked_videos(db, case.id)
    assert len(linked) == 1
    assert linked[0]["video_id"] == dummy_video.id
    assert linked[0]["filename"] == dummy_video.original_filename

    # Unlink video
    case_svc.unlink_video(db, case.id, dummy_video.id)
    linked_after = case_svc.get_linked_videos(db, case.id)
    assert len(linked_after) == 0


def test_camera_linking_and_offsets(db, case_svc, dummy_camera):
    case = case_svc.create_case(db, title="Camera Linking Case")

    link = case_svc.link_camera(db, case.id, dummy_camera.id, clock_offset_seconds=-3.5, notes="Time is 3.5s slow")
    assert link.camera_id == dummy_camera.id
    assert link.clock_offset_seconds == -3.5

    cams = case_svc.get_linked_cameras(db, case.id)
    assert len(cams) == 1
    assert cams[0]["camera_id"] == dummy_camera.id
    assert cams[0]["camera_label"] == dummy_camera.camera_label
    assert cams[0]["clock_offset_seconds"] == -3.5

    case_svc.unlink_camera(db, case.id, dummy_camera.id)
    assert len(case_svc.get_linked_cameras(db, case.id)) == 0


def test_incident_linking(db, case_svc, dummy_incident):
    case = case_svc.create_case(db, title="Incident Linking Case")

    link = case_svc.link_incident(db, case.id, dummy_incident.id, incident_type="CORRELATED")
    assert link.incident_id == dummy_incident.id

    incidents = case_svc.get_case_incidents(db, case.id)
    assert len(incidents) >= 1
    assert any(i["incident_id"] == dummy_incident.id for i in incidents)

    case_svc.unlink_incident(db, case.id, dummy_incident.id)
