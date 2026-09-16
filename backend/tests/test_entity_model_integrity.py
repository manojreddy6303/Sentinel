"""
backend/tests/test_entity_model_integrity.py

Verifies the Canonical Sentinel Entity Model:
1. Camera registration idempotency (registering existing label updates & does not duplicate).
2. Videos can exist without cameras, or link explicitly to cameras.
3. Multi-camera session reuses physical cameras without duplicating CameraSourceModel rows.
4. Cases start with 0 linked entities and require explicit linking.
5. Case evidence linking & unlinking is explicit and does not destroy the evidence record.
"""
import pytest
from database.session import SessionLocal
from database.models import (
    CameraSourceModel,
    VideoModel,
    CaseModel,
    CorrelatedIncidentModel,
    EvidenceModel,
    CaseEvidenceModel,
    SurveillanceSessionModel,
)
from backend.app.services.case_service import CaseService
from backend.app.services.session_service import SurveillanceSessionService


@pytest.fixture
def db():
    from database.session import init_db
    init_db()
    session = SessionLocal()
    yield session
    session.close()


def test_camera_registration_idempotency(db):
    from backend.app.api.cameras import register_camera, RegisterCameraRequest

    req1 = RegisterCameraRequest(
        camera_label="CAM-TEST-NORTH",
        location="North Gate",
        coverage_description="Facing Road",
    )
    res1 = register_camera(req1)
    cam1_id = res1["id"]

    # Register again with same label (case-insensitive)
    req2 = RegisterCameraRequest(
        camera_label="cam-test-north",
        location="North Gate Updated",
        coverage_description="Facing Highway",
    )
    res2 = register_camera(req2)
    cam2_id = res2["id"]

    assert cam1_id == cam2_id
    assert res2["location"] == "North Gate Updated"

    # Count in DB
    cams = db.query(CameraSourceModel).filter(CameraSourceModel.camera_label == "CAM-TEST-NORTH").all()
    assert len(cams) == 1


def test_session_creation_reuses_existing_cameras(db):
    svc = SurveillanceSessionService()
    # Create or ensure camera exists
    existing = db.query(CameraSourceModel).filter(CameraSourceModel.camera_label == "CAM-REUSE-01").first()
    if not existing:
        import uuid
        existing = CameraSourceModel(
            id=str(uuid.uuid4()),
            camera_label="CAM-REUSE-01",
            position_hint="Perimeter East",
            status="ACTIVE",
        )
        db.add(existing)
        db.commit()

    existing_id = existing.id

    import uuid
    dummy_vid = VideoModel(
        id=str(uuid.uuid4()),
        original_filename="reuse_vid.mp4",
        storage_path="/storage/reuse_vid.mp4",
        status="PROCESSED",
    )
    db.add(dummy_vid)
    db.commit()

    session = svc.create_session(db, name="Test Reuse Session", description="Testing camera reuse")
    cam_link = svc.add_camera(
        db,
        session_id=session.id,
        video_id=dummy_vid.id,
        camera_label="CAM-REUSE-01",
        position_hint="Perimeter East",
    )

    assert cam_link.id == existing_id
    # Ensure no second camera was created with that label
    count = db.query(CameraSourceModel).filter(CameraSourceModel.camera_label == "CAM-REUSE-01").count()
    assert count == 1


def test_case_lifecycle_and_explicit_linking(db):
    case_svc = CaseService()
    case = case_svc.create_case(db, title="Integrity Audit Test Case")
    assert case.id is not None

    # Verify initial counts are 0
    assert len(case.videos) == 0
    assert len(case.cameras) == 0
    assert len(case.incidents) == 0
    assert len(case.evidence_links) == 0

    # Create dummy video & dummy evidence
    import uuid
    dummy_vid = VideoModel(
        id=str(uuid.uuid4()),
        original_filename="integrity_vid.mp4",
        storage_path="/storage/integrity_vid.mp4",
        status="PROCESSED",
    )
    db.add(dummy_vid)

    dummy_ev = EvidenceModel(
        id=str(uuid.uuid4()),
        video_id=dummy_vid.id,
        source_video_name="integrity_vid.mp4",
        timestamp_seconds=12.5,
        evidence_type="snapshot_and_clip",
        object_class="person",
        validation_status="VALID",
    )
    db.add(dummy_ev)
    db.commit()

    # Link evidence
    link = case_svc.link_evidence(db, case.id, dummy_ev.id, notes="Primary suspect frame")
    assert link.evidence_id == dummy_ev.id

    linked = case_svc.get_linked_evidence(db, case.id)
    assert len(linked) == 1
    assert linked[0]["evidence_id"] == dummy_ev.id
    assert linked[0]["object_class"] == "person"

    # Unlink evidence
    case_svc.unlink_evidence(db, case.id, dummy_ev.id)
    linked_after = case_svc.get_linked_evidence(db, case.id)
    assert len(linked_after) == 0

    # Ensure evidence entity itself was NOT deleted
    still_exists = db.query(EvidenceModel).filter(EvidenceModel.id == dummy_ev.id).first()
    assert still_exists is not None
