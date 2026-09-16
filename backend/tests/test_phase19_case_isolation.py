"""
backend/tests/test_phase19_case_isolation.py

Tests enforcing strict Case Isolation and boundary controls:
- Case A cannot access Case B's bookmarks, notes, annotations, or activities
- Attempted cross-case operations raise CaseIsolationError
- Video and camera boundary isolation: invalid IDs or cross-boundary mutations rejected
"""
import uuid
import pytest
from database.session import SessionLocal, init_db
from database.models import VideoModel
from backend.app.services.case_service import (
    CaseService,
    CaseIsolationError,
    CaseNotFoundError,
    CaseEntityNotFoundError,
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
def sample_video(db):
    vid = str(uuid.uuid4())
    v = VideoModel(
        id=vid,
        original_filename="isolation_audit_video.mp4",
        storage_path=f"storage/{vid}.mp4",
        duration_seconds=120.0,
        fps=25.0,
        status="processed",
    )
    db.add(v)
    db.commit()
    db.refresh(v)
    return v


def test_bookmark_cross_case_isolation(db, case_svc, sample_video):
    case_a = case_svc.create_case(db, title="Investigation Case Alpha")
    case_b = case_svc.create_case(db, title="Investigation Case Bravo")

    bm_a = case_svc.create_bookmark(db, case_a.id, sample_video.id, 15.0, "Suspicious person observed")

    # Updating Case A's bookmark under Case B's scope must be blocked
    with pytest.raises(CaseIsolationError):
        case_svc.update_bookmark(db, case_id=case_b.id, bookmark_id=bm_a.id, title="Tampered Title")

    # Deleting Case A's bookmark under Case B's scope must be blocked
    with pytest.raises(CaseIsolationError):
        case_svc.delete_bookmark(db, case_id=case_b.id, bookmark_id=bm_a.id)

    # Bookmark still exists and is untampered in Case A
    bms_a = case_svc.list_bookmarks(db, case_a.id)
    assert len(bms_a) == 1
    assert bms_a[0].title == "Suspicious person observed"

    # Case B has zero bookmarks
    bms_b = case_svc.list_bookmarks(db, case_b.id)
    assert len(bms_b) == 0


def test_note_cross_case_isolation(db, case_svc):
    case_a = case_svc.create_case(db, title="Note Isolation Alpha")
    case_b = case_svc.create_case(db, title="Note Isolation Bravo")

    note_a = case_svc.create_note(db, case_a.id, "Analyst confidential finding A.")

    # Updating note A under case B must be blocked
    with pytest.raises(CaseIsolationError):
        case_svc.update_note(db, case_id=case_b.id, note_id=note_a.id, content="Injected false finding.")

    # Deleting note A under case B must be blocked
    with pytest.raises(CaseIsolationError):
        case_svc.delete_note(db, case_id=case_b.id, note_id=note_a.id)

    # Case B notes list is empty
    notes_b = case_svc.list_notes(db, case_b.id)
    assert len(notes_b) == 0

    # Case A notes list has the original note
    notes_a = case_svc.list_notes(db, case_a.id)
    assert len(notes_a) == 1
    assert notes_a[0].content == "Analyst confidential finding A."


def test_annotation_cross_case_isolation(db, case_svc, sample_video):
    case_a = case_svc.create_case(db, title="Annotation Isolation Alpha")
    case_b = case_svc.create_case(db, title="Annotation Isolation Bravo")

    ann_a = case_svc.create_annotation(
        db,
        case_id=case_a.id,
        video_id=sample_video.id,
        timestamp_seconds=22.4,
        annotation_type="REGION",
        data={"bbox": [100, 100, 250, 300], "label": "Vehicle of interest"},
    )

    # Deleting Case A's annotation under Case B must fail
    with pytest.raises(CaseIsolationError):
        case_svc.delete_annotation(db, case_id=case_b.id, annotation_id=ann_a.id)

    # Annotation remains in Case A
    anns_a = case_svc.list_annotations(db, case_a.id)
    assert len(anns_a) == 1

    # Case B has no annotations
    anns_b = case_svc.list_annotations(db, case_b.id)
    assert len(anns_b) == 0


def test_invalid_entity_ids_rejection(db, case_svc):
    case = case_svc.create_case(db, title="Invalid ID Test Case")

    # Non-existent video linking must be rejected
    with pytest.raises(CaseEntityNotFoundError):
        case_svc.link_video(db, case.id, "nonexistent-video-id-999")

    # Non-existent camera linking must be rejected
    with pytest.raises(CaseEntityNotFoundError):
        case_svc.link_camera(db, case.id, "nonexistent-camera-id-999")

    # Non-existent incident linking must be rejected
    with pytest.raises(CaseEntityNotFoundError):
        case_svc.link_incident(db, case.id, "nonexistent-incident-id-999")
