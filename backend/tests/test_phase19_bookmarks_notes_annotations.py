"""
backend/tests/test_phase19_bookmarks_notes_annotations.py

Tests for Phase 19 Bookmarks, Notes, and Annotations:
- Bookmark pinning, timestamp accuracy, and update/deletion
- Investigator notes, entity associations, and strict ANALYST_NOTE classification
- Overlay annotations: non-mutation invariant (media on disk is never altered)
"""
import os
import uuid
import pytest
from database.session import SessionLocal, init_db
from database.models import VideoModel
from backend.app.services.case_service import CaseService, CaseValidationError


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
def test_video(db, tmp_path):
    vid = str(uuid.uuid4())
    # Create an actual physical dummy video file to test non-mutation
    video_file = tmp_path / f"{vid}.mp4"
    video_file.write_bytes(b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2mp41")
    initial_bytes = video_file.read_bytes()

    v = VideoModel(
        id=vid,
        original_filename="security_cam_hallway.mp4",
        storage_path=str(video_file),
        duration_seconds=90.0,
        fps=30.0,
        status="processed",
    )
    db.add(v)
    db.commit()
    db.refresh(v)
    return v, video_file, initial_bytes


def test_bookmarks_workflow(db, case_svc, test_video):
    video, _, _ = test_video
    case = case_svc.create_case(db, title="Bookmark Workflow Case")

    # Create bookmark
    bm = case_svc.create_bookmark(
        db,
        case_id=case.id,
        video_id=video.id,
        timestamp_seconds=42.75,
        title="Key exchange event observed",
        description="Two subjects paused at corridor intersection.",
        author="Officer Dave",
    )
    assert bm.id is not None
    assert bm.timestamp_seconds == 42.75
    assert bm.author == "Officer Dave"

    # List bookmarks
    bms = case_svc.list_bookmarks(db, case.id)
    assert len(bms) == 1
    assert bms[0].title == "Key exchange event observed"

    # Update bookmark
    updated = case_svc.update_bookmark(
        db,
        case_id=case.id,
        bookmark_id=bm.id,
        title="Package transfer observed",
        timestamp_seconds=43.0,
    )
    assert updated.title == "Package transfer observed"
    assert updated.timestamp_seconds == 43.0

    # Delete bookmark
    case_svc.delete_bookmark(db, case.id, bm.id)
    assert len(case_svc.list_bookmarks(db, case.id)) == 0


def test_investigator_notes_classification(db, case_svc):
    case = case_svc.create_case(db, title="Notes Audit Case")

    # Create case-level note
    n1 = case_svc.create_note(
        db,
        case_id=case.id,
        content="Initial review of access log suggests discrepancy at 14:00.",
        author="Analyst Sarah",
        associated_type="CASE",
    )
    assert n1.note_classification == "ANALYST_NOTE"
    assert n1.associated_type == "CASE"

    # Create incident-associated note
    n2 = case_svc.create_note(
        db,
        case_id=case.id,
        content="Subject entered through south delivery entrance without badge.",
        author="Analyst Sarah",
        associated_type="INCIDENT",
        associated_id="INC-9988",
        timestamp_seconds=125.4,
    )
    assert n2.note_classification == "ANALYST_NOTE"
    assert n2.associated_id == "INC-9988"
    assert n2.timestamp_seconds == 125.4

    # List notes
    all_notes = case_svc.list_notes(db, case.id)
    assert len(all_notes) == 2

    # Filter notes by associated_type
    inc_notes = case_svc.list_notes(db, case.id, associated_type="INCIDENT")
    assert len(inc_notes) == 1
    assert inc_notes[0].associated_id == "INC-9988"

    # Update note
    n2_updated = case_svc.update_note(db, case.id, n2.id, content="Updated: badge scan failed twice.")
    assert n2_updated.content == "Updated: badge scan failed twice."

    # Delete note
    case_svc.delete_note(db, case.id, n1.id)
    assert len(case_svc.list_notes(db, case.id)) == 1


def test_overlay_annotations_non_mutation(db, case_svc, test_video):
    video, video_file, initial_bytes = test_video
    case = case_svc.create_case(db, title="Overlay Non-Mutation Case")

    # Create region annotation
    ann = case_svc.create_annotation(
        db,
        case_id=case.id,
        video_id=video.id,
        timestamp_seconds=18.5,
        end_timestamp_seconds=24.0,
        annotation_type="REGION",
        data={
            "bbox": [50, 50, 200, 200],
            "color": "#10b981",
            "label": "Bag left on bench",
        },
        author="Forensic Examiner",
    )
    assert ann.id is not None
    assert ann.annotation_type == "REGION"
    assert ann.timestamp_seconds == 18.5
    assert ann.end_timestamp_seconds == 24.0

    # Verify original media file on disk was NOT modified at all!
    current_bytes = video_file.read_bytes()
    assert current_bytes == initial_bytes, "SAFETY VIOLATION: Media file was mutated by annotation!"

    # List annotations with time filter
    anns = case_svc.list_annotations(db, case.id, start_time=15.0, end_time=20.0)
    assert len(anns) == 1
    assert anns[0].data["label"] == "Bag left on bench"

    # Delete annotation
    case_svc.delete_annotation(db, case.id, ann.id)
    assert len(case_svc.list_annotations(db, case.id)) == 0

    # Ensure media is still untouched
    assert video_file.read_bytes() == initial_bytes
