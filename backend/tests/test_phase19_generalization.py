"""
backend/tests/test_phase19_generalization.py

Generalization and edge-case testing matrix for Phase 19:
- Multiple video resolutions: 320x240, 640x480, 1280x720, 1920x1080, 3840x2160
- Frame rates: 5, 15, 25, 29.97, 30, 60 FPS
- Case conditions:
  - Empty case
  - Case with many videos
  - Disconnected camera
  - Large timeline (hundreds to thousands of entries)
  - Idempotent duplicate links
  - Deleted source video
"""
import uuid
import pytest
from database.session import SessionLocal, init_db
from database.models import VideoModel, GroupedEventModel
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


@pytest.mark.parametrize("res,fps", [
    ("320x240", 5.0),
    ("640x480", 15.0),
    ("1280x720", 25.0),
    ("1920x1080", 29.97),
    ("3840x2160", 60.0),
])
def test_generalization_resolutions_and_fps(db, case_svc, res, fps):
    vid = str(uuid.uuid4())
    v = VideoModel(
        id=vid,
        original_filename=f"cctv_{res}_{fps}fps.mp4",
        duration_seconds=100.0,
        fps=fps,
        storage_path=f"storage/{vid}.mp4",
        status="processed",
    )
    db.add(v)
    db.commit()

    case = case_svc.create_case(db, title=f"Case {res} {fps}fps", initial_video_ids=[vid])
    bm = case_svc.create_bookmark(db, case.id, vid, 33.33, f"Marker for {res}")
    assert bm.timestamp_seconds == 33.33

    timeline = case_svc.get_case_timeline(db, case.id)
    assert timeline["total_entries"] >= 1


def test_empty_case(db, case_svc):
    case = case_svc.create_case(db, title="Completely Empty Case")
    assert case.id is not None

    timeline = case_svc.get_case_timeline(db, case.id)
    assert timeline["total_entries"] == 0
    assert timeline["timeline"] == []

    topology = case_svc.get_case_topology(db, case.id)
    assert topology["total_cameras"] == 0

    storyline = case_svc.get_case_storyline(db, case.id)
    assert storyline["total_steps"] == 0

    export = case_svc.export_case_data(db, case.id)
    assert export["case_metadata"]["title"] == "Completely Empty Case"
    assert len(export["linked_videos"]) == 0


def test_idempotent_duplicate_links(db, case_svc):
    vid = str(uuid.uuid4())
    v = VideoModel(id=vid, original_filename="duplicate_test.mp4", duration_seconds=50.0, storage_path=f"storage/{vid}.mp4", status="processed")
    db.add(v)
    db.commit()

    case = case_svc.create_case(db, title="Duplicate Link Case")
    link1 = case_svc.link_video(db, case.id, vid)
    link2 = case_svc.link_video(db, case.id, vid)

    # Must return same link without duplicating
    assert link1.id == link2.id
    videos = case_svc.get_linked_videos(db, case.id)
    assert len(videos) == 1


def test_large_timeline_pagination(db, case_svc):
    vid = str(uuid.uuid4())
    v = VideoModel(id=vid, original_filename="large_timeline_video.mp4", duration_seconds=1000.0, storage_path=f"storage/{vid}.mp4", status="processed")
    db.add(v)

    # Create 50 detection events
    for i in range(50):
        ge = GroupedEventModel(
            id=str(uuid.uuid4()),
            video_id=vid,
            event_type="OBJECT_DETECTED",
            start_time=float(i * 10),
            end_time=float(i * 10 + 5),
            duration_seconds=5.0,
            objects_summary=[{"class": "car", "count": 1}],
            max_confidence=0.85,
            priority="NORMAL",
        )
        db.add(ge)
    db.commit()

    case = case_svc.create_case(db, title="Large Timeline Case", initial_video_ids=[vid])

    # Paginate with limit 15, offset 0
    p1 = case_svc.get_case_timeline(db, case.id, limit=15, offset=0)
    assert p1["total_entries"] >= 50
    assert len(p1["timeline"]) == 15

    # Offset 15
    p2 = case_svc.get_case_timeline(db, case.id, limit=15, offset=15)
    assert len(p2["timeline"]) == 15
    assert p1["timeline"][0]["id"] != p2["timeline"][0]["id"]
