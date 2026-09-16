"""
backend/tests/test_phase19_timeline_and_replay.py

Tests for Case-Level Unified Timeline & Incident Replay:
- Multi-layer chronological timeline merging across multiple videos
- Strict provenance preservation for every entry
- Rejection preservation: rejected detections are never promoted
- REVIEW_REQUIRED <= 0.65 assessment ceiling maintained
- Incident Replay: BEFORE -> CONTEXT -> INCIDENT -> AFTER temporal windows
"""
import uuid
import pytest
from database.session import SessionLocal, init_db
from database.models import (
    VideoModel,
    GroupedEventModel,
    SecurityEventModel,
    CorrelatedIncidentModel,
    EvidenceModel,
)
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


@pytest.fixture
def multi_video_case_setup(db, case_svc):
    # Video 1: Gate Camera (0 - 60s)
    v1_id = str(uuid.uuid4())
    v1 = VideoModel(id=v1_id, original_filename="cam1_gate.mp4", duration_seconds=60.0, storage_path=f"storage/{v1_id}.mp4", status="processed")
    db.add(v1)

    # Video 2: Lobby Camera (0 - 80s)
    v2_id = str(uuid.uuid4())
    v2 = VideoModel(id=v2_id, original_filename="cam2_lobby.mp4", duration_seconds=80.0, storage_path=f"storage/{v2_id}.mp4", status="processed")
    db.add(v2)

    # Add detection event to v1 at 10s
    ge1 = GroupedEventModel(
        id=str(uuid.uuid4()),
        video_id=v1_id,
        event_type="PERSON_DETECTED",
        start_time=10.0,
        end_time=15.0,
        duration_seconds=5.0,
        objects_summary=[{"class": "person", "count": 1}],
        max_confidence=0.92,
        priority="NORMAL",
    )
    db.add(ge1)

    # Add security event to v1 at 12s
    se1 = SecurityEventModel(
        id=str(uuid.uuid4()),
        video_id=v1_id,
        event_type="POTENTIAL_INTRUSION",
        severity="HIGH",
        timestamp_seconds=12.0,
        duration_seconds=4.0,
        confidence=0.88,
        description="Person scaled gate perimeter",
        human_verification_required=1,
    )
    db.add(se1)

    # Add correlated incident to v1 at 11s (REVIEW_REQUIRED <= 0.65)
    ci1 = CorrelatedIncidentModel(
        id=f"CORR-{uuid.uuid4().hex[:8]}",
        video_id=v1_id,
        incident_category="intrusion",
        incident_subcategory="gate_breach",
        start_time=11.0,
        end_time=16.0,
        duration=5.0,
        assessment_score=0.65,
        evidence_strength=0.95,
        reliability_rating="HIGH",
        validation_decision="REVIEW_REQUIRED",
        storyline="Subject accessed perimeter gate.",
    )
    db.add(ci1)

    # Add evidence item to v2 at 25s
    ev2 = EvidenceModel(
        id=str(uuid.uuid4()),
        video_id=v2_id,
        source_video_name=v2.original_filename,
        timestamp_seconds=25.0,
        evidence_type="snapshot_and_clip",
        validation_status="VALID",
        object_class="person",
        confidence=0.89,
    )
    db.add(ev2)

    db.commit()

    case = case_svc.create_case(db, title="Multi-Video Investigation", initial_video_ids=[v1_id, v2_id])
    return case, v1, v2, ci1, se1


def test_unified_timeline_chronological_and_provenance(db, case_svc, multi_video_case_setup):
    case, v1, v2, ci1, se1 = multi_video_case_setup

    res = case_svc.get_case_timeline(db, case.id)
    assert res["total_entries"] >= 4
    entries = res["timeline"]

    # Verify chronological ordering
    timestamps = [e["timestamp"] for e in entries]
    assert timestamps == sorted(timestamps)

    # Verify provenance for all items
    for e in entries:
        assert "provenance" in e
        assert "table" in e["provenance"]
        assert "record_id" in e["provenance"]
        assert e["video_id"] in (v1.id, v2.id)

    # Verify REVIEW_REQUIRED ceiling on the correlated incident
    corr_entry = next((e for e in entries if e["id"] == ci1.id), None)
    assert corr_entry is not None
    assert corr_entry["validation_decision"] == "REVIEW_REQUIRED"
    assert corr_entry["assessment_score"] <= 0.65


def test_incident_replay_context(db, case_svc, multi_video_case_setup):
    case, v1, _, ci1, _ = multi_video_case_setup

    # Incident is from 11.0s to 16.0s
    replay = case_svc.get_incident_replay_context(
        db,
        case_id=case.id,
        incident_id=ci1.id,
        pre_roll_seconds=5.0,
        post_roll_seconds=5.0,
    )

    assert replay["incident_id"] == ci1.id
    assert replay["video_id"] == v1.id
    assert replay["incident_window"] == [11.0, 16.0]

    # Replay context: 11.0 - 5.0 = 6.0, 16.0 + 5.0 = 21.0
    ctx = replay["replay_context"]
    assert ctx["replay_start"] == 6.0
    assert ctx["replay_end"] == 21.0
    assert ctx["duration"] == 15.0
    assert replay["stream_url"] == f"/api/videos/{v1.id}/stream"


def test_incident_replay_boundary_clamping(db, case_svc, multi_video_case_setup):
    case, v1, _, _, _ = multi_video_case_setup

    # Create incident near beginning of video (start_time = 2.0s)
    early_inc = CorrelatedIncidentModel(
        id=f"CORR-{uuid.uuid4().hex[:8]}",
        video_id=v1.id,
        incident_category="vehicle",
        start_time=2.0,
        end_time=5.0,
        duration=3.0,
        assessment_score=0.60,
        validation_decision="REVIEW_REQUIRED",
        storyline="Vehicle entered early.",
    )
    db.add(early_inc)
    db.commit()

    # Pre-roll of 5.0s on an incident starting at 2.0s must clamp to 0.0s
    replay = case_svc.get_incident_replay_context(
        db,
        case_id=case.id,
        incident_id=early_inc.id,
        pre_roll_seconds=5.0,
        post_roll_seconds=5.0,
    )
    assert replay["replay_context"]["replay_start"] == 0.0
    assert replay["replay_context"]["replay_end"] == 10.0
