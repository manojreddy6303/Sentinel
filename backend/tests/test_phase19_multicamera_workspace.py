"""
backend/tests/test_phase19_multicamera_workspace.py

Tests for Phase 19 Multi-Camera Case Workspace & Camera Topology:
- Visual camera topology graph construction (nodes, adjacency edges, associations)
- Synchronized multi-camera playback offsets
- Provenance and session isolation across cameras
"""
import uuid
import pytest
from database.session import SessionLocal, init_db
from database.models import (
    VideoModel,
    SurveillanceSessionModel,
    CameraSourceModel,
    CrossCameraAssociationModel,
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


def test_camera_topology_graph(db, case_svc):
    sess_id = str(uuid.uuid4())
    sess = SurveillanceSessionModel(id=sess_id, name="Campus Site Alpha")
    db.add(sess)

    # 3 Cameras: Cam A -> Cam B -> Cam C
    v_a_id = str(uuid.uuid4())
    v_a = VideoModel(id=v_a_id, original_filename="cam_a.mp4", duration_seconds=60.0, storage_path=f"storage/{v_a_id}.mp4", status="processed")
    db.add(v_a)

    v_b_id = str(uuid.uuid4())
    v_b = VideoModel(id=v_b_id, original_filename="cam_b.mp4", duration_seconds=60.0, storage_path=f"storage/{v_b_id}.mp4", status="processed")
    db.add(v_b)

    v_c_id = str(uuid.uuid4())
    v_c = VideoModel(id=v_c_id, original_filename="cam_c.mp4", duration_seconds=60.0, storage_path=f"storage/{v_c_id}.mp4", status="processed")
    db.add(v_c)

    cam_a = CameraSourceModel(
        id=str(uuid.uuid4()),
        session_id=sess_id,
        video_id=v_a_id,
        camera_label="Main Gate",
        position_hint="North Entrance",
        adjacency_hints=["Lobby Entrance"],
    )
    cam_b = CameraSourceModel(
        id=str(uuid.uuid4()),
        session_id=sess_id,
        video_id=v_b_id,
        camera_label="Lobby Entrance",
        position_hint="Building Foyer",
        adjacency_hints=["Main Gate", "Courtyard"],
    )
    cam_c = CameraSourceModel(
        id=str(uuid.uuid4()),
        session_id=sess_id,
        video_id=v_c_id,
        camera_label="Courtyard",
        position_hint="South Yard",
        adjacency_hints=["Lobby Entrance"],
    )
    db.add_all([cam_a, cam_b, cam_c])

    # Add cross-camera association between Cam A and Cam B (CONFIRMED)
    assoc = CrossCameraAssociationModel(
        id=str(uuid.uuid4()),
        session_id=sess_id,
        source_camera_id=cam_a.id,
        source_video_id=v_a_id,
        source_track_id="TRACK-001",
        target_camera_id=cam_b.id,
        target_video_id=v_b_id,
        target_track_id="TRACK-005",
        confidence=0.88,
        association_type="SAME_OBJECT",
        analyst_verdict="CONFIRMED",
        temporal_gap_seconds=4.2,
    )
    db.add(assoc)
    db.commit()

    case = case_svc.create_case(db, title="Topology Case")
    case_svc.link_camera(db, case.id, cam_a.id, clock_offset_seconds=0.0)
    case_svc.link_camera(db, case.id, cam_b.id, clock_offset_seconds=2.0)
    case_svc.link_camera(db, case.id, cam_c.id, clock_offset_seconds=-1.0)

    topology = case_svc.get_case_topology(db, case.id)
    assert topology["total_cameras"] == 3
    nodes = topology["nodes"]
    assert len(nodes) == 3
    labels = {n["label"] for n in nodes}
    assert "Main Gate" in labels
    assert "Lobby Entrance" in labels
    assert "Courtyard" in labels

    edges = topology["edges"]
    assert len(edges) >= 2
    # Check that confirmed association edge exists
    confirmed_edge = next((e for e in edges if e.get("type") == "CONFIRMED_ASSOCIATION"), None)
    assert confirmed_edge is not None
    assert confirmed_edge["verdict"] == "CONFIRMED"
    assert confirmed_edge["confidence"] == 0.88
