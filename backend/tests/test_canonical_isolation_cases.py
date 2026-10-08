"""
Sentinel Generic Regression Coverage: Canonical Entity Isolation Across Scenarios A-E
======================================================================================
Generic regression coverage validating:
CASE A — SAME VIDEO DUPLICATE TRACKLETS (Reconciles to 1 canonical entity)
CASE B — SAME VIDEO DISTINCT PEOPLE (Remains 2 distinct canonical entities)
CASE C — DIFFERENT VIDEOS, SAME TRACK ID (Absolutely separate, never merged)
CASE D — DIFFERENT VIDEOS, SIMILAR BBOX/COLOR (Identical coordinates/color -> absolutely separate)
CASE E — DIFFERENT VIDEOS, ADJACENT PROCESSING (Sequential queries produce zero leakage)
"""

import uuid
import pytest
from database.session import SessionLocal
from database.models import VideoModel, TrackModel, EventModel
from backend.app.services.investigation_service import InvestigationService


def test_case_a_same_video_duplicate_tracklets():
    """CASE A: Same physical person in same video with strong spatial/temporal continuity reconciles to 1 entity."""
    svc = InvestigationService()
    vid = f"vid_case_a_{uuid.uuid4().hex[:6]}"
    
    t1 = {
        "video_id": vid,
        "track_id": "TRACK-001",
        "object_class": "person",
        "first_seen": 0.0,
        "last_seen": 20.0,
        "duration_seconds": 20.0,
        "detection_count": 20,
        "max_confidence": 0.92,
        "color": "blue",
        "color_confidence": 0.85,
        "current_bbox": {"x1": 200, "y1": 100, "x2": 300, "y2": 300},
        "trajectory": [(10.0, 250, 200), (11.0, 252, 201)],
    }
    t2 = {
        "video_id": vid,
        "track_id": "TRACK-005",
        "object_class": "person",
        "first_seen": 9.5,
        "last_seen": 12.0,
        "duration_seconds": 2.5,
        "detection_count": 3,
        "max_confidence": 0.70,
        "color": None,
        "color_confidence": None,
        "current_bbox": {"x1": 202, "y1": 102, "x2": 298, "y2": 298},
        "trajectory": [(10.0, 251, 200), (11.0, 253, 202)],
    }

    canon = svc._reconcile_canonical_entities([t1, t2], video_id=vid)
    assert len(canon) == 1, f"Expected 1 canonical entity, got {len(canon)}"
    assert canon[0]["canonical_id"] == "TRACK-001"
    assert "TRACK-005" in canon[0]["member_track_ids"]
    assert canon[0]["color"] == "blue"


def test_case_b_same_video_distinct_people():
    """CASE B: Two actual people in the same video remain 2 separate canonical entities."""
    svc = InvestigationService()
    vid = f"vid_case_b_{uuid.uuid4().hex[:6]}"

    t1 = {
        "video_id": vid,
        "track_id": "TRACK-001",
        "object_class": "person",
        "first_seen": 0.0,
        "last_seen": 20.0,
        "duration_seconds": 20.0,
        "detection_count": 20,
        "max_confidence": 0.90,
        "color": "orange",
        "color_confidence": 0.80,
        "current_bbox": {"x1": 100, "y1": 100, "x2": 150, "y2": 250},
        "trajectory": [(10.0, 125, 175)],
    }
    t2 = {
        "video_id": vid,
        "track_id": "TRACK-002",
        "object_class": "person",
        "first_seen": 0.0,
        "last_seen": 20.0,
        "duration_seconds": 20.0,
        "detection_count": 21,
        "max_confidence": 0.93,
        "color": "blue",
        "color_confidence": 0.85,
        "current_bbox": {"x1": 700, "y1": 100, "x2": 750, "y2": 250},
        "trajectory": [(10.0, 725, 175)],
    }

    canon = svc._reconcile_canonical_entities([t1, t2], video_id=vid)
    assert len(canon) == 2, f"Expected 2 canonical entities, got {len(canon)}"
    ids = {c["canonical_id"] for c in canon}
    assert ids == {"TRACK-001", "TRACK-002"}


def test_case_c_different_videos_same_track_id():
    """CASE C: TRACK-001 in Video A and TRACK-001 in Video B are absolutely separate."""
    svc = InvestigationService()
    vid_a = f"vid_a_{uuid.uuid4().hex[:6]}"
    vid_b = f"vid_b_{uuid.uuid4().hex[:6]}"

    t_a = {
        "video_id": vid_a,
        "track_id": "TRACK-001",
        "object_class": "person",
        "first_seen": 0.0,
        "last_seen": 10.0,
        "duration_seconds": 10.0,
        "detection_count": 10,
        "max_confidence": 0.90,
        "color": "blue",
        "color_confidence": 0.80,
    }
    t_b = {
        "video_id": vid_b,
        "track_id": "TRACK-001",
        "object_class": "person",
        "first_seen": 0.0,
        "last_seen": 10.0,
        "duration_seconds": 10.0,
        "detection_count": 10,
        "max_confidence": 0.90,
        "color": "blue",
        "color_confidence": 0.80,
    }

    # Hard invariant: tracks from different videos can NEVER be same entity
    assert not svc._are_tracks_same_entity(t_a, t_b)

    # Reconciling together without video_id filter still keeps them strictly disjoint
    canon_both = svc._reconcile_canonical_entities([t_a, t_b])
    assert len(canon_both) == 2, f"Expected 2 canonical entities, got {len(canon_both)}"
    for c in canon_both:
        assert len(c["member_track_ids"]) == 1

    # Scoped to Video A only returns Video A
    canon_a = svc._reconcile_canonical_entities([t_a, t_b], video_id=vid_a)
    assert len(canon_a) == 1
    assert canon_a[0]["video_id"] == vid_a


def test_case_d_different_videos_similar_bbox_color():
    """CASE D: Tracks in different videos with identical bbox, color, and timestamps remain absolutely separate."""
    svc = InvestigationService()
    vid_a = f"vid_a_{uuid.uuid4().hex[:6]}"
    vid_b = f"vid_b_{uuid.uuid4().hex[:6]}"

    identical_bbox = {"x1": 300, "y1": 150, "x2": 400, "y2": 350}
    t_a = {
        "video_id": vid_a,
        "track_id": "TRACK-001",
        "object_class": "person",
        "first_seen": 5.0,
        "last_seen": 15.0,
        "duration_seconds": 10.0,
        "detection_count": 15,
        "max_confidence": 0.95,
        "color": "red",
        "color_confidence": 0.90,
        "current_bbox": identical_bbox,
        "trajectory": [(10.0, 350, 250)],
    }
    t_b = {
        "video_id": vid_b,
        "track_id": "TRACK-002",
        "object_class": "person",
        "first_seen": 5.0,
        "last_seen": 15.0,
        "duration_seconds": 10.0,
        "detection_count": 15,
        "max_confidence": 0.95,
        "color": "red",
        "color_confidence": 0.90,
        "current_bbox": identical_bbox,
        "trajectory": [(10.0, 350, 250)],
    }

    assert not svc._are_tracks_same_entity(t_a, t_b)
    canon = svc._reconcile_canonical_entities([t_a, t_b])
    assert len(canon) == 2
    assert canon[0]["video_id"] != canon[1]["video_id"]


def test_case_e_different_videos_adjacent_processing():
    """CASE E: Processing Video A, then Video B, then Video A produces zero canonical mapping or state leakage."""
    db = SessionLocal()
    vid_a = f"vid_adj_a_{uuid.uuid4().hex[:8]}"
    vid_b = f"vid_adj_b_{uuid.uuid4().hex[:8]}"
    svc = InvestigationService()

    try:
        v_a = VideoModel(id=vid_a, original_filename="store_a.mp4", storage_path=f"storage/{vid_a}.mp4", status="processed")
        v_b = VideoModel(id=vid_b, original_filename="store_b.mp4", storage_path=f"storage/{vid_b}.mp4", status="processed")
        db.add_all([v_a, v_b])

        # Video A has 1 blue person track and 2 detections
        ea1 = EventModel(id=str(uuid.uuid4()), video_id=vid_a, object_class="person", class_id=0, timestamp_seconds=5.0, confidence=0.9, bbox_x1=10, bbox_y1=10, bbox_x2=20, bbox_y2=20, frame_number=150, validation_status="VALID")
        ea2 = EventModel(id=str(uuid.uuid4()), video_id=vid_a, object_class="person", class_id=0, timestamp_seconds=6.0, confidence=0.9, bbox_x1=10, bbox_y1=10, bbox_x2=20, bbox_y2=20, frame_number=180, validation_status="VALID")
        ta = TrackModel(id=str(uuid.uuid4()), video_id=vid_a, track_id="TRACK-001", object_class="person", first_seen=5.0, last_seen=6.0, duration_seconds=1.0, detection_count=2, max_confidence=0.9, color="blue", color_confidence=0.85)

        # Video B has 1 yellow person track and 1 detection
        eb1 = EventModel(id=str(uuid.uuid4()), video_id=vid_b, object_class="person", class_id=0, timestamp_seconds=12.0, confidence=0.88, bbox_x1=50, bbox_y1=50, bbox_x2=60, bbox_y2=60, frame_number=360, validation_status="VALID")
        tb = TrackModel(id=str(uuid.uuid4()), video_id=vid_b, track_id="TRACK-001", object_class="person", first_seen=12.0, last_seen=12.0, duration_seconds=0.0, detection_count=1, max_confidence=0.88, color="yellow", color_confidence=0.80)

        db.add_all([ea1, ea2, ta, eb1, tb])
        db.commit()

        # Step 1: Query Video A
        res_a1_people = svc.investigate(vid_a, "how many people")
        assert res_a1_people["count"] == 1
        res_a1_det = svc.investigate(vid_a, "How many person detections were recorded?")
        assert res_a1_det["count"] == 2
        res_a1_blue = svc.investigate(vid_a, "how many people are wearing blue")
        assert res_a1_blue["count"] == 1

        # Step 2: Query Video B (must not see Video A's blue person or count)
        res_b_people = svc.investigate(vid_b, "how many people")
        assert res_b_people["count"] == 1
        res_b_det = svc.investigate(vid_b, "How many person detections were recorded?")
        assert res_b_det["count"] == 1
        res_b_blue = svc.investigate(vid_b, "how many people are wearing blue")
        assert res_b_blue["count"] == 0

        # Step 3: Query Video A again (must remain identical, zero cross-contamination from Video B)
        res_a2_people = svc.investigate(vid_a, "how many people")
        assert res_a2_people["count"] == 1
        assert res_a2_people["canonical_entities"][0]["video_id"] == vid_a
        assert res_a2_people["canonical_entities"][0]["color"] == "blue"
        res_a2_yellow = svc.investigate(vid_a, "how many people in yellow")
        assert res_a2_yellow["count"] == 0

    finally:
        db.query(TrackModel).filter(TrackModel.video_id.in_([vid_a, vid_b])).delete(synchronize_session=False)
        db.query(EventModel).filter(EventModel.video_id.in_([vid_a, vid_b])).delete(synchronize_session=False)
        db.query(VideoModel).filter(VideoModel.id.in_([vid_a, vid_b])).delete(synchronize_session=False)
        db.commit()
        db.close()
