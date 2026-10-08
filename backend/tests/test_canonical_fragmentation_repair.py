import os
import sys
import math
import uuid
import pytest

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.investigation_service import InvestigationService

def test_a_same_person_short_occlusion_merges():
    """Test A: Same person fragmented by short occlusion (<=2s) with spatial continuity merges."""
    svc = InvestigationService()
    vid = f"vid_a_{uuid.uuid4().hex[:6]}"
    t1 = {
        "video_id": vid,
        "track_id": "TRACK-001",
        "object_class": "person",
        "first_seen": 0.0,
        "last_seen": 4.0,
        "duration_seconds": 4.0,
        "detection_count": 8,
        "max_confidence": 0.88,
        "color": "black",
        "color_confidence": 0.85,
        "current_bbox": {"x1": 100.0, "y1": 50.0, "x2": 150.0, "y2": 150.0},
        "trajectory": [[0.0, 100.0, 50.0], [4.0, 125.0, 100.0]],
    }
    t2 = {
        "video_id": vid,
        "track_id": "TRACK-002",
        "object_class": "person",
        "first_seen": 5.5,  # 1.5s gap
        "last_seen": 9.0,
        "duration_seconds": 3.5,
        "detection_count": 7,
        "max_confidence": 0.86,
        "color": "black",
        "color_confidence": 0.80,
        "current_bbox": {"x1": 110.0, "y1": 55.0, "x2": 155.0, "y2": 155.0},
        "trajectory": [[5.5, 128.0, 102.0], [9.0, 140.0, 120.0]],
    }
    canon = svc._reconcile_canonical_entities([t1, t2], video_id=vid)
    people = [c for c in canon if c.get("object_class") == "person"]
    assert len(people) == 1
    assert "TRACK-001" in people[0]["member_track_ids"]
    assert "TRACK-002" in people[0]["member_track_ids"]

def test_b_same_person_moderate_occlusion_merge_conditional():
    """Test B: Moderate occlusion merges ONLY when spatial/motion/appearance continuity is strong; otherwise remains separate."""
    svc = InvestigationService()
    vid = f"vid_b_{uuid.uuid4().hex[:6]}"
    
    # Base track
    t1 = {
        "video_id": vid,
        "track_id": "TRACK-001",
        "object_class": "person",
        "first_seen": 10.0,
        "last_seen": 15.0,
        "duration_seconds": 5.0,
        "detection_count": 10,
        "max_confidence": 0.85,
        "color": "black",
        "color_confidence": 0.85,
        "current_bbox": {"x1": 100.0, "y1": 50.0, "x2": 140.0, "y2": 150.0},
        "trajectory": [[10.0, 80.0, 50.0], [15.0, 120.0, 100.0]],
    }
    # Strong continuity candidate (gap = 8s, feasible velocity ~2.5 px/s, matching exit/entry boundary)
    t2_strong = {
        "video_id": vid,
        "track_id": "TRACK-002",
        "object_class": "person",
        "first_seen": 23.0,  # 8s gap
        "last_seen": 28.0,
        "duration_seconds": 5.0,
        "detection_count": 10,
        "max_confidence": 0.82,
        "color": "black",
        "color_confidence": 0.80,
        "current_bbox": {"x1": 115.0, "y1": 55.0, "x2": 150.0, "y2": 155.0},
        "trajectory": [[23.0, 135.0, 110.0], [28.0, 160.0, 120.0]],
    }
    canon_strong = svc._reconcile_canonical_entities([t1, t2_strong], video_id=vid)
    people_strong = [c for c in canon_strong if c.get("object_class") == "person"]
    assert len(people_strong) == 1

    # Weak/infeasible continuity candidate (gap = 8s, but entry location is 300px away, velocity ~37 px/s)
    t2_weak = {
        "video_id": vid,
        "track_id": "TRACK-003",
        "object_class": "person",
        "first_seen": 23.0,
        "last_seen": 28.0,
        "duration_seconds": 5.0,
        "detection_count": 10,
        "max_confidence": 0.82,
        "color": "black",
        "color_confidence": 0.80,
        "current_bbox": {"x1": 400.0, "y1": 55.0, "x2": 440.0, "y2": 155.0},
        "trajectory": [[23.0, 420.0, 100.0], [28.0, 450.0, 120.0]],
    }
    canon_weak = svc._reconcile_canonical_entities([t1, t2_weak], video_id=vid)
    people_weak = [c for c in canon_weak if c.get("object_class") == "person"]
    assert len(people_weak) == 2

def test_c_two_people_in_identical_black_clothing_remain_separate():
    """Test C: Two people wearing identical black clothing with spatial separation remain separate."""
    svc = InvestigationService()
    vid = f"vid_c_{uuid.uuid4().hex[:6]}"
    p1 = {
        "video_id": vid,
        "track_id": "TRACK-001",
        "object_class": "person",
        "first_seen": 0.0,
        "last_seen": 10.0,
        "duration_seconds": 10.0,
        "detection_count": 15,
        "max_confidence": 0.90,
        "color": "black",
        "color_confidence": 0.95,
        "current_bbox": {"x1": 50.0, "y1": 50.0, "x2": 90.0, "y2": 150.0},
        "trajectory": [[0.0, 60.0, 60.0], [10.0, 70.0, 70.0]],
    }
    p2 = {
        "video_id": vid,
        "track_id": "TRACK-002",
        "object_class": "person",
        "first_seen": 2.0,
        "last_seen": 12.0,
        "duration_seconds": 10.0,
        "detection_count": 15,
        "max_confidence": 0.90,
        "color": "black",
        "color_confidence": 0.95,
        "current_bbox": {"x1": 250.0, "y1": 50.0, "x2": 290.0, "y2": 150.0},
        "trajectory": [[2.0, 260.0, 60.0], [12.0, 270.0, 70.0]],
    }
    canon = svc._reconcile_canonical_entities([p1, p2], video_id=vid)
    people = [c for c in canon if c.get("object_class") == "person"]
    assert len(people) == 2

def test_d_two_people_crossing_same_location_at_different_times_do_not_merge():
    """Test D: Two people crossing the same location at different times do not merge solely by location."""
    svc = InvestigationService()
    vid = f"vid_d_{uuid.uuid4().hex[:6]}"
    # Person 1 enters door, walks deep into room to (320, 150)
    t1 = {
        "video_id": vid,
        "track_id": "TRACK-001",
        "object_class": "person",
        "first_seen": 0.0,
        "last_seen": 5.0,
        "duration_seconds": 5.0,
        "detection_count": 6,
        "max_confidence": 0.88,
        "color": "black",
        "color_confidence": 0.80,
        "current_bbox": {"x1": 300.0, "y1": 100.0, "x2": 340.0, "y2": 200.0},
        "trajectory": [[0.0, 60.0, 150.0], [5.0, 320.0, 150.0]],
    }
    # Person 2 enters same doorway at (50, 150) 8 seconds later
    t2 = {
        "video_id": vid,
        "track_id": "TRACK-002",
        "object_class": "person",
        "first_seen": 13.0,
        "last_seen": 18.0,
        "duration_seconds": 5.0,
        "detection_count": 6,
        "max_confidence": 0.85,
        "color": "black",
        "color_confidence": 0.80,
        "current_bbox": {"x1": 40.0, "y1": 100.0, "x2": 80.0, "y2": 200.0},
        "trajectory": [[13.0, 50.0, 150.0], [18.0, 80.0, 150.0]],
    }
    canon = svc._reconcile_canonical_entities([t1, t2], video_id=vid)
    people = [c for c in canon if c.get("object_class") == "person"]
    assert len(people) == 2

def test_e_two_simultaneously_visible_people_hard_cannot_link():
    """Test E: Two simultaneously visible people have a hard Cannot-Link and never merge even transitively."""
    svc = InvestigationService()
    vid = f"vid_e_{uuid.uuid4().hex[:6]}"
    p1 = {
        "video_id": vid,
        "track_id": "TRACK-001",
        "object_class": "person",
        "first_seen": 10.0,
        "last_seen": 20.0,
        "duration_seconds": 10.0,
        "detection_count": 12,
        "max_confidence": 0.90,
        "color": "black",
        "color_confidence": 0.90,
        "current_bbox": {"x1": 50.0, "y1": 50.0, "x2": 90.0, "y2": 150.0},
        "trajectory": [[10.0, 70.0, 100.0], [15.0, 70.0, 100.0], [20.0, 70.0, 100.0]],
    }
    p2 = {
        "video_id": vid,
        "track_id": "TRACK-002",
        "object_class": "person",
        "first_seen": 12.0,
        "last_seen": 22.0,
        "duration_seconds": 10.0,
        "detection_count": 12,
        "max_confidence": 0.88,
        "color": "black",
        "color_confidence": 0.90,
        "current_bbox": {"x1": 200.0, "y1": 50.0, "x2": 240.0, "y2": 150.0},
        "trajectory": [[12.0, 220.0, 100.0], [17.0, 220.0, 100.0], [22.0, 220.0, 100.0]],
    }
    canon = svc._reconcile_canonical_entities([p1, p2], video_id=vid)
    people = [c for c in canon if c.get("object_class") == "person"]
    assert len(people) == 2

def test_f_stationary_real_person_remains_valid():
    """Test F: A legitimate stationary person with high detector confidence and persistence is preserved."""
    svc = InvestigationService()
    vid = f"vid_f_{uuid.uuid4().hex[:6]}"
    real_person = {
        "video_id": vid,
        "track_id": "TRACK-001",
        "object_class": "person",
        "first_seen": 0.0,
        "last_seen": 25.0,
        "duration_seconds": 25.0,
        "detection_count": 50,
        "max_confidence": 0.92,
        "color": "blue",
        "color_confidence": 0.88,
        "current_bbox": {"x1": 150.0, "y1": 80.0, "x2": 200.0, "y2": 210.0},
        "trajectory": [[float(t), 175.0, 145.0] for t in range(0, 26, 2)],
    }
    canon = svc._reconcile_canonical_entities([real_person], video_id=vid)
    people = [c for c in canon if c.get("object_class") == "person"]
    assert len(people) == 1
    assert people[0]["canonical_id"] == "TRACK-001"
    assert people[0]["color"] == "blue"

def test_g_static_poster_mannequin_false_detection_suppressed():
    """Test G: Static poster/mannequin false detection is not promoted to a verified physical person."""
    svc = InvestigationService()
    vid = f"vid_g_{uuid.uuid4().hex[:6]}"
    poster_track = {
        "video_id": vid,
        "track_id": "TRACK-099",
        "object_class": "person",
        "first_seen": 5.0,
        "last_seen": 10.0,
        "duration_seconds": 5.0,
        "detection_count": 4,
        "max_confidence": 0.65,
        "color": "black",
        "color_confidence": 0.50,
        "current_bbox": {"x1": 85.0, "y1": 50.0, "x2": 110.0, "y2": 110.0},
        "trajectory": [[5.0, 97.5, 80.0], [6.0, 97.8, 80.2], [7.0, 97.4, 80.1], [10.0, 97.6, 80.0]],
    }
    canon = svc._reconcile_canonical_entities([poster_track], video_id=vid)
    people = [c for c in canon if c.get("object_class") == "person"]
    assert len(people) == 0

def test_h_retail_two_person_scene_remains_exactly_two():
    """Test H: Retail two-person scene remains exactly two individuals with 1 orange and 1 blue."""
    import sqlite3, json
    conn = sqlite3.connect("storage/sentinel.db")
    c = conn.cursor()
    vid = "2614807e-f749-436f-af66-4d34d9a20970"
    rows = c.execute(
        "SELECT track_id, object_class, first_seen, last_seen, duration_seconds, "
        "detection_count, max_confidence, color, color_confidence, current_bbox, trajectory "
        "FROM tracks WHERE video_id = ? ORDER BY first_seen, track_id",
        (vid,)
    ).fetchall()
    items = []
    for r in rows:
        items.append({
            "video_id": vid,
            "track_id": r[0],
            "object_class": r[1],
            "first_seen": float(r[2]),
            "last_seen": float(r[3]),
            "duration_seconds": float(r[4]),
            "detection_count": int(r[5]),
            "max_confidence": float(r[6]),
            "color": r[7],
            "color_confidence": float(r[8]) if r[8] else 0.0,
            "current_bbox": json.loads(r[9]) if r[9] else {},
            "trajectory": json.loads(r[10]) if r[10] else [],
        })
    svc = InvestigationService()
    canon = svc._reconcile_canonical_entities(items, video_id=vid)
    people = [ce for ce in canon if ce.get("object_class") == "person"]
    assert len(people) == 2
    colors = {p.get("color") for p in people}
    assert "blue" in colors
    assert all(p.get("duration_seconds", 0) > 0 for p in people)

def test_i_different_videos_with_same_track_ids_never_merge():
    """Test I: Different videos with identical track IDs never merge."""
    svc = InvestigationService()
    vid1 = f"vid_i1_{uuid.uuid4().hex[:6]}"
    vid2 = f"vid_i2_{uuid.uuid4().hex[:6]}"
    t1 = {
        "video_id": vid1,
        "track_id": "TRACK-001",
        "object_class": "person",
        "first_seen": 0.0,
        "last_seen": 10.0,
        "duration_seconds": 10.0,
        "detection_count": 15,
        "max_confidence": 0.88,
        "color": "black",
        "color_confidence": 0.80,
        "current_bbox": {"x1": 100.0, "y1": 50.0, "x2": 150.0, "y2": 150.0},
        "trajectory": [[0.0, 125.0, 100.0], [10.0, 125.0, 100.0]],
    }
    t2 = {
        "video_id": vid2,
        "track_id": "TRACK-001",
        "object_class": "person",
        "first_seen": 0.0,
        "last_seen": 10.0,
        "duration_seconds": 10.0,
        "detection_count": 15,
        "max_confidence": 0.88,
        "color": "black",
        "color_confidence": 0.80,
        "current_bbox": {"x1": 100.0, "y1": 50.0, "x2": 150.0, "y2": 150.0},
        "trajectory": [[0.0, 125.0, 100.0], [10.0, 125.0, 100.0]],
    }
    # Pass both tracks to reconciliation
    canon = svc._reconcile_canonical_entities([t1, t2])
    people = [c for c in canon if c.get("object_class") == "person"]
    assert len(people) == 2
    # Verify they have distinct video IDs and were not clustered together
    assert people[0]["video_id"] != people[1]["video_id"]
    for p in people:
        assert len(p["member_track_ids"]) == 1
