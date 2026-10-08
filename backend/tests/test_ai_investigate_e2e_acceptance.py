"""
End-to-End API Regression Tests for Sentinel AI Investigation Endpoint
(POST /api/videos/{video_id}/ai-investigate)

Verifies:
1. Retail video canonical person count (2), blue clothing count (1), orange clothing count (1),
   and clothing color breakdown summary.
2. Burglary video low-light color accuracy: blue count = 0, black count = 2 (canonical physical actors),
   and clothing color summary returns clothing attributes, NOT theft summary.
3. Multi-query sequential isolation across video transitions.
"""

import pytest
import uuid
from fastapi.testclient import TestClient
from backend.app.main import app
from database.session import SessionLocal
from database.models import VideoModel, TrackModel, EventModel, GroupedEventModel, EvidenceModel

client = TestClient(app)


@pytest.fixture(scope="module")
def setup_acceptance_fixtures():
    db = SessionLocal()
    vid_retail = f"test_retail_{uuid.uuid4().hex[:8]}"
    vid_burglary = f"test_burglary_{uuid.uuid4().hex[:8]}"

    try:
        # Create Video records
        db.add(VideoModel(
            id=vid_retail,
            original_filename="acceptance_retail.mp4",
            storage_path=f"storage/uploads/{vid_retail}.mp4",
            status="processed",
            fps=30.0,
            duration_seconds=20.0,
        ))
        db.add(VideoModel(
            id=vid_burglary,
            original_filename="acceptance_burglary.mp4",
            storage_path=f"storage/uploads/{vid_burglary}.mp4",
            status="processed",
            fps=30.0,
            duration_seconds=180.0,
        ))

        # RETAIL TRACKS: Person A (orange), Person B (blue)
        db.add(TrackModel(
            id=str(uuid.uuid4()),
            video_id=vid_retail,
            track_id="TRACK-001",
            object_class="person",
            first_seen=0.0,
            last_seen=20.0,
            duration_seconds=20.0,
            detection_count=20,
            max_confidence=0.88,
            color="orange",
            color_confidence=0.75,
            current_bbox='{"x1": 450.0, "y1": 50.0, "x2": 520.0, "y2": 220.0}',
            trajectory='[[0.0, 485.0, 135.0], [20.0, 485.0, 135.0]]',
            active=False,
        ))
        db.add(TrackModel(
            id=str(uuid.uuid4()),
            video_id=vid_retail,
            track_id="TRACK-002",
            object_class="person",
            first_seen=0.0,
            last_seen=20.0,
            duration_seconds=20.0,
            detection_count=21,
            max_confidence=0.91,
            color="blue",
            color_confidence=0.78,
            current_bbox='{"x1": 500.0, "y1": 80.0, "x2": 580.0, "y2": 250.0}',
            trajectory='[[0.0, 540.0, 165.0], [20.0, 540.0, 165.0]]',
            active=False,
        ))

        # BURGLARY TRACKS: 2 physical actors fragmented across temporal tracklets
        # Actor 1 fragments (all black/dark under ATM light)
        db.add(TrackModel(
            id=str(uuid.uuid4()),
            video_id=vid_burglary,
            track_id="TRACK-001",
            object_class="person",
            first_seen=95.0,
            last_seen=107.0,
            duration_seconds=12.0,
            detection_count=11,
            max_confidence=0.85,
            color="black",
            color_confidence=0.65,
            current_bbox='{"x1": 126.9, "y1": 65.4, "x2": 163.2, "y2": 139.7}',
            trajectory='[[95.0, 145.0, 102.5], [107.0, 145.0, 102.5]]',
            active=False,
        ))
        db.add(TrackModel(
            id=str(uuid.uuid4()),
            video_id=vid_burglary,
            track_id="TRACK-005",
            object_class="person",
            first_seen=107.0,
            last_seen=117.0,
            duration_seconds=10.0,
            detection_count=9,
            max_confidence=0.82,
            color="black",
            color_confidence=0.70,
            current_bbox='{"x1": 127.2, "y1": 56.6, "x2": 155.3, "y2": 133.3}',
            trajectory='[[107.0, 141.2, 95.0], [117.0, 141.2, 95.0]]',
            active=False,
        ))

        # Actor 2 fragments (black clothing near ATM kiosk)
        db.add(TrackModel(
            id=str(uuid.uuid4()),
            video_id=vid_burglary,
            track_id="TRACK-002",
            object_class="person",
            first_seen=97.0,
            last_seen=101.0,
            duration_seconds=4.0,
            detection_count=5,
            max_confidence=0.84,
            color="black",
            color_confidence=0.68,
            current_bbox='{"x1": 76.5, "y1": 57.1, "x2": 105.7, "y2": 123.0}',
            trajectory='[[97.0, 91.1, 90.1], [101.0, 91.1, 90.1]]',
            active=False,
        ))
        db.add(TrackModel(
            id=str(uuid.uuid4()),
            video_id=vid_burglary,
            track_id="TRACK-007",
            object_class="person",
            first_seen=102.0,
            last_seen=125.0,
            duration_seconds=23.0,
            detection_count=13,
            max_confidence=0.89,
            color="black",
            color_confidence=0.72,
            current_bbox='{"x1": 61.6, "y1": 66.4, "x2": 88.8, "y2": 132.8}',
            trajectory='[[102.0, 75.2, 99.6], [125.0, 75.2, 99.6]]',
            active=False,
        ))

        db.commit()
        yield vid_retail, vid_burglary
    finally:
        # Cleanup
        db.query(TrackModel).filter(TrackModel.video_id.in_([vid_retail, vid_burglary])).delete(synchronize_session=False)
        db.query(VideoModel).filter(VideoModel.id.in_([vid_retail, vid_burglary])).delete(synchronize_session=False)
        db.commit()
        db.close()


def test_e2e_retail_canonical_counts_and_colors(setup_acceptance_fixtures):
    """Verify Retail video reports exactly 2 people, 1 blue, 1 orange via public API."""
    vid_retail, _ = setup_acceptance_fixtures

    # 1. Total people count query
    resp = client.post(f"/api/videos/{vid_retail}/ai-investigate", json={"query": "How many people are in the video?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert "2" in data["answer"]

    # 2. Blue clothing query
    resp_blue = client.post(f"/api/videos/{vid_retail}/ai-investigate", json={"query": "How many people are wearing blue?"})
    assert resp_blue.status_code == 200
    data_blue = resp_blue.json()
    assert data_blue["count"] == 1
    assert "1" in data_blue["answer"] or "one" in data_blue["answer"].lower()

    # 3. Orange clothing query
    resp_orange = client.post(f"/api/videos/{vid_retail}/ai-investigate", json={"query": "How many people are wearing orange?"})
    assert resp_orange.status_code == 200
    data_orange = resp_orange.json()
    assert data_orange["count"] == 1
    assert "1" in data_orange["answer"] or "one" in data_orange["answer"].lower()

    # 4. Clothing colors summary query
    resp_colors = client.post(f"/api/videos/{vid_retail}/ai-investigate", json={"query": "What colour clothes were the people wearing?"})
    assert resp_colors.status_code == 200
    data_colors = resp_colors.json()
    assert "orange" in data_colors["answer"].lower()
    assert "blue" in data_colors["answer"].lower()


def test_e2e_burglary_low_light_color_and_canonical_consistency(setup_acceptance_fixtures):
    """Verify Burglary video reports 0 blue, 2 black canonical people, and clothing color summary returns colors."""
    _, vid_burglary = setup_acceptance_fixtures

    # 1. Blue clothing count must be 0 (no false blue from cool low-light cast)
    resp_blue = client.post(f"/api/videos/{vid_burglary}/ai-investigate", json={"query": "How many people are wearing blue?"})
    assert resp_blue.status_code == 200
    data_blue = resp_blue.json()
    assert data_blue["count"] == 0
    assert "0" in data_blue["answer"] or "no" in data_blue["answer"].lower()

    # 2. Black clothing count must be 2 canonical physical actors (not 4 fragmented tracks)
    resp_black = client.post(f"/api/videos/{vid_burglary}/ai-investigate", json={"query": "How many people are wearing black?"})
    assert resp_black.status_code == 200
    data_black = resp_black.json()
    assert data_black["count"] == 2
    assert "2" in data_black["answer"] or "two" in data_black["answer"].lower()

    # 3. Clothing color breakdown query must return clothing color info, not theft summary
    resp_colors = client.post(f"/api/videos/{vid_burglary}/ai-investigate", json={"query": "What colour clothes were the people wearing?"})
    assert resp_colors.status_code == 200
    data_colors = resp_colors.json()
    assert "black" in data_colors["answer"].lower()
    # Must NOT return a generic theft takeaway summary for a clothing color query
    assert "potential theft" not in data_colors["answer"].lower()


def test_e2e_sequential_video_isolation(setup_acceptance_fixtures):
    """Verify switching Retail -> Burglary -> Retail preserves complete isolation."""
    vid_retail, vid_burglary = setup_acceptance_fixtures

    # Query Retail
    r1 = client.post(f"/api/videos/{vid_retail}/ai-investigate", json={"query": "How many people are wearing orange?"})
    assert r1.status_code == 200
    assert r1.json()["count"] == 1

    # Query Burglary
    b1 = client.post(f"/api/videos/{vid_burglary}/ai-investigate", json={"query": "How many people are wearing orange?"})
    assert b1.status_code == 200
    assert b1.json()["count"] == 0

    # Query Retail again
    r2 = client.post(f"/api/videos/{vid_retail}/ai-investigate", json={"query": "How many people are wearing orange?"})
    assert r2.status_code == 200
    assert r2.json()["count"] == 1


def test_e2e_crowd_dispersal_and_generic_activity_routing(setup_acceptance_fixtures):
    """
    Verify crowd dispersal inquiries route to specific event intent (POTENTIAL_CROWD_DISPERSAL)
    returning explicit zero/no crowd dispersal for videos without crowd events,
    while generic queries ('What happened in the video?') still return normal activity summary.
    """
    vid_retail, vid_burglary = setup_acceptance_fixtures

    # 1. Retail crowd dispersal query
    resp_ret = client.post(f"/api/videos/{vid_retail}/ai-investigate", json={"query": "Was there crowd dispersal?"})
    assert resp_ret.status_code == 200
    data_ret = resp_ret.json()
    assert data_ret["count"] == 0
    ans_ret = data_ret["answer"].lower()
    assert ("no crowd dispersal" in ans_ret or "0 crowd dispersal" in ans_ret or "no crowd, density" in ans_ret or "zero" in ans_ret)

    # 1b. Test crowd dispersal natural language variants
    variants = [
        "Was there any crowd dispersal?",
        "Did the crowd disperse?",
        "Was a crowd dispersal detected?",
        "Any crowd dispersal?",
        "Did people disperse as a crowd?",
    ]
    for variant in variants:
        resp_v = client.post(f"/api/videos/{vid_retail}/ai-investigate", json={"query": variant})
        assert resp_v.status_code == 200
        data_v = resp_v.json()
        assert data_v["count"] == 0
        ans_v = data_v["answer"].lower()
        assert ("no crowd dispersal" in ans_v or "0 crowd dispersal" in ans_v or "no crowd, density" in ans_v or "zero" in ans_v)

    # 2. Burglary crowd dispersal query
    resp_burg = client.post(f"/api/videos/{vid_burglary}/ai-investigate", json={"query": "Was there crowd dispersal?"})
    assert resp_burg.status_code == 200
    data_burg = resp_burg.json()
    assert data_burg["count"] == 0
    ans_burg = data_burg["answer"].lower()
    assert ("no crowd dispersal" in ans_burg or "0 crowd dispersal" in ans_burg or "no crowd, density" in ans_burg or "zero" in ans_burg)

    # 3. Generic query: must still return normal activity summary
    resp_gen = client.post(f"/api/videos/{vid_retail}/ai-investigate", json={"query": "What happened in the video?"})
    assert resp_gen.status_code == 200
    data_gen = resp_gen.json()
    ans_gen = data_gen["answer"].lower()
    assert ("activity" in ans_gen or "theft" in ans_gen or "takeaway" in ans_gen or "person" in ans_gen)

