"""
Sentinel Retail Entity Consistency & Deduplication Regression Test Suite
Validates:
1. Canonical entity deduplication on duplicate/fragmented person tracklets
2. Zero color attribution contamination across separate tracks
3. Exact count reporting for generic counting variations (A-F)
4. Repeat query consistency across runs
5. Preservation of bystander status for uninvolved individuals
"""

import pytest
from database.session import SessionLocal
from database.models import VideoModel, TrackModel, SecurityEventModel
from backend.app.services.investigation_service import InvestigationService
from backend.app.services.investigation_parser import InvestigationParser
from ai.investigation.orchestrator import InvestigationOrchestrator
from ai.investigation.provider import MockLLMProvider


@pytest.fixture
def test_setup():
    db = SessionLocal()
    video_id = "test_retail_video_001"
    
    # Ensure clean fixture state
    db.query(TrackModel).filter(TrackModel.video_id == video_id).delete()
    db.query(SecurityEventModel).filter(SecurityEventModel.video_id == video_id).delete()
    db.query(VideoModel).filter(VideoModel.id == video_id).delete()
    db.commit()

    # Create video record
    v = VideoModel(
        id=video_id,
        original_filename="WhatsApp Video 2026-09-26 at 06.14.37.mp4",
        saved_filename=f"{video_id}.mp4",
        storage_path=f"storage/uploads/{video_id}.mp4",
        file_size_bytes=1000,
        status="processed",
        duration_seconds=20.0,
        fps=30.0,
    )
    db.add(v)

    # TRACK-001 (orange, primary)
    t1 = TrackModel(
        video_id=video_id,
        track_id="TRACK-001",
        object_class="person",
        first_seen=0.0,
        last_seen=20.0,
        duration_seconds=20.0,
        detection_count=20,
        max_confidence=0.9,
        color="orange",
        color_confidence=0.756,
        current_bbox={"x1": 440, "y1": 50, "x2": 530, "y2": 200},
        trajectory=[(15.0, 485, 125)],
        active=0,
    )
    # TRACK-002 (blue, bystander)
    t2 = TrackModel(
        video_id=video_id,
        track_id="TRACK-002",
        object_class="person",
        first_seen=0.0,
        last_seen=20.0,
        duration_seconds=20.0,
        detection_count=21,
        max_confidence=0.92,
        color="blue",
        color_confidence=0.6923,
        current_bbox={"x1": 200, "y1": 100, "x2": 300, "y2": 300},
        trajectory=[(15.0, 250, 200)],
        active=0,
    )
    # TRACK-005 (duplicate fragment of TRACK-001 at 15s)
    t5 = TrackModel(
        video_id=video_id,
        track_id="TRACK-005",
        object_class="person",
        first_seen=15.015,
        last_seen=17.017,
        duration_seconds=2.0,
        detection_count=2,
        max_confidence=0.48,
        color=None,
        color_confidence=None,
        current_bbox={"x1": 442, "y1": 55, "x2": 528, "y2": 165},
        trajectory=[(15.015, 485, 110)],
        active=0,
    )
    # Security event: POTENTIAL_THEFT for TRACK-001 at 8.01s
    se = SecurityEventModel(
        video_id=video_id,
        event_type="POTENTIAL_THEFT",
        timestamp_seconds=8.01,
        severity="medium",
        confidence=0.9,
        track_id="TRACK-001",
        description="Potential object takeaway pattern observed.",
        incident_metadata={"primary_tracks": ["TRACK-001"]},
    )

    db.add_all([t1, t2, t5, se])
    db.commit()

    svc = InvestigationService()
    orch = InvestigationOrchestrator(provider=MockLLMProvider(is_available_flag=False))
    yield db, video_id, svc, orch

    # Cleanup
    db.query(TrackModel).filter(TrackModel.video_id == video_id).delete()
    db.query(SecurityEventModel).filter(SecurityEventModel.video_id == video_id).delete()
    db.query(VideoModel).filter(VideoModel.id == video_id).delete()
    db.commit()
    db.close()


def test_reconcile_canonical_entities_unit():
    svc = InvestigationService()
    # Synthetic overlapping tracks representing the same person
    tracks = [
        {
            "track_id": "TRACK-001",
            "object_class": "person",
            "first_seen": 0.0,
            "last_seen": 20.0,
            "duration_seconds": 20.0,
            "detection_count": 20,
            "max_confidence": 0.9,
            "color": "orange",
            "color_confidence": 0.75,
            "current_bbox": {"x1": 440, "y1": 50, "x2": 530, "y2": 200},
            "trajectory": [(15.0, 485, 125)],
        },
        {
            "track_id": "TRACK-005",
            "object_class": "person",
            "first_seen": 15.0,
            "last_seen": 17.0,
            "duration_seconds": 2.0,
            "detection_count": 2,
            "max_confidence": 0.48,
            "color": None,
            "color_confidence": None,
            "current_bbox": {"x1": 442, "y1": 55, "x2": 528, "y2": 165},
            "trajectory": [(15.0, 485, 110)],
        },
        {
            "track_id": "TRACK-002",
            "object_class": "person",
            "first_seen": 0.0,
            "last_seen": 20.0,
            "duration_seconds": 20.0,
            "detection_count": 21,
            "max_confidence": 0.92,
            "color": "blue",
            "color_confidence": 0.69,
            "current_bbox": {"x1": 200, "y1": 100, "x2": 300, "y2": 300},
            "trajectory": [(15.0, 250, 200)],
        },
    ]

    canon = svc._reconcile_canonical_entities(tracks)
    assert len(canon) == 2, f"Expected 2 canonical entities, got {len(canon)}"
    
    t1_canon = next(c for c in canon if c["canonical_id"] == "TRACK-001")
    assert "TRACK-005" in t1_canon["member_track_ids"]
    assert t1_canon["color"] == "orange"

    t2_canon = next(c for c in canon if c["canonical_id"] == "TRACK-002")
    assert t2_canon["member_track_ids"] == ["TRACK-002"]
    assert t2_canon["color"] == "blue"


def test_counting_queries_variations(test_setup):
    db, video_id, svc, orch = test_setup

    # Variation A: 'how many people'
    res_a = svc.investigate(video_id, "how many people")
    assert res_a["count"] == 2
    assert res_a["canonical_entity_count"] == 2

    # Variation B: 'how many persons'
    res_b = svc.investigate(video_id, "how many persons")
    assert res_b["count"] == 2

    # Variation C: 'how many people are wearing blue'
    res_c = svc.investigate(video_id, "how many people are wearing blue")
    assert res_c["count"] == 1
    assert res_c["canonical_entity_count"] == 1
    assert res_c["results"][0]["canonical_id"] == "TRACK-002"

    # Variation D: 'how many people in blue'
    res_d = svc.investigate(video_id, "how many people in blue")
    assert res_d["count"] == 1

    # Variation E: 'how many people wearing blue clothing'
    res_e = svc.investigate(video_id, "how many people wearing blue clothing")
    assert res_e["count"] == 1

    # Variation F: 'count of people in the video'
    res_f = svc.investigate(video_id, "count of people in the video")
    assert res_f["count"] == 2

    # Variation G: 'how many tracks'
    res_g = svc.investigate(video_id, "how many tracks")
    assert "distinct physical entities" in res_g["message"]


def test_repeat_query_consistency(test_setup):
    db, video_id, svc, orch = test_setup

    q = "How many people are wearing blue?"
    answers = []
    for _ in range(3):
        ans = orch.process_investigation(video_id, q)
        assert ans["count"] == 1
        answers.append(ans["answer"])

    # All 3 answers identical
    assert answers[0] == answers[1] == answers[2]
    assert "TRACK-007" not in answers[0]
    assert "TRACK-002" in answers[0]


def test_blue_person_bystander_status(test_setup):
    db, video_id, svc, orch = test_setup

    ans = orch.process_investigation(video_id, "What did the person wearing blue do?")
    assert ans["count"] == 1
    assert "TRACK-002" in ans["answer"]
    # Verify bystander status
    assert "bystander" in ans["answer"].lower() or "no security infractions" in ans["answer"].lower()
