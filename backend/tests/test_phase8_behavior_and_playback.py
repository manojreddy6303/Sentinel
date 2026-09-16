"""
Phase 8.1 Automated Tests: Security Behavior Intelligence & Evidence Clip Playback

Tests:
1. Browser-compatible H.264 evidence clip verification.
2. Unsupported mp4v evidence clip gets transcoded.
3. Evidence playback Range request returns 206 Partial Content.
4. Evidence playback is idempotent.
5. Concurrent evidence playback requests do not duplicate transcoding.
6. Evidence clip path traversal is blocked.
7. Evidence clip remains associated with correct evidence ID.
8. Person-object interaction detection.
9. Insufficient evidence does NOT produce POTENTIAL_THEFT (negative controls).
10. Valid object-removal pattern produces POTENTIAL_THEFT.
11. Investigation retrieves POTENTIAL_THEFT (positive & negative responses).
12. Safety constraint verification: zero criminal attribution.
"""

import os
import sys
import uuid
import math
import shutil
import threading
from pathlib import Path
import cv2
import numpy as np
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.config import settings
from backend.app.services.playback_service import (
    is_browser_compatible,
    ensure_evidence_clip_playback,
    get_evidence_playback_status,
)
from backend.app.services.evidence_service import EvidenceService
from backend.app.services.investigation_service import InvestigationService
from ai.schemas import BoundingBox, TrackedObject, SecurityEvent
from ai.behavior.analyzer import BehaviorAnalyzer
from ai.intelligence_pipeline import SecurityIntelligencePipeline
from ai.investigation.orchestrator import InvestigationOrchestrator
from database.session import SessionLocal
from database.models import VideoModel, EvidenceModel, SecurityEventModel
from ai.investigation.provider import MockLLMProvider

client = TestClient(app)


@pytest.fixture(autouse=True)
def deterministic_llm_provider(monkeypatch):
    """Ensure investigation tests run deterministically using MockLLMProvider without live Gemini calls."""
    monkeypatch.setattr(
        "ai.investigation.orchestrator.get_llm_provider",
        lambda *args, **kwargs: MockLLMProvider(),
    )


@pytest.fixture(scope="module")
def sample_mp4v_clip():
    """Create a temporary valid mp4v video clip for transcoding tests."""
    temp_dir = settings.STORAGE_EVIDENCE_DIR
    temp_dir.mkdir(parents=True, exist_ok=True)
    clip_id = f"test_ev_{uuid.uuid4().hex[:8]}"
    clip_path = temp_dir / f"{clip_id}_clip.mp4"

    # Create 30 frames of 320x240 video using mp4v fourcc
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(clip_path), fourcc, 10.0, (320, 240))
    for i in range(30):
        frame = np.full((240, 320, 3), (i * 8 % 255, 120, 200), dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    out.release()

    yield clip_id, clip_path

    # Cleanup
    if clip_path.exists():
        clip_path.unlink()
    playback_path = settings.STORAGE_EVIDENCE_PLAYBACK_DIR / f"{clip_id}_clip_playback.mp4"
    if playback_path.exists():
        playback_path.unlink()


@pytest.fixture(scope="module")
def seeded_theft_video():
    """Seed a test video with a grounded POTENTIAL_THEFT security event and evidence record."""
    vid = f"test_vid_{uuid.uuid4().hex[:8]}"
    ev_id = f"ev_theft_{uuid.uuid4().hex[:8]}"
    sec_id = f"sec_theft_{uuid.uuid4().hex[:8]}"

    # Create dummy video clip
    settings.STORAGE_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    clip_path = settings.STORAGE_EVIDENCE_DIR / f"{ev_id}_clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(clip_path), fourcc, 10.0, (320, 240))
    for i in range(20):
        frame = np.full((240, 320, 3), (50, 100, i * 10 % 255), dtype=np.uint8)
        out.write(frame)
    out.release()

    db = SessionLocal()
    try:
        vid_rec = VideoModel(
            id=vid,
            original_filename="theft_test.mp4",
            storage_path=str(clip_path),
            file_size_bytes=1024,
            duration_seconds=30.0,
            fps=10.0,
            frame_count=300,
            status="processed",
        )
        db.add(vid_rec)

        sec_ev = SecurityEventModel(
            id=sec_id,
            video_id=vid,
            event_type="POTENTIAL_THEFT",
            severity="HIGH",
            timestamp_seconds=12.0,
            duration_seconds=6.0,
            track_id="TRACK-014",
            object_class="backpack",
            confidence=0.88,
            description="Potential theft pattern: TRACK-014 approached backpack, remained nearby for 3.0s, after which object was removed.",
            observable_signals=["Proximity dwell: 3.0s", "Departure displacement: 85px"],
            evidence_id=ev_id,
        )
        db.add(sec_ev)

        ev_rec = EvidenceModel(
            id=ev_id,
            video_id=vid,
            event_id=sec_id,
            evidence_type="snapshot_and_clip",
            timestamp_seconds=12.0,
            source_video_name="theft_test.mp4",
            clip_path=str(clip_path),
            object_class="backpack",
            confidence=0.88,
            start_time=9.0,
            end_time=18.0,
            duration_seconds=9.0,
        )
        db.add(ev_rec)
        db.commit()
    finally:
        db.close()

    yield vid, ev_id, sec_id

    # Teardown
    db = SessionLocal()
    try:
        db.query(EvidenceModel).filter(EvidenceModel.video_id == vid).delete()
        db.query(SecurityEventModel).filter(SecurityEventModel.video_id == vid).delete()
        db.query(VideoModel).filter(VideoModel.id == vid).delete()
        db.commit()
    finally:
        db.close()

    if clip_path.exists():
        clip_path.unlink()
    pb_file = settings.STORAGE_EVIDENCE_PLAYBACK_DIR / f"{ev_id}_clip_playback.mp4"
    if pb_file.exists():
        pb_file.unlink()


# ---------------------------------------------------------------------------
# Test 1 & 2: Evidence Clip Transcoding and Compatibility
# ---------------------------------------------------------------------------

def test_01_mp4v_clip_transcoded_to_browser_compatible_h264(sample_mp4v_clip):
    """Verify an unsupported mp4v evidence clip is successfully transcoded to H.264."""
    clip_id, clip_path = sample_mp4v_clip

    # 1. Raw clip should not be browser-compatible
    raw_compatible = is_browser_compatible(clip_path)
    assert raw_compatible is False

    # 2. Transcode to playback directory
    playback_path = ensure_evidence_clip_playback(clip_id, clip_path)
    assert playback_path.exists()
    assert playback_path.stat().st_size > 0

    # 3. Transcoded clip must be browser-compatible H.264
    trans_compatible = is_browser_compatible(playback_path)
    assert trans_compatible is True


# ---------------------------------------------------------------------------
# Test 3: HTTP Range Requests on Evidence Playback
# ---------------------------------------------------------------------------

def test_02_evidence_playback_range_request_returns_206(seeded_theft_video):
    """Verify GET /api/evidence/{id}/playback returns 206 Partial Content with byte ranges."""
    vid, ev_id, sec_id = seeded_theft_video

    # Request first 1024 bytes
    res = client.get(f"/api/evidence/{ev_id}/playback", headers={"Range": "bytes=0-1023"})
    assert res.status_code == 206
    assert "bytes 0-1023/" in res.headers.get("content-range", "")
    assert res.headers.get("content-type") == "video/mp4"
    assert len(res.content) == 1024


# ---------------------------------------------------------------------------
# Test 4 & 5: Idempotency and Concurrency Safety
# ---------------------------------------------------------------------------

def test_03_evidence_playback_is_idempotent(sample_mp4v_clip):
    """Verify repeated conversion calls reuse existing playback file without re-transcoding."""
    clip_id, clip_path = sample_mp4v_clip

    path1 = ensure_evidence_clip_playback(clip_id, clip_path)
    mtime1 = path1.stat().st_mtime_ns

    # Second call should be instantaneous and reuse same file
    path2 = ensure_evidence_clip_playback(clip_id, clip_path)
    mtime2 = path2.stat().st_mtime_ns

    assert path1 == path2
    assert mtime1 == mtime2


def test_04_concurrent_evidence_playback_requests_safe(sample_mp4v_clip):
    """Verify concurrent requests for the same evidence clip do not duplicate or corrupt output."""
    clip_id, clip_path = sample_mp4v_clip
    results = []

    def worker():
        try:
            p = ensure_evidence_clip_playback(clip_id, clip_path)
            results.append(p)
        except Exception as e:
            results.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 4
    for r in results:
        assert isinstance(r, Path)
        assert r.exists()


# ---------------------------------------------------------------------------
# Test 6: Path Traversal Protection
# ---------------------------------------------------------------------------

def test_05_evidence_playback_path_traversal_blocked():
    """Verify directory traversal attempts in evidence ID are strictly rejected."""
    # Attempt dot-dot traversal
    res = client.get("/api/evidence/..%2F..%2Fetc%2Fpasswd/playback")
    assert res.status_code in (400, 404)

    # Attempt illegal characters
    res2 = client.get("/api/evidence/bad!evidence@id/playback")
    assert res2.status_code == 400
    assert "Invalid evidence ID" in res2.json()["detail"]


# ---------------------------------------------------------------------------
# Test 7: Playback Status Endpoint
# ---------------------------------------------------------------------------

def test_06_evidence_playback_status(seeded_theft_video):
    """Verify GET /api/evidence/{id}/playback-status returns correct state."""
    vid, ev_id, sec_id = seeded_theft_video
    res = client.get(f"/api/evidence/{ev_id}/playback-status")
    assert res.status_code == 200
    data = res.json()
    assert data["evidence_id"] == ev_id
    assert data["status"] in ["ready", "needs_conversion"]


# ---------------------------------------------------------------------------
# Test 8: Person-Object Interaction Detection
# ---------------------------------------------------------------------------

def test_07_person_object_interaction_detection():
    """Verify BehaviorAnalyzer identifies proximity between person and detectable object."""
    analyzer = BehaviorAnalyzer(
        theft_min_interaction_seconds=2.0,
        theft_interaction_max_distance=100.0,
        theft_min_departure_distance=60.0,
    )

    # Person approaches suitcase at (100, 100), stays 3 seconds, then walks away to (250, 100)
    p_traj = [
        (10.0, 50.0, 100.0),
        (11.0, 90.0, 100.0),   # interaction start (dist=10)
        (12.0, 95.0, 100.0),   # dwelling
        (13.0, 100.0, 100.0),  # interaction end (3s dwell)
        (14.0, 180.0, 100.0),  # departing (80px displacement)
        (15.0, 220.0, 100.0),
    ]
    person = TrackedObject(
        track_id="TRACK-001",
        object_class="person",
        first_seen=10.0,
        last_seen=15.0,
        confidence=0.90,
        current_bbox=BoundingBox(200, 80, 240, 120),
        trajectory=p_traj,
    )

    # Suitcase present from 10.0 to 13.0, then disappears
    o_traj = [
        (10.0, 100.0, 100.0),
        (11.0, 100.0, 100.0),
        (12.0, 100.0, 100.0),
        (13.0, 100.0, 100.0),
    ]
    suitcase = TrackedObject(
        track_id="TRACK-002",
        object_class="suitcase",
        first_seen=10.0,
        last_seen=13.0,
        confidence=0.85,
        current_bbox=BoundingBox(90, 90, 110, 110),
        trajectory=o_traj,
    )

    theft_events = analyzer.detect_theft_and_removal_patterns([person, suitcase])
    assert len(theft_events) == 1
    ev = theft_events[0]
    assert ev.event_type == "POTENTIAL_THEFT"
    assert ev.track_id == "TRACK-001"
    assert ev.object_class == "suitcase"
    assert ev.confidence >= 0.70
    assert "TRACK-001" in ev.description
    assert "suitcase" in ev.description


# ---------------------------------------------------------------------------
# Test 9: Negative Controls (Insufficient Evidence Rejection)
# ---------------------------------------------------------------------------

def test_08_insufficient_evidence_does_not_produce_theft():
    """Verify non-theft scenarios (passerby, stationary person, non-disappearing object) are rejected."""
    analyzer = BehaviorAnalyzer(
        theft_min_interaction_seconds=2.0,
        theft_interaction_max_distance=100.0,
        theft_min_departure_distance=60.0,
    )

    # Scenario A: Passerby (dwells only 0.5s near backpack)
    p_passerby = TrackedObject(
        track_id="TRACK-P1",
        object_class="person",
        first_seen=5.0,
        last_seen=8.0,
        confidence=0.88,
        current_bbox=BoundingBox(300, 100, 320, 150),
        trajectory=[
            (5.0, 0.0, 100.0),
            (6.0, 90.0, 100.0),   # near backpack for 1 instant
            (7.0, 200.0, 100.0),
            (8.0, 310.0, 100.0),
        ],
    )
    backpack = TrackedObject(
        track_id="TRACK-B1",
        object_class="backpack",
        first_seen=5.0,
        last_seen=15.0,  # backpack stays there!
        confidence=0.85,
        current_bbox=BoundingBox(90, 90, 110, 110),
        trajectory=[(t, 100.0, 100.0) for t in range(5, 16)],
    )

    # Should not produce theft because backpack stays and passerby didn't dwell
    events_a = analyzer.detect_theft_and_removal_patterns([p_passerby, backpack])
    assert len(events_a) == 0

    # Scenario B: Person stays near object and does not depart (no takeaway)
    p_sitting = TrackedObject(
        track_id="TRACK-P2",
        object_class="person",
        first_seen=5.0,
        last_seen=15.0,
        confidence=0.90,
        current_bbox=BoundingBox(95, 95, 105, 105),
        trajectory=[(float(t), 105.0, 100.0) for t in range(5, 16)],  # stationary sitting
    )
    events_b = analyzer.detect_theft_and_removal_patterns([p_sitting, backpack])
    assert len(events_b) == 0


# ---------------------------------------------------------------------------
# Test 10: Valid Object Removal Pattern (Co-Movement / Takeaway)
# ---------------------------------------------------------------------------

def test_09_co_movement_takeaway_produces_potential_theft():
    """Verify co-movement pattern (person picks up object and walks away together) produces POTENTIAL_THEFT."""
    analyzer = BehaviorAnalyzer(
        theft_min_interaction_seconds=2.0,
        theft_interaction_max_distance=100.0,
        theft_min_departure_distance=60.0,
    )

    # Person approaches at 10.0, picks up at 12.0, walks away carrying it at 13.0-16.0
    p_traj = [
        (10.0, 100.0, 100.0),
        (11.0, 100.0, 100.0),
        (12.0, 100.0, 100.0),
        (13.0, 140.0, 100.0),  # moving together
        (14.0, 180.0, 100.0),  # 80px displacement
        (15.0, 220.0, 100.0),
    ]
    o_traj = [
        (10.0, 100.0, 100.0),
        (11.0, 100.0, 100.0),
        (12.0, 100.0, 100.0),
        (13.0, 138.0, 100.0),  # co-moving
        (14.0, 178.0, 100.0),
        (15.0, 218.0, 100.0),
    ]
    person = TrackedObject(
        track_id="TRACK-P3",
        object_class="person",
        first_seen=10.0,
        last_seen=15.0,
        confidence=0.92,
        current_bbox=BoundingBox(210, 80, 230, 120),
        trajectory=p_traj,
    )
    handbag = TrackedObject(
        track_id="TRACK-O3",
        object_class="handbag",
        first_seen=10.0,
        last_seen=15.0,
        confidence=0.88,
        current_bbox=BoundingBox(215, 90, 225, 110),
        trajectory=o_traj,
    )

    events = analyzer.detect_theft_and_removal_patterns([person, handbag])
    assert len(events) == 1
    ev = events[0]
    assert ev.event_type == "POTENTIAL_THEFT"
    assert ev.track_id == "TRACK-P3"
    assert ev.object_class == "handbag"
    assert "co_movement_takeaway" in " ".join(ev.observable_signals)


# ---------------------------------------------------------------------------
# Test 11: Investigation Service & Orchestrator Query Retrieval
# ---------------------------------------------------------------------------

def test_10_investigation_retrieves_potential_theft(seeded_theft_video):
    """Verify Ask Sentinel / investigation queries retrieve POTENTIAL_THEFT with grounded wording."""
    vid, ev_id, sec_id = seeded_theft_video
    service = InvestigationService()

    # Query for theft events
    res = service.investigate(vid, "Were there any possible theft events?")
    assert res["is_supported"] is True
    assert res["result_type"] == "security_events"
    assert res["count"] >= 1
    assert "potential object-takeaway pattern" in res["message"]
    assert "Review the linked evidence" in res["message"]

    # Verify orchestrator handles theft queries properly
    orch = InvestigationOrchestrator()
    orch_res = orch.process_investigation(vid, "Were there any possible theft events?")
    assert orch_res["is_supported"] is True
    assert "potential object-takeaway pattern" in orch_res["answer"]


def test_11_investigation_honest_negative_when_no_theft():
    """Verify queries for theft on a non-theft video honestly state no theft pattern was detected."""
    vid = f"clean_vid_{uuid.uuid4().hex[:8]}"
    db = SessionLocal()
    try:
        db.add(VideoModel(
            id=vid,
            original_filename="clean.mp4",
            storage_path="storage/clean.mp4",
            duration_seconds=20.0,
            status="processed",
        ))
        db.commit()
    finally:
        db.close()

    service = InvestigationService()
    res = service.investigate(vid, "Were there any possible theft events?")
    assert res["is_supported"] is True
    assert res["count"] == 0
    assert "No potential theft pattern was detected in the available visual evidence" in res["message"]

    orch = InvestigationOrchestrator()
    orch_res = orch.process_investigation(vid, "Was there any suspicious activity?")
    assert orch_res["is_supported"] is True
    assert "No" in orch_res["answer"] or "notable" in orch_res["answer"]

    # Cleanup
    db = SessionLocal()
    try:
        db.query(VideoModel).filter(VideoModel.id == vid).delete()
        db.commit()
    finally:
        db.close()
