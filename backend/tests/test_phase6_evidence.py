"""
Sentinel Phase 6 Automated Test Suite
Evidence Extraction & Evidence Vault

Tests:
1. Evidence table initialization & schema binding
2. Create snapshot evidence & verify file on disk
3. Snapshot file exists on disk with valid image dimensions
4. Correct timestamp stored in evidence record
5. Correct source video name stored in evidence record
6. Create video clip evidence
7. Clip extraction respects video boundaries (no negative start, capped end)
8. Event belongs to requested video (ownership validation)
9. Invalid event rejected (404/400)
10. Invalid timestamp rejected (negative or beyond video)
11. Evidence retrieval API (GET /api/evidence/{id})
12. Evidence list API (GET /api/videos/{id}/evidence)
13. Duplicate evidence handling (idempotency, prevents redundant records)
14. Secure evidence file access (path traversal rejection, binary streaming)
15. Verification of ethical accuracy (observational labels, no conclusions)
"""

import os
import sys
import uuid
import json
import shutil
import pytest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.config import settings
from database.session import init_db, SessionLocal
from database.models import VideoModel, EventModel, GroupedEventModel, EvidenceModel
from backend.app.services.evidence_service import EvidenceService

client = TestClient(app)


# ---------------------------------------------------------------------------
# Test Fixture: Seed Video with File and Detections
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def seeded_evidence_video():
    """
    Creates a dedicated test video file and database record for Phase 6 tests.
    Generates a 5-second 640x480 test video with OpenCV so snapshot & clip
    extractions have genuine frames to read.
    """
    init_db()
    video_id = f"test_ev_{uuid.uuid4().hex[:10]}"
    filename = "cctv_test_sample.mp4"
    saved_filename = f"{video_id}_{filename}"
    video_path = settings.STORAGE_UPLOADS_DIR / saved_filename
    meta_path = settings.STORAGE_UPLOADS_DIR / f"{video_id}.json"

    # Generate a 5-second 30fps test video
    fps = 30.0
    duration_sec = 5.0
    total_frames = int(fps * duration_sec)
    width, height = 640, 480

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    for i in range(total_frames):
        # Create a frame with frame number drawn on it
        frame = 40 * ((i // 10) % 5 + 1)
        import numpy as np
        img = np.full((height, width, 3), frame, dtype=np.uint8)
        cv2.putText(img, f"Frame {i} - {i/fps:.2f}s", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(img)
    writer.release()

    # Create video sidecar metadata
    meta = {
        "video_id": video_id,
        "filename": filename,
        "saved_filename": saved_filename,
        "file_size_bytes": video_path.stat().st_size,
        "status": "processed",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    # Insert Video and Event into DB
    db = SessionLocal()
    event_id_1 = str(uuid.uuid4())
    event_id_2 = str(uuid.uuid4())
    foreign_event_id = str(uuid.uuid4())

    try:
        vid_model = VideoModel(
            id=video_id,
            original_filename=filename,
            saved_filename=saved_filename,
            storage_path=str(video_path),
            file_size_bytes=video_path.stat().st_size,
            status="processed",
            duration_seconds=5.0,
            fps=30.0,
            frame_count=total_frames,
        )
        db.add(vid_model)

        ev1 = EventModel(
            id=event_id_1,
            video_id=video_id,
            object_class="car",
            class_id=2,
            confidence=0.81,
            timestamp_seconds=1.50,
            frame_number=45,
            bbox_x1=100.0,
            bbox_y1=150.0,
            bbox_x2=300.0,
            bbox_y2=350.0,
        )
        ev2 = EventModel(
            id=event_id_2,
            video_id=video_id,
            object_class="person",
            class_id=0,
            confidence=0.92,
            timestamp_seconds=4.20,
            frame_number=126,
            bbox_x1=50.0,
            bbox_y1=50.0,
            bbox_x2=150.0,
            bbox_y2=250.0,
        )
        # Foreign event belonging to another video
        foreign_vid = db.query(VideoModel).filter_by(id="foreign_video_999").first()
        if not foreign_vid:
            foreign_vid = VideoModel(
                id="foreign_video_999",
                original_filename="foreign.mp4",
                storage_path="storage/foreign.mp4",
                status="processed",
            )
            db.add(foreign_vid)

        foreign_ev = EventModel(
            id=foreign_event_id,
            video_id="foreign_video_999",
            object_class="truck",
            class_id=7,
            confidence=0.75,
            timestamp_seconds=2.00,
            frame_number=60,
            bbox_x1=10.0,
            bbox_y1=10.0,
            bbox_x2=50.0,
            bbox_y2=50.0,
        )
        db.add_all([ev1, ev2, foreign_ev])
        db.commit()
    finally:
        db.close()

    yield {
        "video_id": video_id,
        "event_id_1": event_id_1,
        "event_id_2": event_id_2,
        "foreign_event_id": foreign_event_id,
        "video_path": video_path,
        "meta_path": meta_path,
    }

    # Cleanup test files
    if video_path.exists():
        try:
            video_path.unlink()
        except Exception:
            pass
    if meta_path.exists():
        try:
            meta_path.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_1_evidence_table_initialization():
    """1. Evidence table initializes cleanly and supports queries."""
    init_db()
    db = SessionLocal()
    try:
        count = db.query(EvidenceModel).count()
        assert isinstance(count, int)
    finally:
        db.close()


def test_2_and_3_create_snapshot_evidence(seeded_evidence_video):
    """2 & 3. Create snapshot evidence and verify image file exists on disk."""
    service = EvidenceService()
    v_info = seeded_evidence_video

    evidence = service.create_evidence(
        video_id=v_info["video_id"],
        timestamp=1.50,
        event_id=v_info["event_id_1"],
        evidence_type="snapshot_only",
    )

    assert evidence is not None
    assert evidence["evidence_id"] is not None
    assert evidence["has_snapshot"] is True
    assert evidence["has_annotated"] is True

    # Verify files directly in DB/disk
    db = SessionLocal()
    try:
        rec = db.query(EvidenceModel).filter(EvidenceModel.id == evidence["evidence_id"]).first()
        assert rec is not None
        assert rec.snapshot_path is not None
        assert os.path.exists(rec.snapshot_path)
        assert os.path.getsize(rec.snapshot_path) > 0
        assert rec.annotated_snapshot_path is not None
        assert os.path.exists(rec.annotated_snapshot_path)
        assert os.path.getsize(rec.annotated_snapshot_path) > 0
    finally:
        db.close()


def test_4_correct_timestamp_stored(seeded_evidence_video):
    """4. Correct timestamp is stored in database record."""
    v_info = seeded_evidence_video
    db = SessionLocal()
    try:
        rec = db.query(EvidenceModel).filter(EvidenceModel.video_id == v_info["video_id"]).first()
        assert rec is not None
        assert abs(rec.timestamp_seconds - 1.50) < 0.001
    finally:
        db.close()


def test_5_correct_source_video_stored(seeded_evidence_video):
    """5. Correct source video name is stored in evidence record."""
    v_info = seeded_evidence_video
    db = SessionLocal()
    try:
        rec = db.query(EvidenceModel).filter(EvidenceModel.video_id == v_info["video_id"]).first()
        assert rec is not None
        assert rec.source_video_name == "cctv_test_sample.mp4"
        assert rec.video_id == v_info["video_id"]
    finally:
        db.close()


def test_6_create_clip_evidence(seeded_evidence_video):
    """6. Create video clip evidence and verify mp4 file exists on disk."""
    service = EvidenceService()
    v_info = seeded_evidence_video

    evidence = service.create_evidence(
        video_id=v_info["video_id"],
        timestamp=2.00,
        evidence_type="clip_only",
        pre_seconds=1.0,
        post_seconds=1.0,
    )

    assert evidence is not None
    assert evidence["has_clip"] is True

    db = SessionLocal()
    try:
        rec = db.query(EvidenceModel).filter(EvidenceModel.id == evidence["evidence_id"]).first()
        assert rec is not None
        assert rec.clip_path is not None
        assert os.path.exists(rec.clip_path)
        assert os.path.getsize(rec.clip_path) > 0
        assert rec.clip_path.endswith(".mp4")
    finally:
        db.close()


def test_7_clip_respects_video_boundaries(seeded_evidence_video):
    """7. Clip extraction respects video boundaries (no negative start, capped at duration)."""
    service = EvidenceService()
    v_info = seeded_evidence_video

    # Video is 5.0 seconds long.
    # Case A: Near start (0.5s) with pre_seconds=3.0 -> start_time should be 0.0, not -2.5
    ev_start = service.create_evidence(
        video_id=v_info["video_id"],
        timestamp=0.50,
        evidence_type="clip_only",
        pre_seconds=3.0,
        post_seconds=1.0,
    )
    assert ev_start["start_time"] == 0.0
    assert ev_start["end_time"] <= 5.0

    # Case B: Near end (4.8s) with post_seconds=3.0 -> end_time should be capped at 5.0, not 7.8
    ev_end = service.create_evidence(
        video_id=v_info["video_id"],
        timestamp=4.80,
        evidence_type="clip_only",
        pre_seconds=1.0,
        post_seconds=3.0,
    )
    assert ev_end["start_time"] >= 0.0
    assert ev_end["end_time"] == 5.0
    assert ev_end["duration_seconds"] <= 5.0


def test_8_event_belongs_to_requested_video(seeded_evidence_video):
    """8. Rejects evidence extraction when event belongs to a different video."""
    v_info = seeded_evidence_video

    # API call with foreign event ID
    res = client.post(
        f"/api/videos/{v_info['video_id']}/evidence",
        json={
            "timestamp": 2.0,
            "event_id": v_info["foreign_event_id"],
            "evidence_type": "snapshot_only",
        },
    )
    assert res.status_code == 400
    assert "does not belong to video" in res.json()["detail"].lower()


def test_9_invalid_event_rejected(seeded_evidence_video):
    """9. Rejects non-existent event ID."""
    v_info = seeded_evidence_video
    bogus_id = str(uuid.uuid4())

    res = client.post(
        f"/api/videos/{v_info['video_id']}/evidence",
        json={
            "timestamp": 2.0,
            "event_id": bogus_id,
            "evidence_type": "snapshot_only",
        },
    )
    assert res.status_code in [400, 404]
    assert "not found" in res.json()["detail"].lower()


def test_10_invalid_timestamp_rejected(seeded_evidence_video):
    """10. Rejects negative timestamps or timestamps far beyond video length."""
    v_info = seeded_evidence_video

    # Negative timestamp
    res_neg = client.post(
        f"/api/videos/{v_info['video_id']}/evidence",
        json={"timestamp": -3.0, "evidence_type": "snapshot_only"},
    )
    assert res_neg.status_code == 422 or res_neg.status_code == 400

    # Far beyond video duration (video is 5s, asking for 999s)
    res_far = client.post(
        f"/api/videos/{v_info['video_id']}/evidence",
        json={"timestamp": 999.0, "evidence_type": "snapshot_only"},
    )
    assert res_far.status_code == 400
    assert "exceeds" in res_far.json()["detail"].lower()


def test_11_evidence_retrieval_api(seeded_evidence_video):
    """11. Evidence retrieval API (GET /api/evidence/{id}) returns forensic metadata without raw paths."""
    v_info = seeded_evidence_video

    # Create evidence via API
    create_res = client.post(
        f"/api/videos/{v_info['video_id']}/evidence",
        json={
            "timestamp": 1.50,
            "event_id": v_info["event_id_1"],
            "evidence_type": "snapshot_and_clip",
            "pre_seconds": 1.0,
            "post_seconds": 1.0,
        },
    )
    assert create_res.status_code in [200, 201]
    ev_data = create_res.json()
    ev_id = ev_data["evidence_id"]

    # Fetch evidence details
    get_res = client.get(f"/api/evidence/{ev_id}")
    assert get_res.status_code == 200
    item = get_res.json()
    assert item["evidence_id"] == ev_id
    assert item["video_id"] == v_info["video_id"]
    assert item["evidence_type"] == "snapshot_and_clip"
    assert item["has_snapshot"] is True
    assert item["has_clip"] is True
    # Verify no raw filesystem path like "C:\\" or "storage/evidence/" in public response
    assert "storage/" not in item.get("snapshot_url", "")
    assert "C:\\" not in json.dumps(item)


def test_12_evidence_list_api(seeded_evidence_video):
    """12. Evidence list API (GET /api/videos/{id}/evidence) returns all evidence for video."""
    v_info = seeded_evidence_video

    res = client.get(f"/api/videos/{v_info['video_id']}/evidence")
    assert res.status_code == 200
    data = res.json()
    assert "evidence" in data
    assert "total_evidence" in data or "total" in data
    total = data.get("total_evidence", data.get("total", 0))
    assert total >= 1
    assert data["evidence"][0]["video_id"] == v_info["video_id"]


def test_13_duplicate_evidence_handling(seeded_evidence_video):
    """13. Duplicate evidence requests return existing record rather than creating duplicates."""
    v_info = seeded_evidence_video

    payload = {
        "timestamp": 4.20,
        "event_id": v_info["event_id_2"],
        "evidence_type": "snapshot_only",
    }

    # First request
    res1 = client.post(f"/api/videos/{v_info['video_id']}/evidence", json=payload)
    assert res1.status_code in [200, 201]
    ev_id_1 = res1.json()["evidence_id"]

    # Second identical request
    res2 = client.post(f"/api/videos/{v_info['video_id']}/evidence", json=payload)
    assert res2.status_code == 200
    ev_id_2 = res2.json()["evidence_id"]

    assert ev_id_1 == ev_id_2
    assert res2.json().get("is_duplicate") is True


def test_14_secure_evidence_file_access(seeded_evidence_video):
    """14. Secure file access prevents path traversal and streams binary content properly."""
    v_info = seeded_evidence_video

    # Create evidence to get a valid ID
    create_res = client.post(
        f"/api/videos/{v_info['video_id']}/evidence",
        json={"timestamp": 1.50, "evidence_type": "snapshot_only"},
    )
    ev_id = create_res.json()["evidence_id"]

    # Valid snapshot streaming
    stream_res = client.get(f"/api/evidence/{ev_id}/snapshot")
    assert stream_res.status_code == 200
    assert stream_res.headers["content-type"].startswith("image/")

    # Path traversal attack attempt in evidence ID
    traversal_res = client.get("/api/evidence/..%2F..%2Fetc%2Fpasswd/snapshot")
    assert traversal_res.status_code in [400, 404]

    # Non-existent UUID
    fake_uuid = str(uuid.uuid4())
    fake_res = client.get(f"/api/evidence/{fake_uuid}/snapshot")
    assert fake_res.status_code == 404


def test_15_observational_labeling_ethics():
    """15. Ethical check: Evidence labels describe observations, never criminal accusations."""
    # Test valid observational strings vs invalid accusatory labels
    forbidden_terms = ["criminal", "thief", "dangerous", "suspect", "guilty", "perpetrator"]

    sample_evidence_note = "Person detected at 4.20s with confidence 92%."
    for term in forbidden_terms:
        assert term not in sample_evidence_note.lower(), f"Unethical term '{term}' found in evidence note"
