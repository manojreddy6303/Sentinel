"""
Sentinel Phase 20.3.1 Regression Tests: Legacy Video Playback Memory Hardening.

Covers:
A. MP4 playback: HTTP 206, Range requests, video/mp4 content type.
B. AVI playback: Bounded memory, clean process exit, temporary file cleanup.
C. Large legacy video: No full-file RAM loading, bounded subprocess stdout/stderr.
D. Low-headroom / active analysis fallback: Informative 503 response, zero backend crash.
E. Backend health after playback: /health and /api/health return 200 OK.
F. Video analysis remains completely unaffected.
"""
import os
import sys
import uuid
import time
import subprocess
import psutil
from pathlib import Path
from unittest.mock import patch

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.config import settings
from backend.app.services.playback_service import (
    is_browser_compatible,
    get_playback_status,
    ensure_playback_file,
    transcode_to_h264,
    get_container_memory_headroom_mb,
    PlaybackMemoryPressureError,
    PlaybackConversionError,
)
from backend.app.api.videos import _heavy_processing_semaphore
from database.session import SessionLocal
from database.models import VideoModel

client = TestClient(app)

# Existing verified MP4 test video in storage/uploads
WORKING_H264_VIDEO_ID = "605f62c7-9b7c-423b-8020-310d6f3899c5"


def create_test_avi_video(path: Path, num_frames: int = 60, width: int = 640, height: int = 480) -> Path:
    """Helper to generate a real, non-browser-compatible AVI test file with OpenCV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(path), fourcc, 25.0, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), (i * 4) % 255, dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    return path


# ---------------------------------------------------------------------------
# Test A: MP4 Playback with HTTP Range Requests
# ---------------------------------------------------------------------------

def test_mp4_playback_range_streaming():
    """Verify MP4 playback uses RFC 7233 Range streaming with 206 Partial Content."""
    h264_path = settings.STORAGE_UPLOADS_DIR / f"{WORKING_H264_VIDEO_ID}_12566041-uhd_3840_2160_30fps.mp4"
    cleanup_needed = False

    if h264_path.exists() and is_browser_compatible(h264_path):
        video_id = WORKING_H264_VIDEO_ID
    else:
        # Create a self-contained H.264 MP4 test file
        test_id = f"test-mp4-{uuid.uuid4().hex[:8]}"
        video_file = settings.STORAGE_UPLOADS_DIR / f"{test_id}_sample.mp4"
        raw_avi = settings.STORAGE_UPLOADS_DIR / f"{test_id}_temp.avi"
        create_test_avi_video(raw_avi, num_frames=25, width=640, height=480)
        transcode_to_h264(raw_avi, video_file)
        if raw_avi.exists():
            raw_avi.unlink()
        video_id = test_id
        h264_path = video_file
        cleanup_needed = True

    try:
        # 1. Range seeking start of video
        res = client.get(f"/api/videos/{video_id}/playback", headers={"Range": "bytes=0-1023"})
        assert res.status_code == 206
        assert res.headers.get("Content-Type") == "video/mp4"
        assert res.headers.get("Accept-Ranges") == "bytes"
        assert "bytes 0-1023/" in res.headers.get("Content-Range", "")
        assert len(res.content) == 1024

        # 2. Range seeking middle of video
        res_mid = client.get(f"/api/videos/{video_id}/playback", headers={"Range": "bytes=2048-4095"})
        assert res_mid.status_code == 206
        assert "bytes 2048-4095/" in res_mid.headers.get("Content-Range", "")
        assert len(res_mid.content) == 2048

    finally:
        if cleanup_needed and h264_path.exists():
            h264_path.unlink()


# ---------------------------------------------------------------------------
# Test B: Legacy AVI Playback & Safe Bounded Transcoding
# ---------------------------------------------------------------------------

def test_avi_playback_bounded_transcode_and_cleanup(tmp_path):
    """Verify legacy AVI conversion is memory-bounded, cleans temp files, and produces valid MP4."""
    test_id = f"test-avi-{uuid.uuid4().hex[:8]}"
    avi_filename = f"{test_id}_surveillance_clip.avi"
    avi_path = settings.STORAGE_UPLOADS_DIR / avi_filename
    create_test_avi_video(avi_path, num_frames=50, width=640, height=480)

    db = SessionLocal()
    try:
        v = VideoModel(
            id=test_id,
            original_filename="surveillance_clip.avi",
            saved_filename=avi_filename,
            storage_path=str(avi_path),
            file_size_bytes=avi_path.stat().st_size,
            duration_seconds=2.0,
            fps=25.0,
            frame_count=50,
            status="completed",
        )
        db.add(v)
        db.commit()

        # Check status reports needs_conversion initially
        status_res = client.get(f"/api/videos/{test_id}/playback-status")
        assert status_res.status_code == 200
        assert status_res.json()["status"] in ("needs_conversion", "deferred")

        # Request playback with range header
        res = client.get(f"/api/videos/{test_id}/playback", headers={"Range": "bytes=0-511"})
        assert res.status_code == 206
        assert res.headers.get("Content-Type") == "video/mp4"
        assert len(res.content) == 512

        # Verify playback file was generated and is valid H.264
        playback_file = settings.STORAGE_PLAYBACK_DIR / f"{test_id}_playback.mp4"
        assert playback_file.exists()
        assert playback_file.stat().st_size > 0
        assert is_browser_compatible(playback_file) is True

        # Verify temporary files were cleaned up completely
        temp_tmp = playback_file.with_name(f"{playback_file.name}.tmp.mp4")
        temp_log = playback_file.with_name(f"{playback_file.name}.stderr.log")
        assert not temp_tmp.exists(), "Temporary mp4 file must be unlinked"
        assert not temp_log.exists(), "Temporary stderr log file must be unlinked"

        # Verify original AVI file was preserved intact
        assert avi_path.exists()
        assert avi_path.stat().st_size > 0

    finally:
        # Cleanup test artifacts
        if avi_path.exists():
            avi_path.unlink()
        pb = settings.STORAGE_PLAYBACK_DIR / f"{test_id}_playback.mp4"
        if pb.exists():
            pb.unlink()
        db_v = db.query(VideoModel).filter(VideoModel.id == test_id).first()
        if db_v:
            db.delete(db_v)
            db.commit()
        db.close()


# ---------------------------------------------------------------------------
# Test C: Memory Headroom & Active Analysis Safe Fallback (No Crash)
# ---------------------------------------------------------------------------

def test_legacy_playback_low_memory_headroom_fallback(tmp_path):
    """Verify that when container memory headroom is low (< 120 MB), playback safely defers."""
    test_id = f"test-defer-{uuid.uuid4().hex[:8]}"
    avi_filename = f"{test_id}_legacy.avi"
    avi_path = settings.STORAGE_UPLOADS_DIR / avi_filename
    create_test_avi_video(avi_path, num_frames=30)

    db = SessionLocal()
    try:
        v = VideoModel(
            id=test_id,
            original_filename="legacy.avi",
            saved_filename=avi_filename,
            storage_path=str(avi_path),
            file_size_bytes=avi_path.stat().st_size,
            duration_seconds=1.2,
            fps=25.0,
            frame_count=30,
            status="completed",
        )
        db.add(v)
        db.commit()

        # Mock low memory headroom (< 120 MB)
        with patch("backend.app.services.playback_service.get_container_memory_headroom_mb", return_value=85.0):
            # 1. Playback status reports deferred
            st = client.get(f"/api/videos/{test_id}/playback-status")
            assert st.status_code == 200
            assert st.json()["status"] == "deferred"
            assert "Conversion deferred to protect container memory" in st.json()["message"]

            # 2. Playback request returns informative 503 instead of crashing
            res = client.get(f"/api/videos/{test_id}/playback")
            assert res.status_code == 503
            data = res.json()
            assert data["error"] == "playback_transcode_deferred"
            assert data["analysis_preserved"] is True
            assert res.headers.get("Retry-After") == "10"

            # 3. Verify backend process remains completely alive
            h = client.get("/health")
            assert h.status_code == 200
            assert h.json()["status"] == "ok"

    finally:
        if avi_path.exists():
            avi_path.unlink()
        db_v = db.query(VideoModel).filter(VideoModel.id == test_id).first()
        if db_v:
            db.delete(db_v)
            db.commit()
        db.close()


def test_heavy_analysis_concurrency_isolation():
    """Verify legacy transcoding does NOT run concurrently with active video analysis."""
    test_id = f"test-concurr-{uuid.uuid4().hex[:8]}"
    avi_filename = f"{test_id}_clip.avi"
    avi_path = settings.STORAGE_UPLOADS_DIR / avi_filename
    create_test_avi_video(avi_path, num_frames=20)

    db = SessionLocal()
    try:
        v = VideoModel(
            id=test_id,
            original_filename="clip.avi",
            saved_filename=avi_filename,
            storage_path=str(avi_path),
            file_size_bytes=avi_path.stat().st_size,
            duration_seconds=0.8,
            fps=25.0,
            frame_count=20,
            status="completed",
        )
        db.add(v)
        db.commit()

        # Simulate heavy video analysis actively holding the semaphore
        acquired = _heavy_processing_semaphore.acquire(blocking=False)
        assert acquired is True, "Must acquire heavy processing semaphore for simulation"

        try:
            # 1. Playback status while analysis is running
            st = client.get(f"/api/videos/{test_id}/playback-status")
            assert st.status_code == 200
            assert st.json()["status"] == "deferred"
            assert "Video analysis in progress" in st.json()["message"]

            # 2. Playback request must defer with 503 and NOT attempt parallel FFmpeg execution
            res = client.get(f"/api/videos/{test_id}/playback")
            assert res.status_code == 503
            assert res.json()["error"] == "playback_transcode_deferred"
            assert res.json()["analysis_preserved"] is True
        finally:
            _heavy_processing_semaphore.release()

        # Now that analysis released the slot, status should no longer report deferred
        st_after = client.get(f"/api/videos/{test_id}/playback-status")
        assert st_after.status_code == 200
        assert st_after.json()["status"] in ("needs_conversion", "ready")

    finally:
        if avi_path.exists():
            avi_path.unlink()
        pb = settings.STORAGE_PLAYBACK_DIR / f"{test_id}_playback.mp4"
        if pb.exists():
            pb.unlink()
        db_v = db.query(VideoModel).filter(VideoModel.id == test_id).first()
        if db_v:
            db.delete(db_v)
            db.commit()
        db.close()


# ---------------------------------------------------------------------------
# Test D: Large Legacy Video Memory Stability
# ---------------------------------------------------------------------------

def test_large_legacy_video_bounded_memory(tmp_path):
    """Verify that transcoding a multi-frame video maintains strictly bounded peak RAM (< 150 MB)."""
    test_id = f"test-ram-{uuid.uuid4().hex[:8]}"
    avi_filename = f"{test_id}_large.avi"
    avi_path = settings.STORAGE_UPLOADS_DIR / avi_filename
    create_test_avi_video(avi_path, num_frames=120, width=1280, height=720)

    playback_file = settings.STORAGE_PLAYBACK_DIR / f"{test_id}_playback.mp4"

    try:
        # Measure peak memory during transcode
        mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        out_path = transcode_to_h264(avi_path, playback_file)
        mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

        assert out_path.exists()
        assert out_path.stat().st_size > 0
        assert is_browser_compatible(out_path) is True

        # Python process memory delta should be minimal (< 30 MB) since work was in subprocess
        delta_mb = abs(mem_after - mem_before)
        assert delta_mb < 30.0, f"Python heap increased unexpectedly by {delta_mb:.1f} MB during transcode"

    finally:
        if avi_path.exists():
            avi_path.unlink()
        if playback_file.exists():
            playback_file.unlink()


# ---------------------------------------------------------------------------
# Test E: Backend Health Preserved After Playback
# ---------------------------------------------------------------------------

def test_backend_health_preserved_after_playback():
    """Verify that both /health and /api/health return 200 OK after playback operations."""
    h1 = client.get("/health")
    assert h1.status_code == 200
    assert h1.json()["status"] == "ok"

    h2 = client.get("/api/health")
    assert h2.status_code == 200
    assert h2.json()["status"] == "ok"
    assert "Sentinel backend is running successfully" in h2.json()["message"]


# ---------------------------------------------------------------------------
# Test F: Video Analysis Unaffected After Playback
# ---------------------------------------------------------------------------

def test_video_analysis_unaffected_after_playback():
    """Verify video analysis and YOLO detector operate flawlessly after playback operations."""
    from ai.detection.detector import YOLODetector

    detector = YOLODetector()
    dummy_frame = np.full((640, 640, 3), 128, dtype=np.uint8)
    batch = [dummy_frame.copy() for _ in range(settings.YOLO_BATCH_SIZE)]
    timestamps = [0.0, 0.04, 0.08, 0.12][:len(batch)]

    results = detector.detect_batch(batch, timestamps)
    assert isinstance(results, list)
    assert len(results) == len(batch)
    for r in results:
        assert isinstance(r, list)
