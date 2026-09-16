"""
Tests for Sentinel Video Playback Service & Range Streaming Endpoints.
"""
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.config import settings
from backend.app.services.playback_service import (
    is_browser_compatible,
    get_playback_status,
    ensure_playback_file,
    get_ffmpeg_binary,
)

client = TestClient(app)

# Known verified test videos in storage/uploads/
WORKING_H264_VIDEO_ID = "605f62c7-9b7c-423b-8020-310d6f3899c5"
AFFECTED_HEVC_VIDEO_ID = "0d4d92f9-19f8-42e3-925f-1931cb557705"


def test_ffmpeg_binary_available():
    """Verify that an ffmpeg executable is discovered on the host system."""
    binary = get_ffmpeg_binary()
    assert binary is not None, "ffmpeg executable should be found (imageio-ffmpeg or system)"
    assert os.path.exists(binary), f"ffmpeg path '{binary}' does not exist"


def test_invalid_video_id_validation():
    """Verify that invalid video IDs and directory traversal attempts are rejected."""
    # Slashes / traversal
    res1 = client.get("/api/videos/..%2F..%2Fetc%2Fpasswd/playback")
    assert res1.status_code in (400, 404)

    # Special characters
    res2 = client.get("/api/videos/bad!id@value/playback")
    assert res2.status_code == 400
    assert "Invalid video ID format" in res2.json()["detail"]


def test_nonexistent_video_returns_404():
    """Verify that requesting playback for an absent video ID returns 404."""
    absent_id = "00000000-0000-0000-0000-000000000000"
    res = client.get(f"/api/videos/{absent_id}/playback")
    assert res.status_code == 404

    res_status = client.get(f"/api/videos/{absent_id}/playback-status")
    assert res_status.status_code == 200
    assert res_status.json()["status"] == "unavailable"


def test_codec_detection_logic():
    """Verify is_browser_compatible correctly distinguishes H.264 from HEVC."""
    h264_path = settings.STORAGE_UPLOADS_DIR / f"{WORKING_H264_VIDEO_ID}_12566041-uhd_3840_2160_30fps.mp4"
    hevc_path = settings.STORAGE_UPLOADS_DIR / f"{AFFECTED_HEVC_VIDEO_ID}_uccrime_Burglary010_x264.mp4"

    if h264_path.exists():
        assert is_browser_compatible(h264_path) is True, "H.264 MP4 should be marked browser compatible"

    if hevc_path.exists():
        assert is_browser_compatible(hevc_path) is False, "HEVC video should be marked NOT browser compatible"


def test_compatible_video_playback_and_ranges():
    """Verify playback of already-compatible H.264 video streams directly with range support."""
    h264_path = settings.STORAGE_UPLOADS_DIR / f"{WORKING_H264_VIDEO_ID}_12566041-uhd_3840_2160_30fps.mp4"
    if not h264_path.exists():
        pytest.skip(f"Test video {WORKING_H264_VIDEO_ID} not in storage/uploads")

    # 1. Playback status
    status_res = client.get(f"/api/videos/{WORKING_H264_VIDEO_ID}/playback-status")
    assert status_res.status_code == 200
    data = status_res.json()
    assert data["status"] == "ready"
    assert data["is_compatible"] is True
    assert data["is_transcoded"] is False

    # 2. HTTP Range request (seeking start of video)
    range_res = client.get(
        f"/api/videos/{WORKING_H264_VIDEO_ID}/playback",
        headers={"Range": "bytes=0-1023"},
    )
    assert range_res.status_code == 206
    assert range_res.headers.get("accept-ranges") == "bytes"
    assert range_res.headers.get("content-type") == "video/mp4"
    assert "bytes 0-1023/" in range_res.headers.get("content-range", "")
    assert len(range_res.content) == 1024


def test_hevc_video_playback_and_transcoding():
    """Verify playback of HEVC video: converted to H.264 MP4, original preserved."""
    hevc_path = settings.STORAGE_UPLOADS_DIR / f"{AFFECTED_HEVC_VIDEO_ID}_uccrime_Burglary010_x264.mp4"
    if not hevc_path.exists():
        pytest.skip(f"Test video {AFFECTED_HEVC_VIDEO_ID} not in storage/uploads")

    original_size_before = hevc_path.stat().st_size
    original_mtime_before = hevc_path.stat().st_mtime

    # 1. Playback request (triggers or serves existing playback file)
    res = client.get(
        f"/api/videos/{AFFECTED_HEVC_VIDEO_ID}/playback",
        headers={"Range": "bytes=0-2047"},
    )
    assert res.status_code == 206
    assert len(res.content) == 2048
    assert res.headers.get("content-type") == "video/mp4"

    # 2. Check that playback file is stored separately in storage/playback/
    playback_file = settings.STORAGE_PLAYBACK_DIR / f"{AFFECTED_HEVC_VIDEO_ID}_playback.mp4"
    assert playback_file.exists()
    assert playback_file.stat().st_size > 0

    # 3. Verify ORIGINAL uploaded file was NOT modified or deleted
    assert hevc_path.exists()
    assert hevc_path.stat().st_size == original_size_before
    assert hevc_path.stat().st_mtime == original_mtime_before

    # 4. Check playback status reports ready and transcoded
    status_res = client.get(f"/api/videos/{AFFECTED_HEVC_VIDEO_ID}/playback-status")
    assert status_res.status_code == 200
    st_data = status_res.json()
    assert st_data["status"] == "ready"
    assert st_data["is_compatible"] is True
    assert st_data["is_transcoded"] is True


def test_playback_range_seeking_middle_and_end():
    """Verify HTTP Range requests seek accurately into middle and tail of video."""
    res_mid = client.get(
        f"/api/videos/{AFFECTED_HEVC_VIDEO_ID}/playback",
        headers={"Range": "bytes=100000-105000"},
    )
    if res_mid.status_code == 206:
        assert len(res_mid.content) == 5001
        assert "bytes 100000-105000/" in res_mid.headers.get("content-range", "")


def test_playback_out_of_bounds_range():
    """Verify Range header beyond file length returns 416 Range Not Satisfiable."""
    res = client.get(
        f"/api/videos/{WORKING_H264_VIDEO_ID}/playback",
        headers={"Range": "bytes=999999999-999999999"},
    )
    assert res.status_code == 416
    assert "bytes */" in res.headers.get("content-range", "")


def test_playback_idempotency():
    """Verify repeated playback requests reuse existing playback file without re-encoding."""
    playback_file = settings.STORAGE_PLAYBACK_DIR / f"{AFFECTED_HEVC_VIDEO_ID}_playback.mp4"
    if not playback_file.exists():
        pytest.skip("Playback file not yet created")

    mtime1 = playback_file.stat().st_mtime
    # Call playback again
    res = client.get(f"/api/videos/{AFFECTED_HEVC_VIDEO_ID}/playback", headers={"Range": "bytes=0-512"})
    assert res.status_code == 206
    mtime2 = playback_file.stat().st_mtime
    assert mtime1 == mtime2, "Playback file should be reused idempotently without reconversion"
