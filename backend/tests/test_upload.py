"""
Tests for Sentinel Video Upload & Streaming API
"""
import io
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

# Synthesize a minimal valid MP4 file byte stream
# Contains valid ftyp atom box header
SAMPLE_MP4_HEADER = (
    b"\x00\x00\x00\x20"         # 32 bytes box length
    b"ftyp"                     # ftyp box type
    b"isom"                     # major brand
    b"\x00\x00\x02\x00"         # minor version
    b"isomiso2mp41"             # compatible brands
    b"\x00\x00\x00\x08free"     # free atom (8 bytes)
    b"\x00\x00\x00\x08mdat"     # mdat atom (8 bytes)
)


def test_health_check():
    """Verify GET /api/health."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "Sentinel Backend"


def test_upload_valid_video():
    """Verify POST /api/videos/upload with valid video bytes."""
    file_bytes = SAMPLE_MP4_HEADER + (b"\x00" * 1024)  # ~1 KB video payload
    files = {
        "file": ("surveillance_sample.mp4", io.BytesIO(file_bytes), "video/mp4")
    }
    response = client.post("/api/videos/upload", files=files)
    assert response.status_code == 201, f"Expected 201, got {response.status_code}: {response.text}"
    data = response.json()
    assert "video_id" in data
    assert data["filename"] == "surveillance_sample.mp4"
    assert data["status"] == "uploaded"
    assert "stream_url" in data

    video_id = data["video_id"]

    # Verify file physically stored in storage/uploads/
    uploads_dir = Path("storage/uploads")
    matching_files = list(uploads_dir.glob(f"{video_id}_*"))
    assert len(matching_files) > 0, "Uploaded video file not found in storage/uploads/"

    # Test metadata info endpoint
    info_resp = client.get(f"/api/videos/{video_id}")
    assert info_resp.status_code == 200
    info_data = info_resp.json()
    assert info_data["video_id"] == video_id

    # Test video streaming endpoint
    stream_resp = client.get(f"/api/videos/{video_id}/stream")
    assert stream_resp.status_code == 200
    assert len(stream_resp.content) == len(file_bytes)


def test_upload_empty_file():
    """Verify upload fails gracefully on empty file."""
    files = {
        "file": ("empty.mp4", io.BytesIO(b""), "video/mp4")
    }
    response = client.post("/api/videos/upload", files=files)
    assert response.status_code == 400
    assert "empty" in response.text.lower()


def test_upload_unsupported_format():
    """Verify upload rejects unsupported formats like .txt or .exe."""
    files = {
        "file": ("malicious.exe", io.BytesIO(b"MZ\x90\x00fake_executable"), "application/octet-stream")
    }
    response = client.post("/api/videos/upload", files=files)
    assert response.status_code == 400
    assert "unsupported" in response.text.lower()


def test_upload_path_traversal_sanitized():
    """Verify path traversal filenames are sanitized."""
    file_bytes = SAMPLE_MP4_HEADER + (b"\x00" * 512)
    files = {
        "file": ("../../etc/passwd.mp4", io.BytesIO(file_bytes), "video/mp4")
    }
    response = client.post("/api/videos/upload", files=files)
    assert response.status_code == 201
    data = response.json()
    video_id = data["video_id"]

    # Ensure saved file is inside storage/uploads/ and does not escape
    uploads_dir = Path("storage/uploads").resolve()
    for item in uploads_dir.glob(f"{video_id}_*"):
        assert item.resolve().parent == uploads_dir


if __name__ == "__main__":
    print("Running Sentinel Video Upload Tests...")
    test_health_check()
    print("[PASS] test_health_check")
    test_upload_valid_video()
    print("[PASS] test_upload_valid_video")
    test_upload_empty_file()
    print("[PASS] test_upload_empty_file")
    test_upload_unsupported_format()
    print("[PASS] test_upload_unsupported_format")
    test_upload_path_traversal_sanitized()
    print("[PASS] test_upload_path_traversal_sanitized")
    print("ALL TESTS PASSED SUCCESSFULLY!")
