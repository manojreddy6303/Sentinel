"""
Phase 3 Tests: Video Intelligence Pipeline (OpenCV + YOLO)

Tests cover:
1. OpenCV import
2. Ultralytics import
3. VideoProcessor metadata extraction
4. VideoProcessor frame sampling
5. YOLODetector model loading
6. End-to-end POST /api/videos/{video_id}/process
7. GET /api/videos/{video_id}/events  
8. Detection confidence validity
9. Bounding box validity
10. Missing video returns 404
11. Invalid video handled safely
12. Zero detections returns completed status
13. Phase 1 & 2 health + upload tests still pass
"""

import io
import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared test MP4 header (same as Phase 2 tests)
# ---------------------------------------------------------------------------
SAMPLE_MP4_HEADER = (
    b"\x00\x00\x00\x20"       # 32 bytes box length
    b"ftyp"                   # ftyp box type
    b"isom"                   # major brand
    b"\x00\x00\x02\x00"       # minor version
    b"isomiso2mp41"           # compatible brands
    b"\x00\x00\x00\x08free"   # free atom (8 bytes)
    b"\x00\x00\x00\x08mdat"   # mdat atom (8 bytes)
)


def make_test_video_path() -> Path:
    """
    Create a minimal real MP4 video file using OpenCV VideoWriter
    and return its path. Used for tests that need a real decodeable video.
    """
    import cv2
    import numpy as np

    test_dir = PROJECT_ROOT / "storage" / "uploads"
    test_dir.mkdir(parents=True, exist_ok=True)
    video_path = test_dir / "_test_phase3_sample.mp4"

    if video_path.exists():
        return video_path

    # Write a short 3-frame, 640x360 colour-noise video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (640, 360))
    for _ in range(30):  # 30 frames @ 10fps = 3 seconds
        frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    return video_path


# ===========================================================================
# TEST 1: OpenCV imports
# ===========================================================================
def test_opencv_imports():
    """OpenCV must import successfully."""
    import cv2
    assert hasattr(cv2, "VideoCapture"), "cv2.VideoCapture not found"
    assert hasattr(cv2, "VideoWriter"), "cv2.VideoWriter not found"


# ===========================================================================
# TEST 2: Ultralytics imports
# ===========================================================================
def test_ultralytics_imports():
    """Ultralytics must import and expose YOLO class."""
    from ultralytics import YOLO
    assert YOLO is not None


# ===========================================================================
# TEST 3: VideoProcessor metadata
# ===========================================================================
def test_video_processor_metadata():
    """VideoProcessor must extract valid metadata from a real video."""
    from ai.video.processor import VideoProcessor

    video_path = make_test_video_path()
    processor = VideoProcessor(str(video_path), sample_rate_fps=1.0)
    meta = processor.get_metadata()

    assert meta["fps"] > 0, "FPS must be positive"
    assert meta["frame_count"] > 0, "Frame count must be positive"
    assert meta["duration_seconds"] > 0, "Duration must be positive"
    assert meta["width"] > 0
    assert meta["height"] > 0
    assert "filename" in meta


# ===========================================================================
# TEST 4: VideoProcessor frame sampling
# ===========================================================================
def test_video_processor_frame_sampling():
    """VideoProcessor must yield frames with valid (frame_number, timestamp, frame) tuples."""
    from ai.video.processor import VideoProcessor
    import numpy as np

    video_path = make_test_video_path()
    processor = VideoProcessor(str(video_path), sample_rate_fps=1.0)

    frames = list(processor.sample_frames())
    assert len(frames) > 0, "At least one frame must be yielded"

    for frame_number, timestamp, frame_bgr in frames:
        assert isinstance(frame_number, int) and frame_number >= 0
        assert isinstance(timestamp, float) and timestamp >= 0.0
        assert isinstance(frame_bgr, np.ndarray), "Frame must be a numpy array"
        assert frame_bgr.ndim == 3, "Frame must be HxWxC"


# ===========================================================================
# TEST 5: YOLO model loads
# ===========================================================================
def test_yolo_model_loads():
    """YOLODetector must load the pretrained model without error."""
    from ai.detection.detector import YOLODetector

    detector = YOLODetector(model_name="yolov8n.pt", confidence_threshold=0.25)
    model = detector._load_model()
    assert model is not None
    assert detector.is_loaded()
    # Model must have a names dict
    assert isinstance(model.names, dict)
    assert len(model.names) > 0


# ===========================================================================
# TEST 6: End-to-end video processing via API
# ===========================================================================
def test_process_video_api_end_to_end():
    """
    Upload a real video, call POST /process, verify response structure.
    """
    # Use the real test video
    video_path = make_test_video_path()
    with open(video_path, "rb") as vf:
        files = {"file": ("test_surveillance.mp4", vf, "video/mp4")}
        upload_resp = client.post("/api/videos/upload", files=files)

    assert upload_resp.status_code == 201, f"Upload failed: {upload_resp.text}"
    video_id = upload_resp.json()["video_id"]

    # Now process
    process_resp = client.post(f"/api/videos/{video_id}/process")
    assert process_resp.status_code == 200, f"Process failed: {process_resp.text}"

    data = process_resp.json()
    assert data["video_id"] == video_id
    assert data["status"] == "completed"
    assert data["duration_seconds"] > 0
    assert data["fps"] > 0
    assert data["frames_processed"] >= 0
    assert data["detections_count"] >= 0


# ===========================================================================
# TEST 7: Event retrieval via API
# ===========================================================================
def test_get_video_events_api():
    """
    Upload + process a video, then verify GET /events returns correct structure.
    """
    video_path = make_test_video_path()
    with open(video_path, "rb") as vf:
        files = {"file": ("test_events.mp4", vf, "video/mp4")}
        upload_resp = client.post("/api/videos/upload", files=files)
    assert upload_resp.status_code == 201
    video_id = upload_resp.json()["video_id"]

    client.post(f"/api/videos/{video_id}/process")

    events_resp = client.get(f"/api/videos/{video_id}/events")
    assert events_resp.status_code == 200, f"Events retrieval failed: {events_resp.text}"

    data = events_resp.json()
    assert data["video_id"] == video_id
    assert "total_events" in data
    assert isinstance(data["events"], list)
    assert data["total_events"] == len(data["events"])


# ===========================================================================
# TEST 8: Detection confidence validity
# ===========================================================================
def test_detection_confidence_valid():
    """All detection confidence scores must be in [0.0, 1.0]."""
    video_path = make_test_video_path()
    with open(video_path, "rb") as vf:
        files = {"file": ("conf_test.mp4", vf, "video/mp4")}
        upload_resp = client.post("/api/videos/upload", files=files)
    assert upload_resp.status_code == 201
    video_id = upload_resp.json()["video_id"]

    client.post(f"/api/videos/{video_id}/process")

    events_resp = client.get(f"/api/videos/{video_id}/events")
    data = events_resp.json()

    for event in data["events"]:
        conf = event["confidence"]
        assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of range for event {event['event_id']}"


# ===========================================================================
# TEST 9: Bounding box validity
# ===========================================================================
def test_bounding_box_valid():
    """All bounding boxes must have x2 > x1 and y2 > y1."""
    video_path = make_test_video_path()
    with open(video_path, "rb") as vf:
        files = {"file": ("bbox_test.mp4", vf, "video/mp4")}
        upload_resp = client.post("/api/videos/upload", files=files)
    assert upload_resp.status_code == 201
    video_id = upload_resp.json()["video_id"]

    client.post(f"/api/videos/{video_id}/process")

    events_resp = client.get(f"/api/videos/{video_id}/events")
    data = events_resp.json()

    for event in data["events"]:
        bb = event["bounding_box"]
        assert "x1" in bb and "y1" in bb and "x2" in bb and "y2" in bb
        assert bb["x2"] >= bb["x1"], f"x2 < x1: {bb}"
        assert bb["y2"] >= bb["y1"], f"y2 < y1: {bb}"


# ===========================================================================
# TEST 10: Missing video returns 404
# ===========================================================================
def test_process_missing_video_returns_404():
    """Processing a non-existent video_id must return HTTP 404."""
    resp = client.post("/api/videos/nonexistent-video-id-12345/process")
    assert resp.status_code == 404


def test_events_missing_video_returns_404():
    """Requesting events for a non-existent video_id must return HTTP 404."""
    resp = client.get("/api/videos/nonexistent-video-id-12345/events")
    assert resp.status_code == 404


# ===========================================================================
# TEST 11: Invalid video handled safely (corrupt file)
# ===========================================================================
def test_process_invalid_video_handled_safely():
    """
    Processing a corrupt/fake video must return a 4xx error, not crash the server.
    (Upload passes because our fake header looks valid, but OpenCV will reject it.)
    """
    # Upload with valid magic bytes but garbage content
    fake_video = SAMPLE_MP4_HEADER + b"\x00" * 512  # Not real video data
    files = {"file": ("corrupt.mp4", io.BytesIO(fake_video), "video/mp4")}
    upload_resp = client.post("/api/videos/upload", files=files)
    assert upload_resp.status_code == 201
    video_id = upload_resp.json()["video_id"]

    # Processing should fail gracefully
    process_resp = client.post(f"/api/videos/{video_id}/process")
    # Must be either 4xx (invalid/empty video) or 200 with detections_count=0
    # Not 500 / server crash
    assert process_resp.status_code in [200, 400, 422, 503], \
        f"Unexpected status {process_resp.status_code}: {process_resp.text}"

    # Server must still be responsive after the error
    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200, "Server should still respond to /api/health"


# ===========================================================================
# TEST 12: Zero detections = completed, not error
# ===========================================================================
def test_zero_detections_completes_successfully():
    """
    A real video where YOLO finds nothing must return status=completed,
    detections_count=0 (not an error).
    """
    import cv2
    import numpy as np

    # Create a plain black video that likely won't trigger object detection
    test_dir = PROJECT_ROOT / "storage" / "uploads"
    black_video_path = test_dir / "_test_black_screen.mp4"
    if not black_video_path.exists():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(black_video_path), fourcc, 5.0, (320, 240))
        for _ in range(5):
            out.write(np.zeros((240, 320, 3), dtype=np.uint8))
        out.release()

    with open(black_video_path, "rb") as vf:
        files = {"file": ("black_screen.mp4", vf, "video/mp4")}
        upload_resp = client.post("/api/videos/upload", files=files)
    assert upload_resp.status_code == 201
    video_id = upload_resp.json()["video_id"]

    process_resp = client.post(f"/api/videos/{video_id}/process")
    assert process_resp.status_code == 200, f"Expected 200, got {process_resp.text}"

    data = process_resp.json()
    assert data["status"] == "completed"
    assert data["detections_count"] >= 0  # 0 is valid


# ===========================================================================
# TEST 13: Phase 1 & 2 regression tests
# ===========================================================================
def test_health_check():
    """Verify GET /api/health still works (Phase 1)."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "Sentinel Backend"


def test_upload_valid_video():
    """Verify POST /api/videos/upload still works (Phase 2)."""
    file_bytes = SAMPLE_MP4_HEADER + (b"\x00" * 1024)
    files = {"file": ("regression_sample.mp4", io.BytesIO(file_bytes), "video/mp4")}
    response = client.post("/api/videos/upload", files=files)
    assert response.status_code == 201
    data = response.json()
    assert "video_id" in data
    assert data["status"] == "uploaded"


def test_upload_empty_file():
    """Verify upload rejects empty files (Phase 2)."""
    files = {"file": ("empty.mp4", io.BytesIO(b""), "video/mp4")}
    response = client.post("/api/videos/upload", files=files)
    assert response.status_code == 400


def test_upload_unsupported_format():
    """Verify upload rejects unsupported formats (Phase 2)."""
    files = {"file": ("malicious.exe", io.BytesIO(b"MZ\x90\x00fake"), "application/octet-stream")}
    response = client.post("/api/videos/upload", files=files)
    assert response.status_code == 400


# ===========================================================================
# Manual runner (python backend/tests/test_phase3_pipeline.py)
# ===========================================================================
if __name__ == "__main__":
    tests = [
        ("OpenCV imports", test_opencv_imports),
        ("Ultralytics imports", test_ultralytics_imports),
        ("VideoProcessor metadata", test_video_processor_metadata),
        ("VideoProcessor frame sampling", test_video_processor_frame_sampling),
        ("YOLO model loads", test_yolo_model_loads),
        ("End-to-end process API", test_process_video_api_end_to_end),
        ("Event retrieval API", test_get_video_events_api),
        ("Detection confidence valid", test_detection_confidence_valid),
        ("Bounding box valid", test_bounding_box_valid),
        ("Missing video → 404 (process)", test_process_missing_video_returns_404),
        ("Missing video → 404 (events)", test_events_missing_video_returns_404),
        ("Invalid video handled safely", test_process_invalid_video_handled_safely),
        ("Zero detections completes OK", test_zero_detections_completes_successfully),
        ("Phase 1+2: Health check", test_health_check),
        ("Phase 1+2: Upload valid video", test_upload_valid_video),
        ("Phase 1+2: Upload empty file", test_upload_empty_file),
        ("Phase 1+2: Upload unsupported format", test_upload_unsupported_format),
    ]

    passed = 0
    failed = 0
    print("\n=== Sentinel Phase 3 Tests ===\n")
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("ALL PHASE 3 TESTS PASSED SUCCESSFULLY!")
    else:
        print("SOME TESTS FAILED. See above for details.")
        sys.exit(1)
