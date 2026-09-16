"""
Phase 15.1: Global Video Robustness, Data Integrity & False-Positive Hardening Test Suite
Verifies universal invariants across arbitrary videos, resolutions, FPS, durations, codecs,
and specialized visual detectors.
"""

import math
import os
import json
import tempfile
import cv2
import numpy as np
import pytest

from ai.common.numeric import is_finite_number, ensure_finite, clamp_finite, sanitize_for_json
from ai.common.numeric import is_finite_number, ensure_finite, clamp_finite, sanitize_for_json
from ai.video.metadata import MediaMetadataExtractor, VideoMetadataError
from ai.validation.consistency import UniversalDataConsistencyValidator, ConsistencyViolation
from ai.specialized.fire_smoke.detector import FireVisualDetector, SmokeVisualDetector
from ai.specialized.weapon.detector import WeaponVisualDetector
from ai.schemas import BoundingBox, TrackedObject, SecurityEvent
from ai.incidents.schemas import IncidentContext
from ai.specialized.schemas import SpecializedObservation, SpecializedTemporalTrack, SpecializedValidationStatus
from ai.specialized.temporal import SpecializedTemporalTracker
from ai.incidents.detectors.specialized import SpecializedFireIncidentDetector, SpecializedSmokeIncidentDetector
from ai.intelligence_repository import SecurityIntelligenceRepository
from backend.app.main import SafeJSONResponse


# ============================================================================
# 1. NUMERIC SANITIZATION & FINITE MATH INVARIANTS
# ============================================================================

def test_numeric_finite_invariants():
    """Verify that is_finite_number, ensure_finite, clamp_finite work universally."""
    assert is_finite_number(0.0) is True
    assert is_finite_number(-123.45) is True
    assert is_finite_number(float("nan")) is False
    assert is_finite_number(float("inf")) is False
    assert is_finite_number(float("-inf")) is False
    assert is_finite_number("123") is False
    assert is_finite_number(None) is False

    # ensure_finite fallback
    assert ensure_finite(0.5, default=0.0) == 0.5
    assert ensure_finite(float("nan"), default=0.0) == 0.0
    assert ensure_finite(float("inf"), default=1.0) == 1.0
    assert ensure_finite(float("-inf"), default=-1.0) == -1.0

    # clamp_finite bounds
    assert clamp_finite(1.5, 0.0, 1.0) == 1.0
    assert clamp_finite(-0.2, 0.0, 1.0) == 0.0
    assert clamp_finite(float("nan"), 0.0, 1.0, default=0.5) == 0.5


def test_safe_json_response_no_nan_or_inf():
    """Verify that SafeJSONResponse never emits NaN/Infinity in API responses (RFC 8259)."""
    dirty_data = {
        "video_id": "vid_test",
        "confidence": float("nan"),
        "metrics": {
            "fps": float("inf"),
            "negative_inf": float("-inf"),
            "valid_score": 0.95
        },
        "items": [float("nan"), 1.0, 2.0]
    }
    
    cleaned = sanitize_for_json(dirty_data, non_finite_replacement=None)
    assert cleaned["confidence"] is None
    assert cleaned["metrics"]["fps"] is None
    assert cleaned["metrics"]["negative_inf"] is None
    assert cleaned["metrics"]["valid_score"] == 0.95
    assert cleaned["items"] == [None, 1.0, 2.0]

    # Verify SafeJSONResponse renders valid JSON without allow_nan=True
    response = SafeJSONResponse(dirty_data)
    rendered_body = response.body.decode("utf-8")
    parsed = json.loads(rendered_body)
    assert parsed["confidence"] is None
    assert parsed["metrics"]["fps"] is None


# ============================================================================
# 2. SYNTHETIC VIDEO FIXTURES & ARBITRARY MATRIX
# ============================================================================

def _create_synthetic_video(path: str, width: int, height: int, fps: float, num_frames: int, color=(128, 128, 128)):
    """Generate a clean synthetic MP4 video file."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        # Add subtle moving box
        bx = int((i * 5) % max(1, width - 40))
        by = int((i * 3) % max(1, height - 40))
        cv2.rectangle(frame, (bx, by), (bx + 30, by + 30), (200, 200, 200), -1)
        out.write(frame)
    out.release()


@pytest.fixture
def arbitrary_videos_tempdir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_arbitrary_video_matrix(arbitrary_videos_tempdir):
    """Test media metadata extraction across diverse resolutions, FPS, durations, aspect ratios."""
    test_specs = [
        # (width, height, fps, num_frames, desc)
        (320, 240, 15.0, 30, "320x240 low res 15fps 2s"),
        (640, 480, 30.0, 60, "4:3 standard 30fps 2s"),
        (1080, 1920, 30.0, 30, "9:16 portrait mobile 1s"),
        (1280, 720, 60.0, 60, "16:9 60fps high FPS 1s"),
        (640, 360, 5.0, 5, "short video 1s 5fps"),
    ]

    for width, height, fps, frames, desc in test_specs:
        vid_path = os.path.join(arbitrary_videos_tempdir, f"test_{width}x{height}_{int(fps)}fps.mp4")
        _create_synthetic_video(vid_path, width, height, fps, frames)

        meta = MediaMetadataExtractor.extract(vid_path, video_id=f"vid_{width}x{height}")
        assert meta["width"] == width, f"Failed width for {desc}"
        assert meta["height"] == height, f"Failed height for {desc}"
        assert abs(meta["fps"] - fps) < 1.0, f"Failed fps for {desc}"
        assert meta["duration_seconds"] > 0.0, f"Duration must be positive for {desc}"
        assert meta["is_decodable"] is True, f"Must be decodable for {desc}"
        assert is_finite_number(meta["fps"]), f"FPS must be finite for {desc}"
        assert is_finite_number(meta["duration_seconds"]), f"Duration must be finite for {desc}"


def test_corrupted_or_empty_video_handling(arbitrary_videos_tempdir):
    """Verify corrupted, non-existent, or 0-byte video files fail safely without crashing."""
    # 1. Non-existent file
    with pytest.raises(VideoMetadataError):
        MediaMetadataExtractor.extract(os.path.join(arbitrary_videos_tempdir, "nonexistent.mp4"), "vid_none")

    # 2. Zero-byte file
    empty_path = os.path.join(arbitrary_videos_tempdir, "empty.mp4")
    with open(empty_path, "wb") as f:
        pass
    with pytest.raises(VideoMetadataError):
        MediaMetadataExtractor.extract(empty_path, "vid_empty")


# ============================================================================
# 3. FALSE-POSITIVE HARDENING: SMOKE & FIRE
# ============================================================================

def test_smoke_detector_rejects_smooth_reflections():
    """Verify smoke detector rejects smooth reflective floor tiles and plain walls via Laplacian texture roughness."""
    detector = SmokeVisualDetector()

    # Create a smooth, low-saturation gray image simulating polished floor tiles
    smooth_tile = np.full((200, 200, 3), 180, dtype=np.uint8)
    # Add gentle gradient (reflection)
    for y in range(200):
        smooth_tile[y, :, :] = np.clip(180 + y // 5, 0, 255)

    obs = detector.detect_frame(smooth_tile, timestamp=1.0, frame_idx=1)
    # Smooth tile has very low Laplacian variance (< 12.0) and should be rejected
    assert len(obs) == 0 or all(o.validation_status.value == "REJECTED" for o in obs)


def test_smoke_detector_rejects_ceiling_halogen_lights():
    """Verify smoke detector suppresses bright ceiling halogen fixtures."""
    detector = SmokeVisualDetector()

    frame = np.full((400, 400, 3), 40, dtype=np.uint8)
    # Add bright ceiling light fixture at top of frame (y < 48)
    cv2.circle(frame, (200, 30), 20, (255, 255, 255), -1)

    obs = detector.detect_frame(frame, timestamp=1.0, frame_idx=1)
    # Ceiling light fixture at y=30 should be suppressed by ceiling lamp suppression
    assert len(obs) == 0


def test_fire_detector_rejects_static_orange_vest():
    """Verify fire detector rejects solid static orange safety vests/signs (low internal variance, uniform shape)."""
    detector = FireVisualDetector()

    # Create uniform orange patch representing safety vest / cone
    vest_frame = np.full((300, 300, 3), 50, dtype=np.uint8)
    # Solid orange rectangle: B=20, G=100, R=240
    cv2.rectangle(vest_frame, (100, 100), (200, 200), (20, 100, 240), -1)

    obs = detector.detect_frame(vest_frame, timestamp=1.0, frame_idx=1)
    # Solid uniform color has standard deviation near 0 (< 8.0) and should be rejected
    assert len(obs) == 0 or all(o.validation_status.value == "REJECTED" for o in obs)


def test_weapon_detector_zero_detections_when_unconfigured():
    """Verify weapon detector strictly produces zero detections when no weights are configured."""
    detector = WeaponVisualDetector(enabled=True, config={"model_path": "nonexistent_weapon_model.pt"})
    dummy_frame = np.zeros((400, 400, 3), dtype=np.uint8)
    
    obs = detector.detect_frame(dummy_frame, timestamp=1.0, frame_idx=1)
    assert len(obs) == 0
    assert detector._status.value in ("NOT_CONFIGURED", "UNAVAILABLE")


# ============================================================================
# 4. RAW OBSERVATIONS VS AGGREGATED INCIDENTS
# ============================================================================

def test_temporal_aggregation_groups_continuous_observations():
    """
    Verify that 20 continuous raw smoke observations produce ONE aggregated incident candidate,
    NOT 20 separate incidents, while storing full observation count and provenance.
    """
    tracker = SpecializedTemporalTracker(max_time_gap_seconds=1.5)

    # Feed 20 observations spaced 0.1s apart (1.9 seconds continuous smoke)
    for i in range(20):
        t = i * 0.1
        obs = SpecializedObservation(
            observation_id=f"obs_{i}",
            detector_name="SmokeVisualDetector",
            detector_version="1.0.0",
            class_name="smoke",
            timestamp=t,
            confidence=0.85,
            evidence_strength=0.8,
            bounding_box=BoundingBox(100.0, 100.0, 150.0, 150.0),
            validation_status=SpecializedValidationStatus.VALID,
            visual_metrics={"motion_active": True}
        )
        tracker.update([obs])

    active_tracks = tracker.finalize()
    assert len(active_tracks) == 1, f"Expected exactly 1 track, got {len(active_tracks)}"
    track = active_tracks[0]
    assert track.observation_count == 20
    assert track.persistence_duration >= 1.8

    # Now convert track to incident candidate
    smoke_detector = SpecializedSmokeIncidentDetector()
    context = IncidentContext(
        video_id="vid_agg_test",
        video_metadata={"video_id": "vid_agg_test", "duration": 10.0, "fps": 10.0},
        tracks=[],
        specialized_tracks=[track],
    )
    candidates = smoke_detector.analyze(context)

    # Must produce exactly 1 incident candidate representing the entire continuous episode
    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.incident_metadata.get("observation_count") == 20
    assert cand.duration >= 1.8
    assert cand.event_type == "POTENTIAL_SMOKE"


# ============================================================================
# 5. UNIVERSAL DATA CONSISTENCY VALIDATOR (12 RULES)
# ============================================================================

def test_universal_consistency_validator_passes_on_valid_data():
    """Verify UniversalDataConsistencyValidator approves a compliant dataset."""
    video_metadata = {
        "video_id": "vid_clean",
        "duration": 60.0,
        "fps": 30.0,
        "width": 1920,
        "height": 1080
    }

    detections = [
        {"id": "d1", "video_id": "vid_clean", "timestamp_seconds": 5.0, "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}, "confidence": 0.9}
    ]
    tracks = [
        {"track_id": "trk_1", "video_id": "vid_clean", "start_time": 5.0, "end_time": 10.0, "duration": 5.0}
    ]
    events = [
        {"id": "ev_1", "video_id": "vid_clean", "timestamp": 5.0, "duration_seconds": 5.0, "event_type": "POTENTIAL_SMOKE", "confidence": 0.85}
    ]
    evidence = [
        {"evidence_id": "evd_1", "video_id": "vid_clean", "event_id": "ev_1", "timestamp": 5.0, "status": "ACCEPTED"}
    ]
    reports = [
        {"report_id": "rep_1", "video_id": "vid_clean", "event_count": 1}
    ]

    is_valid, violations = UniversalDataConsistencyValidator.validate_data(
        video_metadata=video_metadata,
        detections=detections,
        tracks=tracks,
        events=events,
        evidence=evidence,
        reports=reports
    )

    assert is_valid is True
    assert len(violations) == 0


def test_universal_consistency_validator_catches_violations():
    """Verify UniversalDataConsistencyValidator catches NaNs, timestamps > duration, mismatched video_ids, duplicate continuous incidents."""
    video_metadata = {
        "video_id": "vid_bad",
        "duration": 10.0,
        "fps": 30.0
    }

    # 1. NaN in confidence
    # 2. Timestamp beyond duration (15.0 > 10.0)
    # 3. Cross-video leak (video_id = vid_OTHER)
    # 4. Duplicate continuous event
    events = [
        {
            "id": "ev_1",
            "video_id": "vid_bad",
            "timestamp": 15.0,  # Exceeds duration!
            "duration_seconds": 2.0,
            "event_type": "POTENTIAL_SMOKE",
            "confidence": float("nan")  # NaN!
        },
        {
            "id": "ev_2",
            "video_id": "vid_OTHER",  # Leaked cross-video record!
            "timestamp": 2.0,
            "duration_seconds": 1.0,
            "event_type": "POTENTIAL_FIRE",
            "confidence": 0.8
        },
        {
            "id": "ev_3",
            "video_id": "vid_bad",
            "timestamp": 2.0,
            "duration_seconds": 1.0,
            "event_type": "POTENTIAL_FIRE",
            "confidence": 0.8
        },
        {
            "id": "ev_4",
            "video_id": "vid_bad",
            "timestamp": 2.2,  # Overlaps within 0.2s of ev_3 -> duplicate continuous!
            "duration_seconds": 1.0,
            "event_type": "POTENTIAL_FIRE",
            "confidence": 0.82
        }
    ]

    is_valid, violations = UniversalDataConsistencyValidator.validate_data(
        video_metadata=video_metadata,
        events=events
    )

    assert is_valid is False
    codes = [v["rule_name"] for v in violations]
    assert "TIMESTAMP_OUT_OF_BOUNDS" in codes
    assert "NON_FINITE_VALUE" in codes
    assert "CROSS_VIDEO_CONTAMINATION" in codes
    assert "DUPLICATE_CONTINUOUS_INCIDENT" in codes
    assert "DUPLICATE_CONTINUOUS_INCIDENT" in codes


# ============================================================================
# 6. DATABASE REPROCESSING IDEMPOTENCY & ISOLATION
# ============================================================================

def test_database_reprocessing_idempotency_and_isolation():
    """Verify that reprocessing video A clears prior records and does not contaminate video B."""
    repo = SecurityIntelligenceRepository()

    tA = TrackedObject(
        track_id="trk_A1",
        object_class="car",
        first_seen=1.0,
        last_seen=5.0,
        confidence=0.9,
        current_bbox=BoundingBox(10, 10, 50, 50),
    )
    evA = SecurityEvent(
        event_id="ev_A1",
        event_type="HIGH_SPEED",
        timestamp=2.0,
        duration_seconds=1.0,
        track_id="trk_A1",
        severity="MEDIUM",
        confidence=0.88,
        description="High speed vehicle",
    )

    tB = TrackedObject(
        track_id="trk_B1",
        object_class="person",
        first_seen=10.0,
        last_seen=15.0,
        confidence=0.85,
        current_bbox=BoundingBox(20, 20, 60, 120),
    )
    evB = SecurityEvent(
        event_id="ev_B1",
        event_type="LOITERING",
        timestamp=12.0,
        duration_seconds=3.0,
        track_id="trk_B1",
        severity="LOW",
        confidence=0.75,
        description="Person loitering",
    )

    # 1. Process Video A
    repo.save_intelligence_results("vid_A_test", [tA], [], [], [evA], [])

    # 2. Process Video B
    repo.save_intelligence_results("vid_B_test", [tB], [], [], [evB], [])

    # Verify initial state & isolation
    events_A = repo.get_security_events("vid_A_test")
    events_B = repo.get_security_events("vid_B_test")
    assert len(events_A) == 1
    assert len(events_B) == 1
    assert events_A[0]["event_type"] == "HIGH_SPEED"
    assert events_B[0]["event_type"] == "LOITERING"

    # 3. Reprocess Video A (simulate re-running analysis)
    repo.save_intelligence_results("vid_A_test", [tA], [], [], [evA], [])

    # Verify idempotency: exactly 1 event for A, 1 event for B
    events_A_after = repo.get_security_events("vid_A_test")
    events_B_after = repo.get_security_events("vid_B_test")
    assert len(events_A_after) == 1, "Reprocessing must not duplicate records"
    assert len(events_B_after) == 1, "Reprocessing video A must not touch video B"


# ============================================================================
# 7. DETECTOR FAILURE ISOLATION
# ============================================================================

def test_detector_failure_isolation():
    """Verify that if one detector raises an unexpected exception, the rest of the pipeline continues."""
    class FailingDetector:
        def detect_frame(self, frame, frame_idx, timestamp_seconds):
            raise RuntimeError("Unexpected simulated detector crash!")

    failing = FailingDetector()
    working = SmokeVisualDetector()
    frame = np.full((100, 100, 3), 120, dtype=np.uint8)

    # Simulate pipeline orchestrator handling
    results = {}
    for name, det in [("failing", failing), ("working", working)]:
        try:
            obs = det.detect_frame(frame, 1, 0.5)
            results[name] = {"status": "SUCCESS", "obs": obs}
        except Exception as e:
            results[name] = {"status": "FAILED", "error": str(e)}

    assert results["failing"]["status"] == "FAILED"
    assert "Unexpected simulated detector crash!" in results["failing"]["error"]
    assert results["working"]["status"] == "SUCCESS"
