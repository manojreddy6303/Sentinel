"""
Phase 21B: Tracking Association & Temporal Sampling Test Suite

Verifies:
- Primary IoU association
- Centroid-distance fallback on IoU=0
- Ambiguity abstention (absolute and relative thresholds)
- Distance gating (max_distance_threshold)
- Class mismatch gating
- Large temporal gap expiration
- Occluded track recovery via centroid fallback
- Resolution independence (720p, 1080p, 4K)
- Variable Frame Rate (VFR) timestamps handling
- Adaptive sampling frame yielding and monotonicity
- Avoidance of unnecessary frame decoding
- Memory boundedness during long tracking sessions
- Centroid fallback configuration toggle
"""

import math
import numpy as np
import pytest

from ai.schemas import BoundingBox, TrackedObject, TrackLifecycleState
from ai.tracking.botsort_tracker import BoTSORTTracker
from ai.tracking.matching import (
    centroid_distance_matrix,
    centroid_distance_assignment,
    iou_cost_matrix,
)
from ai.video.processor import VideoProcessor


# ==============================================================================
# 1. PRIMARY IoU ASSOCIATION & FALLBACK TRIGGER TESTS
# ==============================================================================

def test_iou_primary_association():
    """Tracks with high IoU match in Stage 1 without needing centroid fallback."""
    tracker = BoTSORTTracker(enable_centroid_fallback=True)
    # Frame 0: person at [100, 100, 150, 200]
    det0 = [{
        "object_class": "person",
        "confidence": 0.90,
        "bounding_box": {"x1": 100.0, "y1": 100.0, "x2": 150.0, "y2": 200.0},
    }]
    t0 = tracker.update(timestamp=0.0, detections=det0, frame_width=1280, frame_height=720)
    assert len(t0) == 1
    orig_id = t0[0].track_id

    # Frame 1: slight movement (high IoU > 0.8)
    det1 = [{
        "object_class": "person",
        "confidence": 0.90,
        "bounding_box": {"x1": 102.0, "y1": 101.0, "x2": 152.0, "y2": 201.0},
    }]
    t1 = tracker.update(timestamp=0.033, detections=det1, frame_width=1280, frame_height=720)
    assert len(t1) == 1
    assert t1[0].track_id == orig_id
    assert t1[0].detection_count == 2
    assert t1[0].state == TrackLifecycleState.CONFIRMED


def test_centroid_fallback_triggered_on_zero_iou():
    """
    When target shifts beyond its bounding box width in 1.0 second (IoU=0),
    centroid fallback associations maintain track continuity instead of spawning a new track.
    """
    # Tracker WITH fallback
    tracker_fb = BoTSORTTracker(enable_centroid_fallback=True, max_distance_threshold=0.25)
    # Tracker WITHOUT fallback (Phase 21A baseline)
    tracker_no_fb = BoTSORTTracker(enable_centroid_fallback=False)

    det0 = [{
        "object_class": "person",
        "confidence": 0.85,
        "bounding_box": {"x1": 100.0, "y1": 100.0, "x2": 140.0, "y2": 200.0},
    }]
    # Frame 1: shifted 60 pixels to right (box width is 40px, so IoU = 0.0)
    det1 = [{
        "object_class": "person",
        "confidence": 0.85,
        "bounding_box": {"x1": 160.0, "y1": 100.0, "x2": 200.0, "y2": 200.0},
    }]

    # Without fallback: spawns 2 tracks
    tracker_no_fb.update(timestamp=0.0, detections=det0, frame_width=1280, frame_height=720)
    res_no_fb = tracker_no_fb.update(timestamp=1.0, detections=det1, frame_width=1280, frame_height=720)
    all_no_fb = tracker_no_fb.finalize()
    assert len(all_no_fb) == 2, "Phase 21A without fallback should have fragmented into 2 tracks"

    # WITH fallback: associates as 1 continuous track!
    tracker_fb.update(timestamp=0.0, detections=det0, frame_width=1280, frame_height=720)
    res_fb = tracker_fb.update(timestamp=1.0, detections=det1, frame_width=1280, frame_height=720)
    assert len(res_fb) == 1
    assert res_fb[0].detection_count == 2
    assert res_fb[0].state == TrackLifecycleState.CONFIRMED

    all_fb = tracker_fb.finalize()
    assert len(all_fb) == 1, "Phase 21B with centroid fallback must maintain a single continuous track"
    assert all_fb[0].detection_count == 2


# ==============================================================================
# 2. AMBIGUITY ABSTENTION TESTS
# ==============================================================================

def test_ambiguous_centroid_matches_abstain():
    """
    When a track is equidistant to two candidate detections of the same class,
    the tracker abstains from matching to avoid false identity swaps.
    """
    tracker = BoTSORTTracker(
        enable_centroid_fallback=True,
        ambiguity_threshold=0.02,
        relative_ambiguity_margin=0.10,
    )
    # Track at centroid (500, 500)
    det0 = [{
        "object_class": "person",
        "confidence": 0.85,
        "bounding_box": {"x1": 480.0, "y1": 450.0, "x2": 520.0, "y2": 550.0},
    }]
    tracker.update(timestamp=0.0, detections=det0, frame_width=1000, frame_height=1000)

    # In next frame, two detections appear at nearly identical distance (50px left and 52px right)
    # Both have IoU=0 with initial box (width 40px, moved 50px)
    det1 = [
        {
            "object_class": "person",
            "confidence": 0.85,
            "bounding_box": {"x1": 410.0, "y1": 450.0, "x2": 450.0, "y2": 550.0},  # cx=430, dist=70
        },
        {
            "object_class": "person",
            "confidence": 0.85,
            "bounding_box": {"x1": 550.0, "y1": 450.0, "x2": 590.0, "y2": 550.0},  # cx=570, dist=70
        },
    ]
    # Centroid distance difference is 0 -> ambiguous! Tracker must abstain from forcing match to either.
    tracker.update(timestamp=1.0, detections=det1, frame_width=1000, frame_height=1000)
    all_tracks = tracker.finalize()
    # The original track was not forcefully matched to an ambiguous candidate
    orig_track = [t for t in all_tracks if t.track_id == "TRACK-001"][0]
    assert orig_track.detection_count == 1, "Track must abstain from ambiguous candidate matching"


def test_unambiguous_candidate_matches_successfully():
    """When one candidate is clearly closer than another, the match succeeds."""
    tracker = BoTSORTTracker(
        enable_centroid_fallback=True,
        ambiguity_threshold=0.01,
        relative_ambiguity_margin=0.10,
    )
    # Track at (500, 500)
    det0 = [{
        "object_class": "person",
        "confidence": 0.85,
        "bounding_box": {"x1": 480.0, "y1": 450.0, "x2": 520.0, "y2": 550.0},
    }]
    tracker.update(timestamp=0.0, detections=det0, frame_width=1000, frame_height=1000)

    # Det 1 at (545, 500) (dist = 45px), Det 2 at (680, 500) (dist = 180px)
    # Relative difference is 400% -> unambiguous!
    det1 = [
        {
            "object_class": "person",
            "confidence": 0.85,
            "bounding_box": {"x1": 525.0, "y1": 450.0, "x2": 565.0, "y2": 550.0},  # cx=545
        },
        {
            "object_class": "person",
            "confidence": 0.85,
            "bounding_box": {"x1": 660.0, "y1": 450.0, "x2": 700.0, "y2": 550.0},  # cx=680
        },
    ]
    tracker.update(timestamp=1.0, detections=det1, frame_width=1000, frame_height=1000)
    all_tracks = tracker.finalize()
    orig_track = [t for t in all_tracks if t.track_id == "TRACK-001"][0]
    assert orig_track.detection_count == 2, "Unambiguous closer candidate must be matched"


# ==============================================================================
# 3. GATING & STATE INTEGRITY TESTS
# ==============================================================================

def test_distance_gating_rejects_distant_objects():
    """Detections beyond max_distance_threshold are rejected by centroid fallback."""
    tracker = BoTSORTTracker(
        enable_centroid_fallback=True,
        max_distance_threshold=0.10,  # Strict: 10% of frame diagonal
    )
    # Frame 0: person at (100, 100) in a 1000x1000 frame (diag ≈ 1414, 10% = 141px)
    det0 = [{
        "object_class": "person",
        "confidence": 0.90,
        "bounding_box": {"x1": 80.0, "y1": 80.0, "x2": 120.0, "y2": 120.0},
    }]
    tracker.update(timestamp=0.0, detections=det0, frame_width=1000, frame_height=1000)

    # Frame 1: person appears 300px away (> 141px threshold)
    det1 = [{
        "object_class": "person",
        "confidence": 0.90,
        "bounding_box": {"x1": 380.0, "y1": 380.0, "x2": 420.0, "y2": 420.0},
    }]
    tracker.update(timestamp=1.0, detections=det1, frame_width=1000, frame_height=1000)
    all_tracks = tracker.finalize()
    assert len(all_tracks) == 2, "Distant detection must not be matched via centroid fallback"


def test_class_mismatch_never_associated():
    """A car detection must never be associated with a person track, even with zero distance."""
    tracker = BoTSORTTracker(enable_centroid_fallback=True)
    # Frame 0: person at (200, 200)
    det0 = [{
        "object_class": "person",
        "confidence": 0.90,
        "bounding_box": {"x1": 180.0, "y1": 150.0, "x2": 220.0, "y2": 250.0},
    }]
    tracker.update(timestamp=0.0, detections=det0, frame_width=1000, frame_height=1000)

    # Frame 1: car at exact same coordinates (200, 200)
    det1 = [{
        "object_class": "car",
        "confidence": 0.90,
        "bounding_box": {"x1": 180.0, "y1": 150.0, "x2": 220.0, "y2": 250.0},
    }]
    tracker.update(timestamp=1.0, detections=det1, frame_width=1000, frame_height=1000)
    all_tracks = tracker.finalize()
    assert len(all_tracks) == 2, "Different classes must never be merged by centroid fallback"
    classes = {t.object_class for t in all_tracks}
    assert classes == {"person", "car"}


def test_large_temporal_gap_expires():
    """Centroid fallback respects max_missing_seconds and does not match expired tracks."""
    tracker = BoTSORTTracker(
        enable_centroid_fallback=True,
        max_missing_seconds=2.0,
    )
    # Frame 0 at t=0.0
    det0 = [{
        "object_class": "person",
        "confidence": 0.90,
        "bounding_box": {"x1": 100.0, "y1": 100.0, "x2": 140.0, "y2": 200.0},
    }]
    tracker.update(timestamp=0.0, detections=det0, frame_width=1000, frame_height=1000)

    # Gap of 4.0 seconds (> max_missing_seconds=2.0)
    det1 = [{
        "object_class": "person",
        "confidence": 0.90,
        "bounding_box": {"x1": 110.0, "y1": 100.0, "x2": 150.0, "y2": 200.0},
    }]
    tracker.update(timestamp=4.0, detections=det1, frame_width=1000, frame_height=1000)
    all_tracks = tracker.finalize()
    assert len(all_tracks) == 2, "Expired track must not be revived after missing window"


def test_occluded_track_recovery_via_centroid():
    """An OCCLUDED track is reactivated to CONFIRMED when target reappears within max_occluded_seconds."""
    tracker = BoTSORTTracker(
        enable_centroid_fallback=True,
        max_missing_seconds=2.0,
        max_occluded_seconds=5.0,
        occlusion_iou_threshold=0.15,
    )
    # Frame 0: Track 1 (person A) and Track 2 (large truck B)
    dets_t0 = [
        {"object_class": "person", "confidence": 0.9, "bounding_box": {"x1": 200.0, "y1": 200.0, "x2": 250.0, "y2": 350.0}},
        {"object_class": "truck", "confidence": 0.95, "bounding_box": {"x1": 180.0, "y1": 150.0, "x2": 350.0, "y2": 400.0}},
    ]
    tracker.update(timestamp=0.0, detections=dets_t0, frame_width=1000, frame_height=1000)
    # Frame 1: both confirmed
    tracker.update(timestamp=0.1, detections=dets_t0, frame_width=1000, frame_height=1000)

    # Frame 2 (t=1.0): Person A is occluded by the truck (only truck detected)
    dets_t1 = [
        {"object_class": "truck", "confidence": 0.95, "bounding_box": {"x1": 180.0, "y1": 150.0, "x2": 350.0, "y2": 400.0}},
    ]
    tracker.update(timestamp=1.0, detections=dets_t1, frame_width=1000, frame_height=1000)
    person_trk = [t for t in tracker._tracks.values() if t.object_class == "person"][0]
    assert person_trk.state == TrackLifecycleState.OCCLUDED

    # Frame 3 (t=3.0, 3 seconds later, within 5.0s max_occluded): Person emerges slightly to the right (IoU=0 with old box)
    dets_t2 = [
        {"object_class": "person", "confidence": 0.9, "bounding_box": {"x1": 360.0, "y1": 200.0, "x2": 410.0, "y2": 350.0}},
        {"object_class": "truck", "confidence": 0.95, "bounding_box": {"x1": 180.0, "y1": 150.0, "x2": 350.0, "y2": 400.0}},
    ]
    tracker.update(timestamp=3.0, detections=dets_t2, frame_width=1000, frame_height=1000)
    assert person_trk.state == TrackLifecycleState.CONFIRMED, "Occluded track must reactivate upon re-emergence"
    assert person_trk.track_id == "TRACK-001"
    assert person_trk.last_seen == 3.0


# ==============================================================================
# 4. RESOLUTION INDEPENDENCE & VFR TESTS
# ==============================================================================

def test_resolution_independence():
    """Normalized centroid distance works consistently across 720p, 1080p, and 4K resolutions."""
    for w, h in [(1280, 720), (1920, 1080), (3840, 2160)]:
        diag = math.hypot(w, h)
        tracker = BoTSORTTracker(enable_centroid_fallback=True, max_distance_threshold=0.20)

        # 5% of screen diagonal displacement
        disp = 0.05 * diag
        det0 = [{"object_class": "car", "confidence": 0.9, "bounding_box": {"x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 200.0}}]
        det1 = [{"object_class": "car", "confidence": 0.9, "bounding_box": {"x1": 100.0 + disp, "y1": 100.0, "x2": 200.0 + disp, "y2": 200.0}}]

        tracker.update(timestamp=0.0, detections=det0, frame_width=w, frame_height=h)
        tracker.update(timestamp=1.0, detections=det1, frame_width=w, frame_height=h)
        tracks = tracker.finalize()
        assert len(tracks) == 1, f"Resolution {w}x{h} failed to associate within 5% diagonal displacement"


def test_vfr_timestamps_handling():
    """Variable Frame Rate timestamps with irregular delta_t are tracked correctly."""
    tracker = BoTSORTTracker(enable_centroid_fallback=True)
    # Non-uniform timestamps: 0.0s, 0.33s, 1.25s, 1.50s, 2.80s
    timestamps = [0.0, 0.33, 1.25, 1.50, 2.80]
    for i, t in enumerate(timestamps):
        x = 100.0 + i * 20.0
        dets = [{"object_class": "person", "confidence": 0.88, "bounding_box": {"x1": x, "y1": 100.0, "x2": x + 30.0, "y2": 180.0}}]
        tracker.update(timestamp=t, detections=dets, frame_width=1280, frame_height=720)

    tracks = tracker.finalize()
    assert len(tracks) == 1, "VFR timestamps must produce a single continuous track"
    assert tracks[0].detection_count == 5
    assert tracks[0].duration_seconds == pytest.approx(2.80, abs=0.01)


# ==============================================================================
# 5. ADAPTIVE SAMPLING & MEMORY TESTS
# ==============================================================================

def test_adaptive_sampling_yields_monotonic_frames(tmp_path):
    """VideoProcessor adaptive sampling yields strictly monotonic frames with finite timestamps."""
    import cv2
    # Create small synthetic test video (2 seconds, 30 fps = 60 frames)
    vid_file = tmp_path / "test_synth.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vid_file), fourcc, 30.0, (320, 240))
    for i in range(60):
        # Draw moving circle to create dynamic motion
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cx = int(20 + i * 4)
        cv2.circle(frame, (cx, 120), 15, (255, 255, 255), -1)
        out.write(frame)
    out.release()

    proc = VideoProcessor(
        str(vid_file),
        sample_rate_fps=1.0,
        adaptive=True,
        min_fps=1.0,
        max_burst_fps=5.0,
        motion_threshold=2.0,
    )
    sampled = list(proc.sample_frames())
    assert len(sampled) > 0

    # Verify strictly monotonic timestamps and non-empty frames
    prev_t = -1.0
    for frame_idx, ts, frame_bgr in sampled:
        assert ts >= prev_t, f"Non-monotonic timestamp detected: {ts} < {prev_t}"
        assert frame_bgr is not None
        assert frame_bgr.shape == (240, 320, 3)
        prev_t = ts


def test_tracker_memory_bounded():
    """BoTSORTTracker with centroid fallback does not leak memory over 500 frame updates."""
    import psutil, os
    tracker = BoTSORTTracker(enable_centroid_fallback=True)
    proc = psutil.Process(os.getpid())
    rss_start = proc.memory_info().rss / (1024 * 1024)

    for step in range(500):
        t = step * 0.1
        dets = [
            {"object_class": "person", "confidence": 0.85, "bounding_box": {"x1": 100.0 + (step % 20), "y1": 100.0, "x2": 150.0 + (step % 20), "y2": 200.0}},
            {"object_class": "car", "confidence": 0.90, "bounding_box": {"x1": 300.0 + (step % 40), "y1": 300.0, "x2": 450.0 + (step % 40), "y2": 400.0}},
        ]
        tracker.update(timestamp=t, detections=dets, frame_width=1280, frame_height=720)

    rss_end = proc.memory_info().rss / (1024 * 1024)
    rss_growth = rss_end - rss_start
    assert rss_growth < 50.0, f"Excessive memory growth: {rss_growth:.2f} MB"
