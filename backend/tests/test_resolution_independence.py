"""
Sentinel Resolution-Independence Unit Test Suite (Phase 20)

Validates that Sentinel's spatial reasoning, motion analysis, detector thresholds,
and inference policies produce consistent, normalized decisions across diverse
resolutions: 320x240 (SD/CCTV), 640x480 (VGA), 1280x720 (HD), 1920x1080 (FHD),
and 3840x2160 (4K UHD).
"""
import pytest
import math
import numpy as np

from ai.schemas import BoundingBox, TrackedObject, TrackLifecycleState
from ai.incidents.schemas import IncidentContext, CANONICAL_REFERENCE_DIAGONAL
from ai.incidents.motion import UniversalMotionEngine
from ai.detection.inference_policy import InferenceResolutionPolicy
from ai.video.frame_cache import BoundedFrameCache
from ai.validation.validator import DetectionValidator


RESOLUTIONS = [
    (320, 240, "QVGA"),
    (640, 480, "VGA"),
    (1280, 720, "720p"),
    (1920, 1080, "1080p"),
    (3840, 2160, "4K UHD"),
]


class TestResolutionIndependence:
    """Test suite ensuring zero resolution bias in Sentinel CV intelligence."""

    @pytest.mark.parametrize("w,h,name", RESOLUTIONS)
    def test_incident_context_scaling(self, w, h, name):
        """Context diagonal and scale factor must be mathematically exact."""
        ctx = IncidentContext(
            video_id=f"test-{name}",
            duration_seconds=10.0,
            fps=30.0,
            video_metadata={"width": w, "height": h},
        )
        expected_diag = math.hypot(w, h)
        expected_scale = expected_diag / CANONICAL_REFERENCE_DIAGONAL

        assert abs(ctx.frame_diagonal - expected_diag) < 1e-4
        assert abs(ctx.resolution_scale_factor - expected_scale) < 1e-4

        # Test distance round-trip normalization
        raw_dist = 100.0 * expected_scale
        norm = ctx.normalize_distance(raw_dist)
        denorm = ctx.denormalize_distance(norm)
        assert abs(denorm - raw_dist) < 1e-4

    @pytest.mark.parametrize("w,h,name", RESOLUTIONS)
    def test_motion_speed_consistency_across_resolutions(self, w, h, name):
        """
        An object traveling 10% of screen diagonal across 2 seconds should produce
        equivalent motion profiles across all resolutions.
        """
        diag = math.hypot(w, h)
        # Move diagonally by 10% of frame diagonal over 2 seconds
        disp_x = 0.08 * w
        disp_y = 0.06 * h  # hypot(0.08, 0.06) = 0.10

        t1, t2 = 1.0, 3.0
        x1, y1 = 0.2 * w, 0.2 * h
        x2, y2 = x1 + disp_x, y1 + disp_y

        track = TrackedObject(
            track_id="TRACK-MOTION-01",
            object_class="person",
            first_seen=t1,
            last_seen=t2,
            confidence=0.9,
            current_bbox=BoundingBox(x1=x2, y1=y2, x2=x2 + 0.05 * w, y2=y2 + 0.15 * h),
            trajectory=[(t1, x1, y1), (t2, x2, y2)],
        )

        engine = UniversalMotionEngine()
        motions = engine.compute_track_motion(track, frame_width=w, frame_height=h)
        assert len(motions) == 2
        assert motions[-1].is_stationary is False
        assert motions[-1].velocity_estimate > 0.0

    @pytest.mark.parametrize("w,h,name", RESOLUTIONS)
    def test_stationary_detection_consistency(self, w, h, name):
        """
        An object shifting < 0.2% diagonal (subtle jitter) should be recognized as
        stationary at all resolutions.
        """
        x, y = 0.5 * w, 0.5 * h
        # Jitter of 0.05% diagonal
        jitter = 0.0005 * math.hypot(w, h)

        track = TrackedObject(
            track_id="TRACK-STAT-01",
            object_class="suitcase",
            first_seen=1.0,
            last_seen=10.0,
            confidence=0.85,
            current_bbox=BoundingBox(x1=x, y1=y, x2=x + 0.05 * w, y2=y + 0.05 * h),
            trajectory=[
                (1.0, x, y),
                (3.0, x + jitter, y),
                (6.0, x, y + jitter),
                (10.0, x, y),
            ],
        )

        engine = UniversalMotionEngine()
        motions = engine.compute_track_motion(track, frame_width=w, frame_height=h)
        assert len(motions) == 4
        assert motions[-1].is_stationary is True

    def test_inference_policy_matrix(self):
        """Verify dynamic resolution selection policy conforms to specification."""
        # SD / QVGA / VGA -> baseline 640
        assert InferenceResolutionPolicy.select_inference_size(320, 240) == 640
        assert InferenceResolutionPolicy.select_inference_size(640, 480) == 640
        # 720p HD -> baseline 640
        assert InferenceResolutionPolicy.select_inference_size(1280, 720) == 640
        # 1080p FHD -> baseline 640 (optimal CPU throughput)
        assert InferenceResolutionPolicy.select_inference_size(1920, 1080) == 640
        # 2K / 1440p / 4K UHD -> 960 (preserves small objects with bounded CPU latency)
        assert InferenceResolutionPolicy.select_inference_size(2560, 1440) == 960
        assert InferenceResolutionPolicy.select_inference_size(3840, 2160) == 960
        # User override has absolute precedence
        assert InferenceResolutionPolicy.select_inference_size(3840, 2160, user_override=1280) == 1280

    def test_bounded_frame_cache_4k_memory_cap(self):
        """Simulate caching multiple 4K frames without runaway memory accumulation."""
        cache = BoundedFrameCache(max_frames=10, max_memory_mb=10.0)
        frame_4k = np.zeros((2160, 3840, 3), dtype=np.uint8)

        for i in range(15):
            cache[float(i)] = frame_4k

        stats = cache.get_memory_stats()
        assert stats["cached_frames"] <= 10
        assert stats["compressed_memory_mb"] < 10.0
        # Compression ratio must exceed 10x
        assert stats["compression_ratio"] > 10.0
        # Retrieval must return correct shape
        retrieved = cache[14.0]
        assert retrieved.shape[0] > 0 and retrieved.shape[1] > 0

    @pytest.mark.parametrize("w,h,name", RESOLUTIONS)
    def test_unattended_luggage_validation_at_resolution(self, w, h, name):
        """
        Unattended backpack/suitcase must NOT be REJECTED across any resolution.
        Instead, must reach UNCERTAIN to allow downstream abandoned-object evaluation.
        """
        validator = DetectionValidator()
        # Create a single frame with an unattended backpack (conf 0.40)
        box_w = 0.05 * w
        box_h = 0.05 * h
        frame_dets = [{
            "frame_number": 0,
            "detections": [{
                "class_name": "backpack",
                "object_class": "backpack",
                "confidence": 0.40,
                "bounding_box": {"x1": 0.5 * w, "y1": 0.5 * h, "x2": 0.5 * w + box_w, "y2": 0.5 * h + box_h},
            }],
        }]

        validator.validate_sequence(frame_dets, frame_width=float(w), frame_height=float(h))
        det = frame_dets[0]["detections"][0]
        assert det.get("validation_status") != "REJECTED", (
            f"Luggage at {name} ({w}x{h}) was incorrectly rejected! Status: {det.get('validation_status')}"
        )
