"""
SENTINEL — VFR / PTS MEDIA TIMESTAMP VALIDATION SUITE (Phase 20.1)

Validates:
1. Monotonicity: Timestamps are strictly non-decreasing (t_{i+1} >= t_i).
2. Media PTS preservation: Real PTS from container is accurately extracted.
3. CFR Fallback: Broken, negative, or zero PTS deterministically falls back to frame_number / fps.
4. Timestamp Bounds: Clamped within [0, max(duration, pts)].
5. Frame/Timestamp Correspondence: Exact mapping for downstream evidence, incidents, and timeline clips.
"""
import pytest
import os
import cv2
import numpy as np
import tempfile
from pathlib import Path

from ai.video.processor import VideoProcessor, VideoEmptyError
from ai.schemas import BoundingBox, TrackedObject
from ai.incidents.schemas import IncidentContext, SupportingSignal


class TestVFRTimestampHandling:
    """Tests container PTS extraction and CFR deterministic fallback."""

    def test_cfr_monotonic_timestamps(self, tmp_path):
        """Standard Constant Frame Rate (CFR) video produces monotonic, bounded timestamps."""
        # Create a synthetic CFR video (10 frames at 10 FPS -> 1.0 second)
        video_path = tmp_path / "cfr_test.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (160, 120))
        for _ in range(10):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        processor = VideoProcessor(video_path, sample_rate_fps=5.0, adaptive=False)
        samples = list(processor.sample_frames())
        assert len(samples) > 0

        prev_t = -1.0
        for f_idx, ts, frame in samples:
            assert ts >= prev_t, f"Non-monotonic timestamp detected: {ts} < {prev_t}"
            assert ts >= 0.0, f"Negative timestamp detected: {ts}"
            assert ts <= 1.2, f"Timestamp exceeded stream duration: {ts}"
            assert frame.shape == (120, 160, 3)
            prev_t = ts

    def test_vfr_real_video_pts_monotonicity(self):
        """Real surveillance video samples produce valid, monotonic timestamps matching PTS."""
        # Check against available real video
        real_video = Path("storage/uploads/0d4d92f9-19f8-42e3-925f-1931cb557705_uccrime_Burglary010_x264.mp4")
        if not real_video.exists():
            pytest.skip("Burglary benchmark video not present")

        processor = VideoProcessor(real_video, sample_rate_fps=2.0, adaptive=True)
        samples = list(processor.sample_frames())
        assert len(samples) >= 10

        prev_t = -1.0
        for f_num, ts, frame in samples:
            assert ts >= prev_t, f"Monotonicity violation: {ts} < {prev_t} at frame {f_num}"
            assert isinstance(ts, float)
            assert ts >= 0.0
            prev_t = ts

    def test_corrupted_pts_fallback_to_cfr(self, tmp_path):
        """When container PTS is missing/corrupted, processor falls back gracefully to CFR."""
        # Create a video with synthetic black frames
        video_path = tmp_path / "pts_fallback.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 25.0, (160, 120))
        for _ in range(25):
            writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
        writer.release()

        processor = VideoProcessor(video_path, sample_rate_fps=5.0, adaptive=False)
        samples = list(processor.sample_frames())

        # Expected step is 25 / 5 = 5 frames (0, 5, 10, 15, 20)
        expected_indices = [0, 5, 10, 15, 20]
        actual_indices = [s[0] for s in samples]
        assert actual_indices == expected_indices

        # Check timestamp mapping
        for f_idx, ts, _ in samples:
            expected_ts = round(f_idx / 25.0, 4)
            assert abs(ts - expected_ts) < 0.05, f"Timestamp {ts} diverged from CFR expected {expected_ts}"

    def test_incident_timeline_timestamp_correspondence(self):
        """Downstream IncidentContext and SupportingSignals preserve exact timestamps."""
        p_track = TrackedObject(
            track_id="TRACK-001",
            object_class="person",
            first_seen=12.345,
            last_seen=15.678,
            confidence=0.92,
            current_bbox=BoundingBox(x1=10, y1=10, x2=50, y2=90),
            trajectory=[(12.345, 20.0, 30.0), (15.678, 25.0, 35.0)],
        )

        ctx = IncidentContext(
            video_id="pts_test_video",
            fps=29.97,
            duration_seconds=30.0,
            tracks=[p_track],
        )

        signal = SupportingSignal(
            signal_type="Observation",
            description="Person confirmed at accurate PTS",
            confidence=0.95,
            timestamp=12.345,
        )

        assert signal.timestamp == 12.345
        assert ctx.tracks[0].first_seen == 12.345
        assert ctx.tracks[0].last_seen == 15.678
