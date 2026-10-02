"""
Video Processing Module

Responsible for:
- Opening and validating video files via OpenCV.
- Extracting video metadata: FPS, duration, resolution, frame count.
- Sampling frames at a configurable interval for downstream YOLO detection.

Design notes:
- Raw frames are never written to disk — everything is in-memory.
- Configurable sample_rate_fps controls how many frames per second are processed.
- VideoCapture is always released, even on error (context manager pattern).
- Raises descriptive exceptions on all failure modes.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Generator, Tuple

logger = logging.getLogger(__name__)


class VideoProcessingError(Exception):
    """Base class for video processing errors."""


class VideoNotFoundError(VideoProcessingError):
    """Raised when the video file does not exist."""


class VideoCorruptedError(VideoProcessingError):
    """Raised when OpenCV cannot open or read the video file."""


class VideoEmptyError(VideoProcessingError):
    """Raised when the video has zero frames or invalid FPS."""


class VideoProcessor:
    """
    Processes a video file using OpenCV.

    Provides metadata extraction and configurable frame sampling without
    permanently writing extracted frames to disk.

    Usage:
        processor = VideoProcessor(video_path, sample_rate_fps=1.0)
        metadata = processor.get_metadata()
        for frame_number, timestamp, frame_bgr in processor.sample_frames():
            ...  # pass frame_bgr to YOLO detector
    """

    def __init__(
        self,
        video_path: str,
        sample_rate_fps: float = 1.0,
        adaptive: bool = False,
        min_fps: float = 1.0,
        max_burst_fps: float = 3.0,
        motion_threshold: float = 12.0,
    ):
        """
        Args:
            video_path: Absolute path to the video file.
            sample_rate_fps: Number of frames to sample per second of video.
                             Use 1.0 to sample 1 frame/second, 0.5 for 1 frame
                             every 2 seconds, etc. Clamped to [0.01, native FPS].
            adaptive: Whether to enable motion-aware adaptive sampling.
            min_fps: Baseline minimum sampling rate under low motion.
            max_burst_fps: Ceiling burst sampling rate during high motion.
            motion_threshold: Inter-frame pixel difference threshold to trigger burst.
        """
        self.video_path = str(video_path)
        self.sample_rate_fps = max(0.01, sample_rate_fps)
        self.adaptive = adaptive
        self.min_fps = max(0.01, min_fps)
        self.max_burst_fps = max(self.min_fps, max_burst_fps)
        self.motion_threshold = motion_threshold

    def _open_capture(self):
        """
        Open and validate the VideoCapture object.
        Returns the opened cv2.VideoCapture.
        Raises descriptive exceptions on failure.
        """
        import cv2

        path = Path(self.video_path)
        if not path.exists():
            raise VideoNotFoundError(
                f"Video file not found: {self.video_path}"
            )
        if not path.is_file():
            raise VideoNotFoundError(
                f"Path is not a file: {self.video_path}"
            )

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise VideoCorruptedError(
                f"OpenCV could not open video file: {self.video_path}. "
                f"The file may be corrupt, use an unsupported codec, or be empty."
            )
        return cap

    def get_metadata(self) -> Dict[str, Any]:
        """
        Extract metadata from the video file using MediaMetadataExtractor.

        Returns:
            Dictionary with keys: fps, frame_count, duration_seconds,
            width, height, video_path, filename, and extended media properties.

        Raises:
            VideoNotFoundError: File does not exist.
            VideoCorruptedError: Cannot open the video.
            VideoEmptyError: FPS is 0 or frame count is 0.
        """
        path = Path(self.video_path)
        if not path.exists():
            raise VideoNotFoundError(f"Video file not found: {self.video_path}")
        if not path.is_file():
            raise VideoNotFoundError(f"Path is not a file: {self.video_path}")

        try:
            from ai.video.metadata import MediaMetadataExtractor, VideoMetadataError
            meta = MediaMetadataExtractor.extract(self.video_path)
        except VideoMetadataError as vme:
            err_msg = str(vme).lower()
            if "empty" in err_msg or "0 frames" in err_msg:
                raise VideoEmptyError(str(vme))
            raise VideoCorruptedError(str(vme))
        except Exception as exc:
            raise VideoCorruptedError(f"Error inspecting media: {exc}")

        fps = meta["fps"]
        frame_count = meta["frame_count"]
        if fps <= 0:
            raise VideoEmptyError(f"Video has invalid FPS ({fps}).")
        if frame_count <= 0 and meta.get("is_decodable"):
            dur = meta.get("duration_seconds", 0.0)
            if dur > 0 and fps > 0:
                frame_count = max(1, int(round(dur * fps)))
                meta["frame_count"] = frame_count
            else:
                frame_count = 1
                meta["frame_count"] = 1
        if frame_count <= 0:
            raise VideoEmptyError("Video has zero frames.")

        return {
            "fps": meta["fps"],
            "frame_count": meta["frame_count"],
            "duration_seconds": meta["duration_seconds"],
            "width": meta["width"],
            "height": meta["height"],
            "aspect_ratio": meta.get("aspect_ratio", 1.0),
            "orientation": meta.get("orientation", "landscape"),
            "container": meta.get("container", "mp4"),
            "codec": meta.get("codec", "unknown"),
            "is_decodable": meta.get("is_decodable", True),
            "seek_capable": meta.get("seek_capable", True),
            "video_path": self.video_path,
            "filename": Path(self.video_path).name,
        }

    def sample_frames(
        self,
        sample_rate_fps: float = None,
        adaptive: bool = None,
        min_fps: float = None,
        max_burst_fps: float = None,
        motion_threshold: float = None,
    ) -> Generator[Tuple[int, float, Any], None, None]:
        """
        Yield sampled frames from the video without writing them to disk.
        Supports both fixed-interval and motion-aware adaptive sampling.
        Enforces strictly monotonic, finite timestamps clamped within [0, duration].

        Yields:
            Tuples of (frame_number, timestamp_seconds, frame_bgr_ndarray).
        """
        import math
        import cv2
        import numpy as np
        from ai.common.numeric import ensure_finite

        is_adaptive = self.adaptive if adaptive is None else bool(adaptive)
        effective_rate = max(0.01, sample_rate_fps or self.sample_rate_fps)
        effective_min_fps = max(0.01, min_fps or self.min_fps or effective_rate)
        effective_max_fps = max(effective_min_fps, max_burst_fps or self.max_burst_fps)
        effective_motion_thresh = self.motion_threshold if motion_threshold is None else float(motion_threshold)

        cap = self._open_capture()
        try:
            fps_prop = cap.get(cv2.CAP_PROP_FPS)
            cnt_prop = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            fps = float(fps_prop) if fps_prop and fps_prop > 0 else 30.0
            frame_count = int(cnt_prop) if cnt_prop and cnt_prop > 0 else 0

            if fps <= 0:
                raise VideoEmptyError(f"Video FPS is {fps}, cannot sample frames.")
            if frame_count < 0:
                raise VideoEmptyError("Video has invalid frame count.")

            duration_seconds = round(frame_count / fps, 4) if frame_count > 0 else 0.0

            # Step calculation
            step = max(1, int(round(fps / effective_rate)))
            min_interval = max(1, int(round(fps / effective_max_fps)))
            base_interval = max(1, int(round(fps / effective_min_fps)))

            frame_number = 0
            frames_yielded = 0
            last_timestamp = 0.0
            last_yielded_frame = -999999
            prev_thumb = None

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # Query container Presentation Time Stamp (PTS) in milliseconds
                pts_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                has_media_pts = False
                media_ts = 0.0
                if pts_msec is not None:
                    try:
                        pts_val = float(pts_msec)
                        if not math.isnan(pts_val) and not math.isinf(pts_val) and pts_val >= 0.0:
                            # Genuine container PTS (non-zero after frame 0)
                            if frame_number == 0 or pts_val > 0.0:
                                media_ts = pts_val / 1000.0
                                has_media_pts = True
                    except (ValueError, TypeError):
                        has_media_pts = False

                should_yield = False
                if not is_adaptive:
                    if frame_number % step == 0:
                        should_yield = True
                else:
                    # Ultra-fast motion score via 64x36 thumbnail difference (<0.05ms)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    thumb = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_NEAREST)
                    motion_score = 0.0
                    if prev_thumb is not None:
                        motion_score = float(np.mean(np.abs(thumb.astype(np.int16) - prev_thumb.astype(np.int16))))
                    prev_thumb = thumb

                    is_base = (frame_number - last_yielded_frame) >= base_interval
                    is_burst = ((frame_number - last_yielded_frame) >= min_interval) and (motion_score >= effective_motion_thresh)

                    if is_base or is_burst:
                        should_yield = True
                        last_yielded_frame = frame_number

                if should_yield:
                    if has_media_pts and media_ts >= last_timestamp:
                        raw_ts = round(media_ts, 4)
                    else:
                        # Deterministic CFR fallback when PTS is unavailable or non-monotonic
                        raw_ts = round(frame_number / fps, 4)

                    # Enforce strict monotonicity and finite bounds
                    ts = max(last_timestamp, ensure_finite(raw_ts, default=last_timestamp))
                    if duration_seconds > 0:
                        max_allowed_duration = max(duration_seconds, media_ts if has_media_pts else duration_seconds)
                        ts = min(max_allowed_duration, ts)
                    last_timestamp = ts

                    yield (frame_number, round(ts, 4), frame)
                    frames_yielded += 1

                frame_number += 1

            logger.debug(
                f"Sampled {frames_yielded} frames from {frame_number} total "
                f"(adaptive={is_adaptive}, rate={effective_rate}fps)"
            )
        finally:
            cap.release()

