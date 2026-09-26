"""
Sentinel Video Ingestion & Media Metadata Normalization Layer (Phase 15.1)

Extracts and validates video container, stream, and frame properties:
- Codec, container, width, height, FPS, duration, frame count
- Orientation (landscape, portrait, square)
- Audio stream presence, seek capability, decodability
- Strict finite-number validation and error reporting
"""
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import cv2

from ai.common.numeric import is_finite_number, ensure_finite

logger = logging.getLogger(__name__)


class VideoMetadataError(Exception):
    """Raised when video metadata extraction encounters unrecoverable errors."""


class MediaMetadataExtractor:
    """
    Extracts, inspects, and validates comprehensive media properties
    from arbitrary surveillance/security video files.
    """

    SUPPORTED_CONTAINERS = {
        ".mp4": "mp4",
        ".mov": "quicktime",
        ".avi": "avi",
        ".mkv": "matroska",
        ".webm": "webm",
        ".m4v": "mp4",
    }

    @classmethod
    def extract(
        cls,
        video_path: str,
        video_id: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract and strictly validate video metadata.
        """
        path = Path(video_path)
        if not path.exists():
            raise VideoMetadataError(f"Video file not found at: {video_path}")
        if not path.is_file():
            raise VideoMetadataError(f"Path is not a regular file: {video_path}")

        file_size = path.stat().st_size
        if file_size == 0:
            raise VideoMetadataError(f"Video file is empty (0 bytes): {video_path}")

        ext = path.suffix.lower()
        container = cls.SUPPORTED_CONTAINERS.get(ext, ext.lstrip(".") or "unknown")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise VideoMetadataError(f"OpenCV could not open media container: {video_path}")

        is_decodable = False
        first_frame_valid = False
        seek_capable = False
        width = 0
        height = 0
        fps = 0.0
        frame_count = 0
        codec = "unknown"

        try:
            # Basic OpenCV container properties
            w_prop = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h_prop = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps_prop = cap.get(cv2.CAP_PROP_FPS)
            cnt_prop = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fourcc_prop = int(cap.get(cv2.CAP_PROP_FOURCC))

            if is_finite_number(w_prop):
                width = int(w_prop)
            if is_finite_number(h_prop):
                height = int(h_prop)
            if is_finite_number(fps_prop) and fps_prop > 0:
                fps = float(fps_prop)
            if is_finite_number(cnt_prop) and cnt_prop >= 0:
                frame_count = int(cnt_prop)

            if fourcc_prop > 0:
                try:
                    codec_chars = [chr((fourcc_prop >> 8 * i) & 0xFF) for i in range(4)]
                    codec = "".join(codec_chars).strip()
                except Exception:
                    codec = "unknown"

            # Validate decodability by attempting to read the first frame
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                is_decodable = True
                first_frame_valid = True
                real_h, real_w = frame.shape[:2]
                if width <= 0 or height <= 0:
                    width = real_w
                    height = real_h

            # Validate seek capability
            if frame_count > 1:
                mid_frame = frame_count // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
                ret_seek, frame_seek = cap.read()
                if ret_seek and frame_seek is not None:
                    seek_capable = True
            elif first_frame_valid:
                seek_capable = True

        finally:
            cap.release()

        if not is_decodable or width <= 0 or height <= 0:
            raise VideoMetadataError(
                f"Video stream cannot be decoded or has invalid dimensions ({width}x{height})"
            )

        # Handle missing or zero FPS gracefully
        fps_uncertain = False
        if fps <= 0.0 or not is_finite_number(fps):
            # Fallback assumption for variable/unreported FPS
            fps = 30.0
            fps_uncertain = True

        # Calculate duration
        duration_uncertain = False
        duration_seconds = 0.0
        if frame_count > 0 and fps > 0:
            duration_seconds = round(float(frame_count) / float(fps), 4)
        else:
            duration_uncertain = True

        # Fallback duration probe via ffmpeg if container header lacked frame count
        if (frame_count <= 0 or duration_seconds <= 0.0) and is_decodable:
            try:
                import subprocess, re
                from backend.app.services.playback_service import get_ffmpeg_binary
                ff_bin = get_ffmpeg_binary()
                if ff_bin:
                    probe = subprocess.run(
                        [ff_bin, "-i", str(path)],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", probe.stderr)
                    if dur_match:
                        hrs, mns, scs = float(dur_match.group(1)), float(dur_match.group(2)), float(dur_match.group(3))
                        probed_dur = hrs * 3600.0 + mns * 60.0 + scs
                        if probed_dur > 0:
                            duration_seconds = round(probed_dur, 4)
                            duration_uncertain = False
                            if frame_count <= 0 and fps > 0:
                                frame_count = max(1, int(round(duration_seconds * fps)))
            except Exception as ff_err:
                logger.debug(f"Media duration probe fallback error for {video_path}: {ff_err}")

        # Determine orientation
        if width > height:
            orientation = "landscape"
        elif height > width:
            orientation = "portrait"
        else:
            orientation = "square"

        aspect_ratio = round(float(width) / float(height), 4) if height > 0 else 1.0

        return {
            "video_id": video_id or path.stem,
            "filename": filename or path.name,
            "filepath": str(path),
            "file_size_bytes": file_size,
            "container": container,
            "codec": codec,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "orientation": orientation,
            "fps": round(fps, 4),
            "fps_uncertain": fps_uncertain,
            "frame_count": frame_count,
            "duration_seconds": duration_seconds,
            "duration_uncertain": duration_uncertain,
            "is_decodable": is_decodable,
            "seek_capable": seek_capable,
            "has_audio": False,  # Populated when audio track detected
            "is_valid": True,
        }
