"""
Video Playback & Transcoding Service for Sentinel.

Ensures that surveillance video files uploaded in non-browser-compatible formats
(e.g., HEVC/H.265, AVI, MPEG-4, high-bitdepth) can be smoothly played in HTML5
<video> elements (Chrome, Edge, Firefox, Safari) via standard H.264 (AVC) + AAC
MP4 streaming with RFC 7233 HTTP Range request support.
"""
import logging
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from fastapi import HTTPException, Response, status
from fastapi.responses import StreamingResponse

try:
    from app.core.config import settings
except ImportError:
    from backend.app.core.config import settings

logger = logging.getLogger(__name__)

# Lock management for thread-safe idempotent conversions
_master_lock = threading.Lock()
_conversion_locks: Dict[str, threading.Lock] = {}
_in_progress_conversions = set()

_evidence_conversion_locks: Dict[str, threading.Lock] = {}
_in_progress_evidence_conversions = set()


class PlaybackError(Exception):
    """Base exception for playback-related failures."""
    pass


class PlaybackNotFoundError(PlaybackError):
    """Raised when original video file is not found on disk."""
    pass


class PlaybackConversionError(PlaybackError):
    """Raised when transcoding fails."""
    pass


def get_ffmpeg_binary() -> Optional[str]:
    """
    Locate a working ffmpeg executable.
    Checks:
    1. imageio_ffmpeg bundled binary
    2. System PATH
    3. Common Windows utility directories
    """
    # 1. imageio_ffmpeg
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass

    # 2. PATH
    path_exe = shutil.which("ffmpeg")
    if path_exe:
        return path_exe

    # 3. Known fallback locations
    candidate_paths = [
        r"C:\Program Files\Softdeluxe\Free Download Manager\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]
    for cand in candidate_paths:
        if os.path.exists(cand):
            return cand

    return None


_compat_cache: Dict[str, Tuple[float, bool]] = {}


def is_browser_compatible(video_path: Path) -> bool:
    """
    Check whether a video is already directly playable by standard HTML5 video in Chrome/Edge.
    Requirements:
    - Container must be .mp4 or .webm
    - Video codec must be H.264 (avc1) or VP8/VP9
    - Cannot be HEVC / H.265 / MPEG4 / AVI / etc.
    """
    if not video_path.exists() or video_path.stat().st_size == 0:
        return False

    suffix = video_path.suffix.lower()
    if suffix not in (".mp4", ".webm"):
        return False

    try:
        mtime = video_path.stat().st_mtime
        cache_key = str(video_path.resolve())
        if cache_key in _compat_cache and _compat_cache[cache_key][0] == mtime:
            return _compat_cache[cache_key][1]
    except Exception:
        mtime = 0.0
        cache_key = str(video_path)

    # 1. Fast-path: OpenCV FourCC inspection (sub-millisecond)
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            cap.release()
            fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).lower()
            if fourcc_str in ("avc1", "h264", "x264") and suffix == ".mp4":
                _compat_cache[cache_key] = (mtime, True)
                return True
            if fourcc_str in ("hevc", "h265", "hev1"):
                _compat_cache[cache_key] = (mtime, False)
                return False
    except Exception as exc:
        logger.warning(f"OpenCV FourCC check failed for {video_path}: {exc}")

    # 2. Secondary path: Probe stream info using ffmpeg
    ffmpeg_exe = get_ffmpeg_binary()
    if ffmpeg_exe:
        try:
            probe = subprocess.run(
                [ffmpeg_exe, "-i", str(video_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            stderr_lower = probe.stderr.lower()

            if "video: hevc" in stderr_lower or "video: h265" in stderr_lower or "hev1" in stderr_lower:
                _compat_cache[cache_key] = (mtime, False)
                return False

            if suffix == ".mp4":
                if ("video: h264" in stderr_lower or "avc1" in stderr_lower) and "yuv420p" in stderr_lower:
                    _compat_cache[cache_key] = (mtime, True)
                    return True
            elif suffix == ".webm":
                if "video: vp8" in stderr_lower or "video: vp9" in stderr_lower or "video: av1" in stderr_lower:
                    _compat_cache[cache_key] = (mtime, True)
                    return True

            _compat_cache[cache_key] = (mtime, False)
            return False
        except Exception as exc:
            logger.warning(f"ffmpeg probe error for {video_path}: {exc}")

    _compat_cache[cache_key] = (mtime, False)
    return False


def transcode_to_h264(input_path: Path, output_path: Path) -> Path:
    """
    Transcode an input video file into a browser-compatible H.264 + AAC MP4 file.
    Uses yuv420p pixel format and +faststart flag for instant streaming.
    Transcodes into a temporary file first and renames atomically upon completion.
    """
    ffmpeg_exe = get_ffmpeg_binary()
    if not ffmpeg_exe:
        raise PlaybackConversionError("No ffmpeg executable available on the system to convert video.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_path.with_name(f"{output_path.name}.tmp.mp4")

    if temp_output.exists():
        try:
            temp_output.unlink()
        except Exception:
            pass

    cmd = [
        ffmpeg_exe,
        "-y",
        "-threads", "0",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(temp_output),
    ]

    logger.info(f"Starting transcode for playback: {input_path.name} -> {output_path.name}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min max for very long surveillance videos
        )
        if result.returncode != 0:
            err_msg = result.stderr[-1000:] if result.stderr else "Unknown ffmpeg error"
            logger.error(f"ffmpeg transcode failed with code {result.returncode}: {err_msg}")
            if temp_output.exists():
                temp_output.unlink()
            raise PlaybackConversionError(f"Video transcoding failed: {err_msg}")

        if not temp_output.exists() or temp_output.stat().st_size == 0:
            raise PlaybackConversionError("Transcode completed but output file is missing or empty.")

        # Atomic rename
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception:
                pass
        temp_output.replace(output_path)
        logger.info(f"Playback video ready: {output_path} ({output_path.stat().st_size} bytes)")
        return output_path

    except subprocess.TimeoutExpired:
        if temp_output.exists():
            try:
                temp_output.unlink()
            except Exception:
                pass
        raise PlaybackConversionError("Video transcoding timed out.")
    except Exception as exc:
        if temp_output.exists():
            try:
                temp_output.unlink()
            except Exception:
                pass
        if isinstance(exc, PlaybackConversionError):
            raise
        raise PlaybackConversionError(f"Transcoding error: {exc}")


def ensure_playback_file(video_id: str, original_path: Path) -> Path:
    """
    Return a browser-compatible playback file for the given video.
    If original is already browser-compatible, returns original_path directly.
    If not, checks for existing playback file. If none, transcodes once and saves
    to storage/playback/{video_id}_playback.mp4.
    """
    if not original_path.exists():
        raise PlaybackNotFoundError(f"Original video for '{video_id}' does not exist.")

    # 1. If converted playback file already exists, use it
    playback_file = settings.STORAGE_PLAYBACK_DIR / f"{video_id}_playback.mp4"
    if playback_file.exists() and playback_file.stat().st_size > 0:
        return playback_file

    # 2. If original is already browser-compatible, use original directly (no conversion)
    if is_browser_compatible(original_path):
        return original_path

    # 3. Transcoding is required. Lock per video_id to avoid duplicate jobs.
    with _master_lock:
        if video_id not in _conversion_locks:
            _conversion_locks[video_id] = threading.Lock()
        lock = _conversion_locks[video_id]

    with lock:
        # Re-check inside lock
        if playback_file.exists() and playback_file.stat().st_size > 0:
            return playback_file

        with _master_lock:
            _in_progress_conversions.add(video_id)

        try:
            return transcode_to_h264(original_path, playback_file)
        finally:
            with _master_lock:
                _in_progress_conversions.discard(video_id)


def get_playback_status(video_id: str, original_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Check the current playback availability and conversion status for a video.
    """
    with _master_lock:
        is_converting = video_id in _in_progress_conversions

    if is_converting:
        return {
            "video_id": video_id,
            "status": "converting",
            "is_compatible": False,
            "is_transcoded": False,
            "message": "Preparing browser-compatible playback...",
        }

    playback_file = settings.STORAGE_PLAYBACK_DIR / f"{video_id}_playback.mp4"
    if playback_file.exists() and playback_file.stat().st_size > 0:
        return {
            "video_id": video_id,
            "status": "ready",
            "is_compatible": True,
            "is_transcoded": True,
            "message": "Browser-compatible playback stream ready.",
        }

    if original_path and original_path.exists():
        if is_browser_compatible(original_path):
            return {
                "video_id": video_id,
                "status": "ready",
                "is_compatible": True,
                "is_transcoded": False,
                "message": "Native browser playback compatible.",
            }
        else:
            return {
                "video_id": video_id,
                "status": "needs_conversion",
                "is_compatible": False,
                "is_transcoded": False,
                "message": "Original format requires conversion for browser playback.",
            }

    return {
        "video_id": video_id,
        "status": "unavailable",
        "is_compatible": False,
        "is_transcoded": False,
        "message": "Video file not found.",
    }


def ensure_evidence_clip_playback(evidence_id: str, original_clip_path: Path) -> Path:
    """
    Return a browser-compatible playback file for an evidence video clip.
    If the original clip is already browser-compatible H.264 MP4, returns original_clip_path.
    If not, checks for existing playback file in storage/evidence_playback/.
    If none, transcodes once to H.264 (yuv420p + AAC + faststart).
    """
    if not original_clip_path.exists():
        raise PlaybackNotFoundError(f"Original evidence clip for '{evidence_id}' not found.")

    playback_file = settings.STORAGE_EVIDENCE_PLAYBACK_DIR / f"{evidence_id}_clip_playback.mp4"
    if playback_file.exists() and playback_file.stat().st_size > 0:
        return playback_file

    if is_browser_compatible(original_clip_path):
        return original_clip_path

    # Transcoding required with per-evidence mutex
    with _master_lock:
        if evidence_id not in _evidence_conversion_locks:
            _evidence_conversion_locks[evidence_id] = threading.Lock()
        lock = _evidence_conversion_locks[evidence_id]

    with lock:
        if playback_file.exists() and playback_file.stat().st_size > 0:
            return playback_file

        with _master_lock:
            _in_progress_evidence_conversions.add(evidence_id)

        try:
            return transcode_to_h264(original_clip_path, playback_file)
        finally:
            with _master_lock:
                _in_progress_evidence_conversions.discard(evidence_id)


def get_evidence_playback_status(evidence_id: str, original_clip_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Check the current playback availability and conversion status for an evidence clip.
    """
    with _master_lock:
        is_converting = evidence_id in _in_progress_evidence_conversions

    if is_converting:
        return {
            "evidence_id": evidence_id,
            "status": "preparing",
            "is_compatible": False,
            "is_transcoded": False,
            "message": "Preparing browser-compatible evidence clip...",
        }

    playback_file = settings.STORAGE_EVIDENCE_PLAYBACK_DIR / f"{evidence_id}_clip_playback.mp4"
    if playback_file.exists() and playback_file.stat().st_size > 0:
        return {
            "evidence_id": evidence_id,
            "status": "ready",
            "is_compatible": True,
            "is_transcoded": True,
            "message": "Browser-compatible evidence clip stream ready.",
        }

    if original_clip_path and original_clip_path.exists():
        if is_browser_compatible(original_clip_path):
            return {
                "evidence_id": evidence_id,
                "status": "ready",
                "is_compatible": True,
                "is_transcoded": False,
                "message": "Native browser playback compatible.",
            }
        else:
            return {
                "evidence_id": evidence_id,
                "status": "needs_conversion",
                "is_compatible": False,
                "is_transcoded": False,
                "message": "Evidence clip requires conversion for browser playback.",
            }

    return {
        "evidence_id": evidence_id,
        "status": "failed",
        "is_compatible": False,
        "is_transcoded": False,
        "message": "Evidence clip not found.",
    }


def stream_video_file_with_ranges(file_path: Path, range_header: Optional[str] = None) -> Response:
    """
    Stream a video file supporting RFC 7233 HTTP Range requests (206 Partial Content).
    Enables seeking/scrubbing across the video timeline in modern web browsers.
    Streams via a chunked generator (64 KB chunks) so the entire file is never
    buffered in memory.
    """
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Playback file not found.",
        )

    file_size = file_path.stat().st_size
    content_type = "video/mp4"
    chunk_size = 64 * 1024  # 64 KB

    if not range_header:
        # Full content response (200 OK)
        def full_file_iterator():
            with open(file_path, "rb") as f:
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    yield data

        return StreamingResponse(
            full_file_iterator(),
            status_code=status.HTTP_200_OK,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
                "Content-Type": content_type,
            },
        )

    # Range header present: parse "bytes=start-end"
    match = re.match(r"bytes=(\d*)-(\d*)", range_header.strip())
    if not match:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid Range header format '{range_header}'.",
        )

    start_str, end_str = match.groups()
    if start_str and end_str:
        start = int(start_str)
        end = int(end_str)
    elif start_str:
        start = int(start_str)
        end = file_size - 1
    elif end_str:
        start = max(0, file_size - int(end_str))
        end = file_size - 1
    else:
        start = 0
        end = file_size - 1

    # Bounds check
    if start >= file_size or start > end:
        return Response(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            headers={"Content-Range": f"bytes */{file_size}"},
        )

    end = min(end, file_size - 1)
    content_length = (end - start) + 1

    def range_file_iterator():
        remaining = content_length
        with open(file_path, "rb") as f:
            f.seek(start)
            while remaining > 0:
                read_len = min(chunk_size, remaining)
                data = f.read(read_len)
                if not data:
                    break
                remaining -= len(data)
                yield data

    return StreamingResponse(
        range_file_iterator(),
        status_code=status.HTTP_206_PARTIAL_CONTENT,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(content_length),
            "Content-Type": content_type,
        },
    )
