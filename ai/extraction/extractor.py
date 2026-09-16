"""
Evidence Clip Extraction Module for Sentinel

Extracts targeted video sub-clips corresponding to detection events
and saves them to storage/evidence/.
"""
from pathlib import Path
from typing import Dict, Any
import cv2
import logging

logger = logging.getLogger(__name__)


class ClipExtractor:
    """Extracts and persists targeted evidence clips without modifying source video."""

    def __init__(self, output_dir: str = "storage/evidence"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_clip(
        self,
        source_video_path: str,
        start_time: float,
        end_time: float,
        clip_id: str,
    ) -> Dict[str, Any]:
        """Extract a sub-clip using OpenCV VideoWriter."""
        source = Path(source_video_path)
        if not source.exists():
            raise FileNotFoundError(f"Source video not found: {source_video_path}")

        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {source_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if total_frames > 0 else end_time

        c_start = max(0.0, start_time)
        c_end = min(duration, end_time) if duration > 0 else end_time
        if c_end <= c_start:
            c_end = c_start + 1.0

        start_frame = int(round(c_start * fps))
        end_frame = int(round(c_end * fps))

        output_file = self.output_dir / f"{clip_id}_clip.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current = start_frame

        while current <= end_frame:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            writer.write(frame)
            current += 1

        writer.release()
        cap.release()

        return {
            "clip_id": clip_id,
            "path": str(output_file),
            "start_time": c_start,
            "end_time": c_end,
            "duration": round(c_end - c_start, 2),
            "file_size": output_file.stat().st_size if output_file.exists() else 0,
        }
