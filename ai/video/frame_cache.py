"""
Bounded Memory-Safe Frame Cache (Phase 20)

Provides a memory-bounded, dict-compatible frame cache designed for surveillance
video processing pipelines. Prevents Out-Of-Memory (OOM) conditions during 4K UHD
and long-duration video intelligence without losing source-grounded frame fidelity.

Features:
- JPEG in-memory compression (typically 15x-30x RAM reduction vs raw BGR).
- Maximum memory budget enforcement (eviction / pruning).
- Small LRU cache for recently accessed uncompressed frames to eliminate decode thrashing.
- Optional video_path fallback for source-grounded retrieval if a frame is evicted.
- Fully compatible with `Dict[float, np.ndarray]` interface.
"""

import collections
import logging
from typing import Dict, Any, Optional, Iterator, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BoundedFrameCache:
    """
    A memory-bounded frame cache storing JPEG-compressed frames with an LRU
    read buffer, preventing raw 4K BGR arrays from exhausting system memory.
    """

    def __init__(
        self,
        max_frames: int = 1200,
        max_memory_mb: float = 512.0,
        jpeg_quality: int = 85,
        max_dimension: Optional[int] = 1920,
        video_path: Optional[str] = None,
    ):
        """
        Args:
            max_frames: Hard upper bound on number of cached frames.
            max_memory_mb: Maximum memory in megabytes for compressed buffers.
            jpeg_quality: JPEG compression quality (1-100). 85 preserves visual fidelity.
            max_dimension: Optional max width/height to downscale before compression
                           (e.g., 1920 bounds 4K to 1080p proxy for visual analytics).
                           None preserves full resolution.
            video_path: Optional path to source video for fallback retrieval.
        """
        self.max_frames = max(1, int(max_frames))
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.jpeg_quality = max(1, min(100, int(jpeg_quality)))
        self.max_dimension = max_dimension
        self.video_path = video_path

        # Primary storage: timestamp -> compressed_bytes (np.ndarray uint8)
        self._compressed_store: Dict[float, np.ndarray] = collections.OrderedDict()
        self._frame_shapes: Dict[float, Tuple[int, int, int]] = {}
        self._current_memory_bytes: int = 0
        self._raw_equivalent_bytes: int = 0

        # LRU cache for recently decoded uncompressed frames (fast repeat access)
        self._lru_decoded: Dict[float, np.ndarray] = collections.OrderedDict()
        self._lru_capacity: int = 8

    def __setitem__(self, timestamp: float, frame_bgr: np.ndarray) -> None:
        """Compress and store frame at given timestamp."""
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3:
            return

        ts_key = round(float(timestamp), 3)

        # Check frame dimensions and downscale proxy if needed
        h, w, c = frame_bgr.shape
        raw_size = frame_bgr.nbytes
        processed_frame = frame_bgr

        if self.max_dimension and (w > self.max_dimension or h > self.max_dimension):
            scale = float(self.max_dimension) / float(max(w, h))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            processed_frame = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # JPEG encode
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        success, enc_buf = cv2.imencode(".jpg", processed_frame, encode_params)
        if not success:
            logger.warning("Failed to encode frame at timestamp %.3f", ts_key)
            return

        # If key already exists, subtract old size
        if ts_key in self._compressed_store:
            self._current_memory_bytes -= self._compressed_store[ts_key].nbytes
            self._raw_equivalent_bytes -= raw_size

        # Enforce frame count limit by evicting oldest
        while len(self._compressed_store) >= self.max_frames:
            oldest_key, oldest_buf = self._compressed_store.popitem(last=False)
            self._current_memory_bytes -= oldest_buf.nbytes
            if oldest_key in self._frame_shapes:
                old_h, old_w, old_c = self._frame_shapes.pop(oldest_key)
                self._raw_equivalent_bytes -= (old_h * old_w * old_c)
            if oldest_key in self._lru_decoded:
                del self._lru_decoded[oldest_key]

        # Enforce memory byte limit
        new_buf_bytes = enc_buf.nbytes
        while self._current_memory_bytes + new_buf_bytes > self.max_memory_bytes and self._compressed_store:
            oldest_key, oldest_buf = self._compressed_store.popitem(last=False)
            self._current_memory_bytes -= oldest_buf.nbytes
            if oldest_key in self._frame_shapes:
                old_h, old_w, old_c = self._frame_shapes.pop(oldest_key)
                self._raw_equivalent_bytes -= (old_h * old_w * old_c)
            if oldest_key in self._lru_decoded:
                del self._lru_decoded[oldest_key]

        # Store
        self._compressed_store[ts_key] = enc_buf
        self._frame_shapes[ts_key] = (h, w, c)
        self._current_memory_bytes += new_buf_bytes
        self._raw_equivalent_bytes += raw_size

        # Update LRU with uncompressed version
        self._lru_decoded[ts_key] = processed_frame
        self._lru_decoded.move_to_end(ts_key)
        if len(self._lru_decoded) > self._lru_capacity:
            self._lru_decoded.popitem(last=False)

    def __getitem__(self, timestamp: float) -> np.ndarray:
        """Retrieve and decode uncompressed BGR frame."""
        ts_key = round(float(timestamp), 3)

        # 1. Check LRU
        if ts_key in self._lru_decoded:
            self._lru_decoded.move_to_end(ts_key)
            return self._lru_decoded[ts_key]

        # 2. Check compressed store
        if ts_key in self._compressed_store:
            enc_buf = self._compressed_store[ts_key]
            decoded = cv2.imdecode(enc_buf, cv2.IMREAD_COLOR)
            if decoded is not None:
                self._lru_decoded[ts_key] = decoded
                self._lru_decoded.move_to_end(ts_key)
                if len(self._lru_decoded) > self._lru_capacity:
                    self._lru_decoded.popitem(last=False)
                return decoded

        # 3. Disk fallback if video_path is provided
        if self.video_path:
            frame_from_disk = self._retrieve_from_disk(ts_key)
            if frame_from_disk is not None:
                return frame_from_disk

        raise KeyError(f"Timestamp {ts_key} not found in frame cache")

    def _retrieve_from_disk(self, timestamp: float) -> Optional[np.ndarray]:
        """Source-grounded fallback retrieval from original video file."""
        import os
        if not self.video_path or not os.path.exists(self.video_path):
            return None
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return frame
        except Exception as exc:
            logger.debug("Disk fallback retrieval failed for %.3f: %s", timestamp, exc)
        return None

    def __contains__(self, timestamp: float) -> bool:
        ts_key = round(float(timestamp), 3)
        return ts_key in self._compressed_store

    def __len__(self) -> int:
        return len(self._compressed_store)

    def __iter__(self) -> Iterator[float]:
        return iter(self._compressed_store)

    def keys(self):
        return self._compressed_store.keys()

    def values(self):
        for k in self._compressed_store:
            yield self[k]

    def items(self):
        for k in self._compressed_store:
            yield (k, self[k])

    def get(self, timestamp: float, default: Any = None) -> Any:
        try:
            return self[timestamp]
        except KeyError:
            return default

    def clear(self) -> None:
        self._compressed_store.clear()
        self._frame_shapes.clear()
        self._lru_decoded.clear()
        self._current_memory_bytes = 0
        self._raw_equivalent_bytes = 0

    def get_memory_stats(self) -> Dict[str, Any]:
        """Diagnostic telemetry on memory consumption and compression ratio."""
        comp_mb = round(self._current_memory_bytes / (1024 * 1024), 2)
        raw_mb = round(self._raw_equivalent_bytes / (1024 * 1024), 2)
        ratio = round(raw_mb / comp_mb, 1) if comp_mb > 0 else 1.0
        return {
            "cached_frames": len(self._compressed_store),
            "compressed_memory_mb": comp_mb,
            "raw_equivalent_mb": raw_mb,
            "compression_ratio": ratio,
            "memory_saved_mb": round(max(0.0, raw_mb - comp_mb), 2),
            "max_memory_mb": round(self.max_memory_bytes / (1024 * 1024), 2),
        }
