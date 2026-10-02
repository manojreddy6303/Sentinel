"""
Phase 20.3: Production Memory Hardening & Railway OOM Prevention Tests

Verifies:
1. Configured memory-safe batch size (YOLO_BATCH_SIZE default = 4).
2. Configured memory-safe frame cache budget (FRAME_CACHE_MAX_MEMORY_MB default <= 64 MB).
3. BoundedFrameCache memory budget enforcement and clear() cleanup.
4. Concurrency guard (_heavy_processing_semaphore) preventing concurrent heavy processing.
5. Post-processing memory cleanup and garbage collection.
6. Crash-safe stale processing job recovery.
7. Sequential video processing memory stability without monotonic accumulation.
"""
import os
import gc
import psutil
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.app.core.config import settings
from backend.app.api.videos import (
    recover_stale_processing_jobs,
    _heavy_processing_semaphore,
    _get_detector,
)
from ai.video.frame_cache import BoundedFrameCache
import numpy as np


def test_memory_hardening_configuration_defaults():
    """Verify production settings match Phase 20.3 memory constraints."""
    assert settings.YOLO_BATCH_SIZE == 4, f"Expected default batch size 4, got {settings.YOLO_BATCH_SIZE}"
    assert settings.FRAME_CACHE_MAX_MEMORY_MB <= 96.0, f"Cache limit too high: {settings.FRAME_CACHE_MAX_MEMORY_MB}"
    assert settings.FRAME_CACHE_MAX_MEMORY_MB == 64.0, f"Expected 64.0 MB default, got {settings.FRAME_CACHE_MAX_MEMORY_MB}"
    assert settings.MAX_CONCURRENT_HEAVY_JOBS == 1, f"Expected single heavy job concurrency, got {settings.MAX_CONCURRENT_HEAVY_JOBS}"
    assert settings.TORCH_NUM_THREADS <= 2, f"Torch threads should be <= 2, got {settings.TORCH_NUM_THREADS}"


def test_frame_cache_budget_and_cleanup():
    """Verify BoundedFrameCache enforces max memory budget and completely clears on cleanup."""
    cache = BoundedFrameCache(max_frames=100, max_memory_mb=2.0)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Insert 30 frames
    for i in range(30):
        cache[float(i)] = dummy_frame

    stats = cache.get_memory_stats()
    assert stats["cached_frames"] > 0
    assert stats["compressed_memory_mb"] <= 2.0, f"Cache exceeded memory cap: {stats['compressed_memory_mb']} MB"

    # Verify clear() releases all memory and references
    cache.clear()
    assert len(cache) == 0
    clean_stats = cache.get_memory_stats()
    assert clean_stats["cached_frames"] == 0
    assert clean_stats["compressed_memory_mb"] == 0.0


def test_heavy_processing_concurrency_guard():
    """Verify heavy processing semaphore serializes concurrent heavy jobs."""
    # When semaphore is acquired, second acquire must fail (blocking=False)
    assert _heavy_processing_semaphore.acquire(blocking=False) is True
    try:
        # Second acquire should fail because MAX_CONCURRENT_HEAVY_JOBS == 1
        assert _heavy_processing_semaphore.acquire(blocking=False) is False
    finally:
        _heavy_processing_semaphore.release()

    # Semaphore should now be free again
    assert _heavy_processing_semaphore.acquire(blocking=False) is True
    _heavy_processing_semaphore.release()


def test_stale_processing_job_recovery(tmp_path):
    """Verify recover_stale_processing_jobs safely transitions stale 'processing' videos to 'interrupted'."""
    from database.session import SessionLocal
    from database.models import VideoModel

    test_video_id = "test_stale_recovery_video_001"
    db = SessionLocal()
    try:
        # Create a test video record in 'processing' status
        v = db.query(VideoModel).filter(VideoModel.id == test_video_id).first()
        if not v:
            v = VideoModel(
                id=test_video_id,
                original_filename="stale_test.mp4",
                storage_path="storage/uploads/stale_test.mp4",
                file_size_bytes=1000,
                status="processing",
            )
            db.add(v)
        else:
            v.status = "processing"
        db.commit()

        # Run recovery
        recovered = recover_stale_processing_jobs()
        assert recovered >= 1

        # Verify the record is now 'interrupted'
        db.refresh(v)
        assert v.status == "interrupted"
    finally:
        # Clean up test record
        try:
            db.query(VideoModel).filter(VideoModel.id == test_video_id).delete()
            db.commit()
        except Exception:
            db.rollback()
        db.close()


def test_sequential_processing_bounded_memory():
    """Verify that multiple sequential inference cycles do not exhibit monotonic memory growth."""
    process = psutil.Process(os.getpid())
    gc.collect()
    mem_start = process.memory_info().rss / (1024 * 1024)

    detector = _get_detector()
    dummy_frame = np.ones((640, 640, 3), dtype=np.uint8) * 128

    mem_measurements = []
    # Run 5 sequential simulated batches of size 4
    for cycle in range(5):
        batch = [dummy_frame.copy() for _ in range(settings.YOLO_BATCH_SIZE)]
        timestamps = [float(i) for i in range(settings.YOLO_BATCH_SIZE)]
        _ = detector.detect_batch(batch, timestamps)

        # Explicit cleanup as done in process_video finally block
        del batch
        del timestamps
        gc.collect()

        current_mem = process.memory_info().rss / (1024 * 1024)
        mem_measurements.append(current_mem)

    # Check that memory after cycle 2 and cycle 5 are within bounded delta (< 50 MB growth)
    growth_between_cycles = mem_measurements[-1] - mem_measurements[1]
    assert growth_between_cycles < 50.0, f"Monotonic memory leak detected: growth of {growth_between_cycles:.1f} MB across cycles"
