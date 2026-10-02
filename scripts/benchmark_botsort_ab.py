"""
SENTINEL — Phase 21A: BoT-SORT vs ByteTrack A/B Benchmark

Runs both trackers on the same detections from each benchmark video
and compares tracking quality, performance, and memory metrics.

Usage:
    python scripts/benchmark_botsort_ab.py

Benchmark videos:
1. Burglary (uccrime_Burglary010_x264.mp4)
2. 4K UHD (12566041-uhd_3840_2160_30fps.mp4)
3. Highway (17.avi)
4. VIRAT (VIRAT_S_010204_05_000856_000890.mp4)
5. WhatsApp mobile footage

IMPORTANT: This script does NOT modify the database.
"""
import gc
import json
import os
import sys
import time
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import psutil

# Ensure project root is on path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ai.tracking.tracker import ObjectTracker
from ai.tracking.botsort_tracker import BoTSORTTracker
from ai.schemas import TrackLifecycleState


# =============================================================================
# BENCHMARK VIDEO REGISTRY
# =============================================================================
BENCHMARK_VIDEOS = {
    "burglary": "0d4d92f9-19f8-42e3-925f-1931cb557705",
    "4k_uhd": "3426f64b-dd44-48a7-8e29-2c5f77b748bb",
    "highway": "a94e46c6-3742-44c1-84b9-24f7be1a6a97",
    "virat": "b118ab48-0fc5-4e14-8524-0205b27a831d",
    "whatsapp": "e63e72b1-6616-4be6-b732-110219a81005",
}


def get_process_rss_mb():
    """Get current process resident set size in MB."""
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def load_detections_from_db(cursor, video_id):
    """Load raw detections for a video, grouped by timestamp."""
    cursor.execute(
        "SELECT timestamp_seconds, object_class, confidence, "
        "bbox_x1, bbox_y1, bbox_x2, bbox_y2, validation_status "
        "FROM events WHERE video_id = ? ORDER BY timestamp_seconds",
        (video_id,),
    )
    rows = cursor.fetchall()
    events_by_time = {}
    for row in rows:
        ts = round(float(row[0]), 3)
        det = {
            "object_class": row[1],
            "confidence": float(row[2]),
            "bounding_box": {
                "x1": float(row[3]),
                "y1": float(row[4]),
                "x2": float(row[5]),
                "y2": float(row[6]),
            },
            "validation_status": row[7] or "VALID",
        }
        events_by_time.setdefault(ts, []).append(det)
    return events_by_time


def load_video_metadata(cursor, video_id):
    """Load video path and metadata."""
    cursor.execute(
        "SELECT original_filename, storage_path, fps, duration_seconds, frame_count "
        "FROM videos WHERE id = ?",
        (video_id,),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return {
        "filename": row[0],
        "storage_path": row[1],
        "fps": float(row[2]) if row[2] else 30.0,
        "duration": float(row[3]) if row[3] else 0.0,
        "frame_count": int(row[4]) if row[4] else 0,
    }


def compute_tracking_metrics(all_tracks):
    """Compute tracking quality metrics from a list of TrackedObject."""
    if not all_tracks:
        return {
            "total_tracks": 0,
            "confirmed_tracks": 0,
            "single_frame_tracks": 0,
            "avg_track_length": 0.0,
            "max_track_duration": 0.0,
            "avg_track_duration": 0.0,
            "validated_tracks": 0,
            "id_switch_proxy": 0,
            "fragmentation_proxy": 0,
        }

    total = len(all_tracks)
    confirmed = sum(1 for t in all_tracks if t.detection_count >= 2)
    single = sum(1 for t in all_tracks if t.detection_count == 1)
    validated = sum(1 for t in all_tracks if t.is_validated)
    avg_len = sum(t.detection_count for t in all_tracks) / total
    durations = [t.duration_seconds for t in all_tracks]
    max_dur = max(durations)
    avg_dur = sum(durations) / total

    # ID switch proxy: same-class tracks that overlap in time and are spatially close
    id_switches = 0
    tracks_by_class = {}
    for t in all_tracks:
        tracks_by_class.setdefault(t.object_class, []).append(t)

    for cls, cls_tracks in tracks_by_class.items():
        for i, t1 in enumerate(cls_tracks):
            for t2 in cls_tracks[i + 1:]:
                # Check temporal overlap
                if t1.last_seen < t2.first_seen or t2.last_seen < t1.first_seen:
                    continue
                # Check spatial proximity (using first/last positions)
                if t1.trajectory and t2.trajectory:
                    _, cx1, cy1 = t1.trajectory[-1]
                    _, cx2, cy2 = t2.trajectory[0]
                    dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                    if dist < 50:  # within 50 pixels
                        id_switches += 1

    # Fragmentation proxy: same-class tracks with <2s gap at similar position
    fragmentations = 0
    for cls, cls_tracks in tracks_by_class.items():
        sorted_tracks = sorted(cls_tracks, key=lambda t: t.first_seen)
        for i in range(len(sorted_tracks) - 1):
            t1 = sorted_tracks[i]
            t2 = sorted_tracks[i + 1]
            gap = t2.first_seen - t1.last_seen
            if 0 < gap < 2.0 and t1.trajectory and t2.trajectory:
                _, cx1, cy1 = t1.trajectory[-1]
                _, cx2, cy2 = t2.trajectory[0]
                dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                if dist < 80:
                    fragmentations += 1

    return {
        "total_tracks": total,
        "confirmed_tracks": confirmed,
        "single_frame_tracks": single,
        "avg_track_length": round(avg_len, 2),
        "max_track_duration": round(max_dur, 2),
        "avg_track_duration": round(avg_dur, 2),
        "validated_tracks": validated,
        "id_switch_proxy": id_switches,
        "fragmentation_proxy": fragmentations,
    }


def run_tracker_on_video(tracker, events_by_time, video_path, fps, video_id):
    """Run a tracker on a video's detections with optional frame loading for GMC."""
    tracker.reset(video_id=video_id)
    sorted_timestamps = sorted(events_by_time.keys())

    # Try to open video for frame loading
    cap = None
    use_frames = hasattr(tracker, 'enable_gmc') and tracker.enable_gmc
    if use_frames and video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap = None

    rss_before = get_process_rss_mb()
    peak_rss = rss_before
    total_update_ms = 0.0
    current_cap_idx = 0

    for t in sorted_timestamps:
        dets = events_by_time[t]

        # Load frame for GMC-capable tracker
        frame_bgr = None
        if cap is not None:
            frame_idx = int(round(t * fps))
            if frame_idx >= current_cap_idx:
                while current_cap_idx < frame_idx:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    current_cap_idx += 1
                    if current_cap_idx == frame_idx:
                        frame_bgr = frame
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_bgr = frame
                    current_cap_idx = frame_idx

        start = time.perf_counter()
        tracker.update(
            timestamp=t,
            detections=dets,
            frame_bgr=frame_bgr,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        total_update_ms += elapsed_ms

        current_rss = get_process_rss_mb()
        peak_rss = max(peak_rss, current_rss)

    if cap is not None:
        cap.release()

    all_tracks = tracker.finalize()
    rss_after = get_process_rss_mb()

    metrics = compute_tracking_metrics(all_tracks)
    metrics["total_update_ms"] = round(total_update_ms, 1)
    metrics["avg_update_ms"] = round(total_update_ms / max(len(sorted_timestamps), 1), 2)
    metrics["peak_rss_mb"] = round(peak_rss, 1)
    metrics["rss_delta_mb"] = round(rss_after - rss_before, 1)
    metrics["num_frames"] = len(sorted_timestamps)

    return metrics, all_tracks


def main():
    db_path = project_root / "storage" / "sentinel.db"
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    print("=" * 90)
    print("SENTINEL — Phase 21A: BoT-SORT vs ByteTrack A/B Benchmark")
    print("=" * 90)
    print()

    all_results = {}

    for name, video_id in BENCHMARK_VIDEOS.items():
        print(f"\n{'=' * 70}")
        print(f"  BENCHMARK: {name.upper()} ({video_id})")
        print(f"{'=' * 70}")

        meta = load_video_metadata(cursor, video_id)
        if meta is None:
            print(f"  [SKIP] Video not found in database")
            continue

        print(f"  File: {meta['filename']}")
        print(f"  FPS: {meta['fps']}, Duration: {meta['duration']}s")

        events = load_detections_from_db(cursor, video_id)
        if not events:
            print(f"  [SKIP] No detections found")
            continue

        total_dets = sum(len(v) for v in events.values())
        print(f"  Detections: {total_dets} across {len(events)} timestamps")

        # Resolve video path
        video_path = None
        if meta["storage_path"]:
            candidate = project_root / meta["storage_path"]
            if candidate.exists():
                video_path = candidate
            else:
                # Try storage/uploads/
                candidate2 = project_root / "storage" / "uploads" / meta["filename"]
                if candidate2.exists():
                    video_path = candidate2

        # Deep-copy detections for each tracker (they mutate raw dicts)
        import copy
        events_bt = copy.deepcopy(events)
        events_bs = copy.deepcopy(events)

        # --- Run ByteTrack ---
        print(f"\n  Running ByteTrack...")
        gc.collect()
        bt_tracker = ObjectTracker(video_id=video_id)
        bt_metrics, bt_tracks = run_tracker_on_video(
            bt_tracker, events_bt, video_path, meta["fps"], video_id
        )

        # --- Run BoT-SORT ---
        print(f"  Running BoT-SORT...")
        gc.collect()
        bs_tracker = BoTSORTTracker(video_id=video_id, enable_gmc=video_path is not None)
        bs_metrics, bs_tracks = run_tracker_on_video(
            bs_tracker, events_bs, video_path, meta["fps"], video_id
        )

        # --- Print comparison ---
        print(f"\n  {'Metric':<30} {'ByteTrack':>12} {'BoT-SORT':>12} {'Delta':>12}")
        print(f"  {'-' * 66}")

        for key in [
            "total_tracks", "confirmed_tracks", "single_frame_tracks",
            "validated_tracks", "avg_track_length", "max_track_duration",
            "avg_track_duration", "id_switch_proxy", "fragmentation_proxy",
            "total_update_ms", "avg_update_ms", "peak_rss_mb", "rss_delta_mb",
        ]:
            bt_val = bt_metrics.get(key, 0)
            bs_val = bs_metrics.get(key, 0)
            if isinstance(bt_val, float):
                delta = bs_val - bt_val
                print(f"  {key:<30} {bt_val:>12.2f} {bs_val:>12.2f} {delta:>+12.2f}")
            else:
                delta = bs_val - bt_val
                print(f"  {key:<30} {bt_val:>12} {bs_val:>12} {delta:>+12}")

        all_results[name] = {
            "bytetrack": bt_metrics,
            "botsort": bs_metrics,
        }

        del bt_tracker, bs_tracker, bt_tracks, bs_tracks
        gc.collect()

    conn.close()

    # --- Summary ---
    print(f"\n\n{'=' * 90}")
    print("SUMMARY")
    print(f"{'=' * 90}")

    if not all_results:
        print("  No benchmarks were run (no videos found in database).")
        return

    # Aggregate comparison
    total_bt_tracks = sum(r["bytetrack"]["total_tracks"] for r in all_results.values())
    total_bs_tracks = sum(r["botsort"]["total_tracks"] for r in all_results.values())
    total_bt_confirmed = sum(r["bytetrack"]["confirmed_tracks"] for r in all_results.values())
    total_bs_confirmed = sum(r["botsort"]["confirmed_tracks"] for r in all_results.values())
    total_bt_fragments = sum(r["bytetrack"]["fragmentation_proxy"] for r in all_results.values())
    total_bs_fragments = sum(r["botsort"]["fragmentation_proxy"] for r in all_results.values())
    total_bt_switches = sum(r["bytetrack"]["id_switch_proxy"] for r in all_results.values())
    total_bs_switches = sum(r["botsort"]["id_switch_proxy"] for r in all_results.values())
    max_peak_rss = max(r["botsort"]["peak_rss_mb"] for r in all_results.values())

    print(f"\n  {'Aggregate Metric':<35} {'ByteTrack':>12} {'BoT-SORT':>12}")
    print(f"  {'-' * 59}")
    print(f"  {'Total tracks':<35} {total_bt_tracks:>12} {total_bs_tracks:>12}")
    print(f"  {'Total confirmed':<35} {total_bt_confirmed:>12} {total_bs_confirmed:>12}")
    print(f"  {'Total fragmentations':<35} {total_bt_fragments:>12} {total_bs_fragments:>12}")
    print(f"  {'Total ID switch proxies':<35} {total_bt_switches:>12} {total_bs_switches:>12}")
    print(f"  {'BoT-SORT peak RSS (MB)':<35} {max_peak_rss:>12.1f}")
    print(f"  {'Railway limit (MB)':<35} {'1024':>12}")

    # Save results JSON
    results_path = project_root / "scratch" / "phase21a_ab_benchmark_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")
    print(f"\n{'=' * 90}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
