"""
SENTINEL — Phase 21B Multi-Strategy Tracking & Sampling Benchmark Matrix

Evaluates:
1. ByteTrack + 1 FPS (Production baseline)
2. BoT-SORT + 1 FPS without fallback (Phase 21A candidate)
3. BoT-SORT + centroid fallback + 1 FPS (Phase 21B candidate)
4. BoT-SORT + centroid fallback + 3 FPS
5. BoT-SORT + centroid fallback + 5 FPS
6. BoT-SORT + centroid fallback + Adaptive sampling

Outputs detailed tracking metrics, latency, memory, and downstream parity.
Saves JSON to scratch/phase21b_benchmark_results.json.
"""

import os
import sys
import time
import math
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Tuple
from statistics import median

import cv2
import numpy as np
import psutil

# Add repository root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.schemas import BoundingBox, TrackedObject, TrackLifecycleState
from ai.tracking.tracker import ObjectTracker
from ai.tracking.botsort_tracker import BoTSORTTracker
from ai.video.processor import VideoProcessor
from ai.detection.detector import YOLODetector


def get_process_rss_mb() -> float:
    """Get current process resident set size in megabytes."""
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def compute_tracking_metrics(all_tracks: List[TrackedObject]) -> Dict[str, Any]:
    """Compute comprehensive tracking metrics."""
    if not all_tracks:
        return {
            "total_tracks": 0,
            "confirmed_tracks": 0,
            "single_frame_tracks": 0,
            "single_frame_pct": 0.0,
            "avg_track_length": 0.0,
            "avg_track_duration": 0.0,
            "median_track_duration": 0.0,
            "max_track_duration": 0.0,
            "fragmentation_proxy": 0,
            "id_switch_proxy": 0,
        }

    total = len(all_tracks)
    confirmed = sum(1 for t in all_tracks if t.detection_count >= 2)
    single = sum(1 for t in all_tracks if t.detection_count == 1)
    single_pct = round((single / total) * 100.0, 1) if total > 0 else 0.0
    avg_len = round(sum(t.detection_count for t in all_tracks) / total, 2)
    durations = [t.duration_seconds for t in all_tracks]
    max_dur = round(max(durations), 2) if durations else 0.0
    avg_dur = round(sum(durations) / total, 2) if durations else 0.0
    med_dur = round(median(durations), 2) if durations else 0.0

    # ID switch proxy: same-class tracks that overlap in time and end/start close
    tracks_by_class: Dict[str, List[TrackedObject]] = {}
    for t in all_tracks:
        tracks_by_class.setdefault(t.object_class, []).append(t)

    id_switches = 0
    for cls, cls_tracks in tracks_by_class.items():
        for i, t1 in enumerate(cls_tracks):
            for t2 in cls_tracks[i + 1:]:
                if t1.last_seen < t2.first_seen or t2.last_seen < t1.first_seen:
                    continue
                if t1.trajectory and t2.trajectory:
                    _, cx1, cy1 = t1.trajectory[-1]
                    _, cx2, cy2 = t2.trajectory[0]
                    dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                    if dist < 50:
                        id_switches += 1

    # Fragmentation proxy: same-class tracks with small time gap and close position
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
        "single_frame_pct": single_pct,
        "avg_track_length": avg_len,
        "avg_track_duration": avg_dur,
        "median_track_duration": med_dur,
        "max_track_duration": max_dur,
        "fragmentation_proxy": fragmentations,
        "id_switch_proxy": id_switches,
    }


def load_db_detections(cursor, video_id: str) -> Dict[float, List[Dict[str, Any]]]:
    """Load precomputed 1.0 FPS detections from database."""
    cursor.execute(
        "SELECT timestamp_seconds, object_class, confidence, "
        "bbox_x1, bbox_y1, bbox_x2, bbox_y2, validation_status "
        "FROM events WHERE video_id = ? ORDER BY timestamp_seconds",
        (video_id,),
    )
    rows = cursor.fetchall()
    events_by_time: Dict[float, List[Dict[str, Any]]] = {}
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


def run_tracker(
    tracker,
    events_by_time: Dict[float, List[Dict[str, Any]]],
    video_path: str,
    fps: float,
    width: float,
    height: float,
    video_id: str,
) -> Tuple[List[TrackedObject], float, float]:
    """Run tracker on events and return (all_tracks, avg_update_ms, peak_rss_mb)."""
    tracker.reset(video_id=video_id)
    sorted_ts = sorted(events_by_time.keys())

    cap = None
    use_frames = hasattr(tracker, "enable_gmc") and tracker.enable_gmc
    if use_frames and video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap = None

    peak_rss = get_process_rss_mb()
    update_times_ms = []
    current_cap_idx = 0

    for t in sorted_ts:
        dets = events_by_time[t]
        frame_bgr = None
        if cap is not None:
            target_idx = int(round(t * fps))
            if target_idx >= current_cap_idx:
                while current_cap_idx < target_idx:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    current_cap_idx += 1
                    if current_cap_idx == target_idx:
                        frame_bgr = frame
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                ret, frame = cap.read()
                if ret:
                    frame_bgr = frame
                    current_cap_idx = target_idx

        t_start = time.perf_counter()
        tracker.update(
            timestamp=t,
            detections=dets,
            frame_width=width,
            frame_height=height,
            frame_bgr=frame_bgr,
        )
        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        update_times_ms.append(t_elapsed)

        current_rss = get_process_rss_mb()
        if current_rss > peak_rss:
            peak_rss = current_rss

    if cap is not None:
        cap.release()

    all_tracks = tracker.finalize()
    avg_update_ms = round(sum(update_times_ms) / max(1, len(update_times_ms)), 3)
    return all_tracks, avg_update_ms, round(peak_rss, 1)


def main():
    print("=" * 90)
    print("SENTINEL — Phase 21B: Multi-Strategy Tracking & Sampling Benchmark Matrix")
    print("=" * 90)

    db_path = "storage/sentinel.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    benchmark_videos = [
        {
            "name": "burglary",
            "video_id": "0d4d92f9-19f8-42e3-925f-1931cb557705",
            "file": "storage/uploads/0d4d92f9-19f8-42e3-925f-1931cb557705_uccrime_Burglary010_x264.mp4",
            "width": 320.0,
            "height": 240.0,
            "fps": 30.0,
        },
        {
            "name": "4k_uhd",
            "video_id": "3426f64b-dd44-48a7-8e29-2c5f77b748bb",
            "file": "storage/uploads/3426f64b-dd44-48a7-8e29-2c5f77b748bb_12566041-uhd_3840_2160_30fps.mp4",
            "width": 3840.0,
            "height": 2160.0,
            "fps": 29.97,
        },
        {
            "name": "highway",
            "video_id": "a94e46c6-3742-44c1-84b9-24f7be1a6a97",
            "file": "storage/uploads/a94e46c6-3742-44c1-84b9-24f7be1a6a97_17.avi",
            "width": 720.0,
            "height": 576.0,
            "fps": 25.0,
        },
        {
            "name": "virat",
            "video_id": "b118ab48-0fc5-4e14-8524-0205b27a831d",
            "file": "storage/uploads/b118ab48-0fc5-4e14-8524-0205b27a831d_VIRAT_S_010204_05_000856_000890.mp4",
            "width": 1280.0,
            "height": 720.0,
            "fps": 23.97,
        },
        {
            "name": "whatsapp",
            "video_id": "e63e72b1-6616-4be6-b732-110219a81005",
            "file": "storage/uploads/e63e72b1-6616-4be6-b732-110219a81005_WhatsApp_Video_2026-09-26_at_06.15.55.mp4",
            "width": 1280.0,
            "height": 720.0,
            "fps": 30.03,
        },
    ]

    all_results = {}

    for bvid in benchmark_videos:
        v_name = bvid["name"]
        v_id = bvid["video_id"]
        v_file = bvid["file"]
        w = bvid["width"]
        h = bvid["height"]
        fps = bvid["fps"]

        print(f"\n{'=' * 75}")
        print(f"  BENCHMARK: {v_name.upper()} ({os.path.basename(v_file)})")
        print(f"{'=' * 75}")

        # Load 1.0 FPS detections from database
        events_1fps = load_db_detections(cursor, v_id)
        print(f"  Loaded {sum(len(d) for d in events_1fps.values())} detections across {len(events_1fps)} frames (1.0 FPS)")

        vid_results = {}

        # -------------------------------------------------------------
        # 1. ByteTrack + 1.0 FPS (Production Default)
        # -------------------------------------------------------------
        bt = ObjectTracker(video_id=v_id)
        bt_tracks, bt_lat, bt_rss = run_tracker(bt, events_1fps, v_file, fps, w, h, v_id)
        m_bt = compute_tracking_metrics(bt_tracks)
        m_bt["update_ms"] = bt_lat
        m_bt["peak_rss_mb"] = bt_rss
        vid_results["bytetrack_1fps"] = m_bt

        # -------------------------------------------------------------
        # 2. BoT-SORT + 1.0 FPS without fallback (Phase 21A baseline)
        # -------------------------------------------------------------
        bs_21a = BoTSORTTracker(video_id=v_id, enable_centroid_fallback=False)
        bs_21a_tracks, bs_21a_lat, bs_21a_rss = run_tracker(bs_21a, events_1fps, v_file, fps, w, h, v_id)
        m_21a = compute_tracking_metrics(bs_21a_tracks)
        m_21a["update_ms"] = bs_21a_lat
        m_21a["peak_rss_mb"] = bs_21a_rss
        vid_results["botsort_21a_1fps"] = m_21a

        # -------------------------------------------------------------
        # 3. BoT-SORT + centroid fallback + 1.0 FPS (Phase 21B candidate)
        # -------------------------------------------------------------
        bs_21b = BoTSORTTracker(video_id=v_id, enable_centroid_fallback=True, max_distance_threshold=0.25)
        bs_21b_tracks, bs_21b_lat, bs_21b_rss = run_tracker(bs_21b, events_1fps, v_file, fps, w, h, v_id)
        m_21b = compute_tracking_metrics(bs_21b_tracks)
        m_21b["update_ms"] = bs_21b_lat
        m_21b["peak_rss_mb"] = bs_21b_rss
        vid_results["botsort_21b_1fps_fallback"] = m_21b

        # Print comparison table for 1.0 FPS strategies
        print(f"\n  {'Strategy':<30} {'Tracks':<8} {'Single':<8} {'Single%':<9} {'AvgDur(s)':<10} {'MedDur(s)':<10} {'Latency':<9} {'PeakRSS'}")
        print("  " + "-" * 90)
        for s_key, s_label in [
            ("bytetrack_1fps", "ByteTrack (Production)"),
            ("botsort_21a_1fps", "BoT-SORT 21A (No Fallback)"),
            ("botsort_21b_1fps_fallback", "BoT-SORT 21B (With Fallback)"),
        ]:
            m = vid_results[s_key]
            print(f"  {s_label:<30} {m['total_tracks']:<8} {m['single_frame_tracks']:<8} {m['single_frame_pct']:<8}% {m['avg_track_duration']:<10} {m['median_track_duration']:<10} {m['update_ms']:<7}ms {m['peak_rss_mb']:<6}MB")

        all_results[v_name] = vid_results

    # -----------------------------------------------------------------
    # Cross-Sampling Rate Evaluation on Representative Benchmark (VIRAT & Highway)
    # -----------------------------------------------------------------
    print(f"\n{'=' * 90}")
    print("  TEMPORAL SAMPLING EVALUATION (1 FPS vs 3 FPS vs 5 FPS vs Adaptive)")
    print(f"{'=' * 90}")

    sampling_videos = [
        {"name": "virat", "file": "storage/uploads/b118ab48-0fc5-4e14-8524-0205b27a831d_VIRAT_S_010204_05_000856_000890.mp4", "w": 1280.0, "h": 720.0, "fps": 23.97, "max_f": 75},
        {"name": "highway", "file": "storage/uploads/a94e46c6-3742-44c1-84b9-24f7be1a6a97_17.avi", "w": 720.0, "h": 576.0, "fps": 25.0, "max_f": 60},
    ]

    detector = YOLODetector(confidence_threshold=0.20)
    sampling_results = {}

    for sv in sampling_videos:
        sname = sv["name"]
        sfile = sv["file"]
        sw = sv["w"]
        sh = sv["h"]
        sfps = sv["fps"]
        max_f = sv["max_f"]

        print(f"\n  Evaluating temporal sampling on: {sname.upper()} ({os.path.basename(sfile)})")
        sv_dict = {}

        for mode, rate_fps, is_adapt in [
            ("1 FPS", 1.0, False),
            ("3 FPS", 3.0, False),
            ("5 FPS", 5.0, False),
            ("Adaptive (1-3 FPS)", 1.0, True),
        ]:
            proc = VideoProcessor(
                video_path=sfile,
                sample_rate_fps=rate_fps,
                adaptive=is_adapt,
                min_fps=1.0,
                max_burst_fps=3.0,
                motion_threshold=3.0,
            )
            # Sample frames (capped to max_f for fast evaluation)
            sampled_frames = []
            for i, f_tuple in enumerate(proc.sample_frames()):
                sampled_frames.append(f_tuple)
                if i >= max_f:
                    break

            # Run YOLO detector on sampled frames
            t_det_start = time.perf_counter()
            frame_images = [f[2] for f in sampled_frames]
            frame_timestamps = [f[1] for f in sampled_frames]
            raw_dets = detector.detect_batch(frame_images, frame_timestamps)
            det_time = time.perf_counter() - t_det_start

            # Group detections by timestamp
            events_by_ts: Dict[float, List[Dict[str, Any]]] = {}
            for f_idx, ts in enumerate(frame_timestamps):
                events_by_ts[round(ts, 3)] = raw_dets[f_idx]

            # Run BoT-SORT with centroid fallback
            tracker = BoTSORTTracker(enable_centroid_fallback=True, max_distance_threshold=0.25)
            tracks, trk_lat, trk_rss = run_tracker(
                tracker, events_by_ts, sfile, sfps, sw, sh, sname
            )
            metrics = compute_tracking_metrics(tracks)
            metrics["frame_count"] = len(sampled_frames)
            metrics["det_fps"] = round(len(sampled_frames) / max(0.01, det_time), 1)
            metrics["trk_latency_ms"] = trk_lat
            metrics["peak_rss_mb"] = trk_rss
            sv_dict[mode] = metrics

            print(f"    {mode:<20}: {len(sampled_frames)} frames | Tracks: {metrics['total_tracks']:<3} | Single: {metrics['single_frame_tracks']:<2} ({metrics['single_frame_pct']}%) | AvgDur: {metrics['avg_track_duration']}s | TrkLat: {trk_lat}ms")

        sampling_results[sname] = sv_dict

    # Save all results to JSON
    scratch_dir = PROJECT_ROOT / "scratch"
    scratch_dir.mkdir(exist_ok=True)
    out_json = scratch_dir / "phase21b_benchmark_results.json"
    with open(out_json, "w") as f:
        json.dump({"strategies_1fps": all_results, "sampling_rates": sampling_results}, f, indent=2)

    print(f"\n{'=' * 90}")
    print(f"  Benchmark complete. Detailed results saved to: {out_json}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
