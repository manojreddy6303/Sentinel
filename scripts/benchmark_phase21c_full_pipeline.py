"""
SENTINEL — Phase 21C Full-Pipeline 1 FPS vs 3 FPS Evaluation Benchmark

Measures the COMPLETE end-to-end security intelligence pipeline:
Decoding -> Sampling -> Detection -> Validation -> Tracking ->
Spatial/Temporal Intelligence -> Security Events -> Correlation -> Incidents -> Evidence

Configurations evaluated:
A. ByteTrack + 1 FPS (Production configuration)
B. ByteTrack + 3 FPS (Experimental configuration)
C. BoT-SORT (with fallback) + 1 FPS
D. BoT-SORT (with fallback) + 3 FPS

Across all 5 benchmark videos:
1. Burglary (uccrime_Burglary010_x264.mp4)
2. 4K UHD (12566041-uhd_3840_2160_30fps.mp4)
3. Highway (17.avi)
4. VIRAT (VIRAT_S_010204_05_000856_000890.mp4)
5. WhatsApp Mobile (WhatsApp Video 2026-09-26 at 06.15.55.mp4)

Outputs structured evaluation metrics to scratch/phase21c_pipeline_benchmark.json.
"""

import os
import sys
import time
import math
import json
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
from ai.video.frame_cache import BoundedFrameCache
from ai.detection.detector import YOLODetector
from ai.detection.inference_policy import InferenceResolutionPolicy
from ai.validation.validator import DetectionValidator
from ai.events.generator import EventGenerator
from ai.intelligence_pipeline import SecurityIntelligencePipeline
from backend.app.core.config import settings


def get_process_rss_mb() -> float:
    """Get current process resident set size in megabytes."""
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def compute_track_stats(all_tracks: List[TrackedObject]) -> Dict[str, Any]:
    """Compute detailed tracking metrics."""
    if not all_tracks:
        return {
            "total_tracks": 0,
            "confirmed_tracks": 0,
            "single_frame_tracks": 0,
            "single_frame_pct": 0.0,
            "avg_track_duration": 0.0,
            "median_track_duration": 0.0,
            "max_track_duration": 0.0,
            "avg_track_length": 0.0,
            "fragmentations": 0,
            "id_switches": 0,
        }

    total = len(all_tracks)
    confirmed = sum(1 for t in all_tracks if t.detection_count >= 2)
    single = sum(1 for t in all_tracks if t.detection_count == 1)
    single_pct = round((single / total) * 100.0, 1) if total > 0 else 0.0
    durations = [t.duration_seconds for t in all_tracks]
    lengths = [t.detection_count for t in all_tracks]

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
                    if ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5 < 50:
                        id_switches += 1

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
                if ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5 < 80:
                    fragmentations += 1

    return {
        "total_tracks": total,
        "confirmed_tracks": confirmed,
        "single_frame_tracks": single,
        "single_frame_pct": single_pct,
        "avg_track_duration": round(sum(durations) / max(1, total), 2),
        "median_track_duration": round(median(durations) if durations else 0.0, 2),
        "max_track_duration": round(max(durations) if durations else 0.0, 2),
        "avg_track_length": round(sum(lengths) / max(1, total), 2),
        "fragmentations": fragmentations,
        "id_switches": id_switches,
    }


def extract_video_detections(
    video_path: str,
    video_id: str,
    sampling_fps: float,
    detector: YOLODetector,
    max_frames: int = 9999,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Any, Dict[str, Any], float, float]:
    """Sample frames, run batch YOLO inference and validation ONCE per sampling rate."""
    processor = VideoProcessor(video_path, sample_rate_fps=sampling_fps)
    meta = processor.get_metadata()
    fps = meta["fps"]
    duration_s = meta["duration_seconds"]
    frame_w = float(meta.get("width") or 1280.0)
    frame_h = float(meta.get("height") or 720.0)
    effective_imgsz = InferenceResolutionPolicy.select_inference_size(frame_w, frame_h)

    t0_sample = time.perf_counter()
    sampled_frames: List[Tuple[int, float, np.ndarray]] = []
    frame_cache = BoundedFrameCache(max_frames=1200, max_memory_mb=64, video_path=video_path)

    for f_num, ts, frame_bgr in processor.sample_frames():
        sampled_frames.append((f_num, ts, frame_bgr))
        frame_cache[ts] = frame_bgr
        if len(sampled_frames) >= max_frames:
            break

    sampling_time = time.perf_counter() - t0_sample
    num_sampled = len(sampled_frames)

    # Batch YOLO inference (batch size 4)
    t0_detect = time.perf_counter()
    batch_size = 4
    all_raw_frame_detections = []

    for b_idx in range(0, num_sampled, batch_size):
        chunk = sampled_frames[b_idx : b_idx + batch_size]
        frames_chunk = [f[2] for f in chunk]
        ts_chunk = [f[1] for f in chunk]
        idx_chunk = [f[0] for f in chunk]

        dets_chunk = detector.detect_batch(
            frames_chunk, ts_chunk, idx_chunk, video_id=video_id, imgsz=effective_imgsz
        )
        for (f_num, ts, _), dets in zip(chunk, dets_chunk):
            all_raw_frame_detections.append({
                "frame_number": f_num,
                "timestamp": ts,
                "detections": dets,
            })

    detection_time = time.perf_counter() - t0_detect

    # Validation
    validator = DetectionValidator()
    validator.validate_sequence(
        frame_detections=all_raw_frame_detections,
        frame_width=frame_w,
        frame_height=frame_h,
    )

    n_raw = 0
    n_valid = 0
    n_uncertain = 0
    n_rejected = 0
    for fd in all_raw_frame_detections:
        for d in fd["detections"]:
            n_raw += 1
            st = str(d.get("validation_status", "VALID")).upper()
            if st == "VALID":
                n_valid += 1
            elif st == "UNCERTAIN":
                n_uncertain += 1
            elif st == "REJECTED":
                n_rejected += 1

    det_stats = {
        "raw_detections": n_raw,
        "valid_detections": n_valid,
        "uncertain_detections": n_uncertain,
        "rejected_detections": n_rejected,
        "frames_sampled": num_sampled,
    }

    generator = EventGenerator()
    events = generator.generate_events(
        video_id=video_id,
        video_filename=Path(video_path).name,
        fps=fps,
        video_duration=duration_s,
        frame_detections=all_raw_frame_detections,
    )

    return meta, events, frame_cache, det_stats, sampling_time, detection_time


def run_pipeline_for_tracker(
    meta: Dict[str, Any],
    events: List[Dict[str, Any]],
    frame_cache: Any,
    det_stats: Dict[str, Any],
    sampling_time: float,
    detection_time: float,
    video_path: str,
    video_id: str,
    sampling_fps: float,
    tracker_type: str,
) -> Dict[str, Any]:
    """Execute tracking and security intelligence pipeline on prepared events."""
    proc = psutil.Process(os.getpid())
    rss_start = proc.memory_info().rss / (1024 * 1024)

    fps = meta["fps"]
    duration_s = meta["duration_seconds"]
    frame_w = float(meta.get("width") or 1280.0)
    frame_h = float(meta.get("height") or 720.0)

    if tracker_type == "botsort":
        tracker = BoTSORTTracker(
            enable_centroid_fallback=True,
            max_distance_threshold=0.25,
            video_id=video_id,
        )
    else:
        tracker = ObjectTracker(video_id=video_id)

    t0_intel = time.perf_counter()
    intel_pipe = SecurityIntelligencePipeline(tracker=tracker)
    intel_res = intel_pipe.process_video_intelligence(
        video_id=video_id,
        video_path=video_path,
        raw_events=events,
        fps=fps,
        duration_seconds=duration_s,
        sample_rate_fps=sampling_fps,
        sampled_frames=frame_cache,
        frame_width=frame_w,
        frame_height=frame_h,
    )
    intel_time = time.perf_counter() - t0_intel

    peak_rss = proc.memory_info().rss / (1024 * 1024)
    total_time = sampling_time + detection_time + intel_time

    all_tracks = intel_res["all_tracks"]
    security_events = intel_res["security_events"]
    candidate_incidents = intel_res["incidents"]
    correlated_incidents = intel_res["correlated_incidents"]
    specialized_obs = intel_res.get("specialized_observations", [])

    false_smoke = sum(1 for o in specialized_obs if getattr(o, "class_name", "") == "smoke")
    false_fire = sum(1 for o in specialized_obs if getattr(o, "class_name", "") == "fire")
    false_weapon = sum(1 for o in specialized_obs if getattr(o, "class_name", "") == "weapon")

    evidence_candidates = 0
    for sev in security_events:
        ev_type = getattr(sev, "event_type", None) or (sev.get("event_type") if isinstance(sev, dict) else "")
        if ev_type in ("POTENTIAL_THEFT", "RESTRICTED_ZONE_INTRUSION", "LOITERING"):
            evidence_candidates += 1

    track_stats = compute_track_stats(all_tracks)

    return {
        "video_id": video_id,
        "tracker_type": tracker_type,
        "sampling_fps_requested": sampling_fps,
        "actual_sampling_rate": round(det_stats["frames_sampled"] / max(0.1, duration_s), 2),
        "source_fps": round(fps, 2),
        "source_resolution": f"{int(frame_w)}x{int(frame_h)}",
        "duration_seconds": round(duration_s, 2),
        "frames_sampled": det_stats["frames_sampled"],
        # Performance
        "total_time_s": round(total_time, 2),
        "pipeline_fps": round(det_stats["frames_sampled"] / max(0.01, total_time), 2),
        "sampling_time_s": round(sampling_time, 2),
        "detection_time_s": round(detection_time, 2),
        "intel_time_s": round(intel_time, 2),
        "peak_rss_mb": round(peak_rss, 1),
        # Detections
        "raw_detections": det_stats["raw_detections"],
        "valid_detections": det_stats["valid_detections"],
        "uncertain_detections": det_stats["uncertain_detections"],
        "rejected_detections": det_stats["rejected_detections"],
        # Tracking
        "total_tracks": track_stats["total_tracks"],
        "confirmed_tracks": track_stats["confirmed_tracks"],
        "single_frame_tracks": track_stats["single_frame_tracks"],
        "single_frame_pct": track_stats["single_frame_pct"],
        "avg_track_duration": track_stats["avg_track_duration"],
        "median_track_duration": track_stats["median_track_duration"],
        "max_track_duration": track_stats["max_track_duration"],
        "avg_track_length": track_stats["avg_track_length"],
        "fragmentations": track_stats["fragmentations"],
        "id_switches": track_stats["id_switches"],
        # Downstream Intelligence
        "security_events_total": len(security_events),
        "candidate_incidents": len(candidate_incidents),
        "correlated_incidents": len(correlated_incidents),
        "evidence_candidates": evidence_candidates,
        "specialized_false_alarms": {
            "smoke": false_smoke,
            "fire": false_fire,
            "weapon": false_weapon,
        },
    }


def main():
    print("=" * 95)
    print("SENTINEL — Phase 21C Full-Pipeline 1 FPS vs 3 FPS Evaluation Benchmark")
    print("=" * 95)

    benchmark_videos = [
        {
            "name": "Burglary",
            "video_id": "0d4d92f9-19f8-42e3-925f-1931cb557705",
            "file": "storage/uploads/0d4d92f9-19f8-42e3-925f-1931cb557705_uccrime_Burglary010_x264.mp4",
        },
        {
            "name": "4K_UHD",
            "video_id": "3426f64b-dd44-48a7-8e29-2c5f77b748bb",
            "file": "storage/uploads/3426f64b-dd44-48a7-8e29-2c5f77b748bb_12566041-uhd_3840_2160_30fps.mp4",
        },
        {
            "name": "Highway",
            "video_id": "a94e46c6-3742-44c1-84b9-24f7be1a6a97",
            "file": "storage/uploads/a94e46c6-3742-44c1-84b9-24f7be1a6a97_17.avi",
        },
        {
            "name": "VIRAT",
            "video_id": "b118ab48-0fc5-4e14-8524-0205b27a831d",
            "file": "storage/uploads/b118ab48-0fc5-4e14-8524-0205b27a831d_VIRAT_S_010204_05_000856_000890.mp4",
        },
        {
            "name": "WhatsApp",
            "video_id": "e63e72b1-6616-4be6-b732-110219a81005",
            "file": "storage/uploads/e63e72b1-6616-4be6-b732-110219a81005_WhatsApp_Video_2026-09-26_at_06.15.55.mp4",
        },
    ]

    detector = YOLODetector(confidence_threshold=0.20)
    all_results = {}

    for bvid in benchmark_videos:
        v_name = bvid["name"]
        v_id = bvid["video_id"]
        v_file = bvid["file"]

        print(f"\n{'=' * 85}")
        print(f"  BENCHMARK VIDEO: {v_name} ({os.path.basename(v_file)})")
        print(f"{'=' * 85}")

        vid_results = {}

        # -------------------------------------------------------------
        # Extract 1.0 FPS detections (shared between ByteTrack & BoT-SORT)
        # -------------------------------------------------------------
        print(f"  [1.0 FPS] Extracting frames, running YOLO and validation...")
        meta_1, ev_1, fc_1, ds_1, st_1, dt_1 = extract_video_detections(
            v_file, v_id, 1.0, detector
        )
        print(f"    Sampled {ds_1['frames_sampled']} frames | Raw: {ds_1['raw_detections']} | Valid: {ds_1['valid_detections']}")

        # 1. ByteTrack + 1.0 FPS (Production)
        print(f"    Running ByteTrack (1.0 FPS)...")
        r_bt1 = run_pipeline_for_tracker(meta_1, ev_1, fc_1, ds_1, st_1, dt_1, v_file, v_id, 1.0, "bytetrack")
        vid_results["bytetrack_1fps"] = r_bt1

        # 2. BoT-SORT + 1.0 FPS
        print(f"    Running BoT-SORT Fallback (1.0 FPS)...")
        r_bs1 = run_pipeline_for_tracker(meta_1, ev_1, fc_1, ds_1, st_1, dt_1, v_file, v_id, 1.0, "botsort")
        vid_results["botsort_1fps"] = r_bs1

        # -------------------------------------------------------------
        # Extract 3.0 FPS detections (shared between ByteTrack & BoT-SORT)
        # -------------------------------------------------------------
        print(f"  [3.0 FPS] Extracting frames, running YOLO and validation...")
        meta_3, ev_3, fc_3, ds_3, st_3, dt_3 = extract_video_detections(
            v_file, v_id, 3.0, detector
        )
        print(f"    Sampled {ds_3['frames_sampled']} frames | Raw: {ds_3['raw_detections']} | Valid: {ds_3['valid_detections']}")

        # 3. ByteTrack + 3.0 FPS (Experimental)
        print(f"    Running ByteTrack (3.0 FPS)...")
        r_bt3 = run_pipeline_for_tracker(meta_3, ev_3, fc_3, ds_3, st_3, dt_3, v_file, v_id, 3.0, "bytetrack")
        vid_results["bytetrack_3fps"] = r_bt3

        # 4. BoT-SORT + 3.0 FPS
        print(f"    Running BoT-SORT Fallback (3.0 FPS)...")
        r_bs3 = run_pipeline_for_tracker(meta_3, ev_3, fc_3, ds_3, st_3, dt_3, v_file, v_id, 3.0, "botsort")
        vid_results["botsort_3fps"] = r_bs3

        # Summary print
        print(f"\n  {'Configuration':<30} {'Frames':<7} {'Tracks':<7} {'Single%':<8} {'AvgDur(s)':<10} {'Incidents':<10} {'Evidence':<9} {'Time(s)':<8} {'PeakRSS'}")
        print("  " + "-" * 95)
        for c_key, c_lbl in [
            ("bytetrack_1fps", "ByteTrack (1 FPS - Prod)"),
            ("bytetrack_3fps", "ByteTrack (3 FPS - Exp)"),
            ("botsort_1fps",   "BoT-SORT (1 FPS - 21B)"),
            ("botsort_3fps",   "BoT-SORT (3 FPS - 21B)"),
        ]:
            r = vid_results[c_key]
            print(f"  {c_lbl:<30} {r['frames_sampled']:<7} {r['total_tracks']:<7} {r['single_frame_pct']:<7}% {r['avg_track_duration']:<10} {r['correlated_incidents']:<10} {r['evidence_candidates']:<9} {r['total_time_s']:<8} {r['peak_rss_mb']:<6}MB")

        all_results[v_name] = vid_results

    # Save to JSON
    scratch_dir = PROJECT_ROOT / "scratch"
    scratch_dir.mkdir(exist_ok=True)
    out_file = scratch_dir / "phase21c_pipeline_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 95}")
    print(f"  PHASE 21C BENCHMARK COMPLETE — Results saved to {out_file}")
    print(f"{'=' * 95}")


if __name__ == "__main__":
    main()
