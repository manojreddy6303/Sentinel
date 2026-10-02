"""
Phase 21D: Sampling-Aware Incident Correlation Full-Pipeline Benchmark
======================================================================
Evaluates all 5 benchmark videos across 4 pipeline modes:
A. ByteTrack + 1 FPS + legacy correlation
B. ByteTrack + 1 FPS + sampling-aware correlation
C. ByteTrack + 3 FPS + sampling-aware correlation
D. ByteTrack + 3 FPS + legacy correlation
"""
import sys
sys.path.insert(0, ".")

import os
import time
import json
import psutil
from pathlib import Path
from typing import Dict, Any, List, Tuple
from statistics import median
import numpy as np
import cv2
import torch

from ai.video.processor import VideoProcessor
from ai.detection.detector import YOLODetector
from ai.detection.inference_policy import InferenceResolutionPolicy
from ai.events.generator import EventGenerator
from ai.tracking.tracker import ObjectTracker
from ai.intelligence_pipeline import SecurityIntelligencePipeline
from ai.incidents.schemas import ValidationDecision
from ai.video.frame_cache import BoundedFrameCache
from backend.app.core.config import settings

BENCHMARK_VIDEOS = {
    "Burglary": {
        "file": "storage/uploads/0d4d92f9-19f8-42e3-925f-1931cb557705_uccrime_Burglary010_x264.mp4",
        "description": "Burglary 320x240 low-resolution CCTV",
    },
    "4K_UHD": {
        "file": "storage/uploads/3426f64b-dd44-48a7-8e29-2c5f77b748bb_12566041-uhd_3840_2160_30fps.mp4",
        "description": "4K UHD 2560x1440 high-resolution footage",
    },
    "Highway": {
        "file": "storage/uploads/a94e46c6-3742-44c1-84b9-24f7be1a6a97_17.avi",
        "description": "Highway traffic AVI",
    },
    "VIRAT": {
        "file": "storage/uploads/b118ab48-0fc5-4e14-8524-0205b27a831d_VIRAT_S_010204_05_000856_000890.mp4",
        "description": "VIRAT multi-actor surveillance",
    },
    "WhatsApp": {
        "file": "storage/uploads/e63e72b1-6616-4be6-b732-110219a81005_WhatsApp_Video_2026-09-26_at_06.15.55.mp4",
        "description": "WhatsApp mobile VFR video",
    },
}


def extract_sampled_and_detected_events(
    video_path: str,
    video_id: str,
    sampling_fps: float,
    detector: YOLODetector,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], float, float]:
    """Sample frames and run batch YOLO inference once per sampling rate."""
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

    sampling_time = time.perf_counter() - t0_sample
    num_sampled = len(sampled_frames)

    # Batch YOLO inference (batch size = 4)
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

    # Validation stats
    n_raw = 0
    n_valid = 0
    for fd in all_raw_frame_detections:
        for det in fd["detections"]:
            n_raw += 1
            st = getattr(det, "validation_status", "VALID")
            if hasattr(st, "value"):
                st = st.value
            if st in ("VALID", "VALIDATED"):
                n_valid += 1

    det_stats = {
        "raw_detections": n_raw,
        "valid_detections": n_valid,
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

    return meta, events, det_stats, sampling_time, detection_time


def run_pipeline_configuration(
    meta: Dict[str, Any],
    events: List[Dict[str, Any]],
    det_stats: Dict[str, Any],
    sampling_time: float,
    detection_time: float,
    video_path: str,
    video_id: str,
    sampling_fps: float,
    sampling_aware: bool,
) -> Dict[str, Any]:
    """Execute pipeline with specific correlation mode."""
    proc = psutil.Process(os.getpid())
    rss_start = proc.memory_info().rss / (1024 * 1024)

    fps = meta["fps"]
    duration_s = meta["duration_seconds"]

    tracker = ObjectTracker(video_id=video_id)
    t0_intel = time.perf_counter()
    intel_pipe = SecurityIntelligencePipeline(
        tracker=tracker,
        sampling_aware_correlation=sampling_aware,
    )
    intel_res = intel_pipe.process_video_intelligence(
        video_id=video_id,
        video_path=video_path,
        raw_events=events,
        fps=fps,
        duration_seconds=duration_s,
        sample_rate_fps=sampling_fps,
        sampling_aware_correlation=sampling_aware,
    )
    intel_time = time.perf_counter() - t0_intel

    rss_end = proc.memory_info().rss / (1024 * 1024)
    peak_rss = max(rss_start, rss_end)

    total_pipeline_time = sampling_time + detection_time + intel_time
    effective_pipeline_fps = det_stats["frames_sampled"] / max(0.001, total_pipeline_time)

    # Tracking metrics
    tracks = intel_res.get("all_tracks", intel_res["tracks"])
    total_tracks = len(tracks)
    confirmed_tracks = sum(1 for t in tracks if t.detection_count >= 2)
    single_frame_tracks = sum(1 for t in tracks if t.detection_count <= 1)
    single_frame_pct = round(100.0 * single_frame_tracks / max(1, total_tracks), 2)
    durations = [t.duration_seconds for t in tracks]
    avg_track_duration = round(sum(durations) / max(1, total_tracks), 2)

    # Incident and Candidate breakdown
    candidates = intel_res["incidents"]
    correlated_incidents = intel_res["correlated_incidents"]
    sec_events = intel_res["security_events"]

    accepted_cands = sum(1 for c in candidates if str(getattr(c, "validation_decision", "")).upper() == "ACCEPTED")
    review_cands = sum(1 for c in candidates if str(getattr(c, "validation_decision", "")).upper() == "REVIEW_REQUIRED")
    rejected_cands = sum(1 for c in candidates if str(getattr(c, "validation_decision", "")).upper() == "REJECTED")

    # Detect duplicate incidents: overlapping temporal interval on exact same primary tracks
    duplicate_count = 0
    for i in range(len(correlated_incidents)):
        ci1 = correlated_incidents[i]
        for j in range(i + 1, len(correlated_incidents)):
            ci2 = correlated_incidents[j]
            if set(ci1.primary_track_ids) == set(ci2.primary_track_ids) and ci1.incident_category == ci2.incident_category:
                overlap = not (ci1.end_time < ci2.start_time or ci2.end_time < ci1.start_time)
                if overlap:
                    duplicate_count += 1

    # Evidence candidates count
    evidence_count = sum(len(getattr(c, "evidence_candidates", [])) for c in candidates)

    return {
        "video_id": video_id,
        "sampling_fps": sampling_fps,
        "sampling_aware": sampling_aware,
        "frames_sampled": det_stats["frames_sampled"],
        "valid_detections": det_stats["valid_detections"],
        "raw_detections": det_stats["raw_detections"],
        "total_tracks": total_tracks,
        "confirmed_tracks": confirmed_tracks,
        "single_frame_tracks": single_frame_tracks,
        "single_frame_pct": single_frame_pct,
        "avg_track_duration": avg_track_duration,
        "security_events_total": len(sec_events),
        "candidate_incidents": len(candidates),
        "accepted_incidents": accepted_cands,
        "review_required_incidents": review_cands,
        "rejected_incidents": rejected_cands,
        "correlated_incidents": len(correlated_incidents),
        "evidence_count": evidence_count,
        "duplicate_incidents": duplicate_count,
        "total_time_s": round(total_pipeline_time, 2),
        "pipeline_fps": round(effective_pipeline_fps, 2),
        "peak_rss_mb": round(peak_rss, 1),
    }


def main():
    torch.set_num_threads(1)
    detector = YOLODetector()
    all_results = {}

    print("================================================================================")
    print("PHASE 21D: FULL-PIPELINE SAMPLING-AWARE INCIDENT CORRELATION BENCHMARK")
    print("================================================================================")

    out_file = "scratch/phase21d_correlation_benchmark.json"
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}

    for v_key, v_info in BENCHMARK_VIDEOS.items():
        if v_key in all_results and len(all_results[v_key]) == 4:
            print(f"Skipping already completed video: {v_key}")
            continue

        v_path = v_info["file"]
        print(f"\n---> Evaluating Video: {v_key} ({v_path})")
        if not os.path.exists(v_path):
            print(f"ERROR: Video file not found: {v_path}")
            continue

        all_results[v_key] = {}

        # 1. Extract 1 FPS detections
        print("  Extracting 1.0 FPS frames and YOLO detections...")
        meta_1f, events_1f, det_stats_1f, t_samp_1f, t_det_1f = extract_sampled_and_detected_events(
            v_path, f"{v_key}_1f", 1.0, detector
        )

        # 2. Extract 3 FPS detections
        print("  Extracting 3.0 FPS frames and YOLO detections...")
        meta_3f, events_3f, det_stats_3f, t_samp_3f, t_det_3f = extract_sampled_and_detected_events(
            v_path, f"{v_key}_3f", 3.0, detector
        )

        # Mode A: 1 FPS Legacy
        print("  Running Mode A: ByteTrack 1 FPS (Legacy Correlation)...")
        res_A = run_pipeline_configuration(
            meta_1f, events_1f, det_stats_1f, t_samp_1f, t_det_1f, v_path, f"{v_key}_A", 1.0, sampling_aware=False
        )
        all_results[v_key]["bytetrack_1fps_legacy"] = res_A

        # Mode B: 1 FPS Sampling-Aware
        print("  Running Mode B: ByteTrack 1 FPS (Sampling-Aware Correlation)...")
        res_B = run_pipeline_configuration(
            meta_1f, events_1f, det_stats_1f, t_samp_1f, t_det_1f, v_path, f"{v_key}_B", 1.0, sampling_aware=True
        )
        all_results[v_key]["bytetrack_1fps_aware"] = res_B

        # Mode C: 3 FPS Sampling-Aware
        print("  Running Mode C: ByteTrack 3 FPS (Sampling-Aware Correlation)...")
        res_C = run_pipeline_configuration(
            meta_3f, events_3f, det_stats_3f, t_samp_3f, t_det_3f, v_path, f"{v_key}_C", 3.0, sampling_aware=True
        )
        all_results[v_key]["bytetrack_3fps_aware"] = res_C

        # Mode D: 3 FPS Legacy
        print("  Running Mode D: ByteTrack 3 FPS (Legacy Correlation)...")
        res_D = run_pipeline_configuration(
            meta_3f, events_3f, det_stats_3f, t_samp_3f, t_det_3f, v_path, f"{v_key}_D", 3.0, sampling_aware=False
        )
        all_results[v_key]["bytetrack_3fps_legacy"] = res_D

        print(f"  Summary for {v_key}:")
        print(f"    Mode A (1F Leg): {res_A['correlated_incidents']} corr incs ({res_A['total_time_s']}s, {res_A['peak_rss_mb']}MB)")
        print(f"    Mode B (1F Awa): {res_B['correlated_incidents']} corr incs ({res_B['total_time_s']}s, {res_B['peak_rss_mb']}MB)")
        print(f"    Mode C (3F Awa): {res_C['correlated_incidents']} corr incs ({res_C['total_time_s']}s, {res_C['peak_rss_mb']}MB)")
        print(f"    Mode D (3F Leg): {res_D['correlated_incidents']} corr incs ({res_D['total_time_s']}s, {res_D['peak_rss_mb']}MB)")

    out_file = "scratch/phase21d_correlation_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nBenchmark completed successfully! Saved results to {out_file}")


if __name__ == "__main__":
    main()
