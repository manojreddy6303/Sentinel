"""
Sentinel Phase 20 Comprehensive Benchmark Evaluator

Runs full end-to-end pipeline on real videos across distinct categories:
- Low-res CCTV (320x240): Burglary010
- 4K UHD (3840x2160): 12566041-uhd
- High-motion traffic (320x240): 17.avi
- Outdoor parking / security (720p): VIRAT
- Mobile / VFR: WhatsApp

Measures:
- Resolution selection (imgsz)
- Peak RAM (MB)
- Execution time and effective processing FPS
- Detection yield (raw, validated, uncertain, rejected)
- Tracking metrics (track count, confirmed, avg track length)
- Frame cache compression ratio & memory saved
"""
import os
import sys
import time
import tracemalloc
from typing import Dict, Any

sys.path.insert(0, os.path.abspath("."))

from ai.video.processor import VideoProcessor
from ai.video.frame_cache import BoundedFrameCache
from ai.detection.detector import YOLODetector
from ai.detection.inference_policy import InferenceResolutionPolicy
from ai.validation import DetectionValidator
from ai.events.generator import EventGenerator
from ai.intelligence_pipeline import SecurityIntelligencePipeline


TEST_VIDEOS = [
    {
        "name": "Burglary010 (320x240 CCTV)",
        "path": "storage/uploads/0d4d92f9-19f8-42e3-925f-1931cb557705_uccrime_Burglary010_x264.mp4",
        "category": "Low-Resolution Surveillance",
    },
    {
        "name": "4K UHD (3840x2160)",
        "path": "storage/uploads/3426f64b-dd44-48a7-8e29-2c5f77b748bb_12566041-uhd_3840_2160_30fps.mp4",
        "category": "Ultra-High Definition",
    },
    {
        "name": "Highway Traffic (320x240)",
        "path": "storage/uploads/a94e46c6-3742-44c1-84b9-24f7be1a6a97_17.avi",
        "category": "High-Motion Highway",
    },
    {
        "name": "VIRAT (1280x720 HD)",
        "path": "storage/uploads/b118ab48-0fc5-4e14-8524-0205b27a831d_VIRAT_S_010204_05_000856_000890.mp4",
        "category": "Standard Surveillance",
    },
    {
        "name": "WhatsApp (Mobile VFR)",
        "path": "storage/uploads/1287e521-1f3a-46bb-8c95-c369f81926dc_WhatsApp_Video_2026-09-26_at_06.14.37.mp4",
        "category": "Mobile / Handheld",
    },
]


def evaluate_video(video_cfg: Dict[str, Any]) -> Dict[str, Any]:
    v_path = video_cfg["path"]
    v_name = video_cfg["name"]
    if not os.path.exists(v_path):
        return {"name": v_name, "error": "File not found"}

    tracemalloc.start()
    t_start = time.perf_counter()

    # 1. Metadata extraction
    processor = VideoProcessor(video_path=v_path, sample_rate_fps=1.0)
    meta = processor.get_metadata()
    w, h = meta["width"], meta["height"]
    fps = meta["fps"]
    duration = meta["duration_seconds"]

    # 2. Dynamic Inference Resolution
    effective_imgsz = InferenceResolutionPolicy.select_inference_size(w, h)

    # 3. Sampling & Detection with BoundedFrameCache
    detector = YOLODetector(model_name="yolov8n.pt", confidence_threshold=0.20)
    frame_cache = BoundedFrameCache(max_frames=1200, max_memory_mb=512.0, video_path=v_path)
    frame_detections = []
    frames_processed = 0

    BATCH_SIZE = 24
    batch_frames = []
    batch_meta = []

    for f_num, ts, frame_bgr in processor.sample_frames():
        batch_frames.append(frame_bgr)
        batch_meta.append((f_num, ts))
        frame_cache[ts] = frame_bgr

        if len(batch_frames) >= BATCH_SIZE:
            results = detector.detect_batch(
                batch_frames,
                [m[1] for m in batch_meta],
                [m[0] for m in batch_meta],
                imgsz=effective_imgsz,
            )
            for (fn, t_stamp), raw_dets in zip(batch_meta, results):
                frame_detections.append({"frame_number": fn, "detections": raw_dets})
                frames_processed += 1
            batch_frames.clear()
            batch_meta.clear()

    if batch_frames:
        results = detector.detect_batch(
            batch_frames,
            [m[1] for m in batch_meta],
            [m[0] for m in batch_meta],
            imgsz=effective_imgsz,
        )
        for (fn, t_stamp), raw_dets in zip(batch_meta, results):
            frame_detections.append({"frame_number": fn, "detections": raw_dets})
            frames_processed += 1
        batch_frames.clear()
        batch_meta.clear()

    # 4. Multi-Signal Detection Validation
    validator = DetectionValidator()
    validator.validate_sequence(
        frame_detections=frame_detections,
        frame_width=float(w) if w else None,
        frame_height=float(h) if h else None,
    )

    # Count validation breakdown
    total_dets = 0
    valid_dets = 0
    uncertain_dets = 0
    rejected_dets = 0
    for fd in frame_detections:
        for d in fd["detections"]:
            total_dets += 1
            st = str(d.get("validation_status", "VALID")).upper()
            if st == "VALID":
                valid_dets += 1
            elif st == "UNCERTAIN":
                uncertain_dets += 1
            elif st == "REJECTED":
                rejected_dets += 1

    # 5. Event Generation
    generator = EventGenerator()
    events = generator.generate_events(
        video_id="eval_run",
        video_filename=os.path.basename(v_path),
        fps=fps,
        video_duration=duration,
        frame_detections=frame_detections,
    )

    # 6. Security Intelligence & ByteTrack
    intel_pipe = SecurityIntelligencePipeline(zones=[])
    intel_res = intel_pipe.process_video_intelligence(
        video_id="eval_run",
        video_path=v_path,
        raw_events=events,
        fps=fps,
        duration_seconds=duration,
        sample_rate_fps=1.0,
        sampled_frames=frame_cache,
        frame_width=float(w) if w else None,
        frame_height=float(h) if h else None,
    )

    t_end = time.perf_counter()
    total_time = t_end - t_start
    proc_fps = round(frames_processed / total_time, 2) if total_time > 0 else 0.0

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = round(peak_mem / (1024 * 1024), 2)

    cache_stats = frame_cache.get_memory_stats()
    tracks = intel_res["tracks"]
    confirmed_tracks = [t for t in tracks if t.is_validated]

    return {
        "name": v_name,
        "category": video_cfg["category"],
        "dimensions": f"{w}x{h}",
        "duration_s": round(duration, 1),
        "frames_sampled": frames_processed,
        "selected_imgsz": effective_imgsz,
        "total_time_s": round(total_time, 2),
        "proc_fps": proc_fps,
        "peak_ram_mb": peak_ram_mb,
        "total_dets": total_dets,
        "valid_dets": valid_dets,
        "uncertain_dets": uncertain_dets,
        "rejected_dets": rejected_dets,
        "total_tracks": len(tracks),
        "confirmed_tracks": len(confirmed_tracks),
        "cache_compressed_mb": cache_stats["compressed_memory_mb"],
        "cache_raw_equiv_mb": cache_stats["raw_equivalent_mb"],
        "compression_ratio": cache_stats["compression_ratio"],
    }


def main():
    print("=" * 80)
    print("SENTINEL PHASE 20.1 MULTI-VIDEO RELIABILITY BENCHMARK SUITE")
    print("=" * 80)

    results = []
    for cfg in TEST_VIDEOS:
        print(f"\nEvaluating: {cfg['name']} ({cfg['category']})...")
        res = evaluate_video(cfg)
        results.append(res)
        if "error" in res:
            print(f"  Error: {res['error']}")
            continue
        print(f"  Dims: {res['dimensions']} | Selected imgsz: {res['selected_imgsz']}")
        print(f"  Frames: {res['frames_sampled']} in {res['total_time_s']}s ({res['proc_fps']} FPS) | Peak RAM: {res['peak_ram_mb']} MB")
        print(f"  Dets: {res['total_dets']} (Valid: {res['valid_dets']}, Uncertain: {res['uncertain_dets']}, Rej: {res['rejected_dets']})")
        print(f"  Tracks: {res['total_tracks']} (Confirmed: {res['confirmed_tracks']})")
        print(f"  Frame Cache: {res['cache_compressed_mb']} MB vs {res['cache_raw_equiv_mb']} MB raw ({res['compression_ratio']}x compression)")

    print("\n" + "=" * 80)
    print("ALL EVALUATIONS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
