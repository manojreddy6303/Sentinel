import time
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import cv2
from ai.video.processor import VideoProcessor
from ai.detection.detector import YOLODetector
from ai.validation import DetectionValidator
from ai.events.generator import EventGenerator
from ai.intelligence_pipeline import SecurityIntelligencePipeline

video_path = "storage/uploads/0d4d92f9-19f8-42e3-925f-1931cb557705_uccrime_Burglary010_x264.mp4"
if not os.path.exists(video_path):
    print("Video file not found")
    exit(1)

print(f"Benchmarking optimized first-time processing pipeline for: {video_path}")
t_start = time.perf_counter()

# 1. Video Processor
processor = VideoProcessor(video_path=video_path, sample_rate_fps=1.0)
metadata = processor.get_metadata()
fps = metadata["fps"]
duration = metadata["duration_seconds"]
width = metadata["width"]
height = metadata["height"]
t_meta = time.perf_counter()
print(f"1. Video metadata extracted: {duration:.2f}s, {fps} FPS, {width}x{height} in {t_meta - t_start:.3f}s")

# 2. YOLO Batched Sampling & Detection
detector = YOLODetector(model_name="yolov8n.pt", confidence_threshold=0.25)
frame_detections = []
sampled_frames_cache = {}
frames_processed = 0

BATCH_SIZE = 24
batch_frames = []
batch_meta = []
effective_imgsz = 320

t_sample_start = time.perf_counter()
for frame_number, timestamp, frame_bgr in processor.sample_frames():
    batch_frames.append(frame_bgr)
    batch_meta.append((frame_number, timestamp))
    if len(sampled_frames_cache) < 1200:
        sampled_frames_cache[round(timestamp, 3)] = frame_bgr
    
    if len(batch_frames) >= BATCH_SIZE:
        results = detector.detect_batch(batch_frames, [m[1] for m in batch_meta], [m[0] for m in batch_meta], imgsz=effective_imgsz)
        for (f_num, ts), raw_dets in zip(batch_meta, results):
            frame_detections.append({"frame_number": f_num, "detections": raw_dets})
            frames_processed += 1
        batch_frames.clear()
        batch_meta.clear()

if batch_frames:
    results = detector.detect_batch(batch_frames, [m[1] for m in batch_meta], [m[0] for m in batch_meta], imgsz=effective_imgsz)
    for (f_num, ts), raw_dets in zip(batch_meta, results):
        frame_detections.append({"frame_number": f_num, "detections": raw_dets})
        frames_processed += 1
    batch_frames.clear()
    batch_meta.clear()

t_sample_end = time.perf_counter()
total_dets = sum(len(f["detections"]) for f in frame_detections)
print(f"2. Sampled & Batched YOLO Detection on {frames_processed} frames: {total_dets} detections in {t_sample_end - t_sample_start:.3f}s (vs 36.72s unbatched)")

# 3. Validation Sequence
t_val_start = time.perf_counter()
validator = DetectionValidator()
validator.validate_sequence(
    frame_detections=frame_detections,
    frame_width=float(width) if width else None,
    frame_height=float(height) if height else None,
)
t_val_end = time.perf_counter()
print(f"3. Sequence validation completed in {t_val_end - t_val_start:.4f}s")

# 4. Event Generation
t_event_start = time.perf_counter()
generator = EventGenerator()
events = generator.generate_events(
    video_id="bench_010",
    video_filename="uccrime_Burglary010_x264.mp4",
    fps=fps,
    video_duration=duration,
    frame_detections=frame_detections,
)
t_event_end = time.perf_counter()
print(f"4. Generated {len(events)} events in {t_event_end - t_event_start:.4f}s")

# 5. Security Intelligence with in-memory frame cache
t_intel_start = time.perf_counter()
intel_pipe = SecurityIntelligencePipeline(zones=[])
intel_res = intel_pipe.process_video_intelligence(
    video_id="bench_010",
    video_path=video_path,
    raw_events=events,
    fps=fps,
    duration_seconds=duration,
    sample_rate_fps=1.0,
    sampled_frames=sampled_frames_cache,
)
t_intel_end = time.perf_counter()
t_total = time.perf_counter() - t_start

print(f"5. Security Intelligence (Tracks: {len(intel_res['tracks'])}, Incidents: {len(intel_res.get('correlated_incidents', []))}) in {t_intel_end - t_intel_start:.3f}s (vs 6.16s re-decode)")
print("=" * 60)
print(f"TOTAL FIRST-TIME PROCESSING TIME: {t_total:.2f} seconds")
print("=" * 60)
