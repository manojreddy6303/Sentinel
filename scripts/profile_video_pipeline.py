"""
Profiling probe for Sentinel Video Processing Pipeline.
Measures exact time spent in:
1. Initial Frame sampling & decoding
2. YOLO inference
3. Validation sequence
4. Event generation
5. Pipeline re-open & frame seeking (cap.set)
6. Media quality gate
7. Specialized detectors
8. Color & Face analysis
9. DB persistence
"""
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
import time
import json
import cv2

from backend.app.core.config import settings
from ai.video.processor import VideoProcessor
from ai.detection.detector import YOLODetector
from ai.validation import DetectionValidator
from ai.events.generator import EventGenerator
from ai.intelligence_pipeline import SecurityIntelligencePipeline
from ai.specialized.media_quality import FrameQualityGate

video_id = "0d4d92f9-19f8-42e3-925f-1931cb557705"
matches = [m for m in settings.STORAGE_UPLOADS_DIR.glob(f"{video_id}_*") if m.suffix != ".json"]
if not matches:
    raise FileNotFoundError(f"Video {video_id} not found")
video_path = str(matches[0])
print(f"Profiling video: {video_path}")

processor = VideoProcessor(video_path=video_path, sample_rate_fps=1.0)
meta = processor.get_metadata()
detector = YOLODetector(
    model_name=settings.YOLO_MODEL_NAME,
    confidence_threshold=settings.YOLO_CONFIDENCE_THRESHOLD,
    surveillance_classes=settings.SURVEILLANCE_CLASSES,
)

frame_detections = []
frames_bgr = []
frames_processed = 0

t_decode_start = time.perf_counter()
t0 = time.perf_counter()
t_yolo_total = 0.0

for frame_number, timestamp, frame_bgr in processor.sample_frames():
    t_y0 = time.perf_counter()
    raw_dets = detector.detect(frame_bgr, timestamp)
    t_y1 = time.perf_counter()
    t_yolo_total += (t_y1 - t_y0)
    frame_detections.append({
        "frame_number": frame_number,
        "timestamp": timestamp,
        "detections": raw_dets,
    })
    frames_bgr.append((frame_number, timestamp, frame_bgr))
    frames_processed += 1

t_decode_and_yolo = time.perf_counter() - t0
print(f"[Stage 1] Sampled & Inferred {frames_processed} frames:")
print(f"  Total Stage 1: {t_decode_and_yolo:.3f}s")
print(f"  YOLO Inference: {t_yolo_total:.3f}s (avg {t_yolo_total/max(1, frames_processed)*1000:.1f}ms/frame)")
print(f"  Pure Decode: {t_decode_and_yolo - t_yolo_total:.3f}s")

# Phase 2: Validation sequence
t_val0 = time.perf_counter()
validator = DetectionValidator()
validator.validate_sequence(frame_detections)
t_val = time.perf_counter() - t_val0
print(f"[Stage 2] Validation sequence: {t_val:.3f}s")

# Phase 3: Event generation
t_gen0 = time.perf_counter()
generator = EventGenerator()
events = generator.generate_events(
    video_id=video_id,
    video_filename="uccrime_Burglary010_x264.mp4",
    fps=meta.get("fps", 30.0),
    video_duration=meta.get("duration_seconds", 30.0),
    frame_detections=frame_detections,
)
t_gen = time.perf_counter() - t_gen0
print(f"[Stage 3] Event generation: {t_gen:.3f}s ({len(events)} events)")

# Phase 4: Measure re-decoding via cap.set vs in-memory frames
cap = cv2.VideoCapture(video_path)
fps = meta.get("fps", 30.0)
t_seek_total = 0.0
t_quality_total = 0.0

for item in frame_detections:
    ts = item["timestamp"]
    frame_idx = int(round(ts * fps))
    t_s0 = time.perf_counter()
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, fr = cap.read()
    t_s1 = time.perf_counter()
    t_seek_total += (t_s1 - t_s0)

    if ret and fr is not None:
        t_q0 = time.perf_counter()
        _ = FrameQualityGate.evaluate(fr)
        t_q1 = time.perf_counter()
        t_quality_total += (t_q1 - t_q0)

cap.release()

print(f"[Stage 4] Secondary re-decode via cap.set: {t_seek_total:.3f}s (avg {t_seek_total/max(1, frames_processed)*1000:.1f}ms/frame)")
print(f"[Stage 5] FrameQualityGate evaluate: {t_quality_total:.3f}s (avg {t_quality_total/max(1, frames_processed)*1000:.1f}ms/frame)")
