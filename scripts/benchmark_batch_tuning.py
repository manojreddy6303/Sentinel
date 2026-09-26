import sys
import os
sys.path.insert(0, os.path.abspath("."))
import time
import cv2
from ai.video.processor import VideoProcessor
from ai.detection.detector import YOLODetector

video_path = "storage/uploads/0d4d92f9-19f8-42e3-925f-1931cb557705_uccrime_Burglary010_x264.mp4"
processor = VideoProcessor(video_path=video_path, sample_rate_fps=1.0)
frames = [frame_bgr for _, _, frame_bgr in processor.sample_frames()]
print(f"Loaded {len(frames)} frames. Dimensions: {frames[0].shape}")

detector = YOLODetector(model_name="yolov8n.pt", confidence_threshold=0.25)
model = detector._load_model()

# Warmup
_ = model(frames[:4], verbose=False)

for bs in [8, 16, 24, 32]:
    t0 = time.perf_counter()
    count = 0
    for i in range(0, len(frames), bs):
        batch = frames[i:i+bs]
        res = model(batch, verbose=False)
        count += sum(len(r.boxes) for r in res)
    t1 = time.perf_counter()
    print(f"Batch size {bs} (default imgsz): {t1 - t0:.3f}s, detections: {count}")

for bs in [16, 24, 32]:
    t0 = time.perf_counter()
    count = 0
    for i in range(0, len(frames), bs):
        batch = frames[i:i+bs]
        res = model(batch, imgsz=320, verbose=False)
        count += sum(len(r.boxes) for r in res)
    t1 = time.perf_counter()
    print(f"Batch size {bs} (imgsz=320): {t1 - t0:.3f}s, detections: {count}")
