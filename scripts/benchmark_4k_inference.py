"""
Benchmark 4K frame inference and seeking.
"""
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
import time
import cv2

from backend.app.core.config import settings
from ai.detection.detector import YOLODetector

v_4k = "3426f64b-dd44-48a7-8e29-2c5f77b748bb"
matches = [m for m in settings.STORAGE_UPLOADS_DIR.glob(f"{v_4k}_*") if m.suffix != ".json"]
if not matches:
    print("4K video not found")
    sys.exit(1)

path_4k = str(matches[0])
cap = cv2.VideoCapture(path_4k)
ret, frame = cap.read()
cap.release()

if not ret or frame is None:
    print("Could not read 4K frame")
    sys.exit(1)

h, w = frame.shape[:2]
print(f"4K Frame shape: {w}x{h}, size in bytes: {frame.nbytes:,}")

detector = YOLODetector(
    model_name=settings.YOLO_MODEL_NAME,
    confidence_threshold=settings.YOLO_CONFIDENCE_THRESHOLD,
    surveillance_classes=settings.SURVEILLANCE_CLASSES,
)

# Warmup
detector.detect(frame, 0.0)

# 1. Direct 4K frame to YOLO
t0 = time.perf_counter()
dets_4k = detector.detect(frame, 0.0)
t_4k = time.perf_counter() - t0
print(f"Direct 4K detect: {t_4k*1000:.1f}ms, found {len(dets_4k)} detections")

# 2. Resized frame inference
t0 = time.perf_counter()
scale_w, scale_h = 1280, int(round(1280 * h / w))
resized = cv2.resize(frame, (scale_w, scale_h), interpolation=cv2.INTER_LINEAR)
dets_resized = detector.detect(resized, 0.0)
t_resized = time.perf_counter() - t0
print(f"Resized (1280x720) detect: {t_resized*1000:.1f}ms, found {len(dets_resized)} detections")
