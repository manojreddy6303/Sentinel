import sys
import cv2
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.specialized.fire_smoke.detector import SmokeVisualDetector

def main():
    video_path = str(PROJECT_ROOT / "storage" / "uploads" / "0d4d92f9-19f8-42e3-925f-1931cb557705_uccrime_Burglary010_x264.mp4")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frames / fps if fps else 0

    print("1. Video metadata")
    print(f"   - resolution: {int(width)}x{int(height)}")
    print(f"   - FPS: {fps}")
    print(f"   - duration: {duration:.2f}s")
    print(f"   - frame/sample count: {int(frames)}")

    detector = SmokeVisualDetector(enabled=True)
    detector.initialize()

    frame_idx = 0
    all_obs = []

    # Process 1 frame per second to speed up
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp = frame_idx / fps
        if frame_idx % int(fps) == 0:
            obs_list = detector.detect_frame(frame, timestamp, frame_idx)
            for obs in obs_list:
                all_obs.append(obs)
                
        frame_idx += 1

    print("\n2. Smoke pipeline observations (raw):")
    print(f"   - raw smoke observations: {len(all_obs)}")
    for i, obs in enumerate(all_obs):
        print(f"   Obs {i}: TS={obs.timestamp:.2f}s, Conf={obs.confidence:.2f}, Box={obs.bounding_box}, Metrics={obs.visual_metrics}")

if __name__ == "__main__":
    main()
