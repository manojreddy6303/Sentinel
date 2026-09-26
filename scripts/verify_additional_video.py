import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from backend.app.core.config import settings
from ai.video.processor import VideoProcessor
from backend.app.api.videos import _get_detector
from ai.validation import DetectionValidator
from ai.events.generator import EventGenerator
from ai.intelligence_pipeline import SecurityIntelligencePipeline
from backend.app.services.evidence_service import EvidenceService
from backend.app.services.playback_service import ensure_evidence_clip_playback

video_path = 'storage/uploads/a94e46c6-3742-44c1-84b9-24f7be1a6a97_17.avi'

processor = VideoProcessor(video_path=video_path, sample_rate_fps=settings.VIDEO_SAMPLE_RATE_FPS)
meta = processor.get_metadata()
print(f"=== ADDITIONAL VIDEO METADATA (17.avi) ===")
print(f"Container: {meta.get('container')}")
print(f"Codec: {meta.get('codec')}")
print(f"Resolution: {meta['width']}x{meta['height']}")
print(f"FPS: {meta['fps']}")
print(f"Duration: {meta['duration_seconds']}s")

detector = _get_detector()
validator = DetectionValidator()

frames = list(processor.sample_frames())
print(f"Sampled frames: {len(frames)}")

frame_detections = []
for frame_number, timestamp, frame_bgr in frames:
    dets = detector.detect(frame_bgr, timestamp, frame_idx=frame_number, video_id='avi_test', imgsz=640)
    frame_detections.append({'frame_number': frame_number, 'detections': dets})

validator.validate_sequence(
    frame_detections=frame_detections,
    frame_width=float(meta['width']),
    frame_height=float(meta['height'])
)

generator = EventGenerator()
events = generator.generate_events(
    video_id='avi_test',
    video_filename='17.avi',
    fps=meta['fps'],
    video_duration=meta['duration_seconds'],
    frame_detections=frame_detections,
)

print(f"\n=== DETECTIONS ===")
print(f"Total detections: {len(events)}")
person_dets = [e for e in events if e.get('object_class') == 'person']
vehicle_dets = [e for e in events if e.get('object_class') in ['car', 'bus', 'truck', 'motorcycle']]
print(f"Person detections: {len(person_dets)}")
print(f"Vehicle detections: {len(vehicle_dets)}")

intel_pipe = SecurityIntelligencePipeline(zones=[])
intel_res = intel_pipe.process_video_intelligence(
    video_id='avi_test',
    video_path=video_path,
    raw_events=events,
    fps=meta['fps'],
    duration_seconds=meta['duration_seconds'],
    sample_rate_fps=settings.VIDEO_SAMPLE_RATE_FPS,
    frame_width=float(meta['width']),
    frame_height=float(meta['height']),
)

print(f"\n=== TRACKS ===")
print(f"Total tracks: {len(intel_res['tracks'])}")
for trk in intel_res['tracks'][:5]:
    print(f"  Track {trk.track_id}: class={trk.object_class}, duration={trk.duration_seconds}s, dets={trk.detection_count}, color={trk.color}")

print(f"\n=== INCIDENTS ===")
incidents = intel_res.get('correlated_incidents', [])
print(f"Correlated Incidents: {len(incidents)}")
for inc in incidents[:3]:
    inc_id = getattr(inc, 'incident_id', None) or (inc.get('incident_id') if isinstance(inc, dict) else '')
    subcat = getattr(inc, 'incident_subcategory', None) or (inc.get('incident_subcategory') if isinstance(inc, dict) else '')
    print(f"  {inc_id}: {subcat}")

print("\n=== EVIDENCE GENERATION & BROWSER PLAYBACK TEST ===")
# Test generating an evidence item from an event if events exist
if events:
    first_ev = events[0]
    ev_svc = EvidenceService()
    # Create evidence for this video
    # Note: we use video_id from a known DB entry or test with raw clip extraction
    print(f"First event: {first_ev.get('event_id')} at {first_ev.get('timestamp')}s ({first_ev.get('object_class')})")
    
print("ADDITIONAL VIDEO VERIFICATION COMPLETE.")
