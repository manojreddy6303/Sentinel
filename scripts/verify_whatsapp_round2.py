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
from backend.app.services.investigation_service import InvestigationService
import sqlite3

video_path = 'storage/uploads/1287e521-1f3a-46bb-8c95-c369f81926dc_WhatsApp_Video_2026-09-26_at_06.14.37.mp4'

processor = VideoProcessor(video_path=video_path, sample_rate_fps=settings.VIDEO_SAMPLE_RATE_FPS)
meta = processor.get_metadata()
print(f"=== METADATA ===")
print(f"Resolution: {meta['width']}x{meta['height']}")
print(f"FPS: {meta['fps']}")
print(f"Duration: {meta['duration_seconds']}s")

detector = _get_detector()
validator = DetectionValidator()

frames = list(processor.sample_frames())
print(f"Sampled frames: {len(frames)}")

frame_detections = []
for frame_number, timestamp, frame_bgr in frames:
    dets = detector.detect(frame_bgr, timestamp, frame_idx=frame_number, video_id='round2_test', imgsz=640)
    frame_detections.append({'frame_number': frame_number, 'detections': dets})

validator.validate_sequence(
    frame_detections=frame_detections,
    frame_width=float(meta['width']),
    frame_height=float(meta['height'])
)

generator = EventGenerator()
events = generator.generate_events(
    video_id='round2_test',
    video_filename='WhatsApp Video 2026-09-26 at 06.14.37.mp4',
    fps=meta['fps'],
    video_duration=meta['duration_seconds'],
    frame_detections=frame_detections,
)

print(f"\n=== DETECTIONS ===")
person_dets = [e for e in events if e.get('object_class') == 'person']
vehicle_dets = [e for e in events if e.get('object_class') in ['car', 'bus', 'truck', 'motorcycle']]
print(f"Total detections: {len(events)}")
print(f"Person detections: {len(person_dets)}")
print(f"Vehicle detections: {len(vehicle_dets)}")

intel_pipe = SecurityIntelligencePipeline(zones=[])
intel_res = intel_pipe.process_video_intelligence(
    video_id='round2_test',
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
for trk in intel_res['tracks']:
    print(f"  Track {trk.track_id}: class={trk.object_class}, duration={trk.duration_seconds}s ({trk.first_seen}s -> {trk.last_seen}s), dets={trk.detection_count}, color={trk.color} (conf={trk.color_confidence})")

print(f"\n=== INCIDENTS ===")
incidents = intel_res.get('correlated_incidents', [])
print(f"Correlated Incidents: {len(incidents)}")
for inc in incidents:
    inc_id = getattr(inc, 'incident_id', None) or (inc.get('incident_id') if isinstance(inc, dict) else '')
    cat = getattr(inc, 'incident_category', None) or (inc.get('incident_category') if isinstance(inc, dict) else '')
    subcat = getattr(inc, 'incident_subcategory', None) or (inc.get('incident_subcategory') if isinstance(inc, dict) else '')
    score = getattr(inc, 'assessment_score', None) or (inc.get('assessment_score') if isinstance(inc, dict) else '')
    print(f"  {inc_id}: category={cat}, subcategory={subcat}, score={score}")

# Test saving to a test video record in DB to test natural language query
from ai.intelligence_repository import SecurityIntelligenceRepository
repo = SecurityIntelligenceRepository()
repo.save_intelligence_results(
    video_id='1287e521-1f3a-46bb-8c95-c369f81926dc',
    tracks=intel_res['tracks'],
    vehicle_attributes=intel_res['vehicle_attributes'],
    face_detections=intel_res['face_detections'],
    security_events=intel_res['security_events'],
    specialized_observations=intel_res.get('specialized_observations', []),
    correlated_incidents=intel_res.get('correlated_incidents', []),
)

# Test investigation queries
service = InvestigationService()
queries = [
    "Is there a person wearing blue?",
    "Show all tracks",
]
print(f"\n=== NATURAL LANGUAGE INVESTIGATION QUERIES ===")
for q in queries:
    ans = service.investigate('1287e521-1f3a-46bb-8c95-c369f81926dc', q)
    print(f"Q: '{q}'")
    print(f"  Supported: {ans.get('is_supported')}")
    print(f"  Message: {ans.get('message')}")
    print(f"  Results count: {len(ans.get('results', []))}")
    if ans.get('results'):
        for r in ans.get('results')[:3]:
            print(f"    - {r.get('track_id')}: {r.get('object_class')} color={r.get('color')} duration={r.get('duration_seconds')}s")
