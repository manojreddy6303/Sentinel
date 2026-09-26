import sys
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from backend.app.core.config import settings
from ai.video.processor import VideoProcessor
from backend.app.api.videos import _get_detector
from ai.validation import DetectionValidator
from ai.events.generator import EventGenerator
from ai.intelligence_pipeline import SecurityIntelligencePipeline

video_path = 'storage/uploads/1287e521-1f3a-46bb-8c95-c369f81926dc_WhatsApp_Video_2026-09-26_at_06.14.37.mp4'

processor = VideoProcessor(video_path=video_path, sample_rate_fps=settings.VIDEO_SAMPLE_RATE_FPS)
meta = processor.get_metadata()
print(f'Metadata: width={meta.get("width")}, height={meta.get("height")}, fps={meta.get("fps")}, duration={meta.get("duration_seconds")}, frames={meta.get("total_frames")}')

detector = _get_detector()
validator = DetectionValidator()

frames = list(processor.sample_frames())
print(f'Sampled frames count: {len(frames)}')

frame_detections = []
for frame_number, timestamp, frame_bgr in frames:
    dets = detector.detect(frame_bgr, timestamp, frame_idx=frame_number, video_id='diag_test', imgsz=640)
    summary = [(d.get('object_class', getattr(d, 'object_class', None)), round(d.get('confidence', getattr(d, 'confidence', 0)), 3), d.get('bbox', getattr(d, 'bbox_dict', {}))) for d in dets]
    print(f't={timestamp:.2f}s (frame {frame_number}): {len(dets)} detections -> {summary}')
    frame_detections.append({'frame_number': frame_number, 'detections': dets})

validator.validate_sequence(
    frame_detections=frame_detections,
    frame_width=float(meta['width']),
    frame_height=float(meta['height'])
)

generator = EventGenerator()
events = generator.generate_events(
    video_id='diag_test',
    video_filename='test.mp4',
    fps=meta['fps'],
    video_duration=meta['duration_seconds'],
    frame_detections=frame_detections,
)

print(f'\nTotal generated events: {len(events)}')
valid_events = [e for e in events if getattr(e, 'validation_status', None) == 'VALID']
print(f'Valid events: {len(valid_events)}')

intel_pipe = SecurityIntelligencePipeline(zones=[])
intel_res = intel_pipe.process_video_intelligence(
    video_id='diag_test',
    video_path=video_path,
    raw_events=events,
    fps=meta['fps'],
    duration_seconds=meta['duration_seconds'],
    sample_rate_fps=settings.VIDEO_SAMPLE_RATE_FPS,
)

print(f'\nTotal Tracks: {len(intel_res["tracks"])}')
for trk in intel_res["tracks"]:
    print(f'  Track {trk.track_id}: class={trk.object_class}, duration={trk.duration_seconds:.1f}s ({trk.first_seen_timestamp:.1f}s -> {trk.last_seen_timestamp:.1f}s), det_count={trk.detection_count}')

print(f'\nCorrelated Incidents: {len(intel_res.get("correlated_incidents", []))}')
for inc in intel_res.get("correlated_incidents", []):
    print(f'  Incident: {inc.get("incident_id")}, subcategory={inc.get("incident_subcategory")}, score={inc.get("assessment_score")}')
