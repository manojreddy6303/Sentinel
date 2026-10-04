"""
SENTINEL — UNIVERSAL FAILURE MATRIX & COMPREHENSIVE PIPELINE DIAGNOSTIC (PHASE 0.1)

Executes the unified Sentinel video intelligence pipeline across all available test videos:
1. Canonical Burglary
2. WhatsApp Mobile Video 1 (Theft-like corridor)
3. WhatsApp Mobile Video 2 (Unseen corridor/doorway)
4. 4K UHD Urban Traffic / Pedestrian
5. Highway 17 Traffic
6. VIRAT Outdoor Parking
7. Scene 02 (Pedestrian transit)
8. Vehicle Crash

Measures and records for EVERY video:
- Metadata
- Detection statistics & classes
- Validation outcomes
- Tracking telemetry
- Interaction candidates
- Security events
- Correlated incidents
- Investigation Q&A
- Performance / Peak RAM
"""
import os
import sys
import time
import math
import tracemalloc
import json
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath("."))

from ai.video.processor import VideoProcessor
from ai.video.frame_cache import BoundedFrameCache
from ai.detection.detector import YOLODetector
from ai.detection.inference_policy import InferenceResolutionPolicy
from ai.validation import DetectionValidator
from ai.events.generator import EventGenerator
from ai.intelligence_pipeline import SecurityIntelligencePipeline
from ai.schemas import DetectionValidationStatus

VIDEOS_TO_AUDIT = [
    {
        "id": "burglary",
        "name": "Canonical Burglary",
        "path": "storage/uploads/0d4d92f9-19f8-42e3-925f-1931cb557705_uccrime_Burglary010_x264.mp4",
        "category": "CCTV / Indoor",
    },
    {
        "id": "whatsapp_1",
        "name": "WhatsApp Mobile 1",
        "path": "storage/uploads/1287e521-1f3a-46bb-8c95-c369f81926dc_WhatsApp_Video_2026-09-26_at_06.14.37.mp4",
        "category": "Mobile Handheld VFR",
    },
    {
        "id": "whatsapp_2",
        "name": "WhatsApp Mobile 2",
        "path": "storage/uploads/e63e72b1-6616-4be6-b732-110219a81005_WhatsApp_Video_2026-09-26_at_06.15.55.mp4",
        "category": "Mobile Handheld VFR",
    },
    {
        "id": "uhd_4k",
        "name": "4K UHD Scene",
        "path": "storage/uploads/3426f64b-dd44-48a7-8e29-2c5f77b748bb_12566041-uhd_3840_2160_30fps.mp4",
        "category": "4K Ultra-HD",
    },
    {
        "id": "highway_17",
        "name": "Highway Traffic 17",
        "path": "storage/uploads/a94e46c6-3742-44c1-84b9-24f7be1a6a97_17.avi",
        "category": "High-Motion Traffic",
    },
    {
        "id": "virat",
        "name": "VIRAT Outdoor Parking",
        "path": "storage/uploads/b118ab48-0fc5-4e14-8524-0205b27a831d_VIRAT_S_010204_05_000856_000890.mp4",
        "category": "Outdoor Surveillance HD",
    },
    {
        "id": "scene_02",
        "name": "Scene 02 Pedestrian Transit",
        "path": "storage/uploads/1b864cdf-edf4-4030-93e7-f128c034402f_02.avi",
        "category": "Outdoor Pedestrian",
    },
    {
        "id": "crash_0",
        "name": "Live Traffic Crash",
        "path": "storage/uploads/a1fd8efb-27e0-4754-aa44-b11f708454dc_livevid_crash0.mp4",
        "category": "Traffic Incident",
    },
]

def run_diagnostic_on_video(v_info: Dict[str, Any]) -> Dict[str, Any]:
    v_path = v_info["path"]
    v_name = v_info["name"]
    v_id = v_info["id"]

    if not os.path.exists(v_path):
        return {"id": v_id, "name": v_name, "error": f"File not found: {v_path}"}

    tracemalloc.start()
    t_start = time.perf_counter()

    processor = VideoProcessor(video_path=v_path, sample_rate_fps=1.0)
    meta = processor.get_metadata()
    w, h = meta["width"], meta["height"]
    fps = meta["fps"]
    duration = meta["duration_seconds"]

    effective_imgsz = InferenceResolutionPolicy.select_inference_size(w, h)
    detector = YOLODetector(model_name="yolov8n.pt", confidence_threshold=0.15)
    frame_cache = BoundedFrameCache(max_frames=600, max_memory_mb=512.0, video_path=v_path)
    
    frame_detections = []
    frames_processed = 0
    raw_class_counts = {}
    small_objects_count = 0
    all_confidences = []

    for fn, ts, frame_bgr in processor.sample_frames():
        frame_cache[ts] = frame_bgr
        raw_dets = detector.detect(frame_bgr, ts, frame_idx=fn, video_id=v_id, imgsz=effective_imgsz)
        
        for d in raw_dets:
            cname = d.get("object_class") or d.get("class_name")
            conf = d.get("confidence", 0.0)
            bbox = d.get("bbox") or d.get("bbox_dict") or {}
            bw = bbox.get("width", 0.0) if isinstance(bbox, dict) else getattr(bbox, "width", 0.0)
            bh = bbox.get("height", 0.0) if isinstance(bbox, dict) else getattr(bbox, "height", 0.0)
            
            raw_class_counts[cname] = raw_class_counts.get(cname, 0) + 1
            all_confidences.append(conf)
            if bw > 0 and bh > 0 and max(bw, bh) < 45.0:
                small_objects_count += 1
                
        frame_detections.append({"frame_number": fn, "detections": raw_dets})
        frames_processed += 1

    # Validation
    validator = DetectionValidator()
    validator.validate_sequence(
        frame_detections=frame_detections,
        frame_width=float(w) if w else None,
        frame_height=float(h) if h else None,
    )

    valid_count = 0
    uncertain_count = 0
    rejected_count = 0

    for fd in frame_detections:
        for d in fd["detections"]:
            st = str(d.get("validation_status", "VALID")).upper()
            if "VALID" in st and "UN" not in st:
                valid_count += 1
            elif "UNCERTAIN" in st:
                uncertain_count += 1
            else:
                rejected_count += 1

    # Event generation
    generator = EventGenerator()
    events = generator.generate_events(
        video_id=v_id,
        video_filename=os.path.basename(v_path),
        fps=fps,
        video_duration=duration,
        frame_detections=frame_detections,
    )

    # Intelligence Pipeline
    intel_pipe = SecurityIntelligencePipeline(zones=[])
    intel_res = intel_pipe.process_video_intelligence(
        video_id=v_id,
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
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = round(peak_mem / (1024 * 1024), 2)

    tracks = intel_res.get("tracks", [])
    person_tracks = [t for t in tracks if t.object_class == "person"]
    object_tracks = [t for t in tracks if t.object_class != "person"]
    single_frame_tracks = [t for t in tracks if t.detection_count <= 1]

    sec_events = intel_res.get("security_events", [])
    event_type_counts = {}
    for se in sec_events:
        etype = se.get("event_type") if isinstance(se, dict) else getattr(se, "event_type", "UNKNOWN")
        event_type_counts[etype] = event_type_counts.get(etype, 0) + 1

    incidents = intel_res.get("correlated_incidents", [])
    incident_subcategories = {}
    for inc in incidents:
        subcat = inc.get("incident_subcategory") if isinstance(inc, dict) else getattr(inc, "incident_subcategory", "unknown")
        incident_subcategories[subcat] = incident_subcategories.get(subcat, 0) + 1

    has_theft_incident = any("theft" in s.lower() or "takeaway" in s.lower() for s in incident_subcategories.keys())
    has_fire_incident = any("fire" in s.lower() for s in incident_subcategories.keys())
    has_altercation_incident = any("altercation" in s.lower() for s in incident_subcategories.keys())

    return {
        "id": v_id,
        "name": v_name,
        "category": v_info["category"],
        "resolution": f"{w}x{h}",
        "duration_s": round(duration, 2),
        "fps": round(fps, 2),
        "frames_processed": frames_processed,
        "proc_fps": proc_fps,
        "peak_ram_mb": peak_ram_mb,
        "raw_detections": sum(raw_class_counts.values()),
        "valid_detections": valid_count,
        "uncertain_detections": uncertain_count,
        "rejected_detections": rejected_count,
        "small_objects_count": small_objects_count,
        "detected_classes": raw_class_counts,
        "total_tracks": len(tracks),
        "person_tracks": len(person_tracks),
        "object_tracks": len(object_tracks),
        "single_frame_tracks": len(single_frame_tracks),
        "security_events": event_type_counts,
        "correlated_incidents": incident_subcategories,
        "has_theft_incident": has_theft_incident,
        "has_fire_incident": has_fire_incident,
        "has_altercation_incident": has_altercation_incident,
    }


if __name__ == "__main__":
    print("=" * 80)
    print("SENTINEL UNIVERSAL VIDEO AI FAILURE MATRIX DIAGNOSTIC")
    print("=" * 80)
    
    matrix_results = []
    for v_info in VIDEOS_TO_AUDIT:
        print(f"\nProcessing [{v_info['id']}]: {v_info['name']} ({v_info['category']})...")
        res = run_diagnostic_on_video(v_info)
        matrix_results.append(res)
        if "error" in res:
            print(f"  ERROR: {res['error']}")
            continue
        print(f"  Resolution: {res['resolution']} | Duration: {res['duration_s']}s | Frames: {res['frames_processed']}")
        print(f"  Detections: Raw={res['raw_detections']}, Valid={res['valid_detections']}, Uncertain={res['uncertain_detections']}, Small(<45px)={res['small_objects_count']}")
        print(f"  Classes: {res['detected_classes']}")
        print(f"  Tracks: Total={res['total_tracks']} (Person={res['person_tracks']}, Object={res['object_tracks']}, SingleFrame={res['single_frame_tracks']})")
        print(f"  Security Events: {res['security_events']}")
        print(f"  Incidents: {res['correlated_incidents']}")
        print(f"  Performance: {res['proc_fps']} FPS | Peak RAM: {res['peak_ram_mb']} MB")

    print("\n" + "=" * 80)
    print("UNIVERSAL MATRIX SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Video':<24} | {'Res':<10} | {'Dets':<6} | {'Tracks (P/O)':<14} | {'Theft':<6} | {'Fire':<6} | {'RAM (MB)':<8}")
    print("-" * 80)
    for r in matrix_results:
        if "error" in r:
            print(f"{r['name']:<24} | {'ERR':<10} | {'-':<6} | {'-':<14} | {'-':<6} | {'-':<6} | {'-':<8}")
        else:
            po = f"{r['person_tracks']}/{r['object_tracks']}"
            print(f"{r['name']:<24} | {r['resolution']:<10} | {r['raw_detections']:<6} | {po:<14} | {str(r['has_theft_incident']):<6} | {str(r['has_fire_incident']):<6} | {r['peak_ram_mb']:<8}")

    with open("docs/universal_matrix_raw.json", "w") as f:
        json.dump(matrix_results, f, indent=2)
    print("\nDiagnostic matrix written to docs/universal_matrix_raw.json")
