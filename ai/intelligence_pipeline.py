"""
Phase 8 Advanced Security Intelligence Pipeline for Sentinel

Coordinates multi-frame tracking, vehicle color analysis, anonymous face detection,
restricted zone monitoring, loitering detection, abandoned object detection, activity
density windowing, and observational anomaly aggregation.
"""
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import cv2
import numpy as np

from ai.schemas import (
    BoundingBox,
    TrackedObject,
    VehicleAttribute,
    FaceDetection,
    ZoneDefinition,
    SecurityEvent,
)
from ai.tracking.tracker import ObjectTracker
from ai.attributes.color_analyzer import VehicleColorAnalyzer
from ai.faces.face_detector import FaceDetector
from ai.zones.zone_manager import ZoneManager
from ai.activity.activity_analyzer import ActivityAnalyzer
from ai.behavior.analyzer import BehaviorAnalyzer
from ai.video.processor import VideoProcessor
from ai.incidents.engine import IncidentIntelligenceEngine
from ai.correlation.engine import AdvancedIncidentCorrelationEngine
from ai.specialized import (
    get_specialized_registry,
    SpecializedDetectorRegistry,
    SpecializedTemporalTracker,
    FireVisualDetector,
    SmokeVisualDetector,
    WeaponVisualDetector,
    PoseActionDetector,
    SpecializedValidationEngine,
    SpecializedValidationStatus,
)
from ai.specialized.episode import SpecializedVisualEpisodeAggregator
from ai.specialized.media_quality import FrameQualityGate
from ai.common.detector_health import (
    DetectorHealthRegistry,
    DetectorOperationalStatus,
    DetectorModelType,
)
from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class SecurityIntelligencePipeline:
    """
    End-to-end security video intelligence pipeline.
    """

    def __init__(
        self,
        zones: Optional[List[ZoneDefinition]] = None,
        color_analyzer: Optional[VehicleColorAnalyzer] = None,
        face_detector: Optional[FaceDetector] = None,
        tracker: Optional[ObjectTracker] = None,
        activity_analyzer: Optional[ActivityAnalyzer] = None,
        behavior_analyzer: Optional[BehaviorAnalyzer] = None,
        incident_engine: Optional[IncidentIntelligenceEngine] = None,
        correlation_engine: Optional[AdvancedIncidentCorrelationEngine] = None,
        specialized_registry: Optional[SpecializedDetectorRegistry] = None,
        episode_aggregator: Optional[SpecializedVisualEpisodeAggregator] = None,
        specialized_validator: Optional[SpecializedValidationEngine] = None,
    ):
        self.tracker = tracker or ObjectTracker()
        self.color_analyzer = color_analyzer or VehicleColorAnalyzer()
        self.face_detector = face_detector or FaceDetector()
        self.zone_manager = ZoneManager(zones=zones)
        self.activity_analyzer = activity_analyzer or ActivityAnalyzer()
        self.behavior_analyzer = behavior_analyzer or BehaviorAnalyzer()
        self.incident_engine = incident_engine or IncidentIntelligenceEngine()
        self.correlation_engine = correlation_engine or AdvancedIncidentCorrelationEngine()
        self.specialized_registry = specialized_registry or get_specialized_registry()
        self.episode_aggregator = episode_aggregator or SpecializedVisualEpisodeAggregator()
        self.specialized_validator = specialized_validator or SpecializedValidationEngine()

        # Ensure default specialized detectors are registered
        if not self.specialized_registry.get("fire_visual_detector"):
            self.specialized_registry.register(FireVisualDetector(enabled=getattr(settings, "FIRE_DETECTION_ENABLED", True)))
        if not self.specialized_registry.get("smoke_visual_detector"):
            self.specialized_registry.register(SmokeVisualDetector(enabled=getattr(settings, "SMOKE_DETECTION_ENABLED", True)))
        if not self.specialized_registry.get("weapon_visual_detector"):
            self.specialized_registry.register(WeaponVisualDetector(enabled=getattr(settings, "WEAPON_DETECTION_ENABLED", True)))
        if not self.specialized_registry.get("pose_action_detector"):
            self.specialized_registry.register(PoseActionDetector(enabled=getattr(settings, "POSE_DETECTION_ENABLED", True)))

    def process_video_intelligence(
        self,
        video_id: str,
        video_path: str,
        raw_events: List[Dict[str, Any]],
        fps: float = 30.0,
        duration_seconds: float = 0.0,
        sample_rate_fps: float = 1.0,
        sampled_frames: Optional[Dict[float, np.ndarray]] = None,
        frame_width: Optional[float] = None,
        frame_height: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute full security intelligence pipeline across a video and its raw detections.
        Accepts optional sampled_frames mapping {timestamp: frame_bgr} to avoid redundant disk I/O.
        """
        # Group raw events by timestamp/frame for sequential tracker feeding
        events_by_time: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
        for ev in raw_events:
            t = round(float(ev.get("timestamp") or ev.get("timestamp_seconds") or 0.0), 3)
            events_by_time[t].append(ev)

        sorted_timestamps = sorted(events_by_time.keys())

        # Open video for visual crops (colors, faces) only if frames not pre-supplied in memory
        cap = None
        current_cap_frame_idx = -1
        video_w: Optional[float] = frame_width
        video_h: Optional[float] = frame_height

        if sampled_frames:
            for smp in sampled_frames.values():
                if hasattr(smp, "shape") and len(smp.shape) >= 2:
                    video_h, video_w = float(smp.shape[0]), float(smp.shape[1])
                    break

        if not sampled_frames and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened() and (video_w is None or video_h is None):
                vw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                vh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if vw and vh and vw > 0 and vh > 0:
                    video_w, video_h = float(vw), float(vh)

        tracks: List[TrackedObject] = []
        vehicle_attributes: List[VehicleAttribute] = []
        face_detections: List[FaceDetection] = []
        security_events: List[SecurityEvent] = []
        specialized_observations: List[Any] = []
        specialized_tracker = SpecializedTemporalTracker()

        health_registry = DetectorHealthRegistry.get_instance()
        health_registry.reset()
        health_registry.register_detector("yolo_detector", version="1.0.0", status=DetectorOperationalStatus.AVAILABLE.value, model_type=DetectorModelType.TRAINED_MODEL.value)
        health_registry.register_detector("vehicle_color_analyzer", version="1.0.0", status=DetectorOperationalStatus.AVAILABLE.value, model_type=DetectorModelType.HEURISTIC_CV.value)
        health_registry.register_detector("face_detector", version="1.0.0", status=DetectorOperationalStatus.AVAILABLE.value, model_type=DetectorModelType.TRAINED_MODEL.value)

        for spec_name in ["fire_visual_detector", "smoke_visual_detector", "weapon_visual_detector", "pose_action_detector"]:
            spec_inst = self.specialized_registry.get(spec_name)
            if spec_inst:
                is_trained = getattr(spec_inst, "has_model", False)
                m_type = DetectorModelType.TRAINED_MODEL.value if is_trained else (
                    DetectorModelType.FOUNDATION_ONLY.value if "weapon" in spec_name else DetectorModelType.HEURISTIC_CV.value
                )
                health_registry.register_detector(
                    name=spec_name,
                    version=getattr(spec_inst, "detector_version", "1.0.0"),
                    status=DetectorOperationalStatus.AVAILABLE.value if getattr(spec_inst, "enabled", True) else DetectorOperationalStatus.NOT_CONFIGURED.value,
                    model_type=m_type,
                    configured=getattr(spec_inst, "enabled", True),
                )

        self.tracker.reset(video_id=video_id)
        for det_inst in self.specialized_registry._detectors.values():
            if hasattr(det_inst, "reset"):
                det_inst.reset()

        # Step 1: Sequential Tracking & Visual Crop Analysis
        for t in sorted_timestamps:
            frame_dets = events_by_time[t]
            for det in frame_dets:
                v_stat = str(det.get("validation_status", "VALID")).upper()
                is_rej = (v_stat == "REJECTED")
                health_registry.record_observation("yolo_detector", is_validated=not is_rej, is_rejected=is_rej)
            # Only feed validated detections into the object tracker to avoid noise contamination
            valid_frame_dets = [
                d for d in frame_dets
                if str(d.get("validation_status", "VALID")).upper() != "REJECTED"
            ]
            active_tracks = self.tracker.update(
                timestamp=t,
                detections=valid_frame_dets,
                frame_width=video_w,
                frame_height=video_h,
            )

            # Retrieve frame image: from memory cache first, or sequential video stream
            frame_bgr = None
            frame_idx = int(round(t * fps))

            if sampled_frames:
                t_key = round(t, 3)
                if t_key in sampled_frames:
                    frame_bgr = sampled_frames[t_key]
                else:
                    best_match = None
                    min_diff = 0.1
                    for k in sampled_frames:
                        diff = abs(k - t)
                        if diff < min_diff:
                            min_diff = diff
                            best_match = k
                    if best_match is not None:
                        frame_bgr = sampled_frames[best_match]

            if frame_bgr is None and cap and cap.isOpened():
                if frame_idx >= current_cap_frame_idx:
                    while current_cap_frame_idx < frame_idx:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            break
                        current_cap_frame_idx += 1
                        if current_cap_frame_idx == frame_idx:
                            frame_bgr = frame
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        frame_bgr = frame
                        current_cap_frame_idx = frame_idx

            # Process vehicle colors, face crops, and specialized visual detections
            if frame_bgr is not None:
                # Specialized Visual Models (Fire, Smoke, Weapon foundation, Pose)
                if getattr(settings, "SPECIALIZED_VISUAL_ENABLED", True):
                    try:
                        # Pre-flight media quality gate: skip frames that are too blurry,
                        # too dark, overexposed, or near-uniform (transition frames).
                        # This prevents garbage observations from entering the pipeline.
                        frame_quality = FrameQualityGate.evaluate(frame_bgr)
                        if not frame_quality.is_usable:
                            logger.debug(
                                "Specialized inference skipped at %.3fs (quality gate): %s",
                                t, "; ".join(frame_quality.reasons),
                            )
                        else:
                            person_bboxes = [
                                det.get("bounding_box", {}) for det in frame_dets
                                if det.get("object_class") == "person"
                            ]
                            spec_obs, _ = self.specialized_registry.execute_all(
                                frame=frame_bgr,
                                timestamp=t,
                                frame_idx=frame_idx,
                                context={"person_bounding_boxes": person_bboxes},
                            )
                            if spec_obs:
                                for so in spec_obs:
                                    validated_so = self.specialized_validator.validate_observation(
                                        so, frame=frame_bgr, timestamp=t
                                    )
                                    specialized_observations.append(validated_so)
                                    d_name = getattr(validated_so, "detector_name", "specialized_detector")
                                    is_valid = (validated_so.validation_status == SpecializedValidationStatus.VALID)
                                    health_registry.record_observation(d_name, is_validated=is_valid)
                                    # Hard validation boundary: ONLY VALID observations enter temporal tracker
                                    if is_valid:
                                        specialized_tracker.update([validated_so])
                    except Exception as spec_err:
                        logger.debug(f"Specialized visual inference error at {t}s: {spec_err}")

                for det in frame_dets:
                    cls = det.get("object_class")
                    bb = det.get("bounding_box", {})
                    bbox = BoundingBox(
                        x1=float(bb.get("x1", 0.0)),
                        y1=float(bb.get("y1", 0.0)),
                        x2=float(bb.get("x2", 0.0)),
                        y2=float(bb.get("y2", 0.0)),
                    )

                    # Find corresponding track ID if available
                    matched_track = None
                    for trk in active_tracks:
                        if trk.object_class == cls and trk.current_bbox.iou(bbox) >= 0.4:
                            matched_track = trk
                            break

                    track_id = matched_track.track_id if matched_track else None

                    # Visual Color Analysis (vehicles and persons)
                    if cls in ["car", "bus", "truck", "motorcycle", "person"]:
                        det_conf = float(det.get("confidence", 0.0))
                        min_color_conf = 0.35 if cls == "person" else 0.40
                        if det_conf >= min_color_conf or (matched_track and matched_track.detection_count >= 2):
                            v_attr = self.color_analyzer.analyze(
                                frame_bgr=frame_bgr,
                                bbox=bbox,
                                timestamp=t,
                                object_class=cls,
                                track_id=track_id,
                            )
                            if cls != "person":
                                vehicle_attributes.append(v_attr)
                            if matched_track and v_attr.color != "unknown":
                                matched_track.color = v_attr.color
                                matched_track.color_confidence = v_attr.confidence

                    # Face Detection (Visual region only; strict safety)
                    if cls == "person":
                        f_dets = self.face_detector.detect_in_person_crop(
                            frame_bgr=frame_bgr,
                            person_bbox=bbox,
                            timestamp=t,
                            track_id=track_id,
                        )
                        face_detections.extend(f_dets)

        if cap:
            cap.release()

        # Finalize tracks: separate all recorded tracks and validated persistent tracks
        all_tracks = self.tracker.finalize()
        validated_tracks = [t for t in all_tracks if t.is_validated]
        specialized_tracks = specialized_tracker.finalize()
        valid_spec_obs = [
            o for o in specialized_observations
            if getattr(o, "validation_status", None) == SpecializedValidationStatus.VALID
        ]
        specialized_episodes = self.episode_aggregator.aggregate(
            video_id=video_id,
            observations=valid_spec_obs,
            temporal_tracks=specialized_tracks,
        )

        # Step 2-7: Phase 10 & Phase 15 Universal Incident Intelligence Engine
        incident_res = self.incident_engine.analyze_incidents(
            video_id=video_id,
            tracks=validated_tracks,
            validated_detections=raw_events,
            fps=fps,
            duration_seconds=duration_seconds,
            sample_rate_fps=sample_rate_fps,
            zones=self.zone_manager.list_zones(),
            vehicle_attributes=vehicle_attributes,
            face_detections=face_detections,
            specialized_observations=valid_spec_obs,
            specialized_tracks=specialized_tracks,
            specialized_episodes=specialized_episodes,
        )

        security_events = incident_res["security_events"]
        incidents = incident_res["incidents"]
        diagnostics = incident_res["diagnostics"]

        # Step 8: Phase 16 Advanced Incident Correlation, Fusion & Storylines
        correlation_res = self.correlation_engine.correlate_incidents(
            video_id=video_id,
            candidates=incidents,
            tracks=validated_tracks,
            fps=fps,
        )
        correlated_incidents = correlation_res["correlated_incidents"]
        correlation_diagnostics = correlation_res["diagnostics"]

        # Activity Density & Peaks for timeline density charts
        density_buckets = self.activity_analyzer.analyze_timeline_density(raw_events, duration_seconds)

        logger.info(
            f"Phase 16 intelligence complete for {video_id}: "
            f"{len(validated_tracks)} validated tracks ({len(all_tracks)} total raw), "
            f"{len(vehicle_attributes)} vehicle attributes, "
            f"{len(face_detections)} face detections, {len(security_events)} security events "
            f"({len(specialized_observations)} specialized visual observations, "
            f"{len(specialized_episodes)} specialized episodes, "
            f"{len(incidents)} fused candidates, "
            f"{len(correlated_incidents)} correlated incidents, {diagnostics['failed_detectors']} failures)."
        )

        return {
            "video_id": video_id,
            "tracks": validated_tracks,
            "all_tracks": all_tracks,
            "vehicle_attributes": vehicle_attributes,
            "face_detections": face_detections,
            "security_events": security_events,
            "incidents": incidents,
            "correlated_incidents": correlated_incidents,
            "correlation_diagnostics": correlation_diagnostics,
            "incident_diagnostics": diagnostics,
            "activity_buckets": density_buckets,
            "specialized_observations": specialized_observations,
            "specialized_tracks": specialized_tracks,
            "specialized_episodes": specialized_episodes,
            "detector_health": health_registry.get_health_report(),
        }
