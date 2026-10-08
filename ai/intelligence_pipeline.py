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
import math

from ai.schemas import (
    BoundingBox,
    TrackedObject,
    VehicleAttribute,
    PersonVisualAttributes,
    FaceDetection,
    ZoneDefinition,
    SecurityEvent,
)
from ai.tracking.tracker import ObjectTracker
from ai.attributes.color_analyzer import VehicleColorAnalyzer
from ai.attributes.person_analyzer import PersonAttributeAnalyzer
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
from ai.detection.micro_object import MicroObjectRecoveryEngine
from backend.app.core.config import settings

logger = logging.getLogger(__name__)


def aggregate_track_clothing_color(track: TrackedObject) -> None:
    """
    Accumulate and smooth clothing color across multi-frame observation history.
    Establishes track.color when consistent evidence exists across >= 2 observations.
    Preserves uncertainty when observations are ambiguous or under low illumination.
    """
    if not track or not track.attribute_history:
        return

    valid_upper = []
    valid_lower = []
    for h in track.attribute_history:
        up = h.get("upper_clothing") or h.get("clothing_color")
        if up and not up.get("is_illumination_uncertain"):
            c = up.get("color_name") or up.get("color")
            if c and c not in ("unknown", "uncertain"):
                valid_upper.append((c, float(up.get("confidence", 0.5))))

        low = h.get("lower_clothing")
        if low and not low.get("is_illumination_uncertain"):
            c = low.get("color_name") or low.get("color")
            if c and c not in ("unknown", "uncertain"):
                valid_lower.append((c, float(low.get("confidence", 0.5))))

    if not valid_upper and not valid_lower:
        return

    color_weights: Dict[str, float] = {}
    color_counts: Dict[str, int] = {}
    for c, conf in valid_upper:
        color_weights[c] = color_weights.get(c, 0.0) + conf
        color_counts[c] = color_counts.get(c, 0) + 1

    # Integrate lower body observations (matching clothing e.g. dress or suit)
    for c, conf in valid_lower:
        weight_factor = 0.5 if c not in color_weights else 0.8
        color_weights[c] = color_weights.get(c, 0.0) + conf * weight_factor
        color_counts[c] = color_counts.get(c, 0) + 1

    if not color_weights:
        return

    total_w = sum(color_weights.values())
    top_color, top_w = max(color_weights.items(), key=lambda kv: kv[1])
    top_count = color_counts[top_color]
    agreement = top_w / total_w if total_w > 0 else 0.0

    # Multi-frame consensus gating:
    # Require at least 2 consistent observations and agreement >= 0.45.
    # Single-frame observations are unconfirmed and must NOT receive false 0.9000 confidence.
    if top_count >= 2 and agreement >= 0.45:
        obs_factor = min(1.0, top_count / 5.0)
        avg_input_conf = top_w / max(1, top_count)
        consensus_conf = min(0.95, max(0.45, agreement * (0.50 + 0.45 * obs_factor) * min(1.0, avg_input_conf / 0.70)))
        track.color = top_color
        track.color_confidence = round(consensus_conf, 4)
        if track.visual_attributes is None:
            track.visual_attributes = {}
        track.visual_attributes["clothing_color"] = {
            "color": top_color,
            "color_name": top_color,
            "confidence": round(consensus_conf, 4),
            "observation_count": top_count,
            "is_confirmed": True,
        }
    else:
        # Insufficient temporal consensus (< 2 observations or low agreement):
        # Do NOT invent or commit an unverified color
        track.color = None
        track.color_confidence = None
        if track.visual_attributes is None:
            track.visual_attributes = {}
        track.visual_attributes["clothing_color"] = {
            "color": top_color,
            "color_name": top_color,
            "confidence": min(0.40, round(agreement * 0.40, 4)),
            "observation_count": top_count,
            "is_confirmed": False,
        }


class SecurityIntelligencePipeline:
    """
    End-to-end security video intelligence pipeline.
    """

    def __init__(
        self,
        zones: Optional[List[ZoneDefinition]] = None,
        color_analyzer: Optional[VehicleColorAnalyzer] = None,
        person_analyzer: Optional[PersonAttributeAnalyzer] = None,
        face_detector: Optional[FaceDetector] = None,
        tracker: Optional[ObjectTracker] = None,
        activity_analyzer: Optional[ActivityAnalyzer] = None,
        behavior_analyzer: Optional[BehaviorAnalyzer] = None,
        incident_engine: Optional[IncidentIntelligenceEngine] = None,
        correlation_engine: Optional[AdvancedIncidentCorrelationEngine] = None,
        specialized_registry: Optional[SpecializedDetectorRegistry] = None,
        episode_aggregator: Optional[SpecializedVisualEpisodeAggregator] = None,
        specialized_validator: Optional[SpecializedValidationEngine] = None,
        sampling_aware_correlation: Optional[bool] = None,
    ):
        self.tracker = tracker or ObjectTracker()
        self.color_analyzer = color_analyzer or VehicleColorAnalyzer()
        self.face_detector = face_detector or FaceDetector()
        self.person_analyzer = person_analyzer or PersonAttributeAnalyzer(face_detector=self.face_detector)
        self.zone_manager = ZoneManager(zones=zones)
        self.activity_analyzer = activity_analyzer or ActivityAnalyzer()
        self.behavior_analyzer = behavior_analyzer or BehaviorAnalyzer()
        self.incident_engine = incident_engine or IncidentIntelligenceEngine()
        self.sampling_aware_correlation = sampling_aware_correlation
        if correlation_engine is not None:
            self.correlation_engine = correlation_engine
        else:
            self.correlation_engine = AdvancedIncidentCorrelationEngine(sampling_aware=sampling_aware_correlation)
        self.specialized_registry = specialized_registry or get_specialized_registry()
        self.episode_aggregator = episode_aggregator or SpecializedVisualEpisodeAggregator()
        self.specialized_validator = specialized_validator or SpecializedValidationEngine()
        self.micro_engine = MicroObjectRecoveryEngine()

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
        sampling_aware_correlation: Optional[bool] = None,
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

        if sampled_frames and (video_w is None or video_h is None):
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
        person_attributes: List[PersonVisualAttributes] = []
        face_detections: List[FaceDetection] = []
        security_events: List[SecurityEvent] = []
        specialized_observations: List[Any] = []
        specialized_tracker = SpecializedTemporalTracker()

        health_registry = DetectorHealthRegistry.get_instance()
        health_registry.reset()
        health_registry.register_detector("yolo_detector", version="1.0.0", status=DetectorOperationalStatus.AVAILABLE.value, model_type=DetectorModelType.TRAINED_MODEL.value)
        health_registry.register_detector("vehicle_color_analyzer", version="1.0.0", status=DetectorOperationalStatus.AVAILABLE.value, model_type=DetectorModelType.HEURISTIC_CV.value)
        health_registry.register_detector("person_analyzer", version="1.0.0", status=DetectorOperationalStatus.AVAILABLE.value, model_type=DetectorModelType.HEURISTIC_CV.value)
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

            # Retrieve frame image BEFORE tracker update so GMC-capable trackers
            # (e.g., BoT-SORT) can use it for camera motion compensation.
            # Frame retrieval depends only on timestamp, not tracker output.
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

            active_tracks = self.tracker.update(
                timestamp=t,
                detections=valid_frame_dets,
                frame_width=video_w,
                frame_height=video_h,
                frame_bgr=frame_bgr,
            )

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

                    # Visual Attribute Analysis
                    # 1. Person Attributes (Anatomical upper/lower clothing, headwear, carried objects, face telemetry)
                    if cls == "person":
                        det_conf = float(det.get("confidence", 0.0))
                        if det_conf >= 0.35 or (matched_track and matched_track.detection_count >= 2):
                            p_attr = self.person_analyzer.analyze(
                                frame_bgr=frame_bgr,
                                person_bbox=bbox,
                                timestamp=t,
                                track_id=track_id,
                            )
                            person_attributes.append(p_attr)
                            if matched_track:
                                matched_track.visual_attributes = p_attr.to_dict()
                                matched_track.attribute_history.append({
                                    "timestamp": t,
                                    "upper_clothing": p_attr.upper_clothing.to_dict() if p_attr.upper_clothing else None,
                                    "lower_clothing": p_attr.lower_clothing.to_dict() if p_attr.lower_clothing else None,
                                    "headwear": p_attr.headwear_present,
                                    "face_present": p_attr.face_telemetry.face_present if p_attr.face_telemetry else False,
                                })
                                # Accumulate clothing color across multi-frame observation history
                                aggregate_track_clothing_color(matched_track)

                    # 2. Vehicle Color & Orientation Analysis
                    elif cls in ["car", "bus", "truck", "motorcycle"]:
                        det_conf = float(det.get("confidence", 0.0))
                        if det_conf >= 0.40 or (matched_track and matched_track.detection_count >= 2):
                            v_attr = self.color_analyzer.analyze(
                                frame_bgr=frame_bgr,
                                bbox=bbox,
                                timestamp=t,
                                object_class=cls,
                                track_id=track_id,
                            )
                            vehicle_attributes.append(v_attr)
                            if matched_track and v_attr.color != "uncertain":
                                matched_track.color = v_attr.color
                                matched_track.color_confidence = v_attr.confidence
                                matched_track.visual_attributes = v_attr.to_dict()
                                matched_track.attribute_history.append({
                                    "timestamp": t,
                                    "color": v_attr.color,
                                    "secondary_color": v_attr.secondary_color,
                                    "is_illumination_uncertain": v_attr.is_illumination_uncertain,
                                })

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
        for trk in all_tracks:
            if trk.object_class == "person":
                aggregate_track_clothing_color(trk)
        validated_tracks = [t for t in all_tracks if t.is_validated]

        # Phase 0.2: Universal Micro-Object & Object-Interaction Recovery Engine
        # Discovers unmodeled small portable objects (<32px or outside COCO)
        # via selective spatio-temporal reach ROI analysis around dwelling person tracks.
        if getattr(settings, "MICRO_OBJECT_RECOVERY_ENABLED", True) and video_w and video_h:
            dwelling_intervals = self.micro_engine.discover_dwelling_person_intervals(validated_tracks, fps)
            if dwelling_intervals:
                recovered_micro_dets = self.micro_engine.recover_micro_objects_from_video(
                    video_path=video_path,
                    dwelling_intervals=dwelling_intervals,
                    fps=fps,
                    frame_width=int(video_w),
                    frame_height=int(video_h),
                    sampled_frames_cache=sampled_frames,
                )
                if recovered_micro_dets:
                    # Save existing visual attributes and clothing color history before tracker reset
                    prior_track_attrs = {
                        t.track_id: {
                            "color": t.color,
                            "color_confidence": t.color_confidence,
                            "visual_attributes": t.visual_attributes,
                            "attribute_history": list(t.attribute_history) if t.attribute_history else [],
                            "object_class": t.object_class,
                            "trajectory": list(t.trajectory) if t.trajectory else [],
                            "history_bboxes": list(t.history_bboxes) if t.history_bboxes else [],
                        }
                        for t in all_tracks
                    }

                    # Re-run tracking to integrate recovered micro-objects alongside baseline detections
                    self.tracker.reset(video_id=video_id)
                    all_ts = sorted(set(list(events_by_time.keys()) + [rd["timestamp"] for rd in recovered_micro_dets]))
                    for ts in all_ts:
                        frame_dets = list(events_by_time.get(ts, []))
                        for rd in recovered_micro_dets:
                            if abs(rd["timestamp"] - ts) < 0.25:
                                frame_dets.append(rd)
                        valid_frame_dets = [
                            d for d in frame_dets
                            if str(d.get("validation_status", "VALID")).upper() != "REJECTED"
                        ]
                        self.tracker.update(
                            timestamp=ts,
                            detections=valid_frame_dets,
                            frame_width=video_w,
                            frame_height=video_h,
                        )
                    all_tracks = self.tracker.finalize()

                    # Strict contemporaneous temporal-spatial attribute restoration (zero cross-track color leakage)
                    for trk in all_tracks:
                        matched_prior = None

                        # Check exact track_id match first
                        prior = prior_track_attrs.get(trk.track_id)
                        if prior and prior["object_class"] == trk.object_class:
                            p_traj = prior.get("trajectory", [])
                            t_traj = trk.trajectory or []
                            shared_ts = set(p[0] for p in p_traj).intersection(set(t[0] for t in t_traj))
                            if shared_ts:
                                p_map = {p[0]: (p[1], p[2]) for p in p_traj}
                                sim_dists = [math.hypot(x - p_map[ts][0], y - p_map[ts][1]) for ts, x, y in t_traj if ts in p_map]
                                if sim_dists and (sum(sim_dists) / len(sim_dists)) <= 40.0:
                                    matched_prior = prior
                            elif not p_traj and not t_traj:
                                matched_prior = prior

                        # If not matched by exact ID, search prior tracks of same class with contemporaneous overlap
                        if not matched_prior:
                            best_candidate = None
                            best_overlap_score = float("inf")
                            t_traj = trk.trajectory or []
                            t_ts_set = set(t[0] for t in t_traj)
                            for p_id, p_data in prior_track_attrs.items():
                                if p_data["object_class"] != trk.object_class:
                                    continue
                                p_traj = p_data.get("trajectory", [])
                                shared_ts = t_ts_set.intersection(set(p[0] for p in p_traj))
                                if len(shared_ts) >= 2 or (len(t_traj) == 1 and len(shared_ts) == 1):
                                    p_map = {p[0]: (p[1], p[2]) for p in p_traj}
                                    dists = [math.hypot(x - p_map[ts][0], y - p_map[ts][1]) for ts, x, y in t_traj if ts in p_map]
                                    avg_dist = sum(dists) / len(dists)
                                    if avg_dist <= 35.0 and avg_dist < best_overlap_score:
                                        best_overlap_score = avg_dist
                                        best_candidate = p_data
                            if best_candidate:
                                matched_prior = best_candidate

                        if matched_prior:
                            trk.color = matched_prior["color"]
                            trk.color_confidence = matched_prior["color_confidence"]
                            trk.visual_attributes = matched_prior["visual_attributes"]
                            trk.attribute_history = matched_prior["attribute_history"]

                        if trk.object_class == "person" and not trk.color:
                            aggregate_track_clothing_color(trk)

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

        # Step 8: Phase 16 & 21D Advanced Incident Correlation, Fusion & Storylines
        resolved_sampling_aware = (
            sampling_aware_correlation
            if sampling_aware_correlation is not None
            else self.sampling_aware_correlation
        )
        correlation_res = self.correlation_engine.correlate_incidents(
            video_id=video_id,
            candidates=incidents,
            tracks=validated_tracks,
            fps=fps,
            sample_rate_fps=sample_rate_fps,
            sampling_aware=resolved_sampling_aware,
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
            "person_attributes": person_attributes,
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
