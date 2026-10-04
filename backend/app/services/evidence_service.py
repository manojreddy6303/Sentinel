"""
Evidence Extraction & Management Service for Sentinel

Responsible for:
- Deterministic extraction of original video frame snapshots at precise timestamps
- Optional annotated snapshot creation with bounding box overlays (non-destructive)
- Bounded short context video clip extraction surrounding detections and grouped events
- Duplicate mitigation to avoid redundant media generation and storage bloating
- Relational database persistence (EvidenceModel) and secure media provenance
"""

import os
import uuid
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
from sqlalchemy import desc

from backend.app.core.config import settings
from database.session import SessionLocal
from database.models import VideoModel, EventModel, GroupedEventModel, EvidenceModel

logger = logging.getLogger(__name__)


class EvidenceService:
    """Service for extracting, indexing, and serving investigation evidence."""

    def __init__(self):
        self.evidence_dir = settings.STORAGE_EVIDENCE_DIR
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_video_source(self, video_id: str) -> Tuple[Path, Dict[str, Any]]:
        """
        Locate source video file on disk and load its metadata sidecar.
        Raises FileNotFoundError if source video is missing.
        """
        meta_path = settings.STORAGE_UPLOADS_DIR / f"{video_id}.json"
        metadata = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception as exc:
                logger.warning(f"Failed to read metadata sidecar for {video_id}: {exc}")

        # Check path from metadata
        video_path = None
        if "storage_path" in metadata:
            candidate = Path(metadata["storage_path"])
            if candidate.exists() and candidate.is_file():
                video_path = candidate

        # Fallback search by video_id in uploads
        if not video_path:
            matches = list(settings.STORAGE_UPLOADS_DIR.glob(f"{video_id}_*"))
            for m in matches:
                if m.is_file() and m.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
                    video_path = m
                    break

        if not video_path or not video_path.exists():
            raise FileNotFoundError(f"Source video file for video_id '{video_id}' not found.")

        return video_path, metadata

    def create_evidence(
        self,
        video_id: str,
        timestamp: float,
        event_id: Optional[str] = None,
        evidence_type: str = "snapshot_and_clip",
        pre_seconds: float = 3.0,
        post_seconds: float = 3.0,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract and record evidence (snapshot, annotated snapshot, and/or clip)
        for a detection or grouped event.
        """
        # Validate evidence_type
        evidence_type = evidence_type.lower()
        if evidence_type not in ["snapshot_and_clip", "snapshot_only", "clip_only"]:
            evidence_type = "snapshot_and_clip"

        db = SessionLocal()
        try:
            # Pre-validate event_id if supplied
            if event_id:
                raw_ev = db.query(EventModel).filter(EventModel.id == event_id).first()
                if raw_ev:
                    if getattr(raw_ev, "validation_status", "VALID") == "REJECTED":
                        raise ValueError(
                            f"Cannot generate evidence for rejected detection '{event_id}' ({getattr(raw_ev, 'validation_reason', 'rejected')})."
                        )
                    if raw_ev.video_id != video_id:
                        raise ValueError(f"Event '{event_id}' does not belong to video '{video_id}'.")

            video_path, metadata = self._resolve_video_source(video_id)
            original_filename = metadata.get("filename", video_path.name)

            # 1. Inspect source video via OpenCV to get geometry and boundaries
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"OpenCV could not open source video at '{video_path}'")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if fps <= 0 or np.isnan(fps):
                fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration_seconds = metadata.get("duration_seconds")
            if duration_seconds is None or duration_seconds <= 0:
                duration_seconds = (total_frames / fps) if total_frames > 0 else 0.0

            # 2. Validate timestamp
            if timestamp < 0:
                cap.release()
                raise ValueError(f"Timestamp cannot be negative (got {timestamp}s).")
            if duration_seconds > 0 and timestamp > duration_seconds + 1.0:
                cap.release()
                raise ValueError(
                    f"Requested timestamp {timestamp}s exceeds video duration of {duration_seconds:.2f}s."
                )

            # Clamp timestamp to valid duration
            effective_ts = max(0.0, min(timestamp, duration_seconds) if duration_seconds > 0 else timestamp)

            # 3. Retrieve event/detection metadata if event_id is supplied
            object_class = None
            confidence = None
            bounding_box = None
            start_time = None
            end_time = None
            matched_sec_ev: Optional[Any] = None

            if event_id:
                # Check raw detection table (EventModel)
                raw_ev = db.query(EventModel).filter(EventModel.id == event_id).first()
                if raw_ev:
                    object_class = raw_ev.object_class
                    confidence = raw_ev.confidence
                    bounding_box = {
                        "x1": round(raw_ev.bbox_x1, 1),
                        "y1": round(raw_ev.bbox_y1, 1),
                        "x2": round(raw_ev.bbox_x2, 1),
                        "y2": round(raw_ev.bbox_y2, 1),
                    }
                    effective_ts = raw_ev.timestamp_seconds
                else:
                    # Check grouped event table (GroupedEventModel)
                    grp_ev = db.query(GroupedEventModel).filter(GroupedEventModel.id == event_id).first()
                    if grp_ev:
                        if grp_ev.video_id != video_id:
                            cap.release()
                            raise ValueError(f"Grouped event '{event_id}' does not belong to video '{video_id}'.")
                        object_class = grp_ev.event_type
                        confidence = grp_ev.max_confidence
                        start_time = grp_ev.start_time
                        end_time = grp_ev.end_time
                        effective_ts = grp_ev.start_time
                    else:
                        from database.models import SecurityEventModel
                        sec_ev = db.query(SecurityEventModel).filter(SecurityEventModel.id == event_id).first()
                        if sec_ev:
                            matched_sec_ev = sec_ev
                            if sec_ev.video_id != video_id:
                                cap.release()
                                raise ValueError(f"Security event '{event_id}' does not belong to video '{video_id}'.")
                            if getattr(sec_ev, "validation_status", "") == "REJECTED":
                                cap.release()
                                raise ValueError(f"Cannot generate evidence for rejected security event '{event_id}'.")
                            object_class = sec_ev.event_type
                            confidence = sec_ev.confidence
                            bounding_box = sec_ev.bounding_box
                            effective_ts = sec_ev.timestamp_seconds
                            start_time = sec_ev.timestamp_seconds
                            end_time = sec_ev.timestamp_seconds + (sec_ev.duration_seconds or 0.0)

                            # If timestamp is at or near 0.0 but evidence candidates or representative timestamps exist, use canonical event timestamp
                            sec_meta = getattr(sec_ev, "incident_metadata", {}) or {}
                            ev_cands = sec_meta.get("evidence_candidates") or []
                            if effective_ts <= 0.05 and ev_cands:
                                cand_ts = ev_cands[0].get("timestamp")
                                if cand_ts is not None and float(cand_ts) > 0.0:
                                    effective_ts = float(cand_ts)
                                    start_time = max(0.0, effective_ts - 2.0)
                                    end_time = effective_ts + max(2.0, sec_ev.duration_seconds or 3.0)
                            elif effective_ts <= 0.05 and sec_meta.get("representative_timestamps"):
                                rep_ts = [float(t) for t in sec_meta["representative_timestamps"] if float(t) > 0.0]
                                if rep_ts:
                                    effective_ts = rep_ts[0]
                                    start_time = max(0.0, effective_ts - 2.0)
                                    end_time = effective_ts + max(2.0, sec_ev.duration_seconds or 3.0)
                        else:
                            from database.models import SpecializedObservationModel
                            spec_obs = db.query(SpecializedObservationModel).filter(SpecializedObservationModel.id == event_id).first()
                            if spec_obs:
                                if spec_obs.video_id != video_id:
                                    cap.release()
                                    raise ValueError(f"Specialized observation '{event_id}' does not belong to video '{video_id}'.")
                                if spec_obs.validation_status == "REJECTED":
                                    cap.release()
                                    raise ValueError(f"Cannot generate evidence for rejected specialized observation '{event_id}'.")
                                object_class = spec_obs.class_name
                                confidence = spec_obs.confidence
                                bounding_box = spec_obs.bounding_box
                                effective_ts = spec_obs.timestamp_seconds
                                start_time = spec_obs.timestamp_seconds
                                end_time = spec_obs.timestamp_seconds
                            else:
                                cap.release()
                                raise ValueError(f"Event with ID '{event_id}' not found in database.")

            # 4. Duplicate Check
            # Prevent creating redundant duplicate evidence files for the exact same event/timestamp
            dup_query = db.query(EvidenceModel).filter(
                EvidenceModel.video_id == video_id,
            )
            if event_id:
                dup_candidates = dup_query.filter(
                    (EvidenceModel.event_id == event_id) |
                    (
                        (EvidenceModel.timestamp_seconds >= effective_ts - 1.0) &
                        (EvidenceModel.timestamp_seconds <= effective_ts + 1.0) &
                        (EvidenceModel.object_class == object_class)
                    )
                ).all()
            else:
                dup_candidates = dup_query.filter(
                    EvidenceModel.timestamp_seconds >= effective_ts - 1.0,
                    EvidenceModel.timestamp_seconds <= effective_ts + 1.0,
                    EvidenceModel.object_class == object_class,
                ).all()

            for existing in dup_candidates:
                has_snap = bool(existing.snapshot_path and Path(existing.snapshot_path).exists())
                has_clip = bool(existing.clip_path and Path(existing.clip_path).exists())
                if existing.evidence_type == evidence_type:
                    if (evidence_type == "snapshot_only" and has_snap) or \
                       (evidence_type == "clip_only" and has_clip) or \
                       (evidence_type == "snapshot_and_clip" and has_snap and has_clip):
                        cap.release()
                        logger.info(f"Duplicate evidence found ({existing.id}) for timestamp {effective_ts:.1f}s, returning existing.")
                        return self._serialize_evidence(existing, is_duplicate=True)
                elif existing.evidence_type == "snapshot_and_clip" and has_snap and has_clip:
                    # Existing snapshot_and_clip subsumes snapshot_only or clip_only requests
                    cap.release()
                    logger.info(f"Existing superset evidence found ({existing.id}), returning existing.")
                    return self._serialize_evidence(existing, is_duplicate=True)

            # 5. Extract Media
            evidence_id = f"ev_{uuid.uuid4().hex[:12]}"
            snapshot_path: Optional[str] = None
            annotated_snapshot_path: Optional[str] = None
            clip_path: Optional[str] = None
            clip_duration: Optional[float] = None

            # --- A. Snapshot Extraction ---
            if evidence_type in ["snapshot_and_clip", "snapshot_only"]:
                target_frame = int(round(effective_ts * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Fallback seek by msec
                    cap.set(cv2.CAP_PROP_POS_MSEC, effective_ts * 1000.0)
                    ret, frame = cap.read()

                if ret and frame is not None:
                    snap_file = self.evidence_dir / f"{evidence_id}_snapshot.jpg"
                    cv2.imwrite(str(snap_file), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    snapshot_path = str(snap_file)

                    # Create Annotated Snapshot with accurate multi-track grounding
                    annotated = frame.copy()
                    frame_h, frame_w = annotated.shape[:2]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    base_scale = max(0.38, min(0.70, frame_h / 480.0))
                    thickness = 1 if frame_h < 400 else 2

                    is_theft = (
                        object_class == "POTENTIAL_THEFT"
                        or (matched_sec_ev and matched_sec_ev.event_type == "POTENTIAL_THEFT")
                    )

                    if is_theft:
                        has_person_box = False
                        has_object_box = False

                        # 1. Person Bounding Box
                        p_x1 = bounding_box.get("x1") if isinstance(bounding_box, dict) else None
                        p_y1 = bounding_box.get("y1") if isinstance(bounding_box, dict) else None
                        p_x2 = bounding_box.get("x2") if isinstance(bounding_box, dict) else None
                        p_y2 = bounding_box.get("y2") if isinstance(bounding_box, dict) else None

                        p_track_id = (
                            (bounding_box.get("person_track_id") if isinstance(bounding_box, dict) else None)
                            or (matched_sec_ev.track_id if matched_sec_ev else None)
                            or "PERSON"
                        )

                        if p_x1 is not None and p_y1 is not None and p_x2 is not None and p_y2 is not None:
                            px1 = max(0, min(frame_w - 1, int(round(p_x1))))
                            py1 = max(0, min(frame_h - 1, int(round(p_y1))))
                            px2 = max(0, min(frame_w - 1, int(round(p_x2))))
                            py2 = max(0, min(frame_h - 1, int(round(p_y2))))

                            if px2 > px1 and py2 > py1:
                                has_person_box = True
                                # High-visibility emerald green box around person
                                cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 230, 115), 2)
                                p_lbl = f"PERSON • {p_track_id}"
                                self._draw_box_label(annotated, p_lbl, px1, py1, (0, 230, 115), (0, 0, 0), font, base_scale, thickness)

                        # 2. Object Bounding Box (if available)
                        o_bbox = bounding_box.get("object_bbox") if isinstance(bounding_box, dict) else None
                        o_track_id = bounding_box.get("object_track_id") if isinstance(bounding_box, dict) else None
                        o_cls = (bounding_box.get("object_class") if isinstance(bounding_box, dict) else None) or "OBJECT"
                        is_prior = bounding_box.get("is_prior_object_observation", False) if isinstance(bounding_box, dict) else False
                        o_ts = bounding_box.get("object_observation_timestamp") if isinstance(bounding_box, dict) else None

                        if o_bbox and isinstance(o_bbox, dict):
                            ox1 = max(0, min(frame_w - 1, int(round(o_bbox.get("x1", 0)))))
                            oy1 = max(0, min(frame_h - 1, int(round(o_bbox.get("y1", 0)))))
                            ox2 = max(0, min(frame_w - 1, int(round(o_bbox.get("x2", 0)))))
                            oy2 = max(0, min(frame_h - 1, int(round(o_bbox.get("y2", 0)))))

                            if ox2 > ox1 and oy2 > oy1:
                                has_object_box = True
                                # Vibrant cyan box around object
                                cv2.rectangle(annotated, (ox1, oy1), (ox2, oy2), (255, 200, 0), 2)
                                if is_prior and o_ts is not None:
                                    o_lbl = f"{o_cls.upper()} • {o_track_id or 'TRACK'} (Prior Obs {o_ts:.1f}s)"
                                else:
                                    o_lbl = f"{o_cls.upper()} • {o_track_id or 'TRACK'}"
                                self._draw_box_label(annotated, o_lbl, ox1, oy1, (255, 200, 0), (0, 0, 0), font, base_scale, thickness)

                        # 3. Top Banner: Authoritative Score & Validation Semantics
                        meta = getattr(matched_sec_ev, "incident_metadata", {}) or {}
                        p_str = meta.get("pattern_evidence_strength")
                        a_score = meta.get("assessment_score")
                        v_dec = meta.get("validation_decision")
                        if p_str is None:
                            p_str = 0.95
                        if a_score is None:
                            a_score = confidence if confidence is not None else 0.65
                        if v_dec is None:
                            v_dec = "REVIEW_REQUIRED" if a_score <= 0.65 else "ACCEPTED"

                        p_pct = int(round(p_str * 100))
                        a_pct = int(round(a_score * 100))

                        banner_title = "POTENTIAL THEFT PATTERN"
                        if v_dec == "REVIEW_REQUIRED":
                            banner_sub = f"Pattern Evidence Strength: {p_pct}% • Final Assessment: {a_pct}% (REVIEW_REQUIRED) — Human verification required"
                        elif v_dec == "ACCEPTED":
                            banner_sub = f"Pattern Evidence Strength: {p_pct}% • Final Assessment: {a_pct}% (ACCEPTED)"
                        else:
                            banner_sub = f"Pattern Evidence Strength: {p_pct}% • Final Assessment: {a_pct}% ({v_dec})"

                        if not has_person_box and not has_object_box:
                            if v_dec == "REVIEW_REQUIRED":
                                banner_sub = f"Pattern: {p_pct}% • Final: {a_pct}% (REVIEW_REQUIRED) — Human verification required (no box)"
                            else:
                                banner_sub = f"Pattern: {p_pct}% • Final: {a_pct}% ({v_dec}) (no box)"

                        self._draw_banner(annotated, banner_title, banner_sub, font, base_scale, frame_w, frame_h)

                        ann_file = self.evidence_dir / f"{evidence_id}_annotated.jpg"
                        cv2.imwrite(str(ann_file), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        annotated_snapshot_path = str(ann_file)

                    elif bounding_box and isinstance(bounding_box, dict) and "x1" in bounding_box:
                        x1 = max(0, min(frame_w - 1, int(round(bounding_box["x1"]))))
                        y1 = max(0, min(frame_h - 1, int(round(bounding_box["y1"]))))
                        x2 = max(0, min(frame_w - 1, int(round(bounding_box["x2"]))))
                        y2 = max(0, min(frame_h - 1, int(round(bounding_box["y2"]))))

                        if x2 > x1 and y2 > y1:
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 230, 115), 2)
                            conf_str = f" {int(confidence * 100)}%" if confidence is not None else ""
                            lbl = f"{(object_class or 'OBJECT').upper()}{conf_str} | {effective_ts:.2f}s"
                            self._draw_box_label(annotated, lbl, x1, y1, (0, 230, 115), (0, 0, 0), font, base_scale, thickness)

                            ann_file = self.evidence_dir / f"{evidence_id}_annotated.jpg"
                            cv2.imwrite(str(ann_file), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                            annotated_snapshot_path = str(ann_file)
                else:
                    logger.warning(f"Could not extract frame at {effective_ts}s from {video_path}")

            # --- B. Video Clip Extraction ---
            if evidence_type in ["snapshot_and_clip", "clip_only"]:
                # Determine clip time window
                if start_time is not None and end_time is not None:
                    c_start = max(0.0, start_time - pre_seconds)
                    c_end = min(duration_seconds, end_time + post_seconds) if duration_seconds > 0 else (end_time + post_seconds)
                else:
                    c_start = max(0.0, effective_ts - pre_seconds)
                    c_end = min(duration_seconds, effective_ts + post_seconds) if duration_seconds > 0 else (effective_ts + post_seconds)

                if c_end <= c_start:
                    c_end = c_start + 1.0

                start_time = c_start
                end_time = c_end
                clip_duration = round(c_end - c_start, 2)
                start_frame = int(round(c_start * fps))
                end_frame = int(round(c_end * fps))

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

                clip_file = self.evidence_dir / f"{evidence_id}_clip.mp4"
                extracted_with_ffmpeg = False
                try:
                    from backend.app.services.playback_service import get_ffmpeg_binary
                    ffmpeg_exe = get_ffmpeg_binary()
                    if ffmpeg_exe and Path(video_path).exists():
                        cmd = [
                            ffmpeg_exe, "-y",
                            "-ss", f"{c_start:.3f}",
                            "-i", str(video_path),
                            "-t", f"{clip_duration:.3f}",
                            "-c:v", "libx264",
                            "-pix_fmt", "yuv420p",
                            "-preset", "veryfast",
                            "-crf", "23",
                            "-movflags", "+faststart",
                            "-threads", "2",
                            "-an",
                            str(clip_file),
                        ]
                        res = subprocess.run(
                            cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=60,
                        )
                        if res.returncode == 0 and clip_file.exists() and clip_file.stat().st_size > 0:
                            extracted_with_ffmpeg = True
                except Exception as ff_err:
                    logger.warning(f"Direct FFmpeg clip extraction failed for {evidence_id}, falling back to OpenCV: {ff_err}")

                if not extracted_with_ffmpeg:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(clip_file), fourcc, fps, (width, height))

                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    current_frame = start_frame

                    while current_frame <= end_frame:
                        ret, f = cap.read()
                        if not ret or f is None:
                            break
                        writer.write(f)
                        current_frame += 1

                    writer.release()

                if clip_file.exists() and clip_file.stat().st_size > 0:
                    clip_path = str(clip_file)
                    try:
                        from backend.app.services.playback_service import ensure_evidence_clip_playback
                        ensure_evidence_clip_playback(evidence_id, clip_file)
                    except Exception as exc:
                        logger.warning(f"Could not pre-warm evidence playback clip for {evidence_id}: {exc}")
                else:
                    logger.warning(f"Failed to generate clip file at {clip_file}")

            cap.release()

            # 6. Persist Evidence Model
            record = EvidenceModel(
                id=evidence_id,
                video_id=video_id,
                event_id=event_id,
                evidence_type=evidence_type,
                validation_status="VALID",
                timestamp_seconds=round(effective_ts, 2),
                source_video_name=original_filename,
                snapshot_path=snapshot_path,
                annotated_snapshot_path=annotated_snapshot_path,
                clip_path=clip_path,
                object_class=object_class,
                confidence=round(confidence, 4) if confidence is not None else None,
                bounding_box=bounding_box,
                start_time=round(start_time, 2) if start_time is not None else None,
                end_time=round(end_time, 2) if end_time is not None else None,
                duration_seconds=clip_duration,
                pre_seconds=pre_seconds,
                post_seconds=post_seconds,
                notes=notes,
            )
            db.add(record)
            db.commit()
            db.refresh(record)

            logger.info(f"Successfully generated evidence {evidence_id} for video {video_id}")
            return self._serialize_evidence(record, is_duplicate=False)

        finally:
            db.close()

    def reconcile_evidence_validation(self, video_id: str) -> None:
        """
        Synchronize EvidenceModel validation statuses with underlying EventModel
        and SecurityEventModel records for a video.

        Ensures that if a detection was rejected during validation (e.g. false Bus),
        any existing evidence records derived from it are transitioned to REJECTED
        so they are excluded from normal user-facing Evidence Vault and report results.
        Preserves genuine security intelligence evidence (e.g. POTENTIAL_THEFT).
        """
        db = SessionLocal()
        try:
            from ai.events.repository import ensure_video_events_validated
            ensure_video_events_validated(video_id, db)

            evidence_records = db.query(EvidenceModel).filter(EvidenceModel.video_id == video_id).all()
            if not evidence_records:
                return

            # Query all detection events for this video
            detection_events = db.query(EventModel).filter(EventModel.video_id == video_id).all()
            if not detection_events:
                return

            event_map = {e.id: e for e in detection_events}
            valid_classes = {
                e.object_class.lower()
                for e in detection_events
                if e.object_class and getattr(e, "validation_status", "VALID") == "VALID"
            }
            rejected_classes = {
                e.object_class.lower()
                for e in detection_events
                if e.object_class and getattr(e, "validation_status", "VALID") == "REJECTED"
            }

            # Security events mapping
            from database.models import SecurityEventModel
            sec_event_ids = {
                s.id for s in db.query(SecurityEventModel.id).filter(SecurityEventModel.video_id == video_id).all()
            }

            changed = False
            for ev in evidence_records:
                # 1. High-priority security evidence (e.g. POTENTIAL_THEFT) remains VALID
                if ev.event_id and ev.event_id in sec_event_ids:
                    if getattr(ev, "validation_status", "VALID") != "VALID":
                        ev.validation_status = "VALID"
                        changed = True
                    continue
                if ev.notes and "POTENTIAL_THEFT" in ev.notes:
                    if getattr(ev, "validation_status", "VALID") != "VALID":
                        ev.validation_status = "VALID"
                        changed = True
                    continue

                # 2. Linked to specific EventModel
                if ev.event_id and ev.event_id in event_map:
                    target_event = event_map[ev.event_id]
                    ev_status = getattr(target_event, "validation_status", "VALID")
                    if getattr(ev, "validation_status", "VALID") != ev_status:
                        ev.validation_status = ev_status
                        changed = True
                    continue

                # 3. Object-class based check
                cls = (ev.object_class or "").lower()
                if cls:
                    # If this class has NO valid events in the video and has rejected events, mark REJECTED
                    if cls not in valid_classes and cls in rejected_classes:
                        if getattr(ev, "validation_status", "VALID") != "REJECTED":
                            ev.validation_status = "REJECTED"
                            changed = True
                        continue

                    # Match by timestamp proximity (+- 1.5s)
                    nearby_events = [
                        e for e in detection_events
                        if (e.object_class or "").lower() == cls
                        and abs(e.timestamp_seconds - ev.timestamp_seconds) <= 1.5
                    ]
                    if nearby_events:
                        if any(getattr(e, "validation_status", "VALID") == "VALID" for e in nearby_events):
                            if getattr(ev, "validation_status", "VALID") != "VALID":
                                ev.validation_status = "VALID"
                                changed = True
                        elif all(getattr(e, "validation_status", "VALID") == "REJECTED" for e in nearby_events):
                            if getattr(ev, "validation_status", "VALID") != "REJECTED":
                                ev.validation_status = "REJECTED"
                                changed = True

            if changed:
                db.commit()
                logger.info(f"Reconciled evidence validation statuses for video {video_id}.")
        except Exception as exc:
            db.rollback()
            logger.warning(f"Failed to reconcile evidence validation for {video_id}: {exc}")
        finally:
            db.close()

    def get_video_evidence(self, video_id: str) -> List[Dict[str, Any]]:
        """Retrieve all valid evidence items for a given video sorted by created_at DESC."""
        self.reconcile_evidence_validation(video_id)
        db = SessionLocal()
        try:
            records = (
                db.query(EvidenceModel)
                .filter(EvidenceModel.video_id == video_id, EvidenceModel.validation_status == "VALID")
                .order_by(desc(EvidenceModel.created_at))
                .all()
            )
            return [self._serialize_evidence(r) for r in records]
        finally:
            db.close()

    def get_evidence_by_id(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve single evidence item by ID."""
        db = SessionLocal()
        try:
            record = db.query(EvidenceModel).filter(EvidenceModel.id == evidence_id).first()
            return self._serialize_evidence(record) if record else None
        finally:
            db.close()

    def _serialize_evidence(self, item: EvidenceModel, is_duplicate: bool = False) -> Dict[str, Any]:
        """Convert EvidenceModel into an investigator-friendly JSON response."""
        return {
            "evidence_id": item.id,
            "video_id": item.video_id,
            "event_id": item.event_id,
            "evidence_type": item.evidence_type,
            "validation_status": getattr(item, "validation_status", "VALID"),
            "timestamp": item.timestamp_seconds,
            "source_video_name": item.source_video_name,
            "object_class": item.object_class,
            "confidence": item.confidence,
            "bounding_box": item.bounding_box,
            "start_time": item.start_time,
            "end_time": item.end_time,
            "duration_seconds": item.duration_seconds,
            "has_snapshot": bool(item.snapshot_path and Path(item.snapshot_path).exists()),
            "has_annotated": bool(item.annotated_snapshot_path and Path(item.annotated_snapshot_path).exists()),
            "has_clip": bool(item.clip_path and Path(item.clip_path).exists()),
            "snapshot_url": f"/api/evidence/{item.id}/snapshot" if item.snapshot_path else None,
            "annotated_snapshot_url": f"/api/evidence/{item.id}/annotated" if item.annotated_snapshot_path else None,
            "clip_url": f"/api/evidence/{item.id}/clip" if item.clip_path else None,
            "created_at": item.created_at.isoformat() if item.created_at else None,
            "is_duplicate": is_duplicate,
        }

    @staticmethod
    def _draw_box_label(
        img: np.ndarray,
        text: str,
        x1: int,
        y1: int,
        bg_color: Tuple[int, int, int],
        text_color: Tuple[int, int, int],
        font: int,
        font_scale: float,
        thickness: int,
    ) -> None:
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        bg_y1 = max(0, y1 - th - 6)
        bg_y2 = y1
        cv2.rectangle(img, (x1, bg_y1), (x1 + tw + 6, bg_y2), bg_color, -1)
        cv2.putText(img, text, (x1 + 3, y1 - 3), font, font_scale, text_color, thickness, cv2.LINE_AA)

    @staticmethod
    def _draw_banner(
        img: np.ndarray,
        title: str,
        sub: str,
        font: int,
        font_scale: float,
        frame_w: int,
        frame_h: int,
    ) -> None:
        banner_h = max(34, int(frame_h * 0.12))
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (frame_w, banner_h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        cv2.line(img, (0, banner_h), (frame_w, banner_h), (0, 230, 115), 1)

        t_scale = max(0.36, font_scale * 0.95)
        s_scale = max(0.28, font_scale * 0.78)
        (sw, _), _ = cv2.getTextSize(sub, font, s_scale, 1)
        if sw > frame_w - 16:
            s_scale = max(0.22, s_scale * ((frame_w - 16) / max(1, sw)))
        cv2.putText(img, title, (8, int(banner_h * 0.44)), font, t_scale, (0, 230, 115), 1, cv2.LINE_AA)
        cv2.putText(img, sub, (8, int(banner_h * 0.84)), font, s_scale, (220, 220, 220), 1, cv2.LINE_AA)
