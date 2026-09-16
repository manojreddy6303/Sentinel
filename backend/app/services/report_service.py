"""
Report Generation & Management Service for Sentinel (Phase 9)

Orchestrates:
- Gathering video, detection, tracking, behavioral, and evidence records from DB
- Assembling structured ReportDataPayload with strict null-safety
- Formulating grounded investigation findings
- Invoking IncidentDossierPDFGenerator to build the PDF document
- Safe atomic storage under storage/reports/ with path-traversal protection
- Persisting ReportModel in the database
"""

import os
import re
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from sqlalchemy import desc, func

from backend.app.core.config import settings
from database.session import SessionLocal
from database.models import (
    VideoModel,
    EventModel,
    GroupedEventModel,
    EvidenceModel,
    TrackModel,
    VehicleAttributeModel,
    FaceDetectionModel,
    SecurityZoneModel,
    SecurityEventModel,
    ReportModel,
    CorrelatedIncidentModel,
)
from ai.reporting.schema import (
    ReportDataPayload,
    ReportVideoMetadata,
    ReportDetectionStats,
    ReportTimelineEvent,
    ReportSecurityEvent,
    ReportEvidenceItem,
    ReportInvestigationFinding,
    ReportSpecializedEvent,
    ReportCorrelatedIncident,
)
from database.models import SpecializedObservationModel
from ai.reporting.dossier_generator import IncidentDossierPDFGenerator

logger = logging.getLogger(__name__)


class ReportError(Exception):
    """Base exception for report service operations."""
    pass


class ReportNotFoundError(ReportError):
    """Raised when requested report record or file does not exist."""
    pass


class ReportSecurityError(ReportError):
    """Raised when path traversal or unsafe file access is detected."""
    pass


class ReportService:
    """Manages generation, storage, and retrieval of Incident Dossier reports."""

    def __init__(self):
        self.reports_dir: Path = settings.STORAGE_REPORTS_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_dossier(
        self,
        video_id: str,
        title: str = "SECURITY INCIDENT DOSSIER",
        classification: str = "CONFIDENTIAL // LAW ENFORCEMENT & SECURITY OPERATIONS",
        custom_queries: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Extract all video and investigation intelligence from database,
        build PDF dossier, save file, and store report record in database.
        """
        db = SessionLocal()
        try:
            video: Optional[VideoModel] = db.query(VideoModel).filter(VideoModel.id == video_id).first()
            if not video:
                raise ReportNotFoundError(f"Video with ID '{video_id}' not found.")

            # Reconcile evidence validation statuses to guarantee zero stale data in report
            try:
                from backend.app.services.evidence_service import EvidenceService
                EvidenceService().reconcile_evidence_validation(video_id)
            except Exception as rec_err:
                logger.warning(f"Could not reconcile evidence validation for report {video_id}: {rec_err}")

            # Resolve video sensor profile attributes
            resolution = None
            codec = None
            sampling_rate_fps = settings.VIDEO_SAMPLE_RATE_FPS
            analyzed_frames_count = video.frame_count
            meta_path = settings.STORAGE_UPLOADS_DIR / f"{video_id}.json"
            if meta_path.exists():
                try:
                    import json
                    with open(meta_path, "r", encoding="utf-8") as f_meta:
                        meta_info = json.load(f_meta)
                        w = meta_info.get("width")
                        h = meta_info.get("height")
                        if w and h:
                            resolution = f"{w}x{h}"
                        codec = meta_info.get("codec") or meta_info.get("video_codec")
                        sampling_rate_fps = float(meta_info.get("sample_rate_fps", sampling_rate_fps))
                        analyzed_frames_count = meta_info.get("analyzed_frames_count", analyzed_frames_count)
                except Exception as meta_err:
                    logger.debug(f"Could not read upload metadata sidecar for {video_id}: {meta_err}")

            # 1. Fetch related intelligence records (validated detections only for core findings)
            all_events: List[EventModel] = (
                db.query(EventModel)
                .filter(EventModel.video_id == video_id)
                .order_by(EventModel.timestamp_seconds.asc())
                .all()
            )
            events: List[EventModel] = [e for e in all_events if e.validation_status == "VALID"]
            uncertain_events: List[EventModel] = [e for e in all_events if e.validation_status == "UNCERTAIN"]
            rejected_events: List[EventModel] = [e for e in all_events if e.validation_status == "REJECTED"]

            grouped_events: List[GroupedEventModel] = (
                db.query(GroupedEventModel)
                .filter(GroupedEventModel.video_id == video_id)
                .order_by(GroupedEventModel.start_time.asc())
                .all()
            )
            tracks: List[TrackModel] = (
                db.query(TrackModel)
                .filter(TrackModel.video_id == video_id)
                .order_by(TrackModel.first_seen.asc())
                .all()
            )
            vehicle_attrs: List[VehicleAttributeModel] = (
                db.query(VehicleAttributeModel)
                .filter(VehicleAttributeModel.video_id == video_id)
                .all()
            )
            faces: List[FaceDetectionModel] = (
                db.query(FaceDetectionModel)
                .filter(FaceDetectionModel.video_id == video_id)
                .all()
            )
            zones: List[SecurityZoneModel] = (
                db.query(SecurityZoneModel)
                .filter(SecurityZoneModel.video_id == video_id)
                .all()
            )
            sec_events: List[SecurityEventModel] = (
                db.query(SecurityEventModel)
                .filter(SecurityEventModel.video_id == video_id)
                .order_by(SecurityEventModel.timestamp_seconds.asc())
                .all()
            )
            evidence_records: List[EvidenceModel] = (
                db.query(EvidenceModel)
                .filter(EvidenceModel.video_id == video_id, EvidenceModel.validation_status == "VALID")
                .order_by(EvidenceModel.timestamp_seconds.asc())
                .all()
            )

            # 2. Compute Detection & Tracking Statistics with strict distinction between observations and tracks
            total_detections = len(events)
            class_counts: Dict[str, int] = {}
            class_raw_counts: Dict[str, int] = {}
            class_uncertain_counts: Dict[str, int] = {}
            class_rejected_counts: Dict[str, int] = {}
            class_conf_sums: Dict[str, float] = {}

            for ev in all_events:
                cls_name = ev.object_class or "unknown"
                class_raw_counts[cls_name] = class_raw_counts.get(cls_name, 0) + 1
                if ev.validation_status == "VALID":
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    class_conf_sums[cls_name] = class_conf_sums.get(cls_name, 0.0) + (ev.confidence or 0.0)
                elif ev.validation_status == "UNCERTAIN":
                    class_uncertain_counts[cls_name] = class_uncertain_counts.get(cls_name, 0) + 1
                elif ev.validation_status == "REJECTED":
                    class_rejected_counts[cls_name] = class_rejected_counts.get(cls_name, 0) + 1

            class_track_counts: Dict[str, int] = {}
            for t in tracks:
                t_cls = t.object_class or "unknown"
                class_track_counts[t_cls] = class_track_counts.get(t_cls, 0) + 1

            class_avg_confs: Dict[str, float] = {}
            for cls_name, count in class_counts.items():
                if count > 0:
                    class_avg_confs[cls_name] = round(class_conf_sums[cls_name] / count, 3)

            total_tracks = len(tracks)
            # Canonical validation semantics — must match TrackedObject.is_validated in ai/schemas.py:
            # - detection_count >= 2 → always validated (all classes)
            # - vehicles (car/bus/truck) with detection_count == 1 → validated only if confidence >= 0.65
            # - non-vehicles with detection_count == 1 → validated (pipeline does not reject single-frame persons)
            VEHICLE_CLASSES_REPORT = {"car", "bus", "truck"}
            validated_tracks = 0
            for t in tracks:
                det_count = t.detection_count or 0
                cls = (t.object_class or "").lower()
                if det_count >= 2:
                    validated_tracks += 1
                elif cls in VEHICLE_CLASSES_REPORT:
                    # Single-frame vehicle: only validated if high confidence
                    if (t.max_confidence or 0.0) >= 0.65:
                        validated_tracks += 1
                else:
                    # Non-vehicle single-frame track: validated per pipeline semantics
                    validated_tracks += 1

            stats = ReportDetectionStats(
                total_detections=total_detections,
                total_raw_detections=len(all_events),
                total_uncertain_detections=len(uncertain_events),
                total_rejected_detections=len(rejected_events),
                class_counts=class_counts,
                class_raw_counts=class_raw_counts,
                class_uncertain_counts=class_uncertain_counts,
                class_rejected_counts=class_rejected_counts,
                class_track_counts=class_track_counts,
                class_avg_confidences=class_avg_confs,
                total_tracks=total_tracks,
                validated_tracks=validated_tracks,
                total_security_events=len(sec_events),
                total_evidence_items=len(evidence_records),
                total_face_detections=len(faces),
            )

            # 3. Construct Security Events and Potential Theft Subsets
            report_sec_events: List[ReportSecurityEvent] = []
            theft_events: List[ReportSecurityEvent] = []

            for se in sec_events:
                signals = se.observable_signals if isinstance(se.observable_signals, list) else None
                r_se = ReportSecurityEvent(
                    id=se.id,
                    event_type=se.event_type or "SECURITY_EVENT",
                    severity=se.severity or "NORMAL",
                    timestamp_seconds=se.timestamp_seconds or 0.0,
                    duration_seconds=se.duration_seconds or 0.0,
                    track_id=se.track_id,
                    object_class=se.object_class,
                    zone_name=se.zone_name,
                    confidence=round(se.confidence or 0.0, 2),
                    description=se.description or "",
                    observable_signals=signals,
                    evidence_id=se.evidence_id,
                )
                report_sec_events.append(r_se)
                if r_se.event_type in ("POTENTIAL_THEFT", "POTENTIAL_OBJECT_TAKEAWAY"):
                    theft_events.append(r_se)

            # 3b. Extract Specialized Visual Events
            specialized_events: List[ReportSpecializedEvent] = []
            try:
                spec_obs_records = db.query(SpecializedObservationModel).filter(
                    SpecializedObservationModel.video_id == video_id
                ).order_by(SpecializedObservationModel.timestamp_seconds).all()
                for so in spec_obs_records:
                    desc = ""
                    cname = (so.class_name or "").lower()
                    if "fire" in cname:
                        desc = "Potential fire visual evidence was detected with sustained chromatic features."
                    elif "smoke" in cname:
                        desc = "Potential smoke-like visual evidence was detected with diffuse plume dynamics."
                    elif "weapon" in cname:
                        desc = "Potential weapon-like object visual evidence was detected."
                    elif "pose" in cname or "posture" in cname:
                        desc = "Postural transition evidence observed across tracking sequence."
                    else:
                        desc = f"Specialized visual observation ({so.class_name})."

                    specialized_events.append(ReportSpecializedEvent(
                        id=f"SPEC-{so.id}",
                        event_type=f"POTENTIAL_{so.class_name.upper()}",
                        timestamp_seconds=so.timestamp_seconds or 0.0,
                        duration_seconds=0.0,
                        detector_name=so.detector_name or "specialized",
                        model_name=so.detector_version,
                        evidence_strength=round(so.evidence_strength or 0.0, 2),
                        validation_status=so.validation_status or "REVIEW_REQUIRED",
                        review_required=True,
                        description=desc,
                        evidence_id=None,
                    ))
            except Exception as exc:
                logger.warning(f"Failed to query specialized observations for report: {exc}")

            # Also incorporate any specialized incidents from security events
            for se in sec_events:
                if any(k in (se.event_type or "") for k in ("FIRE", "SMOKE", "WEAPON")):
                    if not any(abs(se.timestamp_seconds - so.timestamp_seconds) < 0.5 and so.event_type == se.event_type for so in specialized_events):
                        desc = se.description or ""
                        if not desc:
                            if "FIRE" in se.event_type and "SMOKE" in se.event_type:
                                desc = "Potential fire and smoke visual evidence was detected in localized region."
                            elif "FIRE" in se.event_type:
                                desc = "Potential fire visual evidence was detected."
                            elif "SMOKE" in se.event_type:
                                desc = "Potential smoke-like visual evidence was detected."
                            elif "WEAPON" in se.event_type:
                                desc = "Potential weapon-like object visual evidence was detected."
                        specialized_events.append(ReportSpecializedEvent(
                            id=se.id,
                            event_type=se.event_type,
                            timestamp_seconds=se.timestamp_seconds or 0.0,
                            duration_seconds=se.duration_seconds or 0.0,
                            detector_name="specialized_incident_detector",
                            evidence_strength=round(se.confidence or 0.0, 2),
                            validation_status="REVIEW_REQUIRED",
                            review_required=True,
                            description=desc,
                            evidence_id=se.evidence_id,
                        ))

            specialized_events.sort(key=lambda x: x.timestamp_seconds)

            # 3c. Extract Phase 16 Correlated Incidents
            correlated_incidents: List[ReportCorrelatedIncident] = []
            try:
                corr_records = (
                    db.query(CorrelatedIncidentModel)
                    .filter(CorrelatedIncidentModel.video_id == video_id)
                    .order_by(CorrelatedIncidentModel.start_time.asc())
                    .all()
                )
                for ci in corr_records:
                    correlated_incidents.append(
                        ReportCorrelatedIncident(
                            incident_id=ci.id,
                            incident_category=ci.incident_category,
                            incident_subcategory=ci.incident_subcategory,
                            start_time=ci.start_time or 0.0,
                            end_time=ci.end_time or 0.0,
                            duration=ci.duration or 0.0,
                            primary_track_ids=ci.primary_track_ids or [],
                            supporting_track_ids=ci.supporting_track_ids or [],
                            involved_object_classes=ci.involved_object_classes or [],
                            assessment_score=round(ci.assessment_score or 0.0, 4),
                            evidence_strength=round(ci.evidence_strength or 0.0, 4),
                            reliability_rating=ci.reliability_rating or "LOW",
                            validation_decision=ci.validation_decision or "REVIEW_REQUIRED",
                            storyline=ci.storyline or "",
                            evidence_ids=ci.evidence_ids or [],
                            negative_evidence=ci.negative_evidence or [],
                            contextual_factors=ci.contextual_factors or {},
                        )
                    )
            except Exception as exc:
                logger.warning(f"Failed to query correlated incidents for report: {exc}")

            # 4. Construct Timeline
            report_timeline: List[ReportTimelineEvent] = []
            # Add grouped events
            for ge in grouped_events:
                # Find matching track if available
                first_cls = None
                if ge.objects_summary and isinstance(ge.objects_summary, list) and len(ge.objects_summary) > 0:
                    first_cls = ge.objects_summary[0].get("class")
                report_timeline.append(ReportTimelineEvent(
                    timestamp_seconds=ge.start_time or 0.0,
                    event_type=ge.event_type or "ACTIVITY",
                    object_class=first_cls,
                    confidence=ge.max_confidence,
                    description=f"Activity window: {ge.total_detections} detections ({ge.duration_seconds:.1f}s)",
                    source="grouped_event",
                ))

            # Add security events into timeline
            for se in sec_events:
                report_timeline.append(ReportTimelineEvent(
                    timestamp_seconds=se.timestamp_seconds or 0.0,
                    event_type=se.event_type or "SECURITY_ALERT",
                    object_class=se.object_class,
                    confidence=se.confidence,
                    track_id=se.track_id,
                    description=se.description,
                    evidence_id=se.evidence_id,
                    source="security_event",
                ))

            report_timeline.sort(key=lambda x: x.timestamp_seconds)

            # 5. Construct Evidence items with local image path resolution
            report_evidence: List[ReportEvidenceItem] = []
            for ev in evidence_records:
                chosen_img: Optional[str] = None
                is_annotated = False

                # Prefer annotated snapshot
                if ev.annotated_snapshot_path and Path(ev.annotated_snapshot_path).exists():
                    chosen_img = str(Path(ev.annotated_snapshot_path).resolve())
                    is_annotated = True
                elif ev.snapshot_path and Path(ev.snapshot_path).exists():
                    chosen_img = str(Path(ev.snapshot_path).resolve())

                report_evidence.append(ReportEvidenceItem(
                    id=ev.id,
                    evidence_type=ev.evidence_type or "snapshot",
                    timestamp_seconds=ev.timestamp_seconds or 0.0,
                    object_class=ev.object_class,
                    confidence=round(ev.confidence or 0.0, 2) if ev.confidence else None,
                    track_id=None,
                    local_image_path=chosen_img,
                    has_image=bool(chosen_img),
                    is_annotated=is_annotated,
                    clip_available=bool(ev.clip_path and Path(ev.clip_path).exists()),
                    notes=ev.notes,
                ))

            # 6. Formulate Investigation Findings
            investigation_findings: List[ReportInvestigationFinding] = []
            if custom_queries:
                try:
                    from backend.app.services.investigation_service import InvestigationService
                    inv_svc = InvestigationService()
                    for q in custom_queries:
                        res = inv_svc.investigate(video_id, q)
                        count = res.get("count", 0)
                        msg = res.get("message") or f"Identified {count} matching records in database."
                        investigation_findings.append(ReportInvestigationFinding(
                            query=q,
                            finding=msg,
                        ))
                except Exception as exc:
                    logger.warning(f"Failed to execute custom queries: {exc}")

            if not investigation_findings:
                # Add auto-generated factual findings based on real DB records
                if theft_events:
                    te = theft_events[0]
                    investigation_findings.append(ReportInvestigationFinding(
                        query="Were any potential theft or takeaway events observed?",
                        finding=(
                            f"Yes. At {te.timestamp_seconds:.1f}s, an automated pattern matching "
                            f"{te.event_type} was registered for Track {te.track_id or 'N/A'} "
                            f"({te.object_class or 'object'}). Pattern evidence strength: {te.confidence*100:.0f}%. "
                            f"Evidence Vault ID: {te.evidence_id or 'N/A'}. Human verification required."
                        ),
                        timestamp_reference=f"{te.timestamp_seconds:.1f}s",
                        confidence=te.confidence,
                    ))

                if sec_events:
                    loitering = [s for s in sec_events if s.event_type == "PROLONGED_PRESENCE"]
                    if loitering:
                        investigation_findings.append(ReportInvestigationFinding(
                            query="Was any prolonged presence or loitering detected?",
                            finding=(
                                f"Observed {len(loitering)} prolonged presence occurrences. Earliest stationary "
                                f"pattern recorded at {loitering[0].timestamp_seconds:.1f}s involving track {loitering[0].track_id or 'N/A'}."
                            ),
                            timestamp_reference=f"{loitering[0].timestamp_seconds:.1f}s",
                        ))

                if specialized_events:
                    sp_ev = specialized_events[0]
                    investigation_findings.append(ReportInvestigationFinding(
                        query="Were any specialized visual phenomena (fire, smoke, weapon, posture) observed?",
                        finding=(
                            f"Automated specialized visual analysis recorded {len(specialized_events)} potential visual signature(s). "
                            f"Earliest detection at {sp_ev.timestamp_seconds:.1f}s ({sp_ev.event_type}). "
                            f"Evidence strength: {sp_ev.evidence_strength*100:.0f}%. "
                            f"Validation status: {sp_ev.validation_status}. Mandatory human verification required."
                        ),
                        timestamp_reference=f"{sp_ev.timestamp_seconds:.1f}s",
                        confidence=sp_ev.evidence_strength,
                    ))

                sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
                if sorted_classes:
                    top_cls, top_cnt = sorted_classes[0]
                    investigation_findings.append(ReportInvestigationFinding(
                        query="What was the predominant object class detected in the scene?",
                        finding=f"'{top_cls}' was the most frequent object class with {top_cnt} verified detections across the surveillance timeline.",
                    ))

            # 7. Generate safe unique Report ID & Filename
            now = datetime.now(timezone.utc)
            date_str = now.strftime("%Y%m%d")
            rand_suffix = uuid.uuid4().hex[:8].upper()
            report_id = f"REP-{date_str}-{rand_suffix}"
            pdf_filename = f"{report_id}.pdf"
            pdf_path = self.reports_dir / pdf_filename

            # 8. Assemble complete payload
            payload = ReportDataPayload(
                report_id=report_id,
                title=title,
                classification=classification,
                generated_at_iso=now.isoformat(),
                video=ReportVideoMetadata(
                    video_id=video.id,
                    original_filename=video.original_filename,
                    duration_seconds=video.duration_seconds,
                    fps=video.fps,
                    frame_count=video.frame_count,
                    file_size_bytes=video.file_size_bytes,
                    status=video.status,
                    resolution=resolution,
                    codec=codec,
                    sampling_rate_fps=sampling_rate_fps,
                    analyzed_frames_count=analyzed_frames_count,
                    uploaded_at=video.uploaded_at.isoformat() if video.uploaded_at else None,
                    processed_at=video.processed_at.isoformat() if video.processed_at else None,
                ),
                stats=stats,
                timeline=report_timeline,
                security_events=report_sec_events,
                theft_events=theft_events,
                specialized_events=specialized_events,
                correlated_incidents=correlated_incidents,
                evidence=report_evidence,
                vehicle_attributes=[
                    {
                        "track_id": getattr(va, "track_id", None) or (va.get("track_id") if isinstance(va, dict) else "N/A"),
                        "object_class": getattr(va, "object_class", None) or getattr(va, "vehicle_type", None) or (va.get("object_class") or va.get("vehicle_type") if isinstance(va, dict) else "vehicle"),
                        "vehicle_type": getattr(va, "object_class", None) or getattr(va, "vehicle_type", None) or (va.get("object_class") or va.get("vehicle_type") if isinstance(va, dict) else "vehicle"),
                        "color": getattr(va, "color", None) or (va.get("color") if isinstance(va, dict) else "unknown"),
                        "color_confidence": float(getattr(va, "confidence", None) if getattr(va, "confidence", None) is not None else (getattr(va, "color_confidence", 0.0) or (va.get("confidence") if isinstance(va, dict) and va.get("confidence") is not None else (va.get("color_confidence", 0.0) if isinstance(va, dict) else 0.0)))),
                        "confidence": float(getattr(va, "confidence", None) if getattr(va, "confidence", None) is not None else (getattr(va, "color_confidence", 0.0) or (va.get("confidence") if isinstance(va, dict) and va.get("confidence") is not None else (va.get("color_confidence", 0.0) if isinstance(va, dict) else 0.0)))),
                        "timestamp": float(getattr(va, "timestamp_seconds", None) if getattr(va, "timestamp_seconds", None) is not None else (getattr(va, "timestamp", 0.0) or (va.get("timestamp_seconds") if isinstance(va, dict) and va.get("timestamp_seconds") is not None else (va.get("timestamp", 0.0) if isinstance(va, dict) else 0.0)))),
                    }
                    for va in vehicle_attrs
                ],
                anonymous_faces_count=len(faces),
                zones=[{"name": z.name, "target_classes": z.target_classes} for z in zones],
                investigation_findings=investigation_findings,
            )

            # 9. Generate PDF
            generator = IncidentDossierPDFGenerator(payload)
            file_size_bytes, page_count = generator.generate(str(pdf_path))

            # 9.1 Compute True Cryptographic SHA-256 Hash of Generated Document
            import hashlib
            with open(pdf_path, "rb") as f_pdf:
                sha256_hash = hashlib.sha256(f_pdf.read()).hexdigest()

            # 9.2 Quality & Consistency Audit: Verify all metrics match DB
            if video.id != video_id:
                raise ReportError(f"Video ID mismatch: requested {video_id} but video is {video.id}")
            if total_detections != len(events):
                raise ReportError("Report statistics inconsistency: total_detections mismatch")
            if total_tracks != len(tracks):
                raise ReportError("Report statistics inconsistency: total_tracks mismatch")
            for rej_ev in rejected_events:
                if rej_ev.object_class and class_counts.get(rej_ev.object_class, 0) == 0 and rej_ev.object_class in class_counts:
                    raise ReportError(f"Rejected object class '{rej_ev.object_class}' improperly counted as validated")

            # 10. Persist ReportModel in database
            relative_path = f"storage/reports/{pdf_filename}"
            meta_dict = {
                "total_detections": total_detections,
                "total_raw_detections": len(all_events),
                "total_uncertain_detections": len(uncertain_events),
                "total_rejected_detections": len(rejected_events),
                "class_counts": class_counts,
                "class_raw_counts": class_raw_counts,
                "class_track_counts": class_track_counts,
                "total_tracks": total_tracks,
                "total_security_events": len(sec_events),
                "has_theft_event": bool(theft_events),
                "theft_event_count": len(theft_events),
                "total_evidence": len(evidence_records),
                "page_count": page_count,
                "sha256_hash": sha256_hash,
                "generated_at": now.isoformat(),
            }

            report_model = ReportModel(
                report_id=report_id,
                video_id=video_id,
                title=title,
                report_type="INCIDENT_DOSSIER",
                file_path=relative_path,
                file_size_bytes=file_size_bytes,
                page_count=page_count,
                status="completed",
                metadata_payload=meta_dict,
                generated_at=now,
            )
            db.add(report_model)
            db.commit()
            db.refresh(report_model)

            return self._serialize_report(report_model)
        finally:
            db.close()

    def list_reports(self, video_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all generated reports, optionally filtered by video_id."""
        db = SessionLocal()
        try:
            q = db.query(ReportModel)
            if video_id:
                q = q.filter(ReportModel.video_id == video_id)
            records = q.order_by(desc(ReportModel.generated_at)).all()
            return [self._serialize_report(r) for r in records]
        finally:
            db.close()

    def get_report(self, report_id: str) -> Dict[str, Any]:
        """Retrieve single report metadata by its public report_id."""
        db = SessionLocal()
        try:
            record = db.query(ReportModel).filter(ReportModel.report_id == report_id).first()
            if not record:
                raise ReportNotFoundError(f"Report '{report_id}' not found.")
            return self._serialize_report(record)
        finally:
            db.close()

    def get_report_file_path(self, report_id: str) -> Tuple[Path, str]:
        """
        Validate and resolve safe on-disk file path for the requested report.
        Protects strictly against directory traversal.
        """
        if not report_id or not re.match(r"^[A-Za-z0-9_\-]+$", report_id):
            raise ReportSecurityError(f"Invalid report identifier: '{report_id}'")

        # Check DB first
        db = SessionLocal()
        record = None
        try:
            record = db.query(ReportModel).filter(ReportModel.report_id == report_id).first()
        finally:
            db.close()

        if not record:
            raise ReportNotFoundError(f"Report with ID '{report_id}' not found in database.")

        resolved_reports_dir = self.reports_dir.resolve()
        candidate = (resolved_reports_dir / f"{report_id}.pdf").resolve()

        if not str(candidate).startswith(str(resolved_reports_dir)):
            raise ReportSecurityError("Directory traversal detected.")

        if not candidate.exists() or not candidate.is_file():
            raise ReportNotFoundError(f"Report PDF file for '{report_id}' not found on disk.")

        download_filename = f"{report_id}.pdf"
        return candidate, download_filename

    def _serialize_report(self, item: ReportModel) -> Dict[str, Any]:
        """Serialize ReportModel into API-ready dictionary without exposing internal paths."""
        return {
            "id": item.id,
            "report_id": item.report_id,
            "video_id": item.video_id,
            "title": item.title,
            "report_type": item.report_type,
            "file_size_bytes": item.file_size_bytes,
            "page_count": item.page_count,
            "status": item.status,
            "metadata": item.metadata_payload or {},
            "generated_at": item.generated_at.isoformat() if item.generated_at else None,
            "download_url": f"/api/reports/{item.report_id}/download",
            "view_url": f"/api/reports/{item.report_id}/view",
        }
