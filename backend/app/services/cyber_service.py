"""
Sentinel Cybersecurity Extension Service
backend/app/services/cyber_service.py

Provides business logic for:
1. Cybersecurity event ingestion, listing, filtering, and detail retrieval
2. Security asset inventory (leveraging physical Camera Sources as digital/physical assets)
3. Cyber-Physical Correlation Engine correlating IT telemetry with CCTV physical incidents
4. Deterministic demo/replay telemetry generator with explicit provenance marking
5. Strict adherence to Sentinel safety invariants: zero identity claims, human review mandate
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import func, or_

from database.models import (
    CyberSecurityEventModel,
    CameraSourceModel,
    VideoModel,
    CorrelatedIncidentModel,
    SecurityEventModel,
    EvidenceModel,
)

logger = logging.getLogger(__name__)

# Valid event types
VALID_EVENT_TYPES = {
    "AUTHENTICATION_ANOMALY",
    "UNAUTHORIZED_ACCESS_ATTEMPT",
    "CONFIGURATION_CHANGE",
    "CONNECTION_ANOMALY",
    "STREAM_INTEGRITY_ANOMALY",
    "DIGITAL_INTEGRITY_ANOMALY",
    "SECURITY_POLICY_VIOLATION",
}

# Valid severities
VALID_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

# Valid statuses
VALID_STATUSES = {"NEW", "ACTIVE", "INVESTIGATING", "CONFIRMED", "DISMISSED", "RESOLVED"}

# Valid provenance types
VALID_PROVENANCE_TYPES = {
    "LIVE_TELEMETRY",
    "REPLAYED_TELEMETRY",
    "MANUAL_ANALYST_EVENT",
    "SYSTEM_GENERATED",
}


class CyberSecurityService:
    """Service orchestrating cybersecurity telemetry and cyber-physical correlation."""

    # -------------------------------------------------------------------------
    # 1. Event Operations
    # -------------------------------------------------------------------------

    def list_events(
        self,
        db: Session,
        asset_id: Optional[str] = None,
        asset_label: Optional[str] = None,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        provenance: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Query cybersecurity events with rich multi-field filtering."""
        query = db.query(CyberSecurityEventModel)

        if asset_id:
            query = query.filter(CyberSecurityEventModel.asset_id == asset_id)
        if asset_label and asset_label != "all":
            query = query.filter(func.lower(CyberSecurityEventModel.asset_label) == asset_label.strip().lower())
        if event_type and event_type != "all":
            query = query.filter(CyberSecurityEventModel.event_type == event_type.strip().upper())
        if severity and severity != "all":
            query = query.filter(CyberSecurityEventModel.severity == severity.strip().upper())
        if status and status != "all":
            query = query.filter(CyberSecurityEventModel.status == status.strip().upper())
        if provenance and provenance != "all":
            query = query.filter(CyberSecurityEventModel.provenance == provenance.strip().upper())
        if start_time:
            query = query.filter(CyberSecurityEventModel.timestamp >= start_time)
        if end_time:
            query = query.filter(CyberSecurityEventModel.timestamp <= end_time)
        if search:
            s = f"%{search.strip()}%"
            query = query.filter(
                or_(
                    CyberSecurityEventModel.description.ilike(s),
                    CyberSecurityEventModel.asset_label.ilike(s),
                    CyberSecurityEventModel.event_type.ilike(s),
                    CyberSecurityEventModel.source_ip.ilike(s),
                )
            )

        total = query.count()
        events = (
            query.order_by(CyberSecurityEventModel.timestamp.desc(), CyberSecurityEventModel.created_at.desc())
            .offset(offset)
            .limit(min(limit, 500))
            .all()
        )

        return {
            "total": total,
            "events": [self.to_dict(e) for e in events],
            "limit": limit,
            "offset": offset,
        }

    def get_event(self, db: Session, event_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full details for a single cybersecurity event."""
        ev = db.query(CyberSecurityEventModel).filter(CyberSecurityEventModel.id == event_id).first()
        if not ev:
            return None
        return self.to_dict(ev, include_relations=True, db=db)

    def create_event(self, db: Session, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new cybersecurity event record."""
        ev_type = str(data.get("event_type", "")).strip().upper()
        if ev_type not in VALID_EVENT_TYPES:
            raise ValueError(f"Invalid event_type '{ev_type}'. Must be one of {sorted(VALID_EVENT_TYPES)}")

        sev = str(data.get("severity", "MEDIUM")).strip().upper()
        if sev not in VALID_SEVERITIES:
            raise ValueError(f"Invalid severity '{sev}'. Must be one of {sorted(VALID_SEVERITIES)}")

        prov = str(data.get("provenance", "REPLAYED_TELEMETRY")).strip().upper()
        if prov not in VALID_PROVENANCE_TYPES:
            prov = "REPLAYED_TELEMETRY"

        stat = str(data.get("status", "NEW")).strip().upper()
        if stat not in VALID_STATUSES:
            stat = "NEW"

        # Resolve asset
        asset_label = str(data.get("asset_label", "")).strip()
        asset_id = data.get("asset_id")

        if not asset_label and not asset_id:
            raise ValueError("Either asset_label or asset_id is required.")

        cam = None
        if asset_id:
            cam = db.query(CameraSourceModel).filter(CameraSourceModel.id == asset_id).first()
        elif asset_label:
            cam = db.query(CameraSourceModel).filter(
                func.lower(CameraSourceModel.camera_label) == asset_label.lower()
            ).first()

        if cam:
            asset_id = cam.id
            asset_label = cam.camera_label
        else:
            asset_label = asset_label or asset_id or "UNKNOWN_ASSET"
            asset_id = None

        # Resolve timestamp
        ts = data.get("timestamp")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                ts = datetime.now(timezone.utc)
        elif not isinstance(ts, datetime):
            ts = datetime.now(timezone.utc)

        # Optional link to physical video
        video_id = data.get("video_id") or (cam.video_id if cam else None)
        incident_id = data.get("incident_id")
        evidence_id = data.get("evidence_id")

        event = CyberSecurityEventModel(
            event_type=ev_type,
            severity=sev,
            timestamp=ts,
            timestamp_seconds=data.get("timestamp_seconds"),
            asset_id=asset_id,
            asset_label=asset_label,
            status=stat,
            description=str(data.get("description", "")).strip(),
            source_ip=data.get("source_ip"),
            destination_port=data.get("destination_port"),
            structured_metadata=data.get("structured_metadata") or {},
            provenance=prov,
            video_id=video_id,
            incident_id=incident_id,
            evidence_id=evidence_id,
            created_at=datetime.now(timezone.utc),
        )
        db.add(event)
        db.commit()
        db.refresh(event)
        return self.to_dict(event)

    # -------------------------------------------------------------------------
    # 2. Security Asset Operations (Reusing CameraSourceModel)
    # -------------------------------------------------------------------------

    def list_security_assets(self, db: Session) -> Dict[str, Any]:
        """
        List all security assets (CCTV cameras acting as cyber-physical endpoints).
        Aggregates digital threat telemetry with physical surveillance metadata.
        """
        cameras = db.query(CameraSourceModel).order_by(CameraSourceModel.camera_label.asc()).all()
        assets = []

        for cam in cameras:
            # Query cyber events count for this asset
            cyber_count = db.query(CyberSecurityEventModel).filter(
                (CyberSecurityEventModel.asset_id == cam.id) |
                (func.lower(CyberSecurityEventModel.asset_label) == cam.camera_label.lower())
            ).count()

            high_sev_count = db.query(CyberSecurityEventModel).filter(
                ((CyberSecurityEventModel.asset_id == cam.id) |
                 (func.lower(CyberSecurityEventModel.asset_label) == cam.camera_label.lower())),
                CyberSecurityEventModel.severity.in_(["HIGH", "CRITICAL"])
            ).count()

            latest_event = db.query(CyberSecurityEventModel).filter(
                (CyberSecurityEventModel.asset_id == cam.id) |
                (func.lower(CyberSecurityEventModel.asset_label) == cam.camera_label.lower())
            ).order_by(CyberSecurityEventModel.timestamp.desc()).first()

            # Physical video & incidents count
            vid = db.query(VideoModel).filter(VideoModel.id == cam.video_id).first() if cam.video_id else None
            inc_count = 0
            ev_count = 0
            if cam.video_id:
                inc_count = db.query(CorrelatedIncidentModel).filter(CorrelatedIncidentModel.video_id == cam.video_id).count()
                ev_count = db.query(EvidenceModel).filter(EvidenceModel.video_id == cam.video_id).count()

            assets.append({
                "asset_id": cam.id,
                "asset_label": cam.camera_label,
                "position_hint": cam.position_hint or "Main Facility",
                "field_of_view_hint": cam.field_of_view_hint or "Standard Perimeter",
                "status": getattr(cam, "status", "ACTIVE") or "ACTIVE",
                "video_id": cam.video_id,
                "video_filename": vid.original_filename if vid else None,
                "total_cyber_events": cyber_count,
                "high_severity_cyber_events": high_sev_count,
                "physical_incident_count": inc_count,
                "physical_evidence_count": ev_count,
                "latest_cyber_event": self.to_dict(latest_event) if latest_event else None,
                "has_active_threat": high_sev_count > 0,
            })

        return {
            "total_assets": len(assets),
            "assets": assets,
        }

    # -------------------------------------------------------------------------
    # 3. Cyber + Physical Correlation Engine
    # -------------------------------------------------------------------------

    def correlate_cyber_physical(
        self,
        db: Session,
        asset_label_or_id: Optional[str] = None,
        time_window_seconds: float = 900.0,
        video_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Cross-correlate digital cyber telemetry with physical CCTV intelligence.

        Correlation Criteria:
        1. Asset Identity Alignment: Cyber event asset_label matches physical camera_label.
        2. Video Correlation: Cyber event associated with same video_id.
        3. Temporal Proximity: Cyber event occurs within delta-t of physical incident
           (preceding, concurrent, or immediately succeeding).
        4. Observable Evidence Binding: Extracts actual CCTV snapshots/clips associated
           with physical events occurring in the correlated window.

        Safety Directive:
        NEVER claim attacker identity, guilt, or unsupported causation. Reports observable
        correlations and flags 'REVIEW_REQUIRED'.
        """
        # Resolve target asset if provided
        cam = None
        if asset_label_or_id:
            cam = db.query(CameraSourceModel).filter(
                or_(
                    CameraSourceModel.id == asset_label_or_id,
                    func.lower(CameraSourceModel.camera_label) == asset_label_or_id.strip().lower()
                )
            ).first()

        # Query cyber events
        cyber_query = db.query(CyberSecurityEventModel)
        if cam:
            cyber_query = cyber_query.filter(
                or_(
                    CyberSecurityEventModel.asset_id == cam.id,
                    func.lower(CyberSecurityEventModel.asset_label) == cam.camera_label.lower()
                )
            )
        elif asset_label_or_id:
            cyber_query = cyber_query.filter(
                func.lower(CyberSecurityEventModel.asset_label) == asset_label_or_id.strip().lower()
            )
        if video_id:
            cyber_query = cyber_query.filter(CyberSecurityEventModel.video_id == video_id)

        cyber_events = cyber_query.order_by(CyberSecurityEventModel.timestamp.asc()).all()

        correlations: List[Dict[str, Any]] = []

        for c_ev in cyber_events:
            # Determine target video
            target_vid = c_ev.video_id or (cam.video_id if cam else None)
            if not target_vid:
                # Try finding camera by label
                c_cam = db.query(CameraSourceModel).filter(
                    func.lower(CameraSourceModel.camera_label) == c_ev.asset_label.lower()
                ).first()
                if c_cam:
                    target_vid = c_cam.video_id

            if not target_vid:
                continue

            # Fetch physical incidents for this video
            phys_incidents = (
                db.query(CorrelatedIncidentModel)
                .filter(CorrelatedIncidentModel.video_id == target_vid)
                .order_by(CorrelatedIncidentModel.start_time.asc())
                .all()
            )

            # Also fetch evidence
            evidence_records = (
                db.query(EvidenceModel)
                .filter(EvidenceModel.video_id == target_vid)
                .all()
            )

            cyber_ts_sec = c_ev.timestamp_seconds

            for inc in phys_incidents:
                # If cyber event has relative timestamp_seconds, compute temporal delta
                delta_sec = None
                is_temporally_correlated = True

                if cyber_ts_sec is not None:
                    # delta: physical start - cyber event
                    delta_sec = inc.start_time - cyber_ts_sec
                    # Consider correlated if within time_window_seconds (e.g. -60s to +time_window_seconds)
                    if delta_sec < -60.0 or delta_sec > time_window_seconds:
                        is_temporally_correlated = False

                if is_temporally_correlated:
                    # Match supporting evidence
                    matched_evidence = []
                    for ev in evidence_records:
                        if abs(ev.timestamp_seconds - inc.start_time) <= 15.0 or (
                            inc.end_time and inc.start_time <= ev.timestamp_seconds <= inc.end_time
                        ):
                            matched_evidence.append({
                                "evidence_id": ev.id,
                                "timestamp": ev.timestamp_seconds,
                                "evidence_type": ev.evidence_type,
                                "object_class": ev.object_class,
                                "confidence": ev.confidence,
                                "snapshot_url": f"/api/evidence/{ev.id}/snapshot" if ev.snapshot_path else None,
                                "annotated_snapshot_url": f"/api/evidence/{ev.id}/annotated-snapshot" if ev.annotated_snapshot_path else None,
                                "clip_url": f"/api/evidence/{ev.id}/clip" if ev.clip_path else None,
                            })

                    # Formulate structured objective narrative
                    delta_desc = (
                        f"{abs(delta_sec):.1f}s prior to" if delta_sec is not None and delta_sec > 0
                        else f"{abs(delta_sec):.1f}s after" if delta_sec is not None and delta_sec < 0
                        else "concurrent with"
                    )

                    narrative = (
                        f"Cybersecurity event '{c_ev.event_type}' ({c_ev.severity}) occurred {delta_desc} "
                        f"observable physical incident '{inc.incident_category.replace('_', ' ').title()}' "
                        f"on security asset '{c_ev.asset_label}'. "
                        f"Physical surveillance intelligence registered a verified pattern with assessment score {round(inc.assessment_score * 100)}%. "
                        f"Observable physical evidence is preserved in Sentinel vault. "
                        f"Human analyst verification is required; causation is not established."
                    )

                    correlations.append({
                        "correlation_id": f"CORR-CYBER-PHYS-{c_ev.id[:8]}-{inc.id[:8]}",
                        "security_asset": c_ev.asset_label,
                        "asset_id": c_ev.asset_id,
                        "cyber_event": {
                            "id": c_ev.id,
                            "event_type": c_ev.event_type,
                            "severity": c_ev.severity,
                            "timestamp": c_ev.timestamp.isoformat() if c_ev.timestamp else None,
                            "timestamp_seconds": c_ev.timestamp_seconds,
                            "description": c_ev.description,
                            "source_ip": c_ev.source_ip,
                            "provenance": c_ev.provenance,
                        },
                        "physical_incident": {
                            "id": inc.id,
                            "category": inc.incident_category,
                            "start_time": inc.start_time,
                            "end_time": inc.end_time,
                            "assessment_score": inc.assessment_score,
                            "validation_decision": inc.validation_decision,
                            "storyline": inc.storyline,
                        },
                        "temporal_delta_seconds": delta_sec,
                        "correlation_hypothesis": "CYBER_PRECEDED_PHYSICAL" if (delta_sec and delta_sec > 0) else "CONCURRENT_CYBER_PHYSICAL",
                        "confidence_score": min(0.65, inc.assessment_score),  # Review ceiling enforced
                        "status": "REVIEW_REQUIRED",
                        "human_verification_required": True,
                        "supporting_evidence": matched_evidence,
                        "correlation_narrative": narrative,
                    })

        return {
            "total_correlations": len(correlations),
            "target_asset": asset_label_or_id or "ALL_ASSETS",
            "correlations": correlations,
            "review_ceiling": 0.65,
            "safety_policy": "Strict observational correlation only. No identity, intent, or culpability claims.",
        }

    # -------------------------------------------------------------------------
    # 4. Deterministic Demo & Replay Telemetry
    # -------------------------------------------------------------------------

    def seed_deterministic_demo_telemetry(self, db: Session) -> Dict[str, Any]:
        """
        Seeds deterministic cybersecurity demo events tied directly to Sentinel's
        canonical CCTV camera assets (CAM-NORTH-01, CAM-LOBBY-02, CAM-PERIM-03).
        Guarantees:
        - 100% deterministic & reproducible
        - Provenance is explicitly REPLAYED_TELEMETRY
        - Does NOT alter video, detection, or benchmark counts
        """
        # Fetch canonical cameras
        cam_north = db.query(CameraSourceModel).filter(CameraSourceModel.camera_label == "CAM-NORTH-01").first()
        cam_lobby = db.query(CameraSourceModel).filter(CameraSourceModel.camera_label == "CAM-LOBBY-02").first()
        cam_perim = db.query(CameraSourceModel).filter(CameraSourceModel.camera_label == "CAM-PERIM-03").first()

        # Base reference time: aligned with burglary video timeline context (e.g., 22:14 UTC)
        base_time = datetime(2026, 9, 25, 22, 14, 0, tzinfo=timezone.utc)

        demo_events = [
            # Event 1: CAM-NORTH-01 Configuration Change (precedes physical theft at 163s)
            {
                "id": "cyber-demo-north-01",
                "event_type": "CONFIGURATION_CHANGE",
                "severity": "HIGH",
                "timestamp": base_time + timedelta(seconds=2),
                "timestamp_seconds": 140.0,
                "asset_id": cam_north.id if cam_north else None,
                "asset_label": "CAM-NORTH-01",
                "status": "NEW",
                "description": "Camera stream encoding profile altered and motion masking zones updated via HTTP administrative interface.",
                "source_ip": "192.168.10.88",
                "destination_port": 80,
                "structured_metadata": {
                    "interface": "WebAdmin",
                    "action": "MODIFY_STREAM_PROFILE",
                    "previous_bitrate_kbps": 4096,
                    "new_bitrate_kbps": 512,
                    "motion_mask_cleared": True,
                },
                "provenance": "REPLAYED_TELEMETRY",
                "video_id": cam_north.video_id if cam_north else None,
            },
            # Event 2: CAM-NORTH-01 Authentication Anomaly
            {
                "id": "cyber-demo-north-02",
                "event_type": "AUTHENTICATION_ANOMALY",
                "severity": "MEDIUM",
                "timestamp": base_time + timedelta(seconds=77),
                "timestamp_seconds": 145.0,
                "asset_id": cam_north.id if cam_north else None,
                "asset_label": "CAM-NORTH-01",
                "status": "INVESTIGATING",
                "description": "3 consecutive failed admin authentications followed by session token issuance outside standard operational shift.",
                "source_ip": "192.168.10.88",
                "destination_port": 443,
                "structured_metadata": {
                    "failed_attempts": 3,
                    "target_account": "admin",
                    "auth_method": "Digest",
                    "off_shift_access": True,
                },
                "provenance": "REPLAYED_TELEMETRY",
                "video_id": cam_north.video_id if cam_north else None,
            },
            # Event 3: CAM-NORTH-01 Security Policy Violation
            {
                "id": "cyber-demo-north-03",
                "event_type": "SECURITY_POLICY_VIOLATION",
                "severity": "HIGH",
                "timestamp": base_time + timedelta(seconds=141),
                "timestamp_seconds": 152.0,
                "asset_id": cam_north.id if cam_north else None,
                "asset_label": "CAM-NORTH-01",
                "status": "CONFIRMED",
                "description": "Unencrypted RTSP video feed requested directly from non-allowlisted VLAN segment.",
                "source_ip": "10.0.4.15",
                "destination_port": 554,
                "structured_metadata": {
                    "protocol": "RTSP",
                    "segment": "GUEST_WIFI",
                    "expected_segment": "SECURITY_MGMT_VLAN",
                    "policy_rule_id": "POL-CCTV-04",
                },
                "provenance": "REPLAYED_TELEMETRY",
                "video_id": cam_north.video_id if cam_north else None,
            },
            # Event 4: CAM-LOBBY-02 Connection Anomaly
            {
                "id": "cyber-demo-lobby-01",
                "event_type": "CONNECTION_ANOMALY",
                "severity": "LOW",
                "timestamp": base_time + timedelta(minutes=5, seconds=10),
                "timestamp_seconds": 30.0,
                "asset_id": cam_lobby.id if cam_lobby else None,
                "asset_label": "CAM-LOBBY-02",
                "status": "NEW",
                "description": "Camera network heartbeat missed 4 consecutive polling intervals before TCP renegotiation.",
                "source_ip": "192.168.10.12",
                "destination_port": 8080,
                "structured_metadata": {
                    "missed_heartbeats": 4,
                    "disconnect_duration_ms": 4200,
                    "reconnect_successful": True,
                },
                "provenance": "REPLAYED_TELEMETRY",
                "video_id": cam_lobby.video_id if cam_lobby else None,
            },
            # Event 5: CAM-PERIM-03 Stream Integrity Anomaly
            {
                "id": "cyber-demo-perim-01",
                "event_type": "STREAM_INTEGRITY_ANOMALY",
                "severity": "MEDIUM",
                "timestamp": base_time + timedelta(minutes=8, seconds=45),
                "timestamp_seconds": 12.0,
                "asset_id": cam_perim.id if cam_perim else None,
                "asset_label": "CAM-PERIM-03",
                "status": "NEW",
                "description": "High RTP packet loss rate (14.2%) and sequence discontinuity detected during 4K surveillance stream capture.",
                "source_ip": "192.168.10.15",
                "destination_port": 5004,
                "structured_metadata": {
                    "packet_loss_percentage": 14.2,
                    "dropped_frames": 9,
                    "resolution": "3840x2160",
                },
                "provenance": "REPLAYED_TELEMETRY",
                "video_id": cam_perim.video_id if cam_perim else None,
            },
        ]

        created_count = 0
        updated_count = 0

        for edata in demo_events:
            existing = db.query(CyberSecurityEventModel).filter(CyberSecurityEventModel.id == edata["id"]).first()
            if existing:
                for k, v in edata.items():
                    setattr(existing, k, v)
                updated_count += 1
            else:
                new_ev = CyberSecurityEventModel(**edata)
                db.add(new_ev)
                created_count += 1

        db.commit()
        logger.info("Seeded deterministic cyber demo events: created=%d, updated=%d", created_count, updated_count)

        return {
            "status": "success",
            "created_count": created_count,
            "updated_count": updated_count,
            "total_demo_events": len(demo_events),
            "provenance": "REPLAYED_TELEMETRY",
            "message": "Deterministic cybersecurity telemetry seeded successfully.",
        }

    # -------------------------------------------------------------------------
    # Serializer Helper
    # -------------------------------------------------------------------------

    def to_dict(
        self,
        event: Optional[CyberSecurityEventModel],
        include_relations: bool = False,
        db: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """Convert a CyberSecurityEventModel instance to a clean API dictionary."""
        if not event:
            return {}

        res = {
            "id": event.id,
            "event_id": event.id,
            "event_type": event.event_type,
            "severity": event.severity,
            "timestamp": event.timestamp.isoformat() if event.timestamp else None,
            "timestamp_seconds": event.timestamp_seconds,
            "asset_id": event.asset_id,
            "asset_label": event.asset_label,
            "status": event.status,
            "description": event.description,
            "source_ip": event.source_ip,
            "destination_port": event.destination_port,
            "structured_metadata": event.structured_metadata or {},
            "provenance": event.provenance,
            "video_id": event.video_id,
            "incident_id": event.incident_id,
            "evidence_id": event.evidence_id,
            "created_at": event.created_at.isoformat() if event.created_at else None,
        }

        if include_relations and db:
            if event.video_id:
                vid = db.query(VideoModel).filter(VideoModel.id == event.video_id).first()
                if vid:
                    res["video_filename"] = vid.original_filename
                    res["video_status"] = vid.status

            if event.asset_id:
                cam = db.query(CameraSourceModel).filter(CameraSourceModel.id == event.asset_id).first()
                if cam:
                    res["asset_location"] = cam.position_hint
                    res["asset_coverage"] = cam.field_of_view_hint

        return res


# ---------------------------------------------------------------------------
# Module-Level Convenience Functions
# ---------------------------------------------------------------------------

_default_service = CyberSecurityService()


def create_cyber_event(
    db: Session,
    asset_id: Optional[str] = None,
    asset_label: Optional[str] = None,
    event_type: str = "AUTHENTICATION_ANOMALY",
    severity: str = "MEDIUM",
    status: str = "NEW",
    description: str = "",
    source_ip: Optional[str] = None,
    destination_port: Optional[int] = None,
    structured_metadata: Optional[Dict[str, Any]] = None,
    provenance: str = "REPLAYED_TELEMETRY",
    timestamp: Optional[datetime] = None,
    video_offset_seconds: Optional[float] = None,
    video_id: Optional[str] = None,
    incident_id: Optional[str] = None,
    evidence_id: Optional[str] = None,
) -> CyberSecurityEventModel:
    """Create and return a CyberSecurityEventModel."""
    ev_type = str(event_type).strip().upper()
    if ev_type not in VALID_EVENT_TYPES:
        raise ValueError(f"Invalid event_type '{ev_type}'. Must be one of {sorted(VALID_EVENT_TYPES)}")
    sev = str(severity).strip().upper()
    if sev not in VALID_SEVERITIES:
        raise ValueError(f"Invalid severity '{sev}'. Must be one of {sorted(VALID_SEVERITIES)}")
    prov = str(provenance).strip().upper()
    if prov not in VALID_PROVENANCE_TYPES:
        raise ValueError(f"Invalid provenance '{prov}'. Must be one of {sorted(VALID_PROVENANCE_TYPES)}")
    stat = str(status).strip().upper()
    if stat not in VALID_STATUSES:
        stat = "NEW"

    cam = None
    resolved_label = asset_label or asset_id or "UNKNOWN_ASSET"
    resolved_asset_id = asset_id

    if asset_id:
        cam = db.query(CameraSourceModel).filter(CameraSourceModel.id == asset_id).first()
        if not cam:
            cam = db.query(CameraSourceModel).filter(
                func.lower(CameraSourceModel.camera_label) == str(asset_id).lower()
            ).first()
    elif asset_label:
        cam = db.query(CameraSourceModel).filter(
            func.lower(CameraSourceModel.camera_label) == str(asset_label).lower()
        ).first()

    if cam:
        resolved_asset_id = cam.id
        resolved_label = cam.camera_label
    else:
        resolved_label = asset_label or asset_id or "UNKNOWN_ASSET"
        resolved_asset_id = None

    event = CyberSecurityEventModel(
        event_type=ev_type,
        severity=sev,
        timestamp=timestamp or datetime.now(timezone.utc),
        timestamp_seconds=video_offset_seconds,
        asset_id=resolved_asset_id,
        asset_label=resolved_label,
        status=stat,
        description=description,
        source_ip=source_ip,
        destination_port=destination_port,
        structured_metadata=structured_metadata or {},
        provenance=prov,
        video_id=video_id or (cam.video_id if cam else None),
        incident_id=incident_id,
        evidence_id=evidence_id,
        created_at=datetime.now(timezone.utc),
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


def get_cyber_event(db: Session, event_id: str) -> Optional[CyberSecurityEventModel]:
    return db.query(CyberSecurityEventModel).filter(CyberSecurityEventModel.id == event_id).first()


def list_cyber_events(
    db: Session,
    asset_id: Optional[str] = None,
    asset_label: Optional[str] = None,
    event_type: Optional[str] = None,
    severity: Optional[str] = None,
    status: Optional[str] = None,
    provenance: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[CyberSecurityEventModel]:
    query = db.query(CyberSecurityEventModel)
    if asset_id:
        query = query.filter(
            or_(
                CyberSecurityEventModel.asset_id == asset_id,
                CyberSecurityEventModel.asset_label == asset_id,
            )
        )
    if asset_label and asset_label != "all":
        query = query.filter(func.lower(CyberSecurityEventModel.asset_label) == asset_label.strip().lower())
    if event_type and event_type != "all":
        query = query.filter(CyberSecurityEventModel.event_type == event_type.strip().upper())
    if severity and severity != "all":
        query = query.filter(CyberSecurityEventModel.severity == severity.strip().upper())
    if status and status != "all":
        query = query.filter(CyberSecurityEventModel.status == status.strip().upper())
    if provenance and provenance != "all":
        query = query.filter(CyberSecurityEventModel.provenance == provenance.strip().upper())
    if start_time:
        query = query.filter(CyberSecurityEventModel.timestamp >= start_time)
    if end_time:
        query = query.filter(CyberSecurityEventModel.timestamp <= end_time)
    if search:
        s = f"%{search.strip()}%"
        query = query.filter(
            or_(
                CyberSecurityEventModel.description.ilike(s),
                CyberSecurityEventModel.asset_label.ilike(s),
                CyberSecurityEventModel.event_type.ilike(s),
            )
        )
    return query.order_by(CyberSecurityEventModel.timestamp.desc()).offset(offset).limit(limit).all()


def list_security_assets(db: Session) -> List[Dict[str, Any]]:
    res = _default_service.get_security_assets(db)
    return res.get("assets", [])


def get_security_asset_events(db: Session, asset_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    res = _default_service.get_asset_events(db, asset_id, limit=limit)
    return res.get("events", [])


def correlate_cyber_physical(
    db: Session,
    asset_id: Optional[str] = None,
    asset_label_or_id: Optional[str] = None,
    time_window_seconds: float = 900.0,
    video_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    target = asset_id or asset_label_or_id
    res = _default_service.correlate_cyber_physical(
        db=db,
        asset_label_or_id=target,
        time_window_seconds=time_window_seconds,
        video_id=video_id,
    )
    return res.get("correlations", [])


def seed_demo_telemetry(db: Session) -> Dict[str, Any]:
    return _default_service.seed_deterministic_demo(db)

