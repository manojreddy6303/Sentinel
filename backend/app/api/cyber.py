"""
Sentinel Cybersecurity API Router
backend/app/api/cyber.py

REST API endpoints for:
- Listing and filtering cybersecurity events
- Single event details
- Security asset inventory (leveraging physical camera sources)
- Cyber-physical event correlation
- Deterministic demo/replay telemetry seeding
- AI-assisted cybersecurity investigation inquiries
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from database.session import SessionLocal
from backend.app.services.cyber_service import (
    CyberSecurityService,
    VALID_EVENT_TYPES,
    VALID_SEVERITIES,
    VALID_STATUSES,
    VALID_PROVENANCE_TYPES,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cyber", tags=["Cybersecurity Extension"])
_cyber_svc = CyberSecurityService()


# ---------------------------------------------------------------------------
# Pydantic Request / Response Schemas
# ---------------------------------------------------------------------------

class CreateCyberEventRequest(BaseModel):
    event_type: str = Field(..., description="AUTHENTICATION_ANOMALY, CONFIGURATION_CHANGE, etc.")
    severity: str = Field("MEDIUM", description="LOW, MEDIUM, HIGH, CRITICAL")
    asset_label: str = Field(..., min_length=1, max_length=100, description="e.g. CAM-NORTH-01")
    asset_id: Optional[str] = Field(None, max_length=64)
    description: str = Field(..., min_length=1, max_length=2000)
    status: Optional[str] = Field("NEW", description="NEW, INVESTIGATING, CONFIRMED, DISMISSED, RESOLVED")
    source_ip: Optional[str] = Field(None, max_length=64)
    destination_port: Optional[int] = Field(None, ge=1, le=65535)
    structured_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    provenance: Optional[str] = Field("REPLAYED_TELEMETRY", description="LIVE_TELEMETRY, REPLAYED_TELEMETRY, etc.")
    timestamp: Optional[str] = Field(None, description="ISO format datetime string")
    timestamp_seconds: Optional[float] = Field(None, ge=0.0, description="Optional relative video seconds")
    video_id: Optional[str] = Field(None, max_length=64)
    incident_id: Optional[str] = Field(None, max_length=64)
    evidence_id: Optional[str] = Field(None, max_length=64)


class CorrelateRequest(BaseModel):
    asset_label_or_id: Optional[str] = Field(None, description="Camera source label or ID")
    time_window_seconds: float = Field(900.0, ge=10.0, le=7200.0, description="Correlation window in seconds")
    video_id: Optional[str] = Field(None, description="Optional video ID scope")


class CyberInvestigateRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Investigator inquiry")
    asset_label: Optional[str] = Field(None, description="Target security asset label")
    time_window_seconds: Optional[float] = Field(900.0, description="Analysis time window")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/events", summary="List cybersecurity events with multi-field filtering")
def list_cyber_events(
    asset_id: Optional[str] = Query(None, description="Filter by security asset ID"),
    asset_label: Optional[str] = Query(None, description="Filter by security asset label (e.g. CAM-NORTH-01)"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    severity: Optional[str] = Query(None, description="Filter by severity (LOW, MEDIUM, HIGH, CRITICAL)"),
    status: Optional[str] = Query(None, description="Filter by status (NEW, INVESTIGATING, CONFIRMED, etc.)"),
    provenance: Optional[str] = Query(None, description="Filter by provenance (REPLAYED_TELEMETRY, LIVE_TELEMETRY)"),
    search: Optional[str] = Query(None, description="Search in description, asset label, or IP"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """List cybersecurity events ordered by timestamp descending."""
    db = SessionLocal()
    try:
        return _cyber_svc.list_events(
            db=db,
            asset_id=asset_id,
            asset_label=asset_label,
            event_type=event_type,
            severity=severity,
            status=status,
            provenance=provenance,
            search=search,
            limit=limit,
            offset=offset,
        )
    finally:
        db.close()


@router.get("/events/{event_id}", summary="Get cybersecurity event details")
def get_cyber_event(event_id: str) -> Dict[str, Any]:
    """Retrieve full details of a specific cybersecurity event."""
    db = SessionLocal()
    try:
        ev = _cyber_svc.get_event(db, event_id)
        if not ev:
            raise HTTPException(status_code=404, detail=f"Cybersecurity event '{event_id}' not found.")
        return ev
    finally:
        db.close()


@router.post("/events", status_code=status.HTTP_201_CREATED, summary="Create a new cybersecurity event")
def create_cyber_event(payload: CreateCyberEventRequest) -> Dict[str, Any]:
    """Create a cybersecurity event record."""
    db = SessionLocal()
    try:
        data = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
        return _cyber_svc.create_event(db, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to create cyber event: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.get("/assets", summary="List security assets (CCTV cameras as cyber-physical endpoints)")
def list_security_assets() -> Dict[str, Any]:
    """
    List security assets with aggregated digital telemetry and physical surveillance stats.
    Reuses physical CameraSourceModel as the unified asset identity.
    """
    db = SessionLocal()
    try:
        return _cyber_svc.list_security_assets(db)
    finally:
        db.close()


@router.get("/assets/{asset_id}/events", summary="List cybersecurity events for a specific security asset")
def get_asset_events(
    asset_id: str,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """Retrieve all cybersecurity events for an asset (by asset ID or label)."""
    db = SessionLocal()
    try:
        return _cyber_svc.list_events(
            db=db,
            asset_id=asset_id,
            limit=limit,
            offset=offset,
        )
    finally:
        db.close()


@router.post("/correlate", summary="Correlate digital cybersecurity events with physical CCTV intelligence")
def correlate_cyber_physical(payload: CorrelateRequest) -> Dict[str, Any]:
    """
    Cross-correlate digital cyber telemetry with physical CCTV intelligence.
    Identifies temporal and asset proximity between digital anomalies and physical incidents.
    """
    db = SessionLocal()
    try:
        return _cyber_svc.correlate_cyber_physical(
            db=db,
            asset_label_or_id=payload.asset_label_or_id,
            time_window_seconds=payload.time_window_seconds,
            video_id=payload.video_id,
        )
    finally:
        db.close()


@router.post("/seed-demo", summary="Seed deterministic cybersecurity demo telemetry")
def seed_demo_telemetry() -> Dict[str, Any]:
    """
    Seed reproducible, deterministic cybersecurity demo events tied directly to
    Sentinel's canonical camera assets (CAM-NORTH-01, CAM-LOBBY-02, CAM-PERIM-03).
    Provenance is explicitly marked as REPLAYED_TELEMETRY.
    """
    db = SessionLocal()
    try:
        return _cyber_svc.seed_deterministic_demo_telemetry(db)
    finally:
        db.close()


@router.post("/investigate", summary="AI-assisted cybersecurity investigation inquiry")
def investigate_cyber(payload: CyberInvestigateRequest) -> Dict[str, Any]:
    """
    Execute an evidence-grounded agentic investigation inquiry across
    cybersecurity telemetry and physical surveillance context.
    Follows Sentinel's multi-step controlled tool investigation workflow.
    """
    try:
        from ai.investigation.orchestrator import InvestigationOrchestrator
        orchestrator = InvestigationOrchestrator()
        return orchestrator.process_cyber_investigation(
            user_query=payload.query,
            asset_label=payload.asset_label,
            time_window_seconds=payload.time_window_seconds or 900.0,
        )
    except Exception as e:
        logger.error("Cyber investigation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Cyber investigation error: {str(e)}")
