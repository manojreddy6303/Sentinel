"""
backend/app/api/cases.py — Phase 19

REST API router for Forensic Case Management & Investigation Workspace.
Enforces:
  - Case existence & ownership validation
  - Cross-case isolation & boundary checks
  - Input validation & pagination limits
  - Provenance preservation across all case entities
  - Privacy safeguards: anonymous track IDs only, zero biometric/identity claims
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

try:
    from database.session import SessionLocal
    from app.services.case_service import (
        CaseService,
        CaseNotFoundError,
        CaseEntityNotFoundError,
        CaseIsolationError,
        CaseValidationError,
    )
except ImportError:
    from database.session import SessionLocal  # type: ignore
    from backend.app.services.case_service import (  # type: ignore
        CaseService,
        CaseNotFoundError,
        CaseEntityNotFoundError,
        CaseIsolationError,
        CaseValidationError,
    )

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cases", tags=["Case Management"])
_svc = CaseService()

_ID_RE = re.compile(r"^[a-zA-Z0-9\-]{1,64}$")


def _validate_id(value: str, label: str) -> str:
    if not _ID_RE.match(value):
        raise HTTPException(status_code=400, detail=f"Invalid {label}: '{value}'")
    return value


# ---------------------------------------------------------------------------
# Pydantic Request Schemas
# ---------------------------------------------------------------------------

class CreateCaseRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    priority: Optional[str] = Field("MEDIUM", pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    assigned_investigator: Optional[str] = Field(None, max_length=100)
    tags: Optional[List[str]] = Field(None, max_length=20)
    summary: Optional[str] = Field(None, max_length=5000)
    initial_video_ids: Optional[List[str]] = Field(None, max_length=50)
    initial_camera_ids: Optional[List[str]] = Field(None, max_length=50)


class UpdateCaseRequest(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    status: Optional[str] = Field(None, pattern="^(OPEN|INVESTIGATING|REVIEW|CLOSED)$")
    priority: Optional[str] = Field(None, pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    assigned_investigator: Optional[str] = Field(None, max_length=100)
    tags: Optional[List[str]] = Field(None, max_length=20)
    summary: Optional[str] = Field(None, max_length=5000)


class LinkVideoRequest(BaseModel):
    video_id: str = Field(..., min_length=1, max_length=64)
    notes: Optional[str] = Field(None, max_length=500)


class LinkCameraRequest(BaseModel):
    camera_id: str = Field(..., min_length=1, max_length=64)
    clock_offset_seconds: Optional[float] = Field(0.0)
    notes: Optional[str] = Field(None, max_length=500)


class LinkIncidentRequest(BaseModel):
    incident_id: str = Field(..., min_length=1, max_length=64)
    incident_type: Optional[str] = Field("CORRELATED", pattern="^(CORRELATED|SECURITY_EVENT)$")
    notes: Optional[str] = Field(None, max_length=500)


class LinkEvidenceRequest(BaseModel):
    evidence_id: str = Field(..., min_length=1, max_length=64)
    notes: Optional[str] = Field(None, max_length=500)


class CreateBookmarkRequest(BaseModel):
    video_id: str = Field(..., min_length=1, max_length=64)
    timestamp_seconds: float = Field(..., ge=0.0)
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    camera_id: Optional[str] = Field(None, max_length=64)
    linked_incident_id: Optional[str] = Field(None, max_length=64)
    linked_track_id: Optional[str] = Field(None, max_length=64)
    linked_evidence_id: Optional[str] = Field(None, max_length=64)
    author: Optional[str] = Field("Investigator", max_length=100)


class UpdateBookmarkRequest(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    timestamp_seconds: Optional[float] = Field(None, ge=0.0)


class CreateNoteRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=4000)
    author: Optional[str] = Field("Investigator", max_length=100)
    associated_type: Optional[str] = Field("CASE", pattern="^(CASE|INCIDENT|EVIDENCE|TRACK|TIMESTAMP|CAMERA)$")
    associated_id: Optional[str] = Field(None, max_length=64)
    timestamp_seconds: Optional[float] = Field(None, ge=0.0)


class UpdateNoteRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=4000)


class CreateAnnotationRequest(BaseModel):
    video_id: str = Field(..., min_length=1, max_length=64)
    timestamp_seconds: float = Field(..., ge=0.0)
    end_timestamp_seconds: Optional[float] = Field(None, ge=0.0)
    annotation_type: Optional[str] = Field("REGION", pattern="^(POINT|REGION|TIMESTAMP_MARKER|TEXT_NOTE|INCIDENT_MARKER)$")
    data: Dict[str, Any] = Field(default_factory=dict)
    author: Optional[str] = Field("Investigator", max_length=100)
    camera_id: Optional[str] = Field(None, max_length=64)


# ---------------------------------------------------------------------------
# Helper Serializers
# ---------------------------------------------------------------------------

def _case_to_dict(case: Any, include_details: bool = False, db: Optional[Session] = None) -> Dict[str, Any]:
    d = {
        "id": case.id,
        "case_number": case.case_number,
        "title": case.title,
        "description": case.description,
        "status": case.status,
        "priority": case.priority,
        "assigned_investigator": case.assigned_investigator,
        "tags": case.tags or [],
        "summary": case.summary,
        "created_at": case.created_at.isoformat() if case.created_at else None,
        "updated_at": case.updated_at.isoformat() if case.updated_at else None,
        "counts": {
            "videos": len(case.videos) if case.videos else 0,
            "cameras": len(case.cameras) if case.cameras else 0,
            "incidents": len(case.incidents) if case.incidents else 0,
            "evidence": len(getattr(case, "evidence_links", [])) if hasattr(case, "evidence_links") and case.evidence_links else 0,
            "bookmarks": len(case.bookmarks) if case.bookmarks else 0,
            "notes": len(case.notes) if case.notes else 0,
            "annotations": len(case.annotations) if case.annotations else 0,
        },
    }
    if include_details and db:
        d["linked_videos"] = _svc.get_linked_videos(db, case.id)
        d["linked_cameras"] = _svc.get_linked_cameras(db, case.id)
        d["linked_incidents"] = _svc.get_case_incidents(db, case.id)
        d["linked_evidence"] = _svc.get_linked_evidence(db, case.id)
    return d


def _bookmark_to_dict(b: Any) -> Dict[str, Any]:
    return {
        "id": b.id,
        "case_id": b.case_id,
        "video_id": b.video_id,
        "camera_id": b.camera_id,
        "timestamp_seconds": b.timestamp_seconds,
        "title": b.title,
        "description": b.description,
        "linked_incident_id": b.linked_incident_id,
        "linked_track_id": b.linked_track_id,
        "linked_evidence_id": b.linked_evidence_id,
        "author": b.author,
        "created_at": b.created_at.isoformat() if b.created_at else None,
        "updated_at": b.updated_at.isoformat() if b.updated_at else None,
    }


def _note_to_dict(n: Any) -> Dict[str, Any]:
    return {
        "id": n.id,
        "case_id": n.case_id,
        "content": n.content,
        "author": n.author,
        "associated_type": n.associated_type,
        "associated_id": n.associated_id,
        "timestamp_seconds": n.timestamp_seconds,
        "note_classification": n.note_classification,
        "created_at": n.created_at.isoformat() if n.created_at else None,
        "updated_at": n.updated_at.isoformat() if n.updated_at else None,
    }


def _annotation_to_dict(a: Any) -> Dict[str, Any]:
    return {
        "id": a.id,
        "case_id": a.case_id,
        "video_id": a.video_id,
        "camera_id": a.camera_id,
        "timestamp_seconds": a.timestamp_seconds,
        "end_timestamp_seconds": a.end_timestamp_seconds,
        "annotation_type": a.annotation_type,
        "data": a.data,
        "author": a.author,
        "created_at": a.created_at.isoformat() if a.created_at else None,
        "updated_at": a.updated_at.isoformat() if a.updated_at else None,
    }


# ---------------------------------------------------------------------------
# Case CRUD
# ---------------------------------------------------------------------------

@router.post("", status_code=status.HTTP_201_CREATED, summary="Create forensic case")
def create_case(body: CreateCaseRequest) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        case = _svc.create_case(
            db,
            title=body.title,
            description=body.description,
            priority=body.priority or "MEDIUM",
            assigned_investigator=body.assigned_investigator,
            tags=body.tags,
            summary=body.summary,
            initial_video_ids=body.initial_video_ids,
            initial_camera_ids=body.initial_camera_ids,
        )
        return _case_to_dict(case, include_details=True, db=db)
    except CaseValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("create_case error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        db.close()


@router.get("", summary="List all forensic cases")
def list_cases(
    status: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        total = _svc.count_cases(
            db,
            status=status,
            priority=priority,
            tag=tag,
            search=search,
        )
        cases = _svc.list_cases(
            db,
            status=status,
            priority=priority,
            tag=tag,
            search=search,
            limit=limit,
            offset=offset,
        )
        return {
            "total": total,
            "cases": [_case_to_dict(c) for c in cases],
            "limit": limit,
            "offset": offset,
        }
    finally:
        db.close()


@router.get("/{case_id}", summary="Get case detail")
def get_case(case_id: str) -> Dict[str, Any]:
    _validate_id(case_id, "case_id")
    db = SessionLocal()
    try:
        case = _svc.get_case(db, case_id)
        return _case_to_dict(case, include_details=True, db=db)
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    finally:
        db.close()


@router.patch("/{case_id}", summary="Update case attributes")
def update_case(case_id: str, body: UpdateCaseRequest) -> Dict[str, Any]:
    _validate_id(case_id, "case_id")
    db = SessionLocal()
    try:
        case = _svc.update_case(
            db,
            case_id=case_id,
            title=body.title,
            description=body.description,
            status=body.status,
            priority=body.priority,
            assigned_investigator=body.assigned_investigator,
            tags=body.tags,
            summary=body.summary,
        )
        return _case_to_dict(case, include_details=True, db=db)
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    except CaseValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        db.close()


@router.delete("/{case_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete case")
def delete_case(case_id: str):
    _validate_id(case_id, "case_id")
    db = SessionLocal()
    try:
        _svc.delete_case(db, case_id)
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Entity Linking (Videos, Cameras, Incidents)
# ---------------------------------------------------------------------------

@router.post("/{case_id}/videos", status_code=status.HTTP_201_CREATED, summary="Link video to case")
def link_video(case_id: str, body: LinkVideoRequest) -> Dict[str, Any]:
    _validate_id(case_id, "case_id")
    _validate_id(body.video_id, "video_id")
    db = SessionLocal()
    try:
        link = _svc.link_video(db, case_id, body.video_id, notes=body.notes)
        return {
            "link_id": link.id,
            "case_id": link.case_id,
            "video_id": link.video_id,
            "notes": link.notes,
            "added_at": link.added_at.isoformat(),
        }
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    except CaseEntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    finally:
        db.close()


@router.get("/{case_id}/videos", summary="List videos linked to case")
def get_linked_videos(case_id: str) -> Dict[str, Any]:
    _validate_id(case_id, "case_id")
    db = SessionLocal()
    try:
        videos = _svc.get_linked_videos(db, case_id)
        return {"case_id": case_id, "total": len(videos), "videos": videos}
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    finally:
        db.close()


@router.delete("/{case_id}/videos/{video_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Unlink video from case")
def unlink_video(case_id: str, video_id: str):
    _validate_id(case_id, "case_id")
    _validate_id(video_id, "video_id")
    db = SessionLocal()
    try:
        _svc.unlink_video(db, case_id, video_id)
    except (CaseNotFoundError, CaseEntityNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    finally:
        db.close()


@router.post("/{case_id}/cameras", status_code=status.HTTP_201_CREATED, summary="Link camera to case")
def link_camera(case_id: str, body: LinkCameraRequest) -> Dict[str, Any]:
    _validate_id(case_id, "case_id")
    _validate_id(body.camera_id, "camera_id")
    db = SessionLocal()
    try:
        link = _svc.link_camera(
            db,
            case_id,
            body.camera_id,
            clock_offset_seconds=body.clock_offset_seconds or 0.0,
            notes=body.notes,
        )
        return {
            "link_id": link.id,
            "case_id": link.case_id,
            "camera_id": link.camera_id,
            "clock_offset_seconds": link.clock_offset_seconds,
            "notes": link.notes,
            "added_at": link.added_at.isoformat(),
        }
    except (CaseNotFoundError, CaseEntityNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    finally:
        db.close()


@router.get("/{case_id}/cameras", summary="List cameras linked to case")
def get_linked_cameras(case_id: str) -> Dict[str, Any]:
    _validate_id(case_id, "case_id")
    db = SessionLocal()
    try:
        cameras = _svc.get_linked_cameras(db, case_id)
        return {"case_id": case_id, "total": len(cameras), "cameras": cameras}
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    finally:
        db.close()


@router.delete("/{case_id}/cameras/{camera_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Unlink camera from case")
def unlink_camera(case_id: str, camera_id: str):
    _validate_id(case_id, "case_id")
    _validate_id(camera_id, "camera_id")
    db = SessionLocal()
    try:
        _svc.unlink_camera(db, case_id, camera_id)
    except (CaseNotFoundError, CaseEntityNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    finally:
        db.close()


@router.post("/{case_id}/incidents", status_code=status.HTTP_201_CREATED, summary="Link incident to case")
def link_incident(case_id: str, body: LinkIncidentRequest) -> Dict[str, Any]:
    _validate_id(case_id, "case_id")
    _validate_id(body.incident_id, "incident_id")
    db = SessionLocal()
    try:
        link = _svc.link_incident(
            db,
            case_id,
            body.incident_id,
            incident_type=body.incident_type or "CORRELATED",
            notes=body.notes,
        )
        return {
            "link_id": link.id,
            "case_id": link.case_id,
            "incident_id": link.incident_id,
            "incident_type": link.incident_type,
            "notes": link.notes,
            "added_at": link.added_at.isoformat(),
        }
    except (CaseNotFoundError, CaseEntityNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    finally:
        db.close()


@router.get("/{case_id}/incidents", summary="List incidents in case")
def get_case_incidents(case_id: str) -> Dict[str, Any]:
    _validate_id(case_id, "case_id")
    db = SessionLocal()
    try:
        incidents = _svc.get_case_incidents(db, case_id)
        return {"case_id": case_id, "total": len(incidents), "incidents": incidents}
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    finally:
        db.close()


@router.delete("/{case_id}/incidents/{incident_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Unlink incident from case")
def unlink_incident(case_id: str, incident_id: str):
    _validate_id(case_id, "case_id")
    _validate_id(incident_id, "incident_id")
    db = SessionLocal()
    try:
        _svc.unlink_incident(db, case_id, incident_id)
    except (CaseNotFoundError, CaseEntityNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    finally:
        db.close()


@router.post("/{case_id}/evidence", status_code=status.HTTP_201_CREATED, summary="Link evidence to case")
def link_evidence(case_id: str, body: LinkEvidenceRequest) -> Dict[str, Any]:
    _validate_id(case_id, "case_id")
    _validate_id(body.evidence_id, "evidence_id")
    db = SessionLocal()
    try:
        link = _svc.link_evidence(db, case_id, body.evidence_id, notes=body.notes)
        return {
            "link_id": link.id,
            "case_id": link.case_id,
            "evidence_id": link.evidence_id,
            "notes": link.notes,
            "added_at": link.added_at.isoformat(),
        }
    except (CaseNotFoundError, CaseEntityNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    finally:
        db.close()


@router.get("/{case_id}/evidence", summary="List evidence linked to case")
def get_linked_evidence(case_id: str) -> Dict[str, Any]:
    _validate_id(case_id, "case_id")
    db = SessionLocal()
    try:
        evidence = _svc.get_linked_evidence(db, case_id)
        return {"case_id": case_id, "total": len(evidence), "evidence": evidence}
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    finally:
        db.close()


@router.delete("/{case_id}/evidence/{evidence_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Unlink evidence from case")
def unlink_evidence(case_id: str, evidence_id: str):
    _validate_id(case_id, "case_id")
    _validate_id(evidence_id, "evidence_id")
    db = SessionLocal()
    try:
        _svc.unlink_evidence(db, case_id, evidence_id)
    except (CaseNotFoundError, CaseEntityNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    finally:
        db.close()



# ---------------------------------------------------------------------------
# Bookmarks
# ---------------------------------------------------------------------------

@router.post("/{case_id}/bookmarks", status_code=status.HTTP_201_CREATED, summary="Create bookmark")
def create_bookmark(case_id: str, body: CreateBookmarkRequest) -> Dict[str, Any]:
    _validate_id(case_id, "case_id")
    _validate_id(body.video_id, "video_id")
    db = SessionLocal()
    try:
        bm = _svc.create_bookmark(
            db,
            case_id=case_id,
            video_id=body.video_id,
            timestamp_seconds=body.timestamp_seconds,
            title=body.title,
            description=body.description,
            camera_id=body.camera_id,
            linked_incident_id=body.linked_incident_id,
            linked_track_id=body.linked_track_id,
            linked_evidence_id=body.linked_evidence_id,
            author=body.author or "Investigator",
        )
        return _bookmark_to_dict(bm)
    except (CaseNotFoundError, CaseEntityNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except CaseValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        db.close()


@router.get("/{case_id}/bookmarks", summary="List bookmarks")
def list_bookmarks(case_id: str, video_id: Optional[str] = Query(None)) -> Dict[str, Any]:
    _validate_id(case_id, "case_id")
    db = SessionLocal()
    try:
        bms = _svc.list_bookmarks(db, case_id, video_id=video_id)
        return {"case_id": case_id, "total": len(bms), "bookmarks": [_bookmark_to_dict(b) for b in bms]}
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    finally:
        db.close()


@router.patch("/{case_id}/bookmarks/{bookmark_id}", summary="Update bookmark")
def update_bookmark(case_id: str, bookmark_id: str, body: UpdateBookmarkRequest) -> Dict[str, Any]:
    _validate_id(case_id, "case_id")
    _validate_id(bookmark_id, "bookmark_id")
    db = SessionLocal()
    try:
        bm = _svc.update_bookmark(
            db,
            case_id=case_id,
            bookmark_id=bookmark_id,
            title=body.title,
            description=body.description,
            timestamp_seconds=body.timestamp_seconds,
        )
        return _bookmark_to_dict(bm)
    except (CaseNotFoundError, CaseIsolationError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    finally:
        db.close()


@router.delete("/{case_id}/bookmarks/{bookmark_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete bookmark")
def delete_bookmark(case_id: str, bookmark_id: str):
    _validate_id(case_id, "case_id")
    _validate_id(bookmark_id, "bookmark_id")
    db = SessionLocal()
    try:
        _svc.delete_bookmark(db, case_id, bookmark_id)
    except (CaseNotFoundError, CaseIsolationError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

@router.post("/{case_id}/notes", status_code=status.HTTP_201_CREATED, summary="Create investigator note")
def create_note(case_id: str, body: CreateNoteRequest) -> Dict[str, Any]:
    _validate_id(case_id, "case_id")
    db = SessionLocal()
    try:
        note = _svc.create_note(
            db,
            case_id=case_id,
            content=body.content,
            author=body.author or "Investigator",
            associated_type=body.associated_type or "CASE",
            associated_id=body.associated_id,
            timestamp_seconds=body.timestamp_seconds,
        )
        return _note_to_dict(note)
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    except CaseValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        db.close()


@router.get("/{case_id}/notes", summary="List investigator notes")
def list_notes(
    case_id: str,
    associated_type: Optional[str] = Query(None),
    associated_id: Optional[str] = Query(None),
) -> Dict[str, Any]:
    _validate_id(case_id, "case_id")
    db = SessionLocal()
    try:
        notes = _svc.list_notes(db, case_id, associated_type=associated_type, associated_id=associated_id)
        return {"case_id": case_id, "total": len(notes), "notes": [_note_to_dict(n) for n in notes]}
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    finally:
        db.close()


@router.patch("/{case_id}/notes/{note_id}", summary="Update investigator note")
def update_note(case_id: str, note_id: str, body: UpdateNoteRequest) -> Dict[str, Any]:
    _validate_id(case_id, "case_id")
    _validate_id(note_id, "note_id")
    db = SessionLocal()
    try:
        note = _svc.update_note(db, case_id, note_id, body.content)
        return _note_to_dict(note)
    except (CaseNotFoundError, CaseIsolationError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except CaseValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        db.close()


@router.delete("/{case_id}/notes/{note_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete investigator note")
def delete_note(case_id: str, note_id: str):
    _validate_id(case_id, "case_id")
    _validate_id(note_id, "note_id")
    db = SessionLocal()
    try:
        _svc.delete_note(db, case_id, note_id)
    except (CaseNotFoundError, CaseIsolationError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

@router.post("/{case_id}/annotations", status_code=status.HTTP_201_CREATED, summary="Create visual overlay annotation")
def create_annotation(case_id: str, body: CreateAnnotationRequest) -> Dict[str, Any]:
    _validate_id(case_id, "case_id")
    _validate_id(body.video_id, "video_id")
    db = SessionLocal()
    try:
        ann = _svc.create_annotation(
            db,
            case_id=case_id,
            video_id=body.video_id,
            timestamp_seconds=body.timestamp_seconds,
            annotation_type=body.annotation_type or "REGION",
            data=body.data,
            author=body.author or "Investigator",
            camera_id=body.camera_id,
            end_timestamp_seconds=body.end_timestamp_seconds,
        )
        return _annotation_to_dict(ann)
    except (CaseNotFoundError, CaseEntityNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except CaseValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        db.close()


@router.get("/{case_id}/annotations", summary="List visual overlay annotations")
def list_annotations(
    case_id: str,
    video_id: Optional[str] = Query(None),
    start_time: Optional[float] = Query(None),
    end_time: Optional[float] = Query(None),
) -> Dict[str, Any]:
    _validate_id(case_id, "case_id")
    db = SessionLocal()
    try:
        anns = _svc.list_annotations(db, case_id, video_id=video_id, start_time=start_time, end_time=end_time)
        return {"case_id": case_id, "total": len(anns), "annotations": [_annotation_to_dict(a) for a in anns]}
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    finally:
        db.close()


@router.delete("/{case_id}/annotations/{annotation_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete annotation")
def delete_annotation(case_id: str, annotation_id: str):
    _validate_id(case_id, "case_id")
    _validate_id(annotation_id, "annotation_id")
    db = SessionLocal()
    try:
        _svc.delete_annotation(db, case_id, annotation_id)
    except (CaseNotFoundError, CaseIsolationError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Case Unified Timeline
# ---------------------------------------------------------------------------

@router.get("/{case_id}/timeline", summary="Case-level unified chronological timeline")
def get_case_timeline(
    case_id: str,
    start_time: Optional[float] = Query(None, ge=0.0),
    end_time: Optional[float] = Query(None, ge=0.0),
    layers: Optional[str] = Query(None, description="Comma-separated layer names"),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    _validate_id(case_id, "case_id")
    db = SessionLocal()
    try:
        layer_list = [l.strip() for l in layers.split(",")] if layers else None
        return _svc.get_case_timeline(
            db,
            case_id=case_id,
            start_time=start_time,
            end_time=end_time,
            layers=layer_list,
            limit=limit,
            offset=offset,
        )
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Investigation Workflows: Replay, Focus, Explain, Topology, Storyline
# ---------------------------------------------------------------------------

@router.get("/{case_id}/incidents/{incident_id}/replay", summary="Incident replay context")
def get_incident_replay(
    case_id: str,
    incident_id: str,
    pre_roll_seconds: float = Query(5.0, ge=0.0, le=30.0),
    post_roll_seconds: float = Query(5.0, ge=0.0, le=30.0),
) -> Dict[str, Any]:
    _validate_id(case_id, "case_id")
    _validate_id(incident_id, "incident_id")
    db = SessionLocal()
    try:
        return _svc.get_incident_replay_context(
            db,
            case_id=case_id,
            incident_id=incident_id,
            pre_roll_seconds=pre_roll_seconds,
            post_roll_seconds=post_roll_seconds,
        )
    except (CaseNotFoundError, CaseEntityNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    finally:
        db.close()


@router.get("/{case_id}/incidents/{incident_id}/focus", summary="Incident investigation focus mode")
def get_incident_focus(case_id: str, incident_id: str) -> Dict[str, Any]:
    _validate_id(case_id, "case_id")
    _validate_id(incident_id, "incident_id")
    db = SessionLocal()
    try:
        return _svc.get_incident_focus_data(db, case_id=case_id, incident_id=incident_id)
    except (CaseNotFoundError, CaseEntityNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    finally:
        db.close()


@router.get("/{case_id}/incidents/{incident_id}/explain", summary="Why Did Sentinel Flag This?")
def get_incident_explain(case_id: str, incident_id: str) -> Dict[str, Any]:
    _validate_id(case_id, "case_id")
    _validate_id(incident_id, "incident_id")
    db = SessionLocal()
    try:
        return _svc.get_incident_explanation(db, case_id=case_id, incident_id=incident_id)
    except (CaseNotFoundError, CaseEntityNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    finally:
        db.close()


@router.get("/{case_id}/topology", summary="Camera topology view for case")
def get_case_topology(case_id: str) -> Dict[str, Any]:
    _validate_id(case_id, "case_id")
    db = SessionLocal()
    try:
        return _svc.get_case_topology(db, case_id=case_id)
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    finally:
        db.close()


@router.get("/{case_id}/storyline", summary="Investigation storyline / notebook")
def get_case_storyline(case_id: str) -> Dict[str, Any]:
    _validate_id(case_id, "case_id")
    db = SessionLocal()
    try:
        return _svc.get_case_storyline(db, case_id=case_id)
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    finally:
        db.close()


@router.get("/{case_id}/activities", summary="Case activity audit log")
def get_case_activities(
    case_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    _validate_id(case_id, "case_id")
    db = SessionLocal()
    try:
        acts = _svc.get_case_activities(db, case_id=case_id, limit=limit, offset=offset)
        return {
            "case_id": case_id,
            "total": len(acts),
            "activities": [
                {
                    "id": a.id,
                    "action_type": a.action_type,
                    "description": a.description,
                    "details": a.details,
                    "actor": a.actor,
                    "created_at": a.created_at.isoformat(),
                }
                for a in acts
            ],
        }
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    finally:
        db.close()


@router.get("/{case_id}/export", summary="Export full forensic case package")
def export_case_data(case_id: str) -> Dict[str, Any]:
    _validate_id(case_id, "case_id")
    db = SessionLocal()
    try:
        return _svc.export_case_data(db, case_id=case_id)
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    finally:
        db.close()
