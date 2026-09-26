"""
Incident Intelligence API Router
Provides system-wide and scoped querying of correlated security incidents.
Preserves canonical validation decisions, evidence strength metrics, and review ceilings.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from database.session import SessionLocal
from database.models import CorrelatedIncidentModel, VideoModel, SecurityEventModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/incidents", tags=["Incidents"])


@router.get("", summary="List all security incidents across videos or filtered")
def list_incidents(
    video_id: Optional[str] = Query(None, description="Filter by source video ID"),
    category: Optional[str] = Query(None, description="Filter by incident category"),
    decision: Optional[str] = Query(None, description="Filter by validation decision (REVIEW_REQUIRED, ACCEPTED, etc.)"),
    min_score: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum assessment score"),
    search: Optional[str] = Query(None, description="Search in storyline narrative or category"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """
    List all correlated security incidents across Sentinel.
    Maintains authoritative <= 0.65 REVIEW_REQUIRED boundary and human verification notice.
    """
    db = SessionLocal()
    try:
        query = db.query(CorrelatedIncidentModel)
        if video_id:
            query = query.filter(CorrelatedIncidentModel.video_id == video_id)
        if category and category != "all":
            query = query.filter(CorrelatedIncidentModel.incident_category == category.upper())
        if decision and decision != "all":
            query = query.filter(CorrelatedIncidentModel.validation_decision == decision.upper())
        if min_score is not None:
            query = query.filter(CorrelatedIncidentModel.assessment_score >= min_score)
        if search:
            query = query.filter(
                (CorrelatedIncidentModel.storyline.ilike(f"%{search.strip()}%")) |
                (CorrelatedIncidentModel.incident_category.ilike(f"%{search.strip()}%"))
            )

        total = query.count()
        incidents = query.order_by(CorrelatedIncidentModel.start_time.desc()).offset(offset).limit(limit).all()

        results = []
        for inc in incidents:
            vid = db.query(VideoModel).filter(VideoModel.id == inc.video_id).first()
            source_video_name = vid.original_filename if vid else "Unknown Video"

            tracks = list(set((inc.primary_track_ids or []) + (inc.supporting_track_ids or [])))
            
            # Review requirement flag: True if <= 0.65 or decision is REVIEW_REQUIRED
            review_required = (inc.assessment_score <= 0.65) or (inc.validation_decision == "REVIEW_REQUIRED")

            clean_storyline = inc.storyline or ""
            if review_required and clean_storyline:
                clean_storyline = clean_storyline.replace("a verified potential forced movement pattern", "a detected potential forced movement pattern supported by validated physical signals")
                clean_storyline = clean_storyline.replace("a verified prolonged presence pattern", "a detected prolonged presence pattern supported by validated physical signals")
                clean_storyline = clean_storyline.replace("a verified theft pattern", "a potential object-takeaway pattern supported by validated physical signals")
                clean_storyline = clean_storyline.replace("verified potential object-takeaway pattern", "potential object-takeaway pattern supported by validated physical signals")
                clean_storyline = clean_storyline.replace("verified incident", "detected incident pattern")
                clean_storyline = clean_storyline.replace("verified physical signal", "validated physical signal")

            results.append({
                "id": inc.id,
                "incident_id": inc.id,
                "video_id": inc.video_id,
                "source_video_name": source_video_name,
                "incident_category": inc.incident_category,
                "category": inc.incident_category,
                "title": f"{inc.incident_category.replace('_', ' ').title()}",
                "start_time": inc.start_time,
                "end_time": inc.end_time,
                "duration": inc.duration,
                "assessment_score": inc.assessment_score,
                "evidence_strength": inc.evidence_strength,
                "pattern_evidence_strength": (inc.contextual_factors or {}).get("pattern_evidence_strength", inc.evidence_strength) if isinstance(inc.contextual_factors, dict) else inc.evidence_strength,
                "confidence": inc.assessment_score,
                "incident_score": inc.assessment_score,
                "score": inc.assessment_score,
                "reliability_rating": inc.reliability_rating,
                "validation_decision": inc.validation_decision,
                "decision": inc.validation_decision,
                "storyline": clean_storyline,
                "narrative": clean_storyline,
                "primary_track_ids": inc.primary_track_ids or [],
                "supporting_track_ids": inc.supporting_track_ids or [],
                "participating_tracks": tracks,
                "involved_object_classes": inc.involved_object_classes or [],
                "evidence_ids": inc.evidence_ids or [],
                "negative_evidence": inc.negative_evidence or [],
                "contextual_factors": inc.contextual_factors or [],
                "human_verification_required": 1 if review_required else 0,
                "review_required": review_required,
                "created_at": inc.created_at.isoformat() if inc.created_at else None,
            })

        return {
            "total": total,
            "incidents": results,
            "limit": limit,
            "offset": offset,
        }
    finally:
        db.close()
