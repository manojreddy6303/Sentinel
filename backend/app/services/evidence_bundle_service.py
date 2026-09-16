"""
Phase 17: Evidence Bundle Service

Manages creation, storage, and retrieval of investigator-authored evidence
bundles. Bundles collect selected incidents, events, tracks, and evidence
references into a reproducible, video-isolated record.

Physical evidence files are NEVER duplicated — only referenced by ID.
All bundles are scoped to a single video via FK enforcement.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from database.session import SessionLocal
from database.models import (
    CorrelatedIncidentModel,
    EvidenceModel,
    InvestigationBundleModel,
    SecurityEventModel,
    TrackModel,
    VideoModel,
)

logger = logging.getLogger(__name__)


class EvidenceBundleError(Exception):
    """Raised when a bundle operation fails validation or is not found."""
    pass


class EvidenceBundleService:
    """
    Creates and retrieves evidence bundles for Phase 17 investigation workflows.

    Security guarantees:
    - video_id is always required and verified against DB
    - All referenced IDs are verified as belonging to the same video
    - No raw SQL; uses ORM repository pattern
    - Bundles are reproducible by bundle_id
    """

    def create_bundle(
        self,
        video_id: str,
        bundle_name: str,
        selected_incident_ids: Optional[List[str]] = None,
        selected_event_ids: Optional[List[str]] = None,
        selected_track_ids: Optional[List[str]] = None,
        selected_evidence_ids: Optional[List[str]] = None,
        storyline_text: Optional[str] = None,
        notes: Optional[str] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create and persist an evidence bundle.

        All referenced IDs are verified to belong to video_id before saving.
        Returns the canonical bundle record.
        """
        db = SessionLocal()
        try:
            # Verify video exists
            video = db.query(VideoModel).filter(VideoModel.id == video_id).first()
            if not video:
                raise EvidenceBundleError(f"Video '{video_id}' not found.")

            # Validate input lengths
            if not bundle_name or not bundle_name.strip():
                bundle_name = "Investigation Bundle"
            bundle_name = bundle_name.strip()[:255]

            if storyline_text:
                storyline_text = storyline_text[:5000]
            if notes:
                notes = notes[:2000]

            # Verify each referenced ID belongs to this video
            verified_incident_ids = self._verify_incident_ids(
                db, video_id, selected_incident_ids or []
            )
            verified_event_ids = self._verify_event_ids(
                db, video_id, selected_event_ids or []
            )
            verified_track_ids = self._verify_track_ids(
                db, video_id, selected_track_ids or []
            )
            verified_evidence_ids = self._verify_evidence_ids(
                db, video_id, selected_evidence_ids or []
            )

            bundle = InvestigationBundleModel(
                video_id=video_id,
                bundle_name=bundle_name,
                selected_incident_ids=verified_incident_ids,
                selected_event_ids=verified_event_ids,
                selected_track_ids=verified_track_ids,
                selected_evidence_ids=verified_evidence_ids,
                storyline_text=storyline_text,
                notes=notes,
                provenance=provenance or {},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            db.add(bundle)
            db.commit()
            db.refresh(bundle)

            logger.info(
                f"Evidence bundle '{bundle.id}' created for video {video_id} "
                f"with {len(verified_incident_ids)} incidents, "
                f"{len(verified_event_ids)} events, {len(verified_evidence_ids)} evidence items."
            )
            return self._serialize_bundle(bundle)

        except EvidenceBundleError:
            raise
        except Exception as exc:
            db.rollback()
            logger.error(f"Failed to create evidence bundle for video {video_id}: {exc}")
            raise EvidenceBundleError(f"Bundle creation failed: {str(exc)}")
        finally:
            db.close()

    def get_bundle(self, video_id: str, bundle_id: str) -> Dict[str, Any]:
        """
        Retrieve an evidence bundle by ID.
        Raises EvidenceBundleError if not found or video mismatch.
        """
        db = SessionLocal()
        try:
            bundle = (
                db.query(InvestigationBundleModel)
                .filter(
                    InvestigationBundleModel.id == bundle_id,
                    InvestigationBundleModel.video_id == video_id,  # isolation enforced
                )
                .first()
            )
            if not bundle:
                raise EvidenceBundleError(
                    f"Bundle '{bundle_id}' not found for video '{video_id}'."
                )
            return self._serialize_bundle(bundle)
        finally:
            db.close()

    def list_bundles(self, video_id: str) -> List[Dict[str, Any]]:
        """List all evidence bundles for a video."""
        db = SessionLocal()
        try:
            bundles = (
                db.query(InvestigationBundleModel)
                .filter(InvestigationBundleModel.video_id == video_id)
                .order_by(InvestigationBundleModel.created_at.desc())
                .all()
            )
            return [self._serialize_bundle(b) for b in bundles]
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Verification helpers — ensure IDs belong to this video
    # ------------------------------------------------------------------

    def _verify_incident_ids(
        self, db, video_id: str, ids: List[str]
    ) -> List[str]:
        if not ids:
            return []
        owned = db.query(CorrelatedIncidentModel.id).filter(
            CorrelatedIncidentModel.video_id == video_id,
            CorrelatedIncidentModel.id.in_(ids),
        ).all()
        owned_set = {r.id for r in owned}
        unowned = set(ids) - owned_set
        if unowned:
            raise EvidenceBundleError(
                f"Referenced incident(s) {sorted(list(unowned))} do not belong to video '{video_id}'."
            )
        return [r.id for r in owned]

    def _verify_event_ids(
        self, db, video_id: str, ids: List[str]
    ) -> List[str]:
        if not ids:
            return []
        owned = db.query(SecurityEventModel.id).filter(
            SecurityEventModel.video_id == video_id,
            SecurityEventModel.id.in_(ids),
        ).all()
        owned_set = {r.id for r in owned}
        unowned = set(ids) - owned_set
        if unowned:
            raise EvidenceBundleError(
                f"Referenced event(s) {sorted(list(unowned))} do not belong to video '{video_id}'."
            )
        return [r.id for r in owned]

    def _verify_track_ids(
        self, db, video_id: str, track_ids: List[str]
    ) -> List[str]:
        if not track_ids:
            return []
        # Track IDs are strings (e.g. "TRACK-001"), not UUIDs
        owned = db.query(TrackModel.track_id).filter(
            TrackModel.video_id == video_id,
            TrackModel.track_id.in_(track_ids),
        ).distinct().all()
        owned_set = {r.track_id for r in owned}
        unowned = set(track_ids) - owned_set
        if unowned:
            raise EvidenceBundleError(
                f"Referenced track(s) {sorted(list(unowned))} do not belong to video '{video_id}'."
            )
        return [r.track_id for r in owned]

    def _verify_evidence_ids(
        self, db, video_id: str, ids: List[str]
    ) -> List[str]:
        if not ids:
            return []
        owned = db.query(EvidenceModel.id).filter(
            EvidenceModel.video_id == video_id,
            EvidenceModel.id.in_(ids),
        ).all()
        owned_set = {r.id for r in owned}
        unowned = set(ids) - owned_set
        if unowned:
            raise EvidenceBundleError(
                f"Referenced evidence item(s) {sorted(list(unowned))} do not belong to video '{video_id}'."
            )
        return [r.id for r in owned]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _serialize_bundle(self, bundle: InvestigationBundleModel) -> Dict[str, Any]:
        return {
            "bundle_id": bundle.id,
            "video_id": bundle.video_id,
            "bundle_name": bundle.bundle_name,
            "selected_incident_ids": bundle.selected_incident_ids or [],
            "selected_event_ids": bundle.selected_event_ids or [],
            "selected_track_ids": bundle.selected_track_ids or [],
            "selected_evidence_ids": bundle.selected_evidence_ids or [],
            "storyline_text": bundle.storyline_text,
            "notes": bundle.notes,
            "provenance": bundle.provenance or {},
            "created_at": bundle.created_at.isoformat() if bundle.created_at else None,
            "updated_at": bundle.updated_at.isoformat() if bundle.updated_at else None,
        }
