"""
Incident Intelligence Engine (Phase 10)

Main orchestrator facade coordinating:
- Context building from tracks and detections
- Frame-by-frame motion & spatial telemetry derivation
- Detector registry execution with failure isolation
- Multi-signal incident fusion & deduplication
- Dual output: Rich IncidentCandidate models & backward-compatible SecurityEvent models
"""
import logging
from typing import List, Dict, Any, Optional

from ai.schemas import TrackedObject, VehicleAttribute, FaceDetection, ZoneDefinition, SecurityEvent
from ai.incidents.schemas import IncidentCandidate, IncidentContext
from ai.incidents.registry import IncidentDetectorRegistry, get_detector_registry
from ai.incidents.detectors import register_standard_detectors
from ai.incidents.motion import UniversalMotionEngine
from ai.incidents.spatial import SpatialRelationshipEngine
from ai.incidents.temporal import TemporalAnalysisEngine
from ai.incidents.context import IncidentContextBuilder
from ai.incidents.fusion import IncidentFusionEngine
from ai.incidents.scoring import IncidentScorer

from ai.incidents.scene_context import SceneContextEngine
from ai.incidents.validator import IncidentCandidateValidator
from ai.incidents.evidence_gate import EvidenceEligibilityGate
from ai.incidents.schemas import ValidationDecision

logger = logging.getLogger(__name__)


class IncidentIntelligenceEngine:
    """
    Universal Incident Intelligence Engine for Sentinel CCTV streams (Phase 10-R).
    """

    def __init__(
        self,
        registry: Optional[IncidentDetectorRegistry] = None,
        motion_engine: Optional[UniversalMotionEngine] = None,
        spatial_engine: Optional[SpatialRelationshipEngine] = None,
        temporal_engine: Optional[TemporalAnalysisEngine] = None,
        context_builder: Optional[IncidentContextBuilder] = None,
        fusion_engine: Optional[IncidentFusionEngine] = None,
        scorer: Optional[IncidentScorer] = None,
        candidate_validator: Optional[IncidentCandidateValidator] = None,
        evidence_gate: Optional[EvidenceEligibilityGate] = None,
    ):
        self.motion_engine = motion_engine or UniversalMotionEngine()
        self.spatial_engine = spatial_engine or SpatialRelationshipEngine()
        self.temporal_engine = temporal_engine or TemporalAnalysisEngine()
        self.context_builder = context_builder or IncidentContextBuilder(motion_engine=self.motion_engine)
        self.fusion_engine = fusion_engine or IncidentFusionEngine()
        self.scorer = scorer or IncidentScorer()
        self.candidate_validator = candidate_validator or IncidentCandidateValidator()
        self.evidence_gate = evidence_gate or EvidenceEligibilityGate()

        if registry is not None:
            self.registry = registry
        else:
            self.registry = IncidentDetectorRegistry()
            register_standard_detectors(self.registry)

    def analyze_incidents(
        self,
        video_id: str,
        tracks: List[TrackedObject],
        validated_detections: List[Dict[str, Any]],
        fps: float = 30.0,
        duration_seconds: float = 0.0,
        sample_rate_fps: float = 1.0,
        zones: Optional[List[ZoneDefinition]] = None,
        vehicle_attributes: Optional[List[VehicleAttribute]] = None,
        face_detections: Optional[List[FaceDetection]] = None,
        video_metadata: Optional[Dict[str, Any]] = None,
        specialized_observations: Optional[List[Any]] = None,
        specialized_tracks: Optional[List[Any]] = None,
        specialized_episodes: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute full modular incident intelligence across tracking data.
        Returns:
            {
                "incidents": List[IncidentCandidate],
                "security_events": List[SecurityEvent],
                "diagnostics": Dict[str, Any],
                "context": IncidentContext,
            }
        """
        # Step 1: Assemble structured IncidentContext
        context = self.context_builder.build_context(
            video_id=video_id,
            tracks=tracks,
            validated_detections=validated_detections,
            fps=fps,
            duration_seconds=duration_seconds,
            sample_rate_fps=sample_rate_fps,
            zones=zones,
            vehicle_attributes=vehicle_attributes,
            face_detections=face_detections,
            video_metadata=video_metadata,
        )
        context.specialized_observations = specialized_observations or []
        context.specialized_tracks = specialized_tracks or []
        context.specialized_episodes = specialized_episodes

        # Step 2: Infer dynamic SceneContext and behavioral baseline (Phase C & D)
        scene_ctx_data = SceneContextEngine.infer_scene_context(context)
        context.scene_context = scene_ctx_data

        # Step 3: Execute all enabled detectors with complete error isolation
        raw_candidates, diagnostics = self.registry.execute_all(context)

        # Step 4: Candidate Validation & Negative Evidence Challenge (Phase A & B)
        validated_candidates = self.candidate_validator.validate_all(
            raw_candidates, context=context, scene_context=scene_ctx_data
        )

        rejected_count = sum(1 for c in validated_candidates if c.validation_decision == ValidationDecision.REJECTED.value)
        review_count = sum(1 for c in validated_candidates if c.validation_decision == ValidationDecision.REVIEW_REQUIRED.value)
        accepted_count = sum(1 for c in validated_candidates if c.validation_decision == ValidationDecision.ACCEPTED.value)

        # Suppress REJECTED candidates from becoming confirmed Security Events
        actionable_candidates = [
            c for c in validated_candidates
            if c.validation_decision != ValidationDecision.REJECTED.value
        ]

        # Invariant: If validated smoke count == 0 and smoke episodes == 0, suppress any smoke candidates
        has_valid_smoke = any(
            getattr(o, "class_name", "") == "smoke"
            and str(getattr(o, "validation_status", "")).upper() in ("VALID", "VALIDATED", "SPECIALIZEDVALIDATIONSTATUS.VALID")
            for o in (context.specialized_observations or [])
        ) or any(
            getattr(e, "class_name", "") == "smoke"
            for e in (context.specialized_episodes or [])
        )
        if not has_valid_smoke:
            actionable_candidates = [
                c for c in actionable_candidates if "SMOKE" not in c.event_type
            ]

        # Step 5: Multi-signal incident fusion & deduplication
        fused_incidents = self.fusion_engine.fuse_incidents(actionable_candidates)

        # Step 6: Evidence Eligibility Gating (Phase I & J)
        fused_incidents = self.evidence_gate.filter_and_mark_candidates(fused_incidents, context=context)

        # Step 7: Convert to Sentinel SecurityEvents for 100% backward compatibility
        security_events: List[SecurityEvent] = []
        for inc in fused_incidents:
            sec_ev = inc.to_security_event()

            # Carry rich metadata onto SecurityEvent for downstream database & evidence preservation
            if inc.spatial_context and inc.spatial_context.metadata:
                meta = inc.spatial_context.metadata
                if "person_track_id" in meta:
                    sec_ev.person_track_id = meta["person_track_id"]
                if "object_track_id" in meta:
                    sec_ev.object_track_id = meta["object_track_id"]
                if "object_bounding_box" in meta and meta["object_bounding_box"]:
                    from ai.schemas import BoundingBox
                    ob = meta["object_bounding_box"]
                    sec_ev.object_bounding_box = BoundingBox(
                        x1=float(ob.get("x1", 0.0)),
                        y1=float(ob.get("y1", 0.0)),
                        x2=float(ob.get("x2", 0.0)),
                        y2=float(ob.get("y2", 0.0)),
                    )
                if "object_observation_timestamp" in meta:
                    sec_ev.object_observation_timestamp = meta["object_observation_timestamp"]
                if "is_prior_object_observation" in meta:
                    sec_ev.is_prior_object_observation = meta["is_prior_object_observation"]

            security_events.append(sec_ev)

        diagnostics.update({
            "scene_context": scene_ctx_data.to_dict(),
            "raw_candidates_count": len(raw_candidates),
            "rejected_candidates_count": rejected_count,
            "review_required_count": review_count,
            "accepted_candidates_count": accepted_count,
            "actionable_candidates_count": len(actionable_candidates),
            "evidence_eligible_count": sum(1 for inc in fused_incidents if inc.evidence_eligible),
        })

        logger.info(
            f"Incident Intelligence Engine completed for video '{video_id}': "
            f"{len(raw_candidates)} raw -> {rejected_count} rejected, {review_count} review, {accepted_count} accepted -> "
            f"{len(fused_incidents)} fused incidents ({len(security_events)} security events) "
            f"across {diagnostics['executed_detectors']} active detectors."
        )

        return {
            "incidents": fused_incidents,
            "security_events": security_events,
            "diagnostics": diagnostics,
            "context": context,
        }

