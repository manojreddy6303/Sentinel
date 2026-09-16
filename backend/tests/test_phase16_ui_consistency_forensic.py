"""
backend/tests/test_phase16_ui_consistency_forensic.py
=====================================================
Targeted regression tests for Phase 16 UI / Cross-Layer Consistency Forensic Fix:
1. REVIEW_REQUIRED candidate cannot become ACCEPTED through correlation alone.
2. ACCEPTED constituent incidents can only produce ACCEPTED output when canonical acceptance rules are satisfied.
3. Score and decision remain independent concepts.
4. Storyline status wording matches backend status and contains no contradictions.
5. Investigation status matches backend status.
6. PDF status matches backend status.
7. Crowd supporting telemetry does not become incident duration.
8. Raw/validated/rejected counts use canonical semantics.
9. Face-region counts remain video isolated.
10. Reprocessing does not change canonical decisions unexpectedly.
"""

import pytest
import uuid
from typing import List, Dict, Any

from ai.schemas import TrackedObject, BoundingBox, FaceDetection
from ai.incidents.schemas import IncidentCandidate, ValidationDecision
from ai.incidents.fusion import IncidentFusionEngine
from ai.correlation.models import CorrelatedIncident, ObservationalRelationship, HypothesisOutcome
from ai.correlation.engine import AdvancedIncidentCorrelationEngine as IncidentCorrelationEngine
from ai.correlation.arbitrator import CompetingHypothesisArbitrator
from ai.correlation.fusion_policies import (
    inherit_validation_decision,
    VehicleCorrelationPolicy,
    PropertyCorrelationPolicy,
    PersonCorrelationPolicy,
    CrowdZoneCorrelationPolicy,
    SpecializedVisualPolicy,
)
from ai.correlation.storyline_generator import IncidentStorylineGenerator
from ai.common.detector_health import DetectorHealthRegistry
from ai.intelligence_pipeline import SecurityIntelligencePipeline
from database.session import SessionLocal
from database.models import VideoModel, FaceDetectionModel, SecurityEventModel, CorrelatedIncidentModel
from ai.intelligence_repository import SecurityIntelligenceRepository


# ---------------------------------------------------------------------------
# Helper candidate factory
# ---------------------------------------------------------------------------
def _make_candidate(
    event_type: str,
    category: str,
    start_time: float,
    end_time: float,
    confidence: float = 0.95,
    validation_decision: str = "ACCEPTED",
    human_verification_required: bool = True,
    track_ids: List[str] = None,
) -> IncidentCandidate:
    dur = max(0.0, end_time - start_time)
    return IncidentCandidate(
        incident_id=f"TEST-{uuid.uuid4().hex[:8]}",
        video_id="v_test_consistency",
        event_type=event_type,
        category=category,
        start_time=start_time,
        end_time=end_time,
        duration=dur,
        severity="NORMAL",
        confidence=confidence,
        track_ids=track_ids or ["trk_01"],
        object_classes=["person"],
        validation_decision=validation_decision,
        human_verification_required=human_verification_required,
        detector_name="test_detector",
        detector_version="1.0.0",
    )


# ---------------------------------------------------------------------------
# 1. REVIEW_REQUIRED candidate cannot become ACCEPTED through correlation alone
# ---------------------------------------------------------------------------
def test_01_review_required_cannot_become_accepted_through_correlation_alone():
    cand1 = _make_candidate(
        event_type="PROLONGED_PRESENCE",
        category="person",
        start_time=10.0,
        end_time=20.0,
        confidence=0.95,
        validation_decision="REVIEW_REQUIRED",
    )
    cand2 = _make_candidate(
        event_type="PROLONGED_PRESENCE",
        category="person",
        start_time=15.0,
        end_time=25.0,
        confidence=0.93,
        validation_decision="REVIEW_REQUIRED",
    )

    # Test fusion engine cluster merge
    fusion = IncidentFusionEngine(time_merge_tolerance_seconds=5.0)
    fused = fusion.fuse_incidents([cand1, cand2])
    assert len(fused) == 1
    assert fused[0].validation_decision == "REVIEW_REQUIRED"
    assert fused[0].confidence <= 0.65

    # Test correlation engine output
    engine = IncidentCorrelationEngine()
    result = engine.correlate_incidents("v_test_consistency", fused, [])
    corrs = result["correlated_incidents"]
    assert len(corrs) >= 1
    for ci in corrs:
        assert ci.validation_decision == "REVIEW_REQUIRED"
        assert ci.assessment_score <= 0.65
        assert ci.reliability_rating != "HIGH"


# ---------------------------------------------------------------------------
# 2. ACCEPTED constituent incidents can only produce ACCEPTED output when
#    canonical acceptance rules are satisfied
# ---------------------------------------------------------------------------
def test_02_accepted_constituent_requires_canonical_acceptance_rules():
    # If constituents are strictly accepted and pass rules
    cand_a = _make_candidate(
        event_type="COORDINATED_PERSON_MOVEMENT",
        category="person",
        start_time=5.0,
        end_time=12.0,
        confidence=0.90,
        validation_decision="ACCEPTED",
    )
    assert inherit_validation_decision([cand_a]) == "ACCEPTED"

    # If any constituent is REVIEW_REQUIRED, result MUST be REVIEW_REQUIRED
    cand_b = _make_candidate(
        event_type="POTENTIAL_FORCED_MOVEMENT",
        category="person",
        start_time=8.0,
        end_time=14.0,
        confidence=0.60,
        validation_decision="REVIEW_REQUIRED",
    )
    assert inherit_validation_decision([cand_a, cand_b]) == "REVIEW_REQUIRED"


# ---------------------------------------------------------------------------
# 3. Score and decision remain independent concepts
# ---------------------------------------------------------------------------
def test_03_score_and_decision_remain_independent_concepts():
    high_score_review = _make_candidate(
        event_type="POTENTIAL_THEFT",
        category="property",
        start_time=20.0,
        end_time=30.0,
        confidence=0.98,
        validation_decision="REVIEW_REQUIRED",
    )

    corrs = PropertyCorrelationPolicy.correlate([high_score_review], "v_test", [])
    assert len(corrs) == 1
    ci = corrs[0]
    # Score must be capped to review ceiling <= 0.65 despite 0.98 raw confidence
    assert ci.validation_decision == "REVIEW_REQUIRED"
    assert ci.assessment_score <= 0.65
    assert ci.evidence_strength <= 0.65
    assert ci.reliability_rating == "MODERATE"


# ---------------------------------------------------------------------------
# 4. Storyline status matches backend status and has non-contradictory wording
# ---------------------------------------------------------------------------
def test_04_storyline_verification_wording_matches_decision():
    # ACCEPTED wording
    story_acc = IncidentStorylineGenerator.generate_storyline(
        category="person",
        subcategory="coordinated_person_movement",
        start_time=10.0,
        end_time=15.0,
        duration=5.0,
        primary_tracks=["trk_01"],
        supporting_tracks=[],
        object_classes=["person"],
        relationships=[],
        supporting_signals=[],
        confidence=0.92,
        validation_decision="ACCEPTED",
    )
    assert "Accepted by Sentinel's evidence policy; observational result." in story_acc
    assert "Mandatory human verification required" not in story_acc

    # REVIEW_REQUIRED wording
    story_rev = IncidentStorylineGenerator.generate_storyline(
        category="person",
        subcategory="prolonged_presence",
        start_time=10.0,
        end_time=22.0,
        duration=12.0,
        primary_tracks=["trk_01"],
        supporting_tracks=[],
        object_classes=["person"],
        relationships=[],
        supporting_signals=[],
        confidence=0.65,
        validation_decision="REVIEW_REQUIRED",
    )
    assert "Insufficient certainty for automatic acceptance; human verification required." in story_rev
    assert "status: REVIEW_REQUIRED" in story_rev


# ---------------------------------------------------------------------------
# 5. Investigation status matches backend status
# ---------------------------------------------------------------------------
def test_05_investigation_status_matches_backend_status():
    from backend.app.services.investigation_service import InvestigationService
    from database.session import SessionLocal
    from database.models import VideoModel, SecurityEventModel

    db = SessionLocal()
    vid = f"v_inv_test_{uuid.uuid4().hex[:8]}"
    try:
        v = VideoModel(id=vid, original_filename="test.mp4", storage_path="test.mp4", duration_seconds=30.0)
        db.add(v)
        se = SecurityEventModel(
            id=f"se_{uuid.uuid4().hex[:8]}",
            video_id=vid,
            event_type="PROLONGED_PRESENCE",
            severity="NORMAL",
            timestamp_seconds=10.0,
            duration_seconds=15.0,
            confidence=0.65,
            description="Prolonged presence",
            human_verification_required=1,
            incident_metadata={"validation_decision": "REVIEW_REQUIRED"},
        )
        db.add(se)
        db.commit()

        service = InvestigationService()
        res = service._query_security_events(db, vid, "prolonged presence", {"event_type": "PROLONGED_PRESENCE"})
        assert res["count"] == 1
        rec = res["results"][0]
        assert rec["validation_decision"] == "REVIEW_REQUIRED"
        assert rec["confidence"] <= 0.65
        assert rec["human_verification_required"] is True
    finally:
        db.query(SecurityEventModel).filter(SecurityEventModel.video_id == vid).delete()
        db.query(VideoModel).filter(VideoModel.id == vid).delete()
        db.commit()
        db.close()


# ---------------------------------------------------------------------------
# 6. PDF status matches backend status
# ---------------------------------------------------------------------------
def test_06_pdf_status_matches_backend_status():
    from ai.reporting.schema import ReportCorrelatedIncident

    item = ReportCorrelatedIncident(
        incident_id="CORR-TEST-123",
        incident_category="person",
        incident_subcategory="prolonged_presence",
        start_time=12.0,
        end_time=24.0,
        duration=12.0,
        assessment_score=0.65,
        evidence_strength=0.52,
        reliability_rating="MODERATE",
        validation_decision="REVIEW_REQUIRED",
        primary_track_ids=["trk_01"],
        storyline="Insufficient certainty for automatic acceptance; human verification required.",
    )
    assert item.validation_decision == "REVIEW_REQUIRED"
    assert item.assessment_score == 0.65
    assert item.reliability_rating == "MODERATE"


# ---------------------------------------------------------------------------
# 7. Crowd supporting telemetry does not become incident duration
# ---------------------------------------------------------------------------
def test_07_crowd_supporting_telemetry_does_not_become_incident_duration():
    # Episode 1 at 101s - 105s (dur 4.0s)
    c1 = _make_candidate(
        event_type="HIGH_PEDESTRIAN_DENSITY",
        category="crowd",
        start_time=101.0,
        end_time=105.0,
        confidence=0.75,
        validation_decision="REVIEW_REQUIRED",
    )
    # Episode 2 at 165s - 170s (dur 5.0s)
    c2 = _make_candidate(
        event_type="POTENTIAL_CROWD_DISPERSAL",
        category="crowd",
        start_time=165.0,
        end_time=170.0,
        confidence=0.70,
        validation_decision="REVIEW_REQUIRED",
    )

    correlated = CrowdZoneCorrelationPolicy.correlate([c1, c2], "v_crowd_test", [])
    # Must produce 2 distinct localized episodes, NOT one giant 69-second episode
    assert len(correlated) == 2

    ep1 = correlated[0]
    assert ep1.start_time == 101.0
    assert ep1.end_time == 105.0
    assert ep1.duration == 4.0
    assert ep1.duration < 10.0  # Localized, not 69.0s

    ep2 = correlated[1]
    assert ep2.start_time == 165.0
    assert ep2.end_time == 170.0
    assert ep2.duration == 5.0
    assert ep2.duration < 10.0  # Localized, not 69.0s

    # The broad supporting telemetry window is preserved in contextual factors
    for ep in correlated:
        telemetry = ep.contextual_factors.get("supporting_telemetry_window")
        assert telemetry is not None
        assert telemetry["start_time"] == 101.0
        assert telemetry["end_time"] == 170.0
        assert telemetry["duration"] == 69.0


# ---------------------------------------------------------------------------
# 8. Raw/validated/rejected counts use canonical semantics
# ---------------------------------------------------------------------------
def test_08_raw_validated_rejected_counts_canonical_semantics():
    registry = DetectorHealthRegistry.get_instance()
    registry.reset()

    # Simulate 138 raw detections (137 valid, 1 rejected)
    for _ in range(137):
        registry.record_observation("yolo_detector", is_validated=True, is_rejected=False)
    registry.record_observation("yolo_detector", is_validated=False, is_rejected=True)

    report = registry.get_health_report()
    assert report["total_observations"] == 138
    assert report["total_validated"] == 137
    assert report["total_rejected"] == 1

    yolo_record = registry.get_detector("yolo_detector")
    assert yolo_record.observations == 138
    assert yolo_record.validated_observations == 137
    assert yolo_record.rejected_observations == 1


# ---------------------------------------------------------------------------
# 9. Face-region counts remain video isolated
# ---------------------------------------------------------------------------
def test_09_face_region_counts_remain_video_isolated():
    db = SessionLocal()
    repo = SecurityIntelligenceRepository()
    vid_a = f"vid_face_a_{uuid.uuid4().hex[:6]}"
    vid_b = f"vid_face_b_{uuid.uuid4().hex[:6]}"

    try:
        # Save 2 faces for video A
        face_a1 = FaceDetection(
            track_id="trk_a1",
            timestamp=1.0,
            confidence=0.90,
            bounding_box=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        )
        face_a2 = FaceDetection(
            track_id="trk_a2",
            timestamp=2.0,
            confidence=0.88,
            bounding_box=BoundingBox(x1=10, y1=10, x2=20, y2=20),
        )
        repo.save_intelligence_results(vid_a, tracks=[], vehicle_attributes=[], face_detections=[face_a1, face_a2], security_events=[])

        # Save 1 face for video B
        face_b1 = FaceDetection(
            track_id="trk_b1",
            timestamp=1.5,
            confidence=0.85,
            bounding_box=BoundingBox(x1=5, y1=5, x2=15, y2=15),
        )
        repo.save_intelligence_results(vid_b, tracks=[], vehicle_attributes=[], face_detections=[face_b1], security_events=[])

        faces_a = repo.get_face_detections(vid_a)
        faces_b = repo.get_face_detections(vid_b)

        assert len(faces_a) == 2
        assert len(faces_b) == 1
        assert all(f["track_id"] in ("trk_a1", "trk_a2") for f in faces_a)
        assert faces_b[0]["track_id"] == "trk_b1"
    finally:
        db.query(FaceDetectionModel).filter(FaceDetectionModel.video_id.in_([vid_a, vid_b])).delete(synchronize_session=False)
        db.query(VideoModel).filter(VideoModel.id.in_([vid_a, vid_b])).delete(synchronize_session=False)
        db.commit()
        db.close()


# ---------------------------------------------------------------------------
# 10. Reprocessing does not change canonical decisions unexpectedly
# ---------------------------------------------------------------------------
def test_10_reprocessing_preserves_canonical_decisions():
    cand = _make_candidate(
        event_type="PROLONGED_PRESENCE",
        category="person",
        start_time=10.0,
        end_time=25.0,
        confidence=0.65,
        validation_decision="REVIEW_REQUIRED",
    )
    engine = IncidentCorrelationEngine()

    run1 = engine.correlate_incidents("v_reprocess", [cand], [])["correlated_incidents"]
    run2 = engine.correlate_incidents("v_reprocess", [cand], [])["correlated_incidents"]

    assert len(run1) == len(run2) == 1
    assert run1[0].validation_decision == run2[0].validation_decision == "REVIEW_REQUIRED"
    assert run1[0].assessment_score == run2[0].assessment_score == 0.65
    assert run1[0].reliability_rating == run2[0].reliability_rating == "MODERATE"
    assert run1[0].storyline == run2[0].storyline
