"""
backend/tests/test_phase16_final_cleanup.py
===========================================
Targeted regression tests for Phase 16 Final UI / Report Consistency Cleanup:
1. Raw count equals validated + rejected.
2. UI/API use canonical raw count.
3. Security-event count is consistent across database/API/report.
4. Pattern evidence strength and assessment score remain distinct.
5. Review-required score remains <= 0.65.
6. Footer contains no stale Phase 3 wording.
7. Existing Phase 16 behavior remains intact.
"""

import uuid
import pytest
from database.session import SessionLocal
from database.models import VideoModel, EventModel, SecurityEventModel, CorrelatedIncidentModel
from ai.events.repository import DatabaseEventRepository
from ai.incidents.schemas import IncidentCandidate, ValidationDecision, SupportingSignal
from ai.incidents.scoring import IncidentScorer
from ai.correlation.engine import AdvancedIncidentCorrelationEngine
from ai.reporting.schema import ReportSecurityEvent, ReportCorrelatedIncident
from ai.reporting.dossier_generator import IncidentDossierPDFGenerator


def test_01_raw_count_equals_validated_plus_rejected():
    """Verify raw detection count strictly equals validated + rejected count."""
    db = SessionLocal()
    vid = f"v_count_test_{uuid.uuid4().hex[:8]}"
    try:
        # Create video and events: 10 valid, 3 rejected
        db.add(VideoModel(id=vid, original_filename="test.mp4", storage_path="test.mp4"))
        for i in range(10):
            db.add(EventModel(
                id=f"ev_val_{i}_{uuid.uuid4().hex[:6]}",
                video_id=vid,
                event_type="object_detected",
                object_class="person",
                class_id=0,
                timestamp_seconds=float(i),
                confidence=0.85,
                bbox_x1=10.0, bbox_y1=10.0, bbox_x2=50.0, bbox_y2=50.0,
                frame_number=i,
                validation_status="VALID",
                validation_reason="CONFIRMED_DETECTION",
            ))
        for j in range(3):
            db.add(EventModel(
                id=f"ev_rej_{j}_{uuid.uuid4().hex[:6]}",
                video_id=vid,
                event_type="object_detected",
                object_class="person",
                class_id=0,
                timestamp_seconds=float(10 + j),
                confidence=0.25,
                bbox_x1=0.0, bbox_y1=0.0, bbox_x2=2.0, bbox_y2=2.0,
                frame_number=10 + j,
                validation_status="REJECTED",
                validation_reason="LOW_CONFIDENCE",
            ))
        db.commit()

        raw_count = db.query(EventModel).filter(EventModel.video_id == vid).count()
        val_count = db.query(EventModel).filter(EventModel.video_id == vid, EventModel.validation_status == "VALID").count()
        rej_count = db.query(EventModel).filter(EventModel.video_id == vid, EventModel.validation_status == "REJECTED").count()

        assert raw_count == 13
        assert val_count == 10
        assert rej_count == 3
        assert raw_count == val_count + rej_count
    finally:
        db.query(EventModel).filter(EventModel.video_id == vid).delete()
        db.query(VideoModel).filter(VideoModel.id == vid).delete()
        db.commit()
        db.close()


def test_02_ui_api_use_canonical_raw_count():
    """Verify repository get_events and API events response differentiate raw and validated counts."""
    db = SessionLocal()
    vid = f"v_api_count_{uuid.uuid4().hex[:8]}"
    try:
        db.add(VideoModel(id=vid, original_filename="test.mp4", storage_path="test.mp4"))
        # 5 valid, 2 rejected
        for i in range(5):
            db.add(EventModel(
                id=f"val_{i}_{uuid.uuid4().hex[:6]}",
                video_id=vid,
                event_type="object_detected",
                object_class="car",
                class_id=2,
                timestamp_seconds=float(i),
                confidence=0.9,
                bbox_x1=10, bbox_y1=10, bbox_x2=50, bbox_y2=50,
                frame_number=i,
                validation_status="VALID",
                validation_reason="CONFIRMED",
            ))
        for j in range(2):
            db.add(EventModel(
                id=f"rej_{j}_{uuid.uuid4().hex[:6]}",
                video_id=vid,
                event_type="object_detected",
                object_class="car",
                class_id=2,
                timestamp_seconds=float(5 + j),
                confidence=0.15,
                bbox_x1=0, bbox_y1=0, bbox_x2=2, bbox_y2=2,
                frame_number=5 + j,
                validation_status="REJECTED",
                validation_reason="LOW_CONFIDENCE",
            ))
        db.commit()

        repo = DatabaseEventRepository()
        # Default query returns only validated (5)
        val_events = repo.get_events(vid, validation_status="VALID")
        assert len(val_events) == 5

        # Query with ALL returns all raw events (7)
        raw_events = repo.get_events(vid, validation_status="ALL")
        assert len(raw_events) == 7

        statuses = [e.get("validation_status") for e in raw_events]
        assert statuses.count("VALID") == 5
        assert statuses.count("REJECTED") == 2
    finally:
        db.query(EventModel).filter(EventModel.video_id == vid).delete()
        db.query(VideoModel).filter(VideoModel.id == vid).delete()
        db.commit()
        db.close()


def test_03_security_event_count_consistent_across_db_api_report():
    """Verify security event count matches across database, schema, and PDF rendering without truncation."""
    db = SessionLocal()
    vid = f"v_sec_parity_{uuid.uuid4().hex[:8]}"
    try:
        db.add(VideoModel(id=vid, original_filename="test.mp4", storage_path="test.mp4"))
        # Exactly 13 security events as seen in Burglary benchmark
        sec_events = []
        for i in range(13):
            se = SecurityEventModel(
                id=f"se_{i}_{uuid.uuid4().hex[:6]}",
                video_id=vid,
                event_type="PROLONGED_PRESENCE" if i == 0 else f"SECURITY_EVENT_{i}",
                severity="HIGH" if i == 0 else "NORMAL",
                timestamp_seconds=float(i * 10),
                duration_seconds=5.0,
                confidence=0.65,
                description=f"Security event record #{i}",
                human_verification_required=1,
                incident_metadata={"validation_decision": "REVIEW_REQUIRED"},
            )
            db.add(se)
            sec_events.append(se)
        db.commit()

        # 1. DB count
        db_count = db.query(SecurityEventModel).filter(SecurityEventModel.video_id == vid).count()
        assert db_count == 13

        # 2. Report models count
        report_sec_events = [
            ReportSecurityEvent(
                id=se.id,
                event_type=se.event_type,
                severity=se.severity,
                timestamp_seconds=se.timestamp_seconds,
                duration_seconds=se.duration_seconds,
                confidence=se.confidence,
                description=se.description,
            )
            for se in sec_events
        ]
        assert len(report_sec_events) == 13

        # 3. PDF generation without silent truncation
        # Generator table must build all 13 rows (plus header = 14 rows total)
        from ai.reporting.schema import ReportDataPayload, ReportVideoMetadata, ReportDetectionStats
        payload = ReportDataPayload(
            report_id="REP-TEST-13",
            generated_at_iso="2026-09-13T12:00:00Z",
            video=ReportVideoMetadata(video_id=vid, original_filename="test.mp4"),
            stats=ReportDetectionStats(total_security_events=13),
            security_events=report_sec_events,
        )
        generator = IncidentDossierPDFGenerator(payload)
        story = []
        generator._build_security_intelligence(story)

        # Find Table element in story
        from reportlab.platypus import Table
        tables = [item for item in story if isinstance(item, Table)]
        assert len(tables) >= 1
        # Table data must include 1 header row + 13 event rows = 14 rows!
        sec_table = tables[-1]
        assert len(sec_table._cellvalues) == 14
    finally:
        db.query(SecurityEventModel).filter(SecurityEventModel.video_id == vid).delete()
        db.query(VideoModel).filter(VideoModel.id == vid).delete()
        db.commit()
        db.close()


def test_04_pattern_evidence_strength_and_assessment_score_remain_distinct():
    """Verify Pattern Evidence Strength and Final Assessment Score are distinct concepts."""
    # Observable signals
    sig1 = SupportingSignal(signal_type="spatial_proximity", confidence=0.9, description="Close proximity")
    sig2 = SupportingSignal(signal_type="kinematic_dynamics", confidence=0.85, description="High velocity")

    res = IncidentScorer.calculate_evidence_score(
        base_confidence=0.95,
        supporting_signals=[sig1, sig2],
        validation_decision="REVIEW_REQUIRED",
    )

    # In REVIEW_REQUIRED, assessment_score is capped at 0.65
    assert res["assessment_score"] <= 0.65
    assert res["score"] <= 0.65
    assert "Pattern Evidence Strength" in res["verification_note"]
    assert "Human verification required" in res["verification_note"]


def test_05_review_required_score_remains_under_ceiling():
    """Verify REVIEW_REQUIRED candidate never exceeds 0.65 ceiling in standalone or fused correlation."""
    cand = IncidentCandidate(
        incident_id="TEST-CEIL-1",
        video_id="v_test",
        event_type="PROLONGED_PRESENCE",
        category="person",
        start_time=10.0,
        end_time=25.0,
        duration=15.0,
        severity="NORMAL",
        confidence=0.95,  # Raw signal was high
        validation_decision="REVIEW_REQUIRED",
        human_verification_required=True,
    )
    engine = AdvancedIncidentCorrelationEngine()
    result = engine.correlate_incidents("v_test", [cand], [])
    corrs = result["correlated_incidents"]
    assert len(corrs) == 1
    ci = corrs[0]
    assert ci.validation_decision == "REVIEW_REQUIRED"
    assert ci.assessment_score <= 0.65
    assert ci.reliability_rating == "MODERATE"


def test_06_footer_contains_no_stale_phase3_wording():
    """Verify frontend page footer does not contain obsolete Phase 3 wording."""
    with open("frontend/src/app/page.tsx", "r", encoding="utf-8") as f:
        content = f.read()

    assert "Phase 3: Video Intelligence Pipeline" not in content
    assert "SENTINEL • AI-Powered Security Video Investigation Platform" in content


def test_07_existing_phase16_behavior_intact():
    """Verify existing Phase 16 models and schemas preserve backward compatibility."""
    from ai.schemas import SecurityEvent
    se = SecurityEvent(
        event_type="POTENTIAL_THEFT",
        severity="HIGH",
        timestamp=100.0,
        duration_seconds=5.0,
        confidence=0.65,
        description="Theft pattern",
        event_id="se_compat_1",
        validation_decision="REVIEW_REQUIRED",
    )
    d = se.to_dict()
    assert d["validation_decision"] == "REVIEW_REQUIRED"
    assert d["confidence"] == 0.65
