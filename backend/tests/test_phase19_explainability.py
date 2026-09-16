"""
backend/tests/test_phase19_explainability.py

Tests for "Why Did Sentinel Flag This?" Explainability Engine:
- Extraction of genuine supporting signals from detection telemetry & context
- Extraction of limiting / contradicting signals from negative evidence
- Preservation of distinction between Pattern Evidence Strength and Final Assessment
- REVIEW_REQUIRED <= 0.65 ceiling invariant
- Human verification requirement visibility
"""
import uuid
import pytest
from database.session import SessionLocal, init_db
from database.models import VideoModel, CorrelatedIncidentModel, SecurityEventModel
from backend.app.services.case_service import CaseService


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def case_svc():
    return CaseService()


def test_explainability_correlated_incident(db, case_svc):
    vid = str(uuid.uuid4())
    v = VideoModel(id=vid, original_filename="perimeter_cam.mp4", duration_seconds=100.0, storage_path=f"storage/{vid}.mp4", status="processed")
    db.add(v)

    inc_id = f"CORR-{uuid.uuid4().hex[:8]}"
    corr = CorrelatedIncidentModel(
        id=inc_id,
        video_id=vid,
        incident_category="theft",
        incident_subcategory="property_takeaway",
        start_time=30.0,
        end_time=45.0,
        duration=15.0,
        assessment_score=0.65,
        evidence_strength=0.95,
        reliability_rating="HIGH",
        validation_decision="REVIEW_REQUIRED",
        primary_track_ids=["TRACK-001", "TRACK-002"],
        contextual_factors={"lighting": "daylight", "zone": "secure_corridor"},
        negative_evidence=["No confirmed physical damage to enclosure", "Subject remained on marked path initially"],
        storyline="Subject approached object, interacted, and departed.",
    )
    db.add(corr)
    db.commit()

    case = case_svc.create_case(db, title="Explainability Audit", initial_video_ids=[vid])
    explanation = case_svc.get_incident_explanation(db, case.id, inc_id)

    assert explanation["incident_id"] == inc_id
    # Supporting signals should be populated from actual metadata
    assert len(explanation["supporting_signals"]) >= 2
    assert any("theft" in s.lower() for s in explanation["supporting_signals"])
    assert any("TRACK-001" in s for s in explanation["supporting_signals"])

    # Limiting signals should come from negative_evidence
    assert len(explanation["limiting_signals"]) >= 2
    assert any("physical damage" in s.lower() for s in explanation["limiting_signals"])

    # Pattern Evidence Strength vs Final Assessment
    assert explanation["pattern_evidence_strength"] == "95.0%"
    assert explanation["final_assessment_score"] == 0.65
    assert explanation["validation_decision"] == "REVIEW_REQUIRED"
    assert explanation["human_verification_required"] is True
    assert "human" in explanation["human_verification_notice"].lower()


def test_explainability_security_event(db, case_svc):
    vid = str(uuid.uuid4())
    v = VideoModel(id=vid, original_filename="restricted_zone.mp4", duration_seconds=50.0, storage_path=f"storage/{vid}.mp4", status="processed")
    db.add(v)

    sec_id = str(uuid.uuid4())
    sec = SecurityEventModel(
        id=sec_id,
        video_id=vid,
        event_type="POTENTIAL_INTRUSION",
        zone_name="Restricted Vault Door",
        severity="HIGH",
        timestamp_seconds=14.2,
        duration_seconds=3.0,
        confidence=0.88,
        description="Person entered restricted polygon boundary.",
        observable_signals=[{"signal": "polygon_crossing", "value": 0.9}],
        detector_name="ZoneIntrusionDetector",
        human_verification_required=1,
    )
    db.add(sec)
    db.commit()

    case = case_svc.create_case(db, title="Security Event Explainability", initial_video_ids=[vid])
    explanation = case_svc.get_incident_explanation(db, case.id, sec_id)

    assert explanation["incident_id"] == sec_id
    assert any("Restricted Vault Door" in s for s in explanation["supporting_signals"])
    assert explanation["human_verification_required"] is True
    # Ceiled at 0.65 for review required
    assert explanation["final_assessment_score"] <= 0.65
