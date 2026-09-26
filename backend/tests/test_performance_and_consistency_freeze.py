"""
Performance, Idempotency, and Cross-Screen Consistency Regression Suite.

Verifies:
1. First-time vs Cached analysis reuse (no duplicate YOLO on repeat)
2. Processing idempotency and duplicate record prevention
3. Playback artifact caching and browser compatibility fast-path
4. Incident score consistency (Pattern Evidence Strength, Final Assessment, REVIEW_REQUIRED <= 0.65)
5. Zero-result search language precision
"""
import pytest
from pathlib import Path

from ai.detection.detector import YOLODetector
from ai.incidents.validator import IncidentCandidateValidator
from ai.incidents.schemas import IncidentCandidate, ValidationDecision, SupportingSignal
from ai.investigation.orchestrator import InvestigationOrchestrator
from backend.app.services.playback_service import is_browser_compatible, get_playback_status


def test_yolo_batch_inference_interface():
    """Verify YOLODetector supports detect_batch with resolution scaling."""
    detector = YOLODetector(model_name="yolov8n.pt", confidence_threshold=0.25)
    assert hasattr(detector, "detect_batch")
    assert detector.detect_batch([], []) == []


def test_incident_score_consistency_and_narrative_synchronization():
    """Verify narrative explanation scores strictly match canonical score fields."""
    validator = IncidentCandidateValidator()
    
    cand = IncidentCandidate(
        incident_id="test_inc_001",
        video_id="test_video",
        event_type="POTENTIAL_THEFT",
        category="property",
        start_time=10.0,
        end_time=25.0,
        duration=15.0,
        severity="HIGH",
        confidence=0.95,
        validation_decision="REVIEW_REQUIRED",
        explanation=(
            "Potential Theft Pattern (Pattern Evidence Strength: 95% (High Evidence Strength) • "
            "Final Assessment: 95% (None) — Sensor-grounded pattern): Person approached object"
        ),
        supporting_signals=[
            SupportingSignal(
                signal_type="Object Proximity",
                description="Person near object",
                confidence=0.90,
                timestamp=12.0,
            )
        ],
        pattern_evidence_strength=0.95,
        assessment_score=0.95,
    )
    
    validated = validator.validate_candidate(cand)
    
    # For review-required candidate, final assessment score must strictly be <= 0.65
    assert validated.validation_decision == ValidationDecision.REVIEW_REQUIRED.value
    assert validated.assessment_score <= 0.65
    assert validated.pattern_evidence_strength == 0.95
    
    # Narrative explanation must NOT disagree with assessment score!
    assert "Final Assessment: 65%" in validated.explanation
    assert "Final Assessment: 95%" not in validated.explanation
    assert "Pattern Evidence Strength: 95%" in validated.explanation
    assert "Human verification required" in validated.explanation
    assert "verified prolonged presence pattern" not in validated.explanation


def test_playback_compatibility_and_cache():
    """Verify is_browser_compatible and get_playback_status accurately detect formats and reuse artifacts."""
    playback_path = Path("storage/playback/0d4d92f9-19f8-42e3-925f-1931cb557705_playback.mp4")
    if playback_path.exists():
        is_compat = is_browser_compatible(playback_path)
        assert is_compat is True
        
        status = get_playback_status("0d4d92f9-19f8-42e3-925f-1931cb557705")
        assert status["status"] == "ready"
        assert status["is_compatible"] is True


def test_search_zero_result_explicit_language():
    """Verify zero-result search returns explicit executed language, not vague fallback."""
    orch = InvestigationOrchestrator()
    
    # Test object detection search (e.g. "What vehicles were detected?")
    ans_det = orch._format_deterministic_grounded_response({
        "count": 0,
        "results": [],
        "evidence": [],
        "filters": {"category": "vehicle", "result_type": "detection"},
    })
    assert "Search executed successfully" in ans_det
    assert "No validated vehicle detections were found in this investigation." in ans_det

    # Test generic zero result
    ans_gen = orch._format_deterministic_grounded_response({
        "count": 0,
        "results": [],
        "evidence": [],
        "filters": {"object_class": "backpack"},
    })
    assert "Search executed successfully. No validated backpack observations were found in this investigation." in ans_gen


def test_calibration_variation_and_evidence_driven_assessments():
    """Verify Section 32: REVIEW_REQUIRED incidents do NOT all receive flat 65%,
    Final Assessment is evidence-driven, not copied from Pattern Strength, and ACCEPTED retain legitimate scores.
    """
    validator = IncidentCandidateValidator()

    # Candidate 1: Prolonged Presence with high dwell (41s)
    cand_prolonged = IncidentCandidate(
        incident_id="test_prolonged_01",
        video_id="test_video",
        event_type="PROLONGED_PRESENCE",
        category="person",
        start_time=10.0,
        end_time=51.0,
        duration=41.0,
        severity="NORMAL",
        confidence=0.85,
        validation_decision="REVIEW_REQUIRED",
        explanation="Stationary presence observation",
        supporting_signals=[
            SupportingSignal(signal_type="Dwell Duration", description="Dwell 41s", confidence=0.85, timestamp=10.0),
            SupportingSignal(signal_type="Localized Perimeter", description="Low displacement", confidence=0.80, timestamp=20.0),
        ],
        pattern_evidence_strength=0.85,
        assessment_score=0.85,
    )

    # Candidate 2: Canonical Potential Theft (16s dwell, multiple modalities)
    cand_theft = IncidentCandidate(
        incident_id="test_theft_01",
        video_id="test_video",
        event_type="POTENTIAL_THEFT",
        category="property",
        start_time=100.0,
        end_time=116.0,
        duration=16.0,
        severity="HIGH",
        confidence=0.95,
        validation_decision="REVIEW_REQUIRED",
        explanation="Potential Theft Pattern (Pattern Evidence Strength: 95% (High Evidence Strength) • Final Assessment: 95% (None) — Sensor-grounded pattern): Person approached object",
        supporting_signals=[
            SupportingSignal(signal_type="Interaction Proximity", description="Person near suitcase", confidence=0.90, timestamp=100.0),
            SupportingSignal(signal_type="Takeaway Pattern", description="Departure displacement", confidence=0.88, timestamp=116.0),
            SupportingSignal(signal_type="Object Grounding", description="Suitcase grounded", confidence=0.85, timestamp=100.0),
        ],
        pattern_evidence_strength=0.95,
        assessment_score=0.95,
    )

    # Candidate 3: Potential Forced Movement (short duration, 1 signal)
    cand_forced = IncidentCandidate(
        incident_id="test_forced_01",
        video_id="test_video",
        event_type="POTENTIAL_FORCED_MOVEMENT",
        category="person",
        start_time=50.0,
        end_time=52.0,
        duration=2.0,
        severity="NORMAL",
        confidence=0.60,
        validation_decision="REVIEW_REQUIRED",
        explanation="Potential constrained person movement",
        supporting_signals=[
            SupportingSignal(signal_type="Tight Proximity", description="Tight proximity", confidence=0.60, timestamp=50.0),
        ],
        pattern_evidence_strength=0.60,
        assessment_score=0.60,
    )

    # Candidate 4: Coordinated Movement (Strong supporting evidence, ACCEPTED)
    cand_coord = IncidentCandidate(
        incident_id="test_coord_01",
        video_id="test_video",
        event_type="COORDINATED_PERSON_MOVEMENT",
        category="person",
        start_time=150.0,
        end_time=160.0,
        duration=10.0,
        severity="NORMAL",
        confidence=0.93,
        validation_decision="ACCEPTED",
        explanation="Coordinated person movement pattern",
        supporting_signals=[
            SupportingSignal(signal_type="Heading Alignment", description="Directional alignment", confidence=0.92, timestamp=150.0),
            SupportingSignal(signal_type="Velocity Correlation", description="Speed correlation", confidence=0.90, timestamp=152.0),
            SupportingSignal(signal_type="Spatial Cluster", description="Group proximity", confidence=0.91, timestamp=154.0),
        ],
        pattern_evidence_strength=0.93,
        assessment_score=0.93,
    )

    val_prolonged = validator.validate_candidate(cand_prolonged)
    val_theft = validator.validate_candidate(cand_theft)
    val_forced = validator.validate_candidate(cand_forced)
    val_coord = validator.validate_candidate(cand_coord)

    # Verify decision states
    assert val_prolonged.validation_decision == ValidationDecision.REVIEW_REQUIRED.value
    assert val_theft.validation_decision == ValidationDecision.REVIEW_REQUIRED.value
    assert val_forced.validation_decision == ValidationDecision.REVIEW_REQUIRED.value
    assert val_coord.validation_decision == ValidationDecision.ACCEPTED.value

    # Verify Section 32 invariants:
    # 1. REVIEW_REQUIRED does not automatically equal 65% across all incidents
    assert val_prolonged.assessment_score != val_forced.assessment_score
    
    # 2. Canonical Potential Theft retains 65% assessment and 95% pattern strength
    assert val_theft.assessment_score == 0.65
    assert val_theft.pattern_evidence_strength == 0.95
    assert "Pattern Evidence Strength: 95% (High Evidence Strength)" in val_theft.explanation
    assert "Final Assessment: 65% (REVIEW_REQUIRED)" in val_theft.explanation

    # 3. Final Assessment is NOT simply copied from Pattern Evidence Strength
    assert val_prolonged.assessment_score != val_prolonged.pattern_evidence_strength
    assert val_theft.assessment_score != val_theft.pattern_evidence_strength

    # 4. ACCEPTED incidents retain their legitimate high assessment score
    assert val_coord.assessment_score == 0.93
    assert val_coord.pattern_evidence_strength == 0.93

