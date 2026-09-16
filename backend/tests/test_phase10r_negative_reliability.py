"""
Phase 10-R Dedicated Negative & False-Positive Prevention Test Suite

Validates that Sentinel's global reliability architecture correctly suppresses
false-positive incident alarms across:
1. Normal highway traffic (vehicles overtaking/passing in adjacent lanes).
2. Normal moving vehicles on roadways (suppression of false loitering).
3. Normal person/object non-theft interaction (object remains at rest).
4. Normal pedestrian transit in unconfigured or non-restricted zones.
5. Candidate Validator decisions (ACCEPTED, REVIEW_REQUIRED, REJECTED).
6. Evidence Eligibility Gate (suppression of ungrounded or refuted evidence).
"""
import pytest
from typing import List

from ai.schemas import TrackedObject, BoundingBox, ZoneDefinition
from ai.incidents.schemas import (
    IncidentCandidate,
    IncidentContext,
    SupportingSignal,
    EvidenceCandidate,
    SpatialContext,
    ValidationDecision,
)
from ai.incidents.validator import IncidentCandidateValidator
from ai.incidents.evidence_gate import EvidenceEligibilityGate
from ai.incidents.negative_evidence import NegativeEvidenceEngine
from ai.incidents.scene_context import SceneContextEngine
from ai.incidents.detectors.future_stubs import VehicleCollisionDetector
from ai.incidents.detectors.prolonged_presence import ProlongedPresenceDetector
from ai.incidents.detectors.theft_and_takeaway import TheftAndTakeawayDetector
from ai.incidents.detectors.zone_intrusion import ZoneIntrusionDetector
from ai.incidents.engine import IncidentIntelligenceEngine


def test_highway_traffic_negative_collision_abstention():
    """
    Scenario 1: Two vehicles travel along a highway in adjacent lanes.
    Vehicle 1 (overtaking) approaches Vehicle 2 at high relative speed.
    Centroid distance drops to 45px, 2D trajectories cross due to camera perspective.
    HOWEVER:
    - Bounding boxes never overlap (physical clearance maintained).
    - Both vehicles continue traveling post-convergence at speed (30px/s).
    - Trajectory headings are parallel (multi-lane highway).

    EXPECTED:
    VehicleCollisionDetector MUST ABSTAIN and return 0 collision candidates.
    """
    # Create Vehicle 1 (faster overtaking car in lane 1)
    v1_bboxes = [
        {"timestamp": 0.0, "bbox": {"x1": 100, "y1": 200, "x2": 150, "y2": 260}},
        {"timestamp": 1.0, "bbox": {"x1": 150, "y1": 200, "x2": 200, "y2": 260}},
        {"timestamp": 2.0, "bbox": {"x1": 200, "y1": 200, "x2": 250, "y2": 260}},  # Closest point to V2
        {"timestamp": 3.0, "bbox": {"x1": 260, "y1": 200, "x2": 310, "y2": 260}},  # Continues moving at speed
        {"timestamp": 4.0, "bbox": {"x1": 320, "y1": 200, "x2": 370, "y2": 260}},
    ]
    v1 = TrackedObject(
        track_id="veh-101",
        object_class="car",
        first_seen=0.0,
        last_seen=4.0,
        confidence=0.92,
        current_bbox=BoundingBox(x1=320, y1=200, x2=370, y2=260),
        trajectory=[
            (0.0, 125.0, 230.0),
            (1.0, 175.0, 230.0),
            (2.0, 225.0, 230.0),
            (3.0, 285.0, 230.0),
            (4.0, 345.0, 230.0),
        ],
        history_bboxes=v1_bboxes,

    )

    # Create Vehicle 2 (steady car in adjacent lane 2: y is 290, so clearance of 30px always exists)
    v2_bboxes = [
        {"timestamp": 0.0, "bbox": {"x1": 180, "y1": 290, "x2": 230, "y2": 350}},
        {"timestamp": 1.0, "bbox": {"x1": 200, "y1": 290, "x2": 250, "y2": 350}},
        {"timestamp": 2.0, "bbox": {"x1": 220, "y1": 290, "x2": 270, "y2": 350}},  # Simultaneous with V2 at t=2.0
        {"timestamp": 3.0, "bbox": {"x1": 240, "y1": 290, "x2": 290, "y2": 350}},  # Continues moving at speed
        {"timestamp": 4.0, "bbox": {"x1": 260, "y1": 290, "x2": 310, "y2": 350}},
    ]
    v2 = TrackedObject(
        track_id="veh-102",
        object_class="car",
        first_seen=0.0,
        last_seen=4.0,
        confidence=0.90,
        current_bbox=BoundingBox(x1=260, y1=290, x2=310, y2=350),
        trajectory=[
            (0.0, 205.0, 320.0),
            (1.0, 225.0, 320.0),
            (2.0, 245.0, 320.0),
            (3.0, 265.0, 320.0),
            (4.0, 285.0, 320.0),
        ],
        history_bboxes=v2_bboxes,

    )

    engine = IncidentIntelligenceEngine()
    result = engine.analyze_incidents(
        video_id="negative_highway_traffic",
        tracks=[v1, v2],
        validated_detections=[],
        fps=30.0,
        duration_seconds=5.0,
    )

    collision_events = [e for e in result["security_events"] if "COLLISION" in e.event_type]
    assert len(collision_events) == 0, f"False collision generated on normal highway traffic: {collision_events}"
    assert result["diagnostics"]["scene_context"]["scene_type"] == "roadway"


def test_normal_road_transit_suppresses_false_loitering():
    """
    Scenario 2: Moving car on roadway travels continuously with high velocity.
    Must NOT trigger false PROLONGED_PRESENCE / loitering.
    """
    v_bboxes = [
        {"timestamp": float(i), "bbox": {"x1": i * 40, "y1": 100, "x2": i * 40 + 50, "y2": 150}}
        for i in range(10)
    ]
    car_track = TrackedObject(
        track_id="car-transit-1",
        object_class="car",
        first_seen=0.0,
        last_seen=9.0,
        confidence=0.95,
        current_bbox=BoundingBox(x1=360, y1=100, x2=410, y2=150),
        trajectory=[(float(i), i * 40.0 + 25.0, 125.0) for i in range(10)],
        history_bboxes=v_bboxes,

    )

    engine = IncidentIntelligenceEngine()
    result = engine.analyze_incidents(
        video_id="car_transit_video",
        tracks=[car_track],
        validated_detections=[],
        fps=30.0,
        duration_seconds=10.0,
    )

    loitering_events = [e for e in result["security_events"] if "PROLONGED" in e.event_type]
    assert len(loitering_events) == 0, "Normal moving vehicle falsely flagged as prolonged presence"


def test_normal_interaction_object_remains_suppresses_false_theft():
    """
    Scenario 3: Person approaches a suitcase, stands near it, and then walks away.
    The suitcase remains at rest at its original coordinates.
    Must NOT trigger POTENTIAL_THEFT.
    """
    # Suitcase stationary at (150, 150)
    s_bboxes = [
        {"timestamp": float(t), "bbox": {"x1": 130, "y1": 130, "x2": 170, "y2": 170}}
        for t in range(12)
    ]
    suitcase = TrackedObject(
        track_id="bag-1",
        object_class="suitcase",
        first_seen=0.0,
        last_seen=11.0,
        confidence=0.88,
        current_bbox=BoundingBox(x1=130, y1=130, x2=170, y2=170),
        trajectory=[(float(t), 150.0, 150.0) for t in range(12)],
        history_bboxes=s_bboxes,

    )

    # Person approaches at t=2, stays near until t=5, departs to (400, 400) at t=9
    p_trajectory = [
        (0.0, 50.0, 50.0),
        (2.0, 140.0, 140.0),  # Close to bag
        (3.0, 145.0, 145.0),
        (4.0, 150.0, 150.0),
        (5.0, 155.0, 155.0),
        (7.0, 280.0, 280.0),  # Walking away
        (9.0, 400.0, 400.0),  # Departed
    ]
    p_bboxes = [
        {"timestamp": pt[0], "bbox": {"x1": pt[1] - 20, "y1": pt[2] - 40, "x2": pt[1] + 20, "y2": pt[2] + 40}}
        for pt in p_trajectory
    ]
    person = TrackedObject(
        track_id="person-1",
        object_class="person",
        first_seen=0.0,
        last_seen=9.0,
        confidence=0.92,
        current_bbox=BoundingBox(x1=380, y1=360, x2=420, y2=440),
        trajectory=p_trajectory,
        history_bboxes=p_bboxes,

    )

    detector = TheftAndTakeawayDetector()
    context = IncidentContext(
        video_id="false_theft_test",
        tracks=[person, suitcase],
    )
    candidates = detector.analyze(context)
    # The negative evidence engine detects that suitcase remains present at rest -> 0 theft candidates
    assert len(candidates) == 0, f"False theft candidate generated when object was not taken: {candidates}"


def test_unconfigured_zone_suppresses_intrusion():
    """
    Scenario 4: Zone is configured with empty polygon or disabled.
    Must NOT generate POTENTIAL_INTRUSION.
    """
    person = TrackedObject(
        track_id="p-walk-1",
        object_class="person",
        first_seen=1.0,
        last_seen=6.0,
        confidence=0.90,
        current_bbox=BoundingBox(x1=50, y1=50, x2=90, y2=130),
        trajectory=[(1.0, 70.0, 90.0), (3.0, 75.0, 95.0), (6.0, 80.0, 100.0)],

    )
    unconfigured_zone = ZoneDefinition(
        zone_id="zone-disabled",
        name="Disabled Zone",
        polygon=[],  # Empty polygon
        enabled=False,
    )

    detector = ZoneIntrusionDetector()
    context = IncidentContext(
        video_id="zone_test",
        tracks=[person],
        zones=[unconfigured_zone],
    )
    candidates = detector.analyze(context)
    assert len(candidates) == 0, "Intrusion candidate generated on disabled/unconfigured zone"


def test_candidate_validator_decision_matrix():
    """
    Scenario 5: Test IncidentCandidateValidator decisions (ACCEPTED, REVIEW_REQUIRED, REJECTED).
    """
    validator = IncidentCandidateValidator()

    # 1. Invalid timestamps -> REJECTED
    bad_cand = IncidentCandidate(
        incident_id="BAD-1",
        video_id="vid_1",
        event_type="TEST_EVENT",
        category="general",
        start_time=5.0,
        end_time=2.0,  # Invalid: end < start
        duration=-3.0,
        severity="NORMAL",
        confidence=0.80,
    )
    val_bad = validator.validate_candidate(bad_cand)
    assert val_bad.validation_decision == ValidationDecision.REJECTED.value
    assert any("Invalid temporal interval" in r for r in val_bad.validation_reasons)

    # 2. Severe contradictory signals -> REJECTED
    refuted_cand = IncidentCandidate(
        incident_id="REFUTED-1",
        video_id="vid_1",
        event_type="POTENTIAL_VEHICLE_COLLISION",
        category="vehicle",
        start_time=2.0,
        end_time=4.0,
        duration=2.0,
        severity="HIGH",
        confidence=0.85,
        supporting_signals=[
            SupportingSignal(signal_type="Convergence", description="Approached", confidence=0.8)
        ],
        contradictory_signals=[
            SupportingSignal(
                signal_type="zero_physical_overlap",
                description="Zero contact between vehicles",
                confidence=0.95,
            ),
            SupportingSignal(
                signal_type="continued_normal_transit",
                description="Both vehicles continued at speed",
                confidence=0.95,
            ),
        ],
    )
    val_refuted = validator.validate_candidate(refuted_cand)
    assert val_refuted.validation_decision == ValidationDecision.REJECTED.value

    # 3. Clean supporting evidence, zero contradictions -> ACCEPTED
    good_cand = IncidentCandidate(
        incident_id="GOOD-1",
        video_id="vid_1",
        event_type="POTENTIAL_THEFT",
        category="property",
        start_time=1.0,
        end_time=5.0,
        duration=4.0,
        severity="HIGH",
        confidence=0.85,
        supporting_signals=[
            SupportingSignal(signal_type="Interaction", description="Proximity dwell", confidence=0.85),
            SupportingSignal(signal_type="Takeaway", description="Co-movement departure", confidence=0.90),
        ],
    )
    val_good = validator.validate_candidate(good_cand)
    assert val_good.validation_decision == ValidationDecision.ACCEPTED.value


def test_evidence_eligibility_gate():
    """
    Scenario 6: Test EvidenceEligibilityGate blocks REJECTED candidates and unaligned evidence.
    """
    gate = EvidenceEligibilityGate()

    # Rejected candidate cannot generate evidence
    rejected_cand = IncidentCandidate(
        incident_id="REJ-1",
        video_id="vid_1",
        event_type="TEST_EVENT",
        category="general",
        start_time=1.0,
        end_time=3.0,
        duration=2.0,
        severity="NORMAL",
        confidence=0.20,
        validation_decision=ValidationDecision.REJECTED.value,
        evidence_candidates=[EvidenceCandidate(timestamp=2.0, reason="Snapshot")],
    )
    marked = gate.filter_and_mark_candidates([rejected_cand])
    assert marked[0].evidence_eligible is False
    assert len(marked[0].evidence_candidates) == 0, "Evidence was not stripped from rejected candidate"

    # Evidence timestamp out of bounds
    out_of_bounds_cand = IncidentCandidate(
        incident_id="OOB-1",
        video_id="vid_1",
        event_type="TEST_EVENT",
        category="general",
        start_time=1.0,
        end_time=3.0,
        duration=2.0,
        severity="NORMAL",
        confidence=0.80,
        validation_decision=ValidationDecision.ACCEPTED.value,
        evidence_candidates=[EvidenceCandidate(timestamp=15.0, reason="Out of bounds snapshot")],
    )
    marked_oob = gate.filter_and_mark_candidates([out_of_bounds_cand])
    assert marked_oob[0].evidence_eligible is False
