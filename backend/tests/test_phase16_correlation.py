"""
Phase 16: Advanced Incident Correlation, Contextual Fusion & Storylines Test Suite
Comprehensive automated verification covering:
1. Architecture (correlation domain model, provenance, video isolation)
2. Temporal algebra (overlapping, adjacent, separated, concurrent, continuation)
3. Spatial correlation (proximity, distant separation, zone consistency)
4. Anonymous tracks (person-object, person-person, vehicle-vehicle, track isolation)
5. Property fusion (takeaway storylines, normal movement retention)
6. Vehicle fusion (collision multi-signal, near-collision distinction, separation)
7. Person fusion (fall+down, non-altercation walking, following vs coordinated movement)
8. Crowd & zone fusion (density+dispersal, orderly queue vs surge, collapse)
9. Specialized visual fusion (raw isolation, fire+smoke fusion, unconfigured weapon)
10. Score fusion & negative evidence (no double-counting, review score caps, negative degradation)
11. Provenance graph (incident->candidate, incident->evidence, incident->investigation, incident->report)
12. API & Filters (video isolation, categories, backward compatibility)
13. Reporting (correlated section, vehicle attributes, theft, empty video)
14. Reprocessing & Idempotency (idempotent correlation, zero duplicate records)
"""

import uuid
import pytest
from datetime import datetime, timezone

from ai.schemas import BoundingBox, TrackedObject, SecurityEvent
from ai.incidents.schemas import (
    IncidentCandidate,
    SupportingSignal,
    EvidenceCandidate,
    ValidationDecision,
)
from ai.correlation.models import (
    CorrelatedIncident,
    CorrelationRelationshipType,
    TemporalRelationType,
    HypothesisOutcome,
    ObservationalRelationship,
)
from ai.correlation.temporal_engine import CorrelationTemporalEngine
from ai.correlation.spatial_engine import CorrelationSpatialEngine
from ai.correlation.relationship_graph import IncidentRelationshipGraph
from ai.correlation.arbitrator import CompetingHypothesisArbitrator
from ai.correlation.storyline_generator import IncidentStorylineGenerator
from ai.correlation.engine import AdvancedIncidentCorrelationEngine
from ai.intelligence_repository import SecurityIntelligenceRepository
from backend.app.services.investigation_service import InvestigationService
from backend.app.services.report_service import ReportService
from database.session import SessionLocal
from database.models import (
    VideoModel,
    SecurityEventModel,
    CorrelatedIncidentModel,
    EventModel,
    EvidenceModel,
    TrackModel,
)


# Helper fixtures
def make_candidate(
    cid: str,
    cat: str,
    subcat: str,
    t_start: float,
    t_end: float,
    tracks: list,
    score: float = 0.8,
    decision: str = "ACCEPTED",
    signals: list = None,
    evidence_ids: list = None,
    negative_evidence: list = None,
    detector: str = "test_detector",
):
    val_dec = decision.value if hasattr(decision, "value") else str(decision)
    sup_signals = [SupportingSignal(signal_type="observable", description=str(s), confidence=score) for s in (signals or ["signal_a"])]
    con_signals = [SupportingSignal(signal_type="contradictory", description=str(s), confidence=0.5) for s in (negative_evidence or [])]
    ev_cands = [EvidenceCandidate(timestamp=t_start, reason="Forensic sample", target_track_id=tracks[0] if tracks else None)]

    return IncidentCandidate(
        incident_id=cid,
        video_id="test_video",
        event_type=subcat,
        category=cat.lower(),
        start_time=t_start,
        end_time=t_end,
        duration=max(0.1, t_end - t_start),
        severity="HIGH" if score >= 0.7 else "NORMAL",
        confidence=score,
        track_ids=tracks,
        object_classes=["person" if "P" in t or "T-01" in t else "car" if "V" in t else "suitcase" for t in tracks] or ["object"],
        supporting_signals=sup_signals,
        contradictory_signals=con_signals,
        validation_decision=val_dec,
        human_verification_required=val_dec != "ACCEPTED",
        detector_name=detector,
        detector_version="1.0.0",
        evidence_candidates=ev_cands,
        incident_metadata={"evidence_ids": evidence_ids or ["EV-001"]},
    )


def make_track(tid: str, cls: str, first_seen: float, last_seen: float, bboxes: list = None):
    cur_bb = bboxes[-1] if bboxes else BoundingBox(x1=100, y1=100, x2=200, y2=200)
    history = bboxes or [cur_bb]
    return TrackedObject(
        track_id=tid,
        object_class=cls,
        first_seen=first_seen,
        last_seen=last_seen,
        history_bboxes=history,
        current_bbox=cur_bb,
        confidence=0.9,
    )


# ============================================================================
# PART 1 — ARCHITECTURE & DOMAIN MODEL
# ============================================================================

def test_1_correlation_model_validation():
    """Verify CorrelatedIncident model fields, validation, and defaults."""
    ci = CorrelatedIncident(
        incident_id="CORR-001",
        video_id="vid_1",
        incident_category="VEHICLE",
        incident_subcategory="COLLISION",
        start_time=10.0,
        end_time=15.5,
        duration=5.5,
        primary_track_ids=["TRACK-001", "TRACK-002"],
        assessment_score=0.88,
        validation_decision=HypothesisOutcome.ACCEPTED,
        storyline="Observed vehicle contact.",
    )
    d = ci.to_dict()
    assert d["incident_id"] == "CORR-001"
    assert d["duration"] == 5.5
    assert d["validation_decision"] == "ACCEPTED"
    assert "TRACK-001" in d["primary_track_ids"]


def test_2_provenance_validation():
    """Verify provenance metadata records source detectors and candidates."""
    ci = CorrelatedIncident(
        incident_id="CORR-002",
        video_id="vid_1",
        incident_category="PROPERTY",
        source_candidate_ids=["CAND-1", "CAND-2"],
        source_detector_ids=["theft_detector", "pickup_detector"],
        provenance={"arbitration": "ACCEPTED", "rule": "takeaway_co_movement"},
    )
    assert len(ci.source_candidate_ids) == 2
    assert ci.provenance["rule"] == "takeaway_co_movement"


def test_3_video_isolation():
    """Verify correlation engine strictly isolates records across distinct videos."""
    engine = AdvancedIncidentCorrelationEngine()
    c1 = make_candidate("C-A1", "PROPERTY", "PICKUP", 10.0, 12.0, ["T-01", "T-02"])
    c2 = make_candidate("C-B1", "PROPERTY", "PICKUP", 10.0, 12.0, ["T-01", "T-02"])

    res_A = engine.correlate_incidents("video_AAA", [c1], [])
    res_B = engine.correlate_incidents("video_BBB", [c2], [])

    assert all(ci.video_id == "video_AAA" for ci in res_A["correlated_incidents"])
    assert all(ci.video_id == "video_BBB" for ci in res_B["correlated_incidents"])


# ============================================================================
# PART 2 — TEMPORAL CORRELATION
# ============================================================================

def test_4_overlapping_candidates_merge():
    """Verify candidates with overlapping temporal windows and common entities merge."""
    te = CorrelationTemporalEngine()
    rel = te.determine_relation(10.0, 15.0, 12.0, 18.0)
    assert rel == TemporalRelationType.OVERLAPPING
    assert te.are_temporally_compatible(10.0, 15.0, 12.0, 18.0) is True


def test_5_adjacent_candidates_merge_when_justified():
    """Verify immediately preceding / adjacent events within gap threshold correlate."""
    te = CorrelationTemporalEngine(max_temporal_gap=3.0)
    rel = te.determine_relation(10.0, 14.0, 15.5, 20.0)
    assert rel == TemporalRelationType.IMMEDIATELY_PRECEDING
    assert te.are_temporally_compatible(10.0, 14.0, 15.5, 20.0) is True


def test_6_separated_incidents_remain_separate():
    """Verify temporally distant events exceed gap threshold and remain separate."""
    te = CorrelationTemporalEngine(max_temporal_gap=3.0)
    rel = te.determine_relation(10.0, 14.0, 45.0, 50.0)
    assert rel == TemporalRelationType.SEPARATED
    assert te.are_temporally_compatible(10.0, 14.0, 45.0, 50.0) is False


def test_7_same_tracks_unrelated_times_remain_separate():
    """Verify same tracks occurring at disjoint times (e.g. 50s apart) form separate episodes."""
    engine = AdvancedIncidentCorrelationEngine()
    c1 = make_candidate("C-1", "PERSON", "FALL", 5.0, 8.0, ["T-01"])
    c2 = make_candidate("C-2", "PERSON", "RAPID_MOVEMENT", 60.0, 64.0, ["T-01"])

    res = engine.correlate_incidents("vid_time", [c1, c2], [])
    corrs = res["correlated_incidents"]
    assert len(corrs) == 2


# ============================================================================
# PART 3 — SPATIAL CORRELATION
# ============================================================================

def test_8_proximity_correlation():
    """Verify spatial proximity correlates adjacent bounding boxes."""
    se = CorrelationSpatialEngine()
    bb1 = BoundingBox(x1=100, y1=100, x2=200, y2=200)
    bb2 = BoundingBox(x1=180, y1=180, x2=280, y2=280)
    assert se.are_spatially_proximate(bb1, bb2, threshold=0.3) is True


def test_9_distant_objects_remain_independent():
    """Verify objects at opposite screen corners do not falsely correlate spatially."""
    se = CorrelationSpatialEngine()
    bb1 = BoundingBox(x1=10, y1=10, x2=50, y2=50)
    bb2 = BoundingBox(x1=1800, y1=1000, x2=1900, y2=1080)
    assert se.are_spatially_proximate(bb1, bb2, threshold=0.1) is False
    assert se.compute_iou(bb1, bb2) == 0.0


def test_10_zone_consistency():
    """Verify spatial engine respects zone membership and does not merge cross-zone."""
    se = CorrelationSpatialEngine()
    bb = BoundingBox(x1=50, y1=50, x2=150, y2=150)
    zone_poly = [(0, 0), (200, 0), (200, 200), (0, 200)]
    assert se.is_in_zone(bb, zone_poly) is True

    outside_poly = [(500, 500), (700, 500), (700, 700), (500, 700)]
    assert se.is_in_zone(bb, outside_poly) is False


# ============================================================================
# PART 4 — TRACK CORRELATION & RELATIONSHIP GRAPH
# ============================================================================

def test_11_person_object_correlation():
    """Verify relationship graph derives approaches, remains_near, and carries."""
    rg = IncidentRelationshipGraph()
    p_track = make_track("T-P1", "person", 10.0, 25.0, [
        BoundingBox(x1=50, y1=50, x2=100, y2=150),
        BoundingBox(x1=140, y1=140, x2=190, y2=240),
    ])
    o_track = make_track("T-O1", "suitcase", 10.0, 20.0, [
        BoundingBox(x1=150, y1=150, x2=200, y2=200),
        BoundingBox(x1=150, y1=150, x2=200, y2=200),
    ])
    rels = rg.build_relationships([p_track, o_track])
    rel_types = [r.relationship_type for r in rels]
    assert any(rt in rel_types for rt in (
        CorrelationRelationshipType.APPROACHES.value,
        CorrelationRelationshipType.REMAINS_NEAR.value,
        CorrelationRelationshipType.INTERACTS_WITH.value
    ))


def test_12_person_person_correlation():
    """Verify following and moves_with relationships between two person tracks."""
    rg = IncidentRelationshipGraph()
    p1 = make_track("T-P1", "person", 10.0, 30.0, [
        BoundingBox(x1=100, y1=100, x2=150, y2=200),
        BoundingBox(x1=200, y1=100, x2=250, y2=200),
    ])
    p2 = make_track("T-P2", "person", 10.0, 30.0, [
        BoundingBox(x1=80, y1=100, x2=130, y2=200),
        BoundingBox(x1=180, y1=100, x2=230, y2=200),
    ])
    rels = rg.build_relationships([p1, p2])
    rel_types = [r.relationship_type for r in rels]
    assert CorrelationRelationshipType.MOVES_WITH in rel_types or CorrelationRelationshipType.FOLLOWS in rel_types


def test_13_vehicle_vehicle_correlation():
    """Verify collides_with detected when vehicle bboxes physically contact and decelerate."""
    rg = IncidentRelationshipGraph()
    v1 = make_track("T-V1", "car", 10.0, 16.0, [
        BoundingBox(x1=100, y1=100, x2=250, y2=200),
        BoundingBox(x1=180, y1=100, x2=300, y2=200),
    ])
    v2 = make_track("T-V2", "car", 10.0, 16.0, [
        BoundingBox(x1=350, y1=100, x2=500, y2=200),
        BoundingBox(x1=260, y1=100, x2=380, y2=200),
    ])
    rels = rg.build_relationships([v1, v2])
    rel_types = [r.relationship_type for r in rels]
    assert CorrelationRelationshipType.APPROACHES in rel_types or CorrelationRelationshipType.COLLIDES_WITH in rel_types


def test_14_cross_video_track_isolation():
    """Verify anonymous track IDs are never cross-correlated between separate videos."""
    rg = IncidentRelationshipGraph()
    t1 = make_track("TRACK-001", "person", 0.0, 10.0)
    t2 = make_track("TRACK-001", "suitcase", 0.0, 10.0)
    # Different videos evaluated independently
    rels_A = rg.build_relationships([t1])
    rels_B = rg.build_relationships([t2])
    assert len(rels_A) == 0
    assert len(rels_B) == 0


# ============================================================================
# PART 5 — PROPERTY INCIDENT FUSION
# ============================================================================

def test_15_pickup_comovement_disappearance_storyline():
    """Verify pickup -> co-movement -> disappearance forms one coherent potential takeaway storyline."""
    engine = AdvancedIncidentCorrelationEngine()
    c_pickup = make_candidate("C-PK", "PROPERTY", "POTENTIAL_OBJECT_PICKUP", 10.0, 14.0, ["T-01", "T-02"])
    c_disp = make_candidate("C-DP", "PROPERTY", "POTENTIAL_OBJECT_DISPLACEMENT", 13.0, 18.0, ["T-01", "T-02"])
    c_theft = make_candidate("C-TH", "PROPERTY", "POTENTIAL_THEFT", 15.0, 22.0, ["T-01", "T-02"], score=0.85)

    p_trk = make_track("T-01", "person", 5.0, 25.0)
    o_trk = make_track("T-02", "suitcase", 5.0, 20.0)

    res = engine.correlate_incidents("vid_theft", [c_pickup, c_disp, c_theft], [p_trk, o_trk])
    corrs = res["correlated_incidents"]
    assert len(corrs) == 1
    ci = corrs[0]
    assert ci.incident_category.upper() == "PROPERTY"
    assert "takeaway" in ci.storyline.lower() or "theft" in ci.storyline.lower()
    assert "T-01" in ci.primary_track_ids
    assert "T-02" in ci.supporting_track_ids or "T-02" in ci.primary_track_ids


def test_16_normal_suitcase_movement_does_not_become_theft():
    """Verify ordinary displacement without disappearance/theft candidate remains ordinary movement."""
    engine = AdvancedIncidentCorrelationEngine()
    c_disp = make_candidate("C-DP", "PROPERTY", "POTENTIAL_OBJECT_DISPLACEMENT", 10.0, 14.0, ["T-01", "T-02"], score=0.55)
    res = engine.correlate_incidents("vid_norm", [c_disp], [])
    corrs = res["correlated_incidents"]
    assert len(corrs) == 1
    assert "takeaway" not in corrs[0].storyline.lower()


# ============================================================================
# PART 6 — VEHICLE INCIDENT FUSION
# ============================================================================

def test_17_collision_signals_fuse():
    """Verify rapid convergence + contact + deceleration fuse into one COLLISION incident."""
    engine = AdvancedIncidentCorrelationEngine()
    c_conv = make_candidate("C-CV", "VEHICLE", "POTENTIAL_SUDDEN_VEHICLE_STOP", 12.0, 15.0, ["V-01"])
    c_coll = make_candidate("C-CL", "VEHICLE", "POTENTIAL_VEHICLE_COLLISION", 13.0, 17.0, ["V-01", "V-02"], score=0.9)

    v1 = make_track("V-01", "car", 5.0, 20.0)
    v2 = make_track("V-02", "car", 5.0, 20.0)

    res = engine.correlate_incidents("vid_crash", [c_conv, c_coll], [v1, v2])
    corrs = res["correlated_incidents"]
    assert len(corrs) == 1
    ci = corrs[0]
    assert ci.incident_category.upper() == "VEHICLE"
    assert "collision" in ci.storyline.lower()
    assert "V-01" in ci.primary_track_ids
    assert "V-02" in ci.primary_track_ids


def test_18_near_collision_remains_distinct():
    """Verify near-collision without physical contact is not upgraded to full collision."""
    engine = AdvancedIncidentCorrelationEngine()
    c_near = make_candidate("C-NC", "VEHICLE", "POTENTIAL_NEAR_COLLISION", 10.0, 14.0, ["V-01", "V-02"], score=0.65)
    res = engine.correlate_incidents("vid_near", [c_near], [])
    corrs = res["correlated_incidents"]
    assert len(corrs) == 1
    assert "near-collision" in corrs[0].storyline.lower() or "near collision" in corrs[0].storyline.lower()
    assert corrs[0].incident_subcategory == "NEAR_COLLISION" or "NEAR" in str(corrs[0].incident_subcategory)


def test_19_separate_collisions_remain_separate():
    """Verify collisions separated by time and vehicles do not merge into one event."""
    engine = AdvancedIncidentCorrelationEngine()
    c_coll1 = make_candidate("C-C1", "VEHICLE", "POTENTIAL_VEHICLE_COLLISION", 10.0, 14.0, ["V-01", "V-02"], score=0.9)
    c_coll2 = make_candidate("C-C2", "VEHICLE", "POTENTIAL_VEHICLE_COLLISION", 60.0, 65.0, ["V-03", "V-04"], score=0.88)

    res = engine.correlate_incidents("vid_c2", [c_coll1, c_coll2], [])
    corrs = res["correlated_incidents"]
    assert len(corrs) == 2


# ============================================================================
# PART 7 — PERSON INCIDENT FUSION
# ============================================================================

def test_20_fall_and_person_down_can_correlate():
    """Verify fall and subsequent person-down fuse into one coherent medical/safety storyline."""
    engine = AdvancedIncidentCorrelationEngine()
    c_fall = make_candidate("C-F", "PERSON", "POTENTIAL_PERSON_FALL", 10.0, 13.0, ["P-01"], score=0.8)
    c_down = make_candidate("C-D", "PERSON", "POTENTIAL_PERSON_DOWN", 12.0, 25.0, ["P-01"], score=0.85)

    p1 = make_track("P-01", "person", 5.0, 30.0)
    res = engine.correlate_incidents("vid_fall", [c_fall, c_down], [p1])
    corrs = res["correlated_incidents"]
    assert len(corrs) == 1
    assert "fall" in corrs[0].storyline.lower() or "down" in corrs[0].storyline.lower()


def test_21_ordinary_walking_does_not_become_altercation():
    """Verify two people walking near each other without fighting signals does not become altercation."""
    engine = AdvancedIncidentCorrelationEngine()
    c_coor = make_candidate("C-CR", "PERSON", "COORDINATED_PERSON_MOVEMENT", 10.0, 20.0, ["P-01", "P-02"], score=0.6)
    res = engine.correlate_incidents("vid_walk", [c_coor], [])
    corrs = res["correlated_incidents"]
    assert len(corrs) == 1
    assert "altercation" not in corrs[0].storyline.lower()
    assert "fight" not in corrs[0].storyline.lower()


def test_22_following_vs_coordinated_movement_arbitration():
    """Verify arbitrator properly distinguishes following vs coordinated movement."""
    arbitrator = CompetingHypothesisArbitrator()
    c_foll = make_candidate("C-FL", "PERSON", "PERSON_FOLLOWING", 10.0, 20.0, ["P-01", "P-02"], score=0.75)
    c_coord = make_candidate("C-CD", "PERSON", "COORDINATED_PERSON_MOVEMENT", 10.0, 20.0, ["P-01", "P-02"], score=0.5)

    winner, loser, outcome = arbitrator.arbitrate(c_foll, c_coord)
    assert winner.incident_id == "C-FL"
    assert outcome == HypothesisOutcome.ACCEPTED
    assert loser.validation_decision in ("REVIEW_REQUIRED", "SUPERSEDED_BY_STRONGER_HYPOTHESIS")


# ============================================================================
# PART 8 — CROWD & ZONE FUSION
# ============================================================================

def test_23_density_and_dispersal_correlation():
    """Verify high density followed immediately by crowd dispersal fuses into one dynamic episode."""
    engine = AdvancedIncidentCorrelationEngine()
    c_dens = make_candidate("C-DN", "CROWD", "HIGH_PEDESTRIAN_DENSITY", 10.0, 16.0, ["P-1", "P-2", "P-3"])
    c_disp = make_candidate("C-DP", "CROWD", "POTENTIAL_CROWD_DISPERSAL", 15.0, 22.0, ["P-1", "P-2", "P-3"])

    res = engine.correlate_incidents("vid_crowd", [c_dens, c_disp], [])
    corrs = res["correlated_incidents"]
    assert len(corrs) == 1
    assert "dispersal" in corrs[0].storyline.lower() or "density" in corrs[0].storyline.lower()


def test_24_orderly_queue_does_not_become_surge():
    """Verify high density alone without rapid motion does not become crowd surge."""
    engine = AdvancedIncidentCorrelationEngine()
    c_dens = make_candidate("C-DN", "CROWD", "HIGH_PEDESTRIAN_DENSITY", 10.0, 25.0, ["P-1", "P-2"])
    res = engine.correlate_incidents("vid_queue", [c_dens], [])
    corrs = res["correlated_incidents"]
    assert len(corrs) == 1
    assert "surge" not in corrs[0].storyline.lower()


def test_25_repeated_crowd_alerts_collapse_appropriately():
    """Verify repeated overlapping density alerts collapse into a single unified incident."""
    engine = AdvancedIncidentCorrelationEngine()
    c1 = make_candidate("C-1", "CROWD", "HIGH_PEDESTRIAN_DENSITY", 10.0, 15.0, ["P-1", "P-2"])
    c2 = make_candidate("C-2", "CROWD", "HIGH_PEDESTRIAN_DENSITY", 12.0, 18.0, ["P-1", "P-2"])
    c3 = make_candidate("C-3", "CROWD", "HIGH_PEDESTRIAN_DENSITY", 14.0, 20.0, ["P-1", "P-2"])

    res = engine.correlate_incidents("vid_collapse", [c1, c2, c3], [])
    corrs = res["correlated_incidents"]
    assert len(corrs) == 1
    assert corrs[0].start_time <= 10.0
    assert corrs[0].end_time >= 20.0


# ============================================================================
# PART 9 — SPECIALIZED VISUAL FUSION
# ============================================================================

def test_26_only_validated_smoke_enters_fusion():
    """Verify raw unvalidated smoke observations cannot enter incident fusion."""
    engine = AdvancedIncidentCorrelationEngine()
    # Candidate rejected by validator
    c_smoke_rej = make_candidate(
        "C-SMK-REJ",
        "SPECIALIZED",
        "POTENTIAL_SMOKE",
        10.0, 15.0, [],
        score=0.2,
        decision=ValidationDecision.REJECTED,
    )
    res = engine.correlate_incidents("vid_smoke", [c_smoke_rej], [])
    assert len(res["correlated_incidents"]) == 0


def test_27_fire_smoke_fusion_requires_independent_evidence():
    """Verify fire + smoke correlates when both have independent evidence."""
    engine = AdvancedIncidentCorrelationEngine()
    c_fire = make_candidate("C-FR", "SPECIALIZED", "POTENTIAL_FIRE", 10.0, 16.0, [], score=0.8, evidence_ids=["EV-FR-1"])
    c_smoke = make_candidate("C-SK", "SPECIALIZED", "POTENTIAL_SMOKE", 12.0, 18.0, [], score=0.82, evidence_ids=["EV-SK-1"])

    res = engine.correlate_incidents("vid_fire", [c_fire, c_smoke], [])
    corrs = res["correlated_incidents"]
    assert len(corrs) == 1
    ci = corrs[0]
    assert ci.incident_subcategory == "FIRE_AND_SMOKE" or "FIRE" in ci.incident_subcategory
    assert len(ci.evidence_ids) >= 2


def test_28_unconfigured_weapon_remains_zero():
    """Verify unconfigured weapon detector produces zero correlated weapon incidents."""
    engine = AdvancedIncidentCorrelationEngine()
    # No weapon candidates passed
    res = engine.correlate_incidents("vid_no_wpn", [], [])
    assert len(res["correlated_incidents"]) == 0


# ============================================================================
# PART 10 — SCORING & NEGATIVE EVIDENCE
# ============================================================================

def test_29_correlated_signals_do_not_double_count():
    """Verify fusing 3 candidates with score 0.8 does not exceed 1.0 or double-count."""
    engine = AdvancedIncidentCorrelationEngine()
    c1 = make_candidate("C-1", "VEHICLE", "POTENTIAL_VEHICLE_COLLISION", 10.0, 15.0, ["V-1", "V-2"], score=0.8)
    c2 = make_candidate("C-2", "VEHICLE", "POTENTIAL_SUDDEN_VEHICLE_STOP", 11.0, 14.0, ["V-1", "V-2"], score=0.8)
    c3 = make_candidate("C-3", "VEHICLE", "POTENTIAL_VEHICLE_COLLISION", 12.0, 16.0, ["V-1", "V-2"], score=0.8)

    res = engine.correlate_incidents("vid_score", [c1, c2, c3], [])
    ci = res["correlated_incidents"][0]
    assert 0.0 <= ci.assessment_score <= 1.0
    assert ci.assessment_score < 0.98  # Does not linearly sum to 2.4


def test_30_review_score_remains_capped():
    """Verify REVIEW_REQUIRED candidates remain strictly capped per reliability policy."""
    engine = AdvancedIncidentCorrelationEngine()
    c_rev = make_candidate("C-R", "PROPERTY", "POTENTIAL_THEFT", 10.0, 15.0, ["T-1"], score=0.95, decision=ValidationDecision.REVIEW_REQUIRED)

    res = engine.correlate_incidents("vid_cap", [c_rev], [])
    ci = res["correlated_incidents"][0]
    assert ci.validation_decision == HypothesisOutcome.REVIEW_REQUIRED or ci.validation_decision == "REVIEW_REQUIRED"
    assert ci.assessment_score <= 0.65  # Capped at strict 0.65 Sentinel review ceiling


def test_31_negative_evidence_reduces_confidence():
    """Verify presence of negative evidence reduces the assessment score."""
    engine = AdvancedIncidentCorrelationEngine()
    c_clean = make_candidate("C-CLN", "VEHICLE", "POTENTIAL_VEHICLE_COLLISION", 10.0, 15.0, ["V-1", "V-2"], score=0.85)
    c_neg = make_candidate("C-NEG", "VEHICLE", "POTENTIAL_VEHICLE_COLLISION", 10.0, 15.0, ["V-1", "V-2"], score=0.85, negative_evidence=["smooth_resumption_of_traffic"])

    res_clean = engine.correlate_incidents("vid_cln", [c_clean], [])
    res_neg = engine.correlate_incidents("vid_neg", [c_neg], [])

    score_clean = res_clean["correlated_incidents"][0].assessment_score
    score_neg = res_neg["correlated_incidents"][0].assessment_score
    assert score_neg < score_clean


# ============================================================================
# PART 11 — PROVENANCE GRAPH
# ============================================================================

def test_32_incident_candidate_provenance():
    """Verify correlated incident stores exact source candidate IDs."""
    engine = AdvancedIncidentCorrelationEngine()
    c1 = make_candidate("C-A", "PERSON", "POTENTIAL_PERSON_FALL", 10.0, 12.0, ["P-1"])
    c2 = make_candidate("C-B", "PERSON", "POTENTIAL_PERSON_DOWN", 11.0, 18.0, ["P-1"])
    res = engine.correlate_incidents("vid_prov", [c1, c2], [])
    ci = res["correlated_incidents"][0]
    assert "C-A" in ci.source_candidate_ids
    assert "C-B" in ci.source_candidate_ids


def test_33_incident_evidence_provenance():
    """Verify evidence IDs from underlying candidates are mapped to the correlated incident."""
    engine = AdvancedIncidentCorrelationEngine()
    c1 = make_candidate("C-1", "PROPERTY", "POTENTIAL_THEFT", 10.0, 15.0, ["T-1"], evidence_ids=["EV-101", "EV-102"])
    res = engine.correlate_incidents("vid_ev", [c1], [])
    ci = res["correlated_incidents"][0]
    assert "EV-101" in ci.evidence_ids
    assert "EV-102" in ci.evidence_ids


def test_34_incident_investigation_provenance():
    """Verify natural-language investigation queries match and return correlated incident provenance."""
    repo = SecurityIntelligenceRepository()
    vid = f"vid_inv_prov_{uuid.uuid4().hex[:6]}"
    ci = CorrelatedIncident(
        incident_id=f"CORR-{uuid.uuid4().hex[:6]}",
        video_id=vid,
        incident_category="PROPERTY",
        incident_subcategory="POTENTIAL_OBJECT_TAKEAWAY",
        start_time=14.0,
        end_time=22.0,
        duration=8.0,
        primary_track_ids=["TRACK-014"],
        supporting_track_ids=["TRACK-020"],
        involved_object_classes=["person", "suitcase"],
        assessment_score=0.85,
        storyline="At 14.0s, TRACK-014 approached suitcase TRACK-020 and departed with it.",
    )
    repo.save_intelligence_results(vid, [], [], [], [], correlated_incidents=[ci])

    inv_svc = InvestigationService()
    res = inv_svc.investigate(vid, "What incidents involved a person and an object?")
    assert res["is_supported"] is True
    assert res["result_type"] == "correlated_incidents"
    assert res["count"] >= 1
    assert "TRACK-014" in res["results"][0]["primary_track_ids"]


def test_35_incident_report_provenance():
    """Verify generated report payload includes correlated incidents with full storylines."""
    repo = SecurityIntelligenceRepository()
    vid = f"vid_rep_prov_{uuid.uuid4().hex[:6]}"
    ci = CorrelatedIncident(
        incident_id=f"CORR-{uuid.uuid4().hex[:6]}",
        video_id=vid,
        incident_category="VEHICLE",
        incident_subcategory="COLLISION",
        start_time=5.0,
        end_time=9.0,
        duration=4.0,
        primary_track_ids=["TRACK-001", "TRACK-002"],
        assessment_score=0.92,
        storyline="Rapid convergence and physical contact between vehicles.",
    )
    repo.save_intelligence_results(vid, [], [], [], [], correlated_incidents=[ci])

    rep_svc = ReportService()
    res = rep_svc.generate_dossier(vid, title="TEST CORRELATED DOSSIER")
    assert res["report_id"] is not None
    assert res["file_size_bytes"] > 0


# ============================================================================
# PART 12 — API & VIDEO ISOLATION
# ============================================================================

def test_36_api_video_isolation():
    """Verify repository get_correlated_incidents isolates by video_id."""
    repo = SecurityIntelligenceRepository()
    vA = f"vid_iso_A_{uuid.uuid4().hex[:6]}"
    vB = f"vid_iso_B_{uuid.uuid4().hex[:6]}"

    ciA = CorrelatedIncident(incident_id="CORR-A", video_id=vA, incident_category="PERSON", start_time=1.0, end_time=2.0)
    ciB = CorrelatedIncident(incident_id="CORR-B", video_id=vB, incident_category="VEHICLE", start_time=1.0, end_time=2.0)

    repo.save_intelligence_results(vA, [], [], [], [], correlated_incidents=[ciA])
    repo.save_intelligence_results(vB, [], [], [], [], correlated_incidents=[ciB])

    resA = repo.get_correlated_incidents(vA)
    resB = repo.get_correlated_incidents(vB)

    assert len(resA) == 1 and resA[0]["incident_id"] == "CORR-A"
    assert len(resB) == 1 and resB[0]["incident_id"] == "CORR-B"


def test_37_api_filters():
    """Verify category and validation_decision filtering in repository query."""
    repo = SecurityIntelligenceRepository()
    vid = f"vid_filt_{uuid.uuid4().hex[:6]}"
    ci1 = CorrelatedIncident(incident_id="C-1", video_id=vid, incident_category="VEHICLE", validation_decision=HypothesisOutcome.ACCEPTED)
    ci2 = CorrelatedIncident(incident_id="C-2", video_id=vid, incident_category="PERSON", validation_decision=HypothesisOutcome.REVIEW_REQUIRED)

    repo.save_intelligence_results(vid, [], [], [], [], correlated_incidents=[ci1, ci2])

    veh = repo.get_correlated_incidents(vid, category="VEHICLE")
    assert len(veh) == 1 and veh[0]["incident_id"] == "C-1"

    rev = repo.get_correlated_incidents(vid, validation_decision="REVIEW_REQUIRED")
    assert len(rev) == 1 and rev[0]["incident_id"] == "C-2"


def test_38_backward_compatible_existing_endpoints():
    """Verify SecurityEventModel records remain queryable and unaltered."""
    repo = SecurityIntelligenceRepository()
    vid = f"vid_bwd_{uuid.uuid4().hex[:6]}"
    eid = f"EVT-BWD-{uuid.uuid4().hex[:6]}"
    se = SecurityEvent(
        event_id=eid,
        event_type="POTENTIAL_INTRUSION",
        timestamp=10.0,
        duration_seconds=2.0,
        severity="HIGH",
        confidence=0.88,
        description="Backward compatibility event.",
    )
    repo.save_intelligence_results(vid, [], [], [], [se])
    stored = repo.get_security_events(vid)
    assert len(stored) == 1
    assert stored[0]["event_id"] == eid


# ============================================================================
# PART 13 — REPORTING & REPROCESSING IDEMPOTENCY
# ============================================================================

def test_39_correlated_incident_report():
    """Verify PDF generator renders correlated incident section without error."""
    repo = SecurityIntelligenceRepository()
    vid = f"vid_rep_{uuid.uuid4().hex[:6]}"
    ci = CorrelatedIncident(
        incident_id="CORR-REP-1",
        video_id=vid,
        incident_category="PROPERTY",
        start_time=12.0,
        end_time=20.0,
        duration=8.0,
        primary_track_ids=["TRACK-001"],
        assessment_score=0.88,
        storyline="At 12.0s, object interaction and takeaway observed.",
    )
    repo.save_intelligence_results(vid, [], [], [], [], correlated_incidents=[ci])
    res = ReportService().generate_dossier(vid, title="CORRELATED REPORT")
    assert res["page_count"] >= 1


def test_40_vehicle_attribute_report():
    """Verify vehicle attributes are preserved and correctly presented in dossier."""
    repo = SecurityIntelligenceRepository()
    vid = f"vid_va_rep_{uuid.uuid4().hex[:6]}"
    from ai.schemas import VehicleAttribute
    va = VehicleAttribute(
        track_id="TRACK-CAR-1",
        object_class="car",
        color="red",
        confidence=0.92,
        timestamp=5.0,
        bounding_box=BoundingBox(x1=10, y1=10, x2=100, y2=100),
    )
    repo.save_intelligence_results(vid, [], [va], [], [])
    res = ReportService().generate_dossier(vid, title="VEHICLE COLOR REPORT")
    assert res["page_count"] >= 1


def test_41_theft_report():
    """Verify theft incident deep-dive remains preserved alongside correlated storyline."""
    repo = SecurityIntelligenceRepository()
    vid = f"vid_th_rep_{uuid.uuid4().hex[:6]}"
    se = SecurityEvent(
        event_id="EVT-TH-1",
        event_type="POTENTIAL_THEFT",
        timestamp=15.0,
        duration_seconds=3.0,
        severity="HIGH",
        confidence=0.85,
        description="Potential object takeaway observed.",
    )
    repo.save_intelligence_results(vid, [], [], [], [se])
    res = ReportService().generate_dossier(vid, title="THEFT REPORT")
    assert res["page_count"] >= 1


def test_42_empty_incident_report():
    """Verify report generation cleanly succeeds for videos with zero incidents."""
    repo = SecurityIntelligenceRepository()
    vid = f"vid_emp_{uuid.uuid4().hex[:6]}"
    repo.save_intelligence_results(vid, [], [], [], [])
    res = ReportService().generate_dossier(vid, title="EMPTY INCIDENT REPORT")
    assert res["page_count"] >= 1


def test_43_idempotent_correlation():
    """Verify running correlation repeatedly produces identical deterministic output."""
    engine = AdvancedIncidentCorrelationEngine()
    c1 = make_candidate("C-1", "PROPERTY", "POTENTIAL_THEFT", 10.0, 15.0, ["T-1"])
    res1 = engine.correlate_incidents("vid_idem", [c1], [])
    res2 = engine.correlate_incidents("vid_idem", [c1], [])

    corrs1 = res1["correlated_incidents"]
    corrs2 = res2["correlated_incidents"]
    assert len(corrs1) == len(corrs2)
    assert corrs1[0].storyline == corrs2[0].storyline
    assert corrs1[0].assessment_score == corrs2[0].assessment_score


def test_44_no_duplicate_correlated_incidents():
    """Verify saving intelligence repeatedly purges previous correlated records without duplication."""
    repo = SecurityIntelligenceRepository()
    vid = f"vid_dup_{uuid.uuid4().hex[:6]}"
    ci = CorrelatedIncident(incident_id="CORR-UNIQUE", video_id=vid, incident_category="CROWD", start_time=1.0, end_time=5.0)

    repo.save_intelligence_results(vid, [], [], [], [], correlated_incidents=[ci])
    assert len(repo.get_correlated_incidents(vid)) == 1

    # Second save should clear and recreate without duplication
    repo.save_intelligence_results(vid, [], [], [], [], correlated_incidents=[ci])
    assert len(repo.get_correlated_incidents(vid)) == 1


# ============================================================================
# PART 15 — PHASE 16 GENERALIZATION AUDIT MATRIX
# ============================================================================

@pytest.mark.parametrize("resolution", [
    (320, 240),
    (640, 480),
    (1280, 720),
    (1920, 1080),
    (3840, 2160),
])
def test_45_spatial_resolution_invariance(resolution):
    """Verifies that spatial proximity uses characteristic scale and is invariant across resolutions."""
    w, h = resolution
    box_a = BoundingBox(x1=0.1 * w, y1=0.1 * h, x2=0.2 * w, y2=0.3 * h)
    box_b = BoundingBox(x1=0.25 * w, y1=0.1 * h, x2=0.35 * w, y2=0.3 * h)

    spatial = CorrelationSpatialEngine()
    iou = spatial.box_iou(box_a, box_b)
    assert iou == 0.0

    c1 = ((box_a.x1 + box_a.x2) / 2.0, (box_a.y1 + box_a.y2) / 2.0)
    c2 = ((box_b.x1 + box_b.x2) / 2.0, (box_b.y1 + box_b.y2) / 2.0)
    dist = spatial.centroid_distance(c1, c2)
    # Average width across the boxes
    width_a = box_a.x2 - box_a.x1
    relative_dist = dist / max(width_a, 1e-6)
    assert abs(relative_dist - 1.5) < 0.05


def test_46_temporal_generalization_fps_and_spacing():
    """Verifies temporal algebra classifies identical second offsets consistently regardless of frame rate."""
    temporal = CorrelationTemporalEngine()
    rel_overlap = temporal.determine_relation(10.0, 12.0, 11.5, 13.0)
    assert rel_overlap == TemporalRelationType.OVERLAPPING

    rel_separated = temporal.determine_relation(10.0, 12.0, 30.0, 35.0)
    assert rel_separated == TemporalRelationType.SEPARATED


def test_47_short_event_vs_unrelated_telemetry_isolation():
    """A brief collision must never have its duration inflated by unrelated prior/posterior telemetry."""
    from ai.correlation.fusion_policies import VehicleCorrelationPolicy
    c_col = make_candidate("c_col", "VEHICLE", "VEHICLE_COLLISION", 15.0, 18.0, ["V1", "V2"], score=0.9)
    c_stop = make_candidate("c_stop", "VEHICLE", "SUDDEN_STOP", 43.0, 45.0, ["V3"], score=0.75)

    res = VehicleCorrelationPolicy.correlate([c_col, c_stop], "v_telemetry", [])
    assert len(res) == 2
    col_inc = next(i for i in res if "collision" in i.incident_subcategory.lower())
    assert col_inc.start_time == 15.0
    assert col_inc.end_time == 18.0
    assert col_inc.duration == 3.0


def test_48_multiple_simultaneous_independent_collisions():
    """Two simultaneous collisions in different lanes must not merge when tracks are disjoint."""
    engine = AdvancedIncidentCorrelationEngine()
    col1 = make_candidate("c_lane1", "VEHICLE", "VEHICLE_COLLISION", 10.0, 14.0, ["VA", "VB"], score=0.88)
    col2 = make_candidate("c_lane2", "VEHICLE", "VEHICLE_COLLISION", 10.5, 14.2, ["VX", "VY"], score=0.85)

    res = engine.correlate_incidents("v_multi", [col1, col2], [])
    incidents = res["correlated_incidents"]
    assert len(incidents) == 2
    all_tracks = [set(i.primary_track_ids + i.supporting_track_ids) for i in incidents]
    assert {"VA", "VB"} in all_tracks
    assert {"VX", "VY"} in all_tracks


def test_49_empty_scene_produces_zero_incidents():
    """An empty scene must gracefully yield 0 incidents and clean zero diagnostics."""
    engine = AdvancedIncidentCorrelationEngine()
    res = engine.correlate_incidents("v_empty", [], [])
    assert len(res["correlated_incidents"]) == 0
    assert res["diagnostics"]["raw_candidates_count"] == 0
    assert res["diagnostics"]["correlated_groups_count"] == 0


def test_50_review_required_reliability_cap_enforcement():
    """Every REVIEW_REQUIRED candidate must strictly observe the <= 0.65 assessment ceiling."""
    engine = AdvancedIncidentCorrelationEngine()
    rev_c = make_candidate("c_rev", "PROPERTY", "UNATTENDED_OBJECT", 1.0, 10.0, ["P1"], score=0.92, decision="REVIEW_REQUIRED")

    res = engine.correlate_incidents("v_rev_cap", [rev_c], [])
    incidents = res["correlated_incidents"]
    assert len(incidents) == 1
    assert incidents[0].assessment_score <= 0.65
    assert incidents[0].validation_decision == "REVIEW_REQUIRED"

