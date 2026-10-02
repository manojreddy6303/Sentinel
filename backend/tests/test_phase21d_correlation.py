"""
Phase 21D: Sampling-Aware Incident Correlation Test Suite
=========================================================

Verifies:
1. Same continuous event at 1 FPS
2. Same continuous event at 3 FPS
3. Same event with irregular timestamps
4. Adjacent observations that should merge
5. Events separated enough to remain separate
6. Same track with dense observations
7. Different tracks occurring simultaneously
8. Same category but spatially separate events
9. Different categories
10. Occluded/reappearing track where existing semantics support continuity
11. Boundary conditions around the temporal merge threshold
12. No accidental incident multiplication caused only by higher sampling density
13. No over-merging of genuinely separate incidents
14. VFR timestamp handling
15. Legacy 1 FPS behavior preservation
"""
import pytest
from typing import List

from ai.incidents.schemas import (
    IncidentCandidate,
    ValidationDecision,
    SpatialContext,
    SupportingSignal,
)
from ai.schemas import BoundingBox, TrackedObject
from ai.correlation.engine import AdvancedIncidentCorrelationEngine
from ai.correlation.temporal_engine import CorrelationTemporalEngine
from ai.correlation.fusion_policies import PersonCorrelationPolicy, VehicleCorrelationPolicy
from ai.incidents.fusion import IncidentFusionEngine


def _make_candidate(
    cid: str,
    event_type: str,
    category: str,
    start_time: float,
    end_time: float,
    track_ids: List[str],
    confidence: float = 0.80,
    val_dec: str = "ACCEPTED",
    centroid: tuple = (100.0, 100.0),
) -> IncidentCandidate:
    bb = BoundingBox(
        x1=centroid[0] - 20,
        y1=centroid[1] - 40,
        x2=centroid[0] + 20,
        y2=centroid[1] + 40,
    )
    s_ctx = SpatialContext(centroid=centroid, bounding_box=bb)
    dur = max(0.0, end_time - start_time)
    sig = SupportingSignal(
        signal_type="Observation",
        description=f"{event_type} on {track_ids}",
        confidence=confidence,
        timestamp=start_time,
    )
    return IncidentCandidate(
        incident_id=cid,
        video_id="test_video",
        event_type=event_type,
        category=category,
        start_time=start_time,
        end_time=end_time,
        duration=dur,
        severity="NORMAL",
        confidence=confidence,
        track_ids=track_ids,
        object_classes=["person"] if category == "person" else ["car"],
        spatial_context=s_ctx,
        supporting_signals=[sig],
        validation_decision=val_dec,
    )


# ---------------------------------------------------------------------------
# Test 1 & 2: Same Continuous Event at 1 FPS vs 3 FPS (Density Invariance)
# ---------------------------------------------------------------------------
def test_density_invariance_1fps_vs_3fps():
    """10 observations over 3s must NOT produce more correlated incidents than 3 observations."""
    # 1 FPS: 3 observations over 3 seconds
    cands_1fps = [
        _make_candidate("C1_1", "PROLONGED_PRESENCE", "person", 0.0, 1.0, ["T1"]),
        _make_candidate("C1_2", "PROLONGED_PRESENCE", "person", 1.0, 2.0, ["T1"]),
        _make_candidate("C1_3", "PROLONGED_PRESENCE", "person", 2.0, 3.0, ["T1"]),
    ]

    # 3 FPS: 9 observations over 3 seconds
    cands_3fps = [
        _make_candidate(f"C3_{i}", "PROLONGED_PRESENCE", "person", i * 0.33, (i + 1) * 0.33, ["T1"])
        for i in range(9)
    ]

    engine = AdvancedIncidentCorrelationEngine(sampling_aware=True, temporal_cooldown_seconds=4.0)

    res_1 = engine.correlate_incidents("v1", cands_1fps, [])
    res_3 = engine.correlate_incidents("v3", cands_3fps, [])

    # Both must result in exactly ONE correlated incident
    assert len(res_1["correlated_incidents"]) == 1
    assert len(res_3["correlated_incidents"]) == 1
    assert res_1["correlated_incidents"][0].primary_track_ids == ["T1"]
    assert res_3["correlated_incidents"][0].primary_track_ids == ["T1"]


# ---------------------------------------------------------------------------
# Test 3 & 14: Irregular Timestamps and VFR Handling
# ---------------------------------------------------------------------------
def test_irregular_and_vfr_timestamps():
    """Correlator must handle variable frame rate intervals based on elapsed time."""
    vfr_cands = [
        _make_candidate("VFR_1", "PERSON_FOLLOWING", "person", 0.041, 0.412, ["P1", "P2"]),
        _make_candidate("VFR_2", "PERSON_FOLLOWING", "person", 0.985, 1.632, ["P1", "P2"]),
        _make_candidate("VFR_3", "PERSON_FOLLOWING", "person", 2.450, 3.120, ["P1", "P2"]),
    ]

    engine = AdvancedIncidentCorrelationEngine(sampling_aware=True, temporal_cooldown_seconds=4.0)
    res = engine.correlate_incidents("vfr_vid", vfr_cands, [])

    assert len(res["correlated_incidents"]) == 1
    ci = res["correlated_incidents"][0]
    assert ci.start_time == pytest.approx(0.041, abs=0.01)
    assert ci.end_time == pytest.approx(3.120, abs=0.01)


# ---------------------------------------------------------------------------
# Test 4: Adjacent Observations that Should Merge
# ---------------------------------------------------------------------------
def test_adjacent_observations_merge():
    """Adjacent observations on shared track within cooldown window must merge."""
    cands = [
        _make_candidate("ADJ_1", "POTENTIAL_FORCED_MOVEMENT", "person", 10.0, 14.0, ["T1", "T2"]),
        _make_candidate("ADJ_2", "COORDINATED_PERSON_MOVEMENT", "person", 15.5, 18.0, ["T1", "T2"]),
    ]
    engine = AdvancedIncidentCorrelationEngine(sampling_aware=True, temporal_cooldown_seconds=5.0)
    res = engine.correlate_incidents("adj_vid", cands, [])

    assert len(res["correlated_incidents"]) == 1
    ci = res["correlated_incidents"][0]
    assert ci.start_time == 10.0
    assert ci.end_time == 18.0
    assert set(ci.source_candidate_ids) == {"ADJ_1", "ADJ_2"}


# ---------------------------------------------------------------------------
# Test 5 & 13: Genuinely Separate Incidents Must Remain Separate (No Over-Merging)
# ---------------------------------------------------------------------------
def test_separated_events_remain_distinct():
    """Events separated by a meaningful temporal gap must NOT merge."""
    cands = [
        _make_candidate("SEP_1", "PROLONGED_PRESENCE", "person", 10.0, 15.0, ["T1"]),
        # Gap = 45.0 - 15.0 = 30.0s >> 5.0s cooldown
        _make_candidate("SEP_2", "PROLONGED_PRESENCE", "person", 45.0, 50.0, ["T1"]),
    ]
    engine = AdvancedIncidentCorrelationEngine(sampling_aware=True, temporal_cooldown_seconds=5.0)
    res = engine.correlate_incidents("sep_vid", cands, [])

    assert len(res["correlated_incidents"]) == 2
    assert res["correlated_incidents"][0].start_time == 10.0
    assert res["correlated_incidents"][1].start_time == 45.0


# ---------------------------------------------------------------------------
# Test 6: Same Track with Dense Observations
# ---------------------------------------------------------------------------
def test_same_track_dense_observations():
    """Dense stream of observations across same track must unify into single continuous narrative."""
    cands = [
        _make_candidate(f"DENSE_{i}", "PROLONGED_PRESENCE", "person", float(i), float(i + 1), ["T_DENSE"])
        for i in range(12)
    ]
    engine = AdvancedIncidentCorrelationEngine(sampling_aware=True, temporal_cooldown_seconds=4.0)
    res = engine.correlate_incidents("dense_vid", cands, [])

    assert len(res["correlated_incidents"]) == 1
    ci = res["correlated_incidents"][0]
    assert ci.start_time == 0.0
    assert ci.end_time == 12.0
    assert ci.duration == 12.0
    assert len(ci.source_candidate_ids) == 12


# ---------------------------------------------------------------------------
# Test 7: Different Tracks Occurring Simultaneously
# ---------------------------------------------------------------------------
def test_different_tracks_simultaneous_remain_separate():
    """Simultaneous events on different, spatially distant tracks must NOT merge."""
    c1 = _make_candidate("TRK_A", "PROLONGED_PRESENCE", "person", 10.0, 15.0, ["PERSON_A"], centroid=(50.0, 50.0))
    c2 = _make_candidate("TRK_B", "PROLONGED_PRESENCE", "person", 10.5, 15.5, ["PERSON_B"], centroid=(600.0, 600.0))

    engine = AdvancedIncidentCorrelationEngine(sampling_aware=True, temporal_cooldown_seconds=5.0)
    res = engine.correlate_incidents("sim_vid", [c1, c2], [])

    assert len(res["correlated_incidents"]) == 2
    tracks = {ci.primary_track_ids[0] for ci in res["correlated_incidents"]}
    assert tracks == {"PERSON_A", "PERSON_B"}


# ---------------------------------------------------------------------------
# Test 8: Same Category but Spatially Separate Events
# ---------------------------------------------------------------------------
def test_same_category_spatially_distant_remain_separate():
    """Events in same category with no shared track and large spatial distance must NOT merge."""
    c1 = _make_candidate("SPAT_1", "UNUSUAL_RAPID_PERSON_MOVEMENT", "person", 10.0, 12.0, ["RUNNER_EAST"], centroid=(100.0, 100.0))
    c2 = _make_candidate("SPAT_2", "UNUSUAL_RAPID_PERSON_MOVEMENT", "person", 11.0, 13.0, ["RUNNER_WEST"], centroid=(900.0, 800.0))

    engine = AdvancedIncidentCorrelationEngine(sampling_aware=True, temporal_cooldown_seconds=5.0)
    res = engine.correlate_incidents("spat_vid", [c1, c2], [])

    assert len(res["correlated_incidents"]) == 2


# ---------------------------------------------------------------------------
# Test 9: Different Categories Never Merge
# ---------------------------------------------------------------------------
def test_different_categories_never_merge():
    """Vehicle and Person candidates must NEVER merge together."""
    c_person = _make_candidate("P_CAND", "PROLONGED_PRESENCE", "person", 10.0, 15.0, ["P_TRK"])
    c_veh = _make_candidate("V_CAND", "POTENTIAL_NEAR_COLLISION", "vehicle", 10.0, 15.0, ["V_TRK"])

    engine = AdvancedIncidentCorrelationEngine(sampling_aware=True, temporal_cooldown_seconds=5.0)
    res = engine.correlate_incidents("cat_vid", [c_person, c_veh], [])

    assert len(res["correlated_incidents"]) == 2
    cats = {ci.incident_category for ci in res["correlated_incidents"]}
    assert cats == {"person", "vehicle"}


# ---------------------------------------------------------------------------
# Test 10: Occluded / Reappearing Track Continuity
# ---------------------------------------------------------------------------
def test_occluded_reappearing_track_continuity():
    """A track temporarily occluded for 3 seconds reappearing should maintain episode continuity."""
    c1 = _make_candidate("OCC_1", "PROLONGED_PRESENCE", "person", 10.0, 15.0, ["OCC_TRK"])
    # 3 second gap (e.g. walk behind pillar)
    c2 = _make_candidate("OCC_2", "PROLONGED_PRESENCE", "person", 18.0, 23.0, ["OCC_TRK"])

    engine = AdvancedIncidentCorrelationEngine(sampling_aware=True, temporal_cooldown_seconds=5.0)
    res = engine.correlate_incidents("occ_vid", [c1, c2], [])

    # Since gap is 3.0s <= 5.0s, they unify into a continuous episode
    assert len(res["correlated_incidents"]) == 1
    ci = res["correlated_incidents"][0]
    assert ci.start_time == 10.0
    assert ci.end_time == 23.0


# ---------------------------------------------------------------------------
# Test 11: Boundary Conditions Around Temporal Merge Threshold
# ---------------------------------------------------------------------------
def test_boundary_condition_temporal_threshold():
    """Verify exact boundary behavior: gap == threshold merges; gap > threshold separates."""
    cooldown = 5.0

    # Case A: Exactly on boundary (gap = 5.0s) -> should merge
    cA1 = _make_candidate("B1", "PROLONGED_PRESENCE", "person", 10.0, 15.0, ["TRK_BND"])
    cA2 = _make_candidate("B2", "PROLONGED_PRESENCE", "person", 20.0, 25.0, ["TRK_BND"])
    engine = AdvancedIncidentCorrelationEngine(sampling_aware=True, temporal_cooldown_seconds=cooldown)
    res_a = engine.correlate_incidents("bnd_a", [cA1, cA2], [])
    assert len(res_a["correlated_incidents"]) == 1

    # Case B: Just beyond boundary (gap = 5.1s) -> should separate
    cB1 = _make_candidate("B1", "PROLONGED_PRESENCE", "person", 10.0, 15.0, ["TRK_BND"])
    cB2 = _make_candidate("B2", "PROLONGED_PRESENCE", "person", 20.1, 25.0, ["TRK_BND"])
    res_b = engine.correlate_incidents("bnd_b", [cB1, cB2], [])
    assert len(res_b["correlated_incidents"]) == 2


# ---------------------------------------------------------------------------
# Test 12: No Accidental Incident Multiplication
# ---------------------------------------------------------------------------
def test_no_accidental_incident_multiplication():
    """Continuous 60-second activity at 3 FPS must NOT produce 20+ fragmented incidents."""
    cands_dense = []
    # Suspect active from t=10s to t=50s
    for sec in range(10, 50, 2):
        cands_dense.append(
            _make_candidate(f"SEG_{sec}", "COORDINATED_PERSON_MOVEMENT", "person", float(sec), float(sec + 3), ["SUSPECT_1", "SUSPECT_2"])
        )

    # In legacy mode (where person events don't cluster in other_person):
    legacy_engine = AdvancedIncidentCorrelationEngine(sampling_aware=False)
    res_legacy = legacy_engine.correlate_incidents("mult_vid", cands_dense, [])

    # In sampling-aware mode:
    aware_engine = AdvancedIncidentCorrelationEngine(sampling_aware=True, temporal_cooldown_seconds=5.0)
    res_aware = aware_engine.correlate_incidents("mult_vid", cands_dense, [])

    # Legacy creates 20 separate incidents
    assert len(res_legacy["correlated_incidents"]) == 20
    # Sampling-aware unifies into 1 continuous episode
    assert len(res_aware["correlated_incidents"]) == 1
    assert res_aware["correlated_incidents"][0].duration == 41.0


# ---------------------------------------------------------------------------
# Test 15: Legacy 1 FPS Behavior Preservation
# ---------------------------------------------------------------------------
def test_legacy_behavior_preservation_when_disabled():
    """When sampling_aware=False, exact 1-to-1 standalone candidate wrapping is preserved."""
    cands = [
        _make_candidate("L1", "PROLONGED_PRESENCE", "person", 10.0, 15.0, ["T1"]),
        _make_candidate("L2", "PROLONGED_PRESENCE", "person", 12.0, 18.0, ["T1"]),
    ]

    legacy_engine = AdvancedIncidentCorrelationEngine(sampling_aware=False)
    res = legacy_engine.correlate_incidents("leg_vid", cands, [])

    # In legacy mode, other_person candidates are emitted 1-to-1
    assert len(res["correlated_incidents"]) == 2
    assert res["correlated_incidents"][0].source_candidate_ids == ["L1"]
    assert res["correlated_incidents"][1].source_candidate_ids == ["L2"]


# ---------------------------------------------------------------------------
# Test 16: Validation Decision Invariant (Never upgrade REVIEW_REQUIRED)
# ---------------------------------------------------------------------------
def test_validation_decision_never_upgraded():
    """Correlation must NEVER upgrade a REVIEW_REQUIRED decision to ACCEPTED."""
    c_accepted = _make_candidate("C_ACC", "PROLONGED_PRESENCE", "person", 10.0, 14.0, ["T1"], val_dec="ACCEPTED")
    c_review = _make_candidate("C_REV", "PERSON_FOLLOWING", "person", 13.0, 17.0, ["T1", "T2"], val_dec="REVIEW_REQUIRED")

    engine = AdvancedIncidentCorrelationEngine(sampling_aware=True, temporal_cooldown_seconds=5.0)
    res = engine.correlate_incidents("val_vid", [c_accepted, c_review], [])

    assert len(res["correlated_incidents"]) == 1
    ci = res["correlated_incidents"][0]
    assert ci.validation_decision == "REVIEW_REQUIRED"
    assert ci.assessment_score <= 0.65
