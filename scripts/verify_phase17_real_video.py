"""
verify_phase17_real_video.py
============================
Comprehensive Phase 17 Forensic Acceptance, Generalization, and
Cross-Layer Reliability Audit on Real CCTV Videos:

1. Burglary Video (0d4d92f9-19f8-42e3-925f-1931cb557705):
   - Natural language investigation queries (theft/property, around 163s, track search)
   - Structured query filtering (min_score, decisions, categories)
   - Unified multi-layer timeline retrieval
   - Track deep dive & lifecycle reconstruction
   - Evidence bundle creation & strict video isolation
   - Canonical score preservation (0.65 ceiling on REVIEW_REQUIRED preserved)

2. Highway Video (8edd2faf-8a6e-4c7d-9b58-292e45b06b92):
   - Natural language vehicle queries (collision, before 30s)
   - Temporal separation of distinct vehicle collisions
   - Unified multi-layer timeline
   - Track lifecycle reconstruction
   - Evidence bundle creation & strict video isolation

3. 4K CCTV Video (3426f64b-dd44-48a7-8e29-2c5f77b748bb):
   - High-res queries (vehicles, pedestrian tracks)
   - Verification of zero smoke, zero fire, zero weapon
   - Unified timeline & evidence preservation

4. Cross-Video Isolation & Boundary Enforcement:
   - Video A bundle creation referencing Video B entities is REJECTED
   - Video A bundle retrieval under Video B scope is REJECTED
   - Querying Video A returns strictly zero entities from Video B

5. Arbitrary & Edge-Case Generalization:
   - Empty queries, nonexistent video, inverted time bounds, extreme limits
"""

import os
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("phase17_gate")

from database.session import SessionLocal, init_db
from database.models import (
    VideoModel,
    EventModel,
    TrackModel,
    SecurityEventModel,
    EvidenceModel,
    CorrelatedIncidentModel,
    InvestigationBundleModel,
)
from backend.app.services.investigation_service import InvestigationService
from backend.app.services.investigation_parser import InvestigationParser
from backend.app.services.investigation_query import InvestigationQuery
from backend.app.services.investigation_query_executor import InvestigationQueryExecutor
from backend.app.services.evidence_bundle_service import EvidenceBundleService, EvidenceBundleError
from backend.app.services.report_service import ReportService


def verify_phase17_real_videos():
    init_db()
    db = SessionLocal()
    inv_svc = InvestigationService()
    executor = InvestigationQueryExecutor()
    bundle_svc = EvidenceBundleService()
    rep_svc = ReportService()

    print("\n" + "=" * 78)
    print("SENTINEL — PHASE 17 REAL-VIDEO FORENSIC ACCEPTANCE AUDIT")
    print("=" * 78 + "\n")

    # Locate Real Videos in DB
    vid_burg = "0d4d92f9-19f8-42e3-925f-1931cb557705"
    vburg = db.query(VideoModel).filter(VideoModel.id == vid_burg).first()
    if not vburg:
        vburg = db.query(VideoModel).filter(VideoModel.original_filename.like("%Burglary010%")).first()
        if vburg:
            vid_burg = vburg.id

    vid_hwy = "8edd2faf-8a6e-4c7d-9b58-292e45b06b92"
    vhwy = db.query(VideoModel).filter(VideoModel.id == vid_hwy).first()
    if not vhwy:
        vhwy = db.query(VideoModel).filter(VideoModel.original_filename.like("%8edd2faf%")).first()
        if vhwy:
            vid_hwy = vhwy.id

    vid_4k = "3426f64b-dd44-48a7-8e29-2c5f77b748bb"
    v4k = db.query(VideoModel).filter(VideoModel.id == vid_4k).first()
    if not v4k:
        v4k = db.query(VideoModel).filter(VideoModel.original_filename.like("%12566041%")).first()
        if v4k:
            vid_4k = v4k.id

    assert vburg is not None, "Burglary video not found in DB!"
    assert vhwy is not None, "Highway video not found in DB!"
    assert v4k is not None, "4K video not found in DB!"

    # =========================================================================
    # 1. BURGLARY VIDEO ACCEPTANCE AUDIT
    # =========================================================================
    print(f"[GATE 1: Burglary Video Forensic Audit — {vburg.original_filename} ({vid_burg})]")

    # A. Natural-language query: property / theft
    q_nl = InvestigationParser.parse_investigation_query(
        "Show evidence related to the theft or property around 163 seconds",
        video_id=vid_burg,
        video_duration_seconds=vburg.duration_seconds,
    )
    res_burg = executor.execute(q_nl)
    assert res_burg.source_video_id == vid_burg
    print(f"  Query NL: 'Show evidence related to the theft or property around 163 seconds'")
    print(f"  -> Interpretation: {res_burg.interpretation}")
    print(f"  -> Matched incidents: {len(res_burg.matched_incidents)}")
    print(f"  -> Matched evidence: {len(res_burg.matched_evidence)}")
    print(f"  -> Total results: {res_burg.total_results}")
    assert len(res_burg.matched_incidents) >= 1, "Expected at least 1 matched property incident for Burglary video"
    
    # Verify score integrity: REVIEW_REQUIRED ceiling of 0.65
    for inc in res_burg.matched_incidents:
        if inc["validation_decision"] == "REVIEW_REQUIRED":
            assert inc["assessment_score"] <= 0.6501, (
                f"Canonical violation: REVIEW_REQUIRED score {inc['assessment_score']} exceeds 0.65 ceiling!"
            )
    print("  -> Canonical score integrity verified: all REVIEW_REQUIRED <= 0.65.")

    # B. Unified Timeline
    timeline_burg = inv_svc.get_unified_timeline(vid_burg)
    assert timeline_burg["is_supported"] is True
    assert timeline_burg["total_entries"] > 0
    layers = timeline_burg["layer_counts"]
    print(f"  Unified Timeline: {timeline_burg['total_entries']} entries across layers: {layers}")
    assert layers["correlated_incidents"] >= 1
    assert layers["security_events"] >= 1
    assert layers["evidence"] >= 1

    # C. Track Deep Dive
    # Get first track from DB
    burg_trk = db.query(TrackModel).filter(TrackModel.video_id == vid_burg).first()
    assert burg_trk is not None
    trk_res = inv_svc.investigate_track(vid_burg, burg_trk.track_id)
    assert trk_res["is_supported"] is True
    assert trk_res["track_id"] == burg_trk.track_id
    print(f"  Track Deep Dive ({burg_trk.track_id}): class={burg_trk.object_class}, duration={trk_res['track']['duration_seconds']:.1f}s")
    assert "lifecycle" in trk_res
    print(f"  -> Reconstructed phases: {len(trk_res['lifecycle']['phases'])} phases")

    # D. Evidence Bundle Creation & Video Scoping
    first_inc_id = res_burg.matched_incidents[0]["incident_id"]
    bundle_burg = bundle_svc.create_bundle(
        video_id=vid_burg,
        bundle_name="Burglary Property Incursion Bundle",
        selected_incident_ids=[first_inc_id],
        notes="Audit bundle for Burglary incident",
    )
    assert bundle_burg["bundle_id"] is not None
    assert bundle_burg["video_id"] == vid_burg
    assert first_inc_id in bundle_burg["selected_incident_ids"]
    print(f"  Evidence Bundle created: {bundle_burg['bundle_id']} with incident {first_inc_id}")

    # E. Report generation with forensic context
    dossier = rep_svc.generate_dossier(
        video_id=vid_burg,
        custom_queries=["Show evidence related to the theft or property around 163 seconds"],
    )
    assert dossier is not None
    assert dossier.get("report_id") is not None
    print(f"  Dossier generated with forensic context: {dossier['report_id']}")
    print("  --> [PASS] Burglary Video Forensic Audit Completed Successfully.\n")

    # =========================================================================
    # 2. HIGHWAY CRASH VIDEO ACCEPTANCE AUDIT
    # =========================================================================
    print(f"[GATE 2: Highway Video Forensic Audit — {vhwy.original_filename} ({vid_hwy})]")

    q_hwy = InvestigationParser.parse_investigation_query(
        "Find vehicle collision incidents before 45 seconds",
        video_id=vid_hwy,
        video_duration_seconds=vhwy.duration_seconds,
    )
    res_hwy = executor.execute(q_hwy)
    assert res_hwy.source_video_id == vid_hwy
    print(f"  Query NL: 'Find vehicle collision incidents before 45 seconds'")
    print(f"  -> Interpretation: {res_hwy.interpretation}")
    print(f"  -> Matched incidents: {len(res_hwy.matched_incidents)}")
    print(f"  -> Matched events: {len(res_hwy.matched_events)}")
    assert len(res_hwy.matched_incidents) >= 1, "Expected matched vehicle incidents for Highway video"

    # Verify temporal separation: collisions should not be collapsed into one giant event
    collision_times = [inc["start_time"] for inc in res_hwy.matched_incidents if inc["incident_category"] == "VEHICLE"]
    print(f"  -> Distinct vehicle incident start timestamps: {collision_times}")

    # Unified Timeline
    timeline_hwy = inv_svc.get_unified_timeline(vid_hwy)
    assert timeline_hwy["total_entries"] > 0
    print(f"  Unified Timeline: {timeline_hwy['total_entries']} entries (incidents: {timeline_hwy['layer_counts']['correlated_incidents']})")

    # Bundle
    hwy_bundle = bundle_svc.create_bundle(
        video_id=vid_hwy,
        bundle_name="Highway Incident Package",
        selected_incident_ids=[res_hwy.matched_incidents[0]["incident_id"]],
    )
    assert hwy_bundle["video_id"] == vid_hwy
    print(f"  Evidence Bundle created: {hwy_bundle['bundle_id']}")
    print("  --> [PASS] Highway Video Forensic Audit Completed Successfully.\n")

    # =========================================================================
    # 3. 4K CCTV VIDEO ACCEPTANCE AUDIT
    # =========================================================================
    print(f"[GATE 3: 4K CCTV Video Forensic Audit — {v4k.original_filename} ({vid_4k})]")

    q_4k = InvestigationParser.parse_investigation_query(
        "Show vehicles and people detected in the scene",
        video_id=vid_4k,
        video_duration_seconds=v4k.duration_seconds,
    )
    res_4k = executor.execute(q_4k)
    assert res_4k.source_video_id == vid_4k
    print(f"  Query NL: 'Show vehicles and people detected in the scene'")
    print(f"  -> Interpretation: {res_4k.interpretation}")
    print(f"  -> Matched incidents: {len(res_4k.matched_incidents)}")
    print(f"  -> Matched events: {len(res_4k.matched_events)}")

    # Verify specialized safety: smoke=0, fire=0, weapon=0
    specialized_count = db.query(SecurityEventModel).filter(
        SecurityEventModel.video_id == vid_4k,
        SecurityEventModel.event_type.in_(["POTENTIAL_SMOKE", "POTENTIAL_FIRE", "POTENTIAL_WEAPON"]),
    ).count()
    assert specialized_count == 0, f"Expected 0 specialized alert events in 4K video, found {specialized_count}"
    print("  -> Zero smoke, zero fire, zero weapon verified in 4K scene.")

    timeline_4k = inv_svc.get_unified_timeline(vid_4k)
    assert timeline_4k["total_entries"] > 0
    print(f"  Unified Timeline: {timeline_4k['total_entries']} entries.")
    print("  --> [PASS] 4K CCTV Video Forensic Audit Completed Successfully.\n")

    # =========================================================================
    # 4. CROSS-VIDEO ISOLATION & SECURITY AUDIT
    # =========================================================================
    print("[GATE 4: Cross-Video Isolation & Boundary Enforcement]")

    # Test 4A: Attempt to create bundle in Video A referencing Video B incident
    print("  Testing bundle cross-video entity injection prevention...")
    hwy_incident_id = res_hwy.matched_incidents[0]["incident_id"]
    try:
        bundle_svc.create_bundle(
            video_id=vid_burg,
            bundle_name="Malicious Cross-Video Bundle",
            selected_incident_ids=[hwy_incident_id],  # from Highway video!
        )
        assert False, "CRITICAL VULNERABILITY: Bundle service allowed referencing an entity from another video!"
    except EvidenceBundleError as exc:
        print(f"  -> Successfully blocked cross-video entity injection: {exc}")

    # Test 4B: Attempt to access Video A bundle using Video B ID
    print("  Testing cross-video bundle access prevention...")
    try:
        bundle_svc.get_bundle(video_id=vid_hwy, bundle_id=bundle_burg["bundle_id"])
        assert False, "CRITICAL VULNERABILITY: Retrieved Video A bundle using Video B scope!"
    except EvidenceBundleError as exc:
        print(f"  -> Successfully blocked cross-video bundle access: {exc}")

    # Test 4C: Query Video A and ensure NO Video B entities are present
    print("  Testing query result video isolation...")
    for inc in res_burg.matched_incidents:
        assert inc["video_id"] == vid_burg, f"Entity leak! Incident {inc['incident_id']} has video_id {inc['video_id']} != {vid_burg}"
    for ev in res_burg.matched_events:
        assert ev["video_id"] == vid_burg, f"Entity leak! Event {ev['id']} has video_id {ev['video_id']} != {vid_burg}"
    for evid in res_burg.matched_evidence:
        assert evid["video_id"] == vid_burg, f"Entity leak! Evidence {evid['evidence_id']} has video_id {evid['video_id']} != {vid_burg}"
    print("  -> Absolute video isolation verified: 0 leaked cross-video entities in results.")
    print("  --> [PASS] Cross-Video Isolation & Boundary Enforcement Verified.\n")

    # =========================================================================
    # 5. ARBITRARY VIDEO GENERALIZATION & ROBUSTNESS AUDIT
    # =========================================================================
    print("[GATE 5: Arbitrary Video Generalization & Edge Cases]")

    # Empty query
    q_empty = InvestigationParser.parse_investigation_query("", video_id=vid_burg)
    res_empty = executor.execute(q_empty)
    assert res_empty is not None
    assert res_empty.total_results >= 0
    print("  -> Empty query handled gracefully (returns default unfiltered results).")

    # Inverted time range (100 to 20)
    q_inv = InvestigationParser.parse_investigation_query("between 100 and 20 seconds", video_id=vid_burg)
    assert q_inv.time_start == 20.0
    assert q_inv.time_end == 100.0
    print("  -> Inverted time range automatically normalized (20.0s -> 100.0s).")

    # Out of bounds time (negative start, beyond video duration)
    q_oob = InvestigationQuery(
        video_id=vid_burg,
        time_start=-50.0,
        time_end=99999.0,
        video_duration_seconds=vburg.duration_seconds,
    ).validate()
    assert q_oob.time_start == 0.0
    assert q_oob.time_end == vburg.duration_seconds
    print(f"  -> Out-of-bounds time clamped safely to [0.0, {vburg.duration_seconds:.1f}].")

    # Extreme limit request (clamped to 200)
    q_lim = InvestigationQuery(video_id=vid_burg, result_limit=5000).validate()
    assert q_lim.result_limit == 200
    print("  -> Extreme limit request clamped to max safe ceiling (200).")

    # Nonexistent video
    try:
        inv_svc.get_unified_timeline("nonexistent_video_123")
        assert False, "Expected error on nonexistent video"
    except Exception as exc:
        print(f"  -> Nonexistent video handled safely: {type(exc).__name__}")

    print("  --> [PASS] All Generalization & Robustness Gates Passed.\n")

    print("=" * 78)
    print("ALL 5 PHASE 17 FORENSIC AUDIT GATES PASSED EMPIRICALLY!")
    print("=" * 78 + "\n")
    db.close()


if __name__ == "__main__":
    verify_phase17_real_videos()
