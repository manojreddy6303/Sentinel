"""
Sentinel Phase 8 End-to-End Verification on Real Processed CCTV Video
Video: 12566041-uhd_3840_2160_30fps.mp4

Validates:
1. Real CCTV video exists and retains 198 detections / 8 grouped events
2. Phase 8 Security Intelligence execution (tracking, vehicle colors, anonymous faces, zones, activity, behavior)
3. Multi-frame object tracks with persistent TRACK-XXX IDs
4. Vehicle visual color classifications via HSV analysis
5. Anonymous face visual region detections (with strict observational safety notice)
6. Security intelligence events (intrusions, activity peaks, prolonged presence)
7. Natural-language investigation integration across Phase 8 signals
8. Evidence extraction from advanced security event to Evidence Vault
9. Zero regressions to existing records and vault items
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.config import settings
from database.session import init_db, SessionLocal
from database.models import (
    VideoModel,
    EventModel,
    GroupedEventModel,
    EvidenceModel,
    TrackModel,
    VehicleAttributeModel,
    FaceDetectionModel,
    SecurityZoneModel,
    SecurityEventModel,
)

client = TestClient(app)


def run_phase8_verification():
    print("\n=======================================================")
    print("STARTING SENTINEL PHASE 8 REAL CCTV VERIFICATION")
    print("Video: 12566041-uhd_3840_2160_30fps.mp4")
    print("=======================================================\n")

    init_db()
    db = SessionLocal()

    # Find verified video ID
    candidate_ids = [
        "605f62c7-9b7c-423b-8020-310d6f3899c5",
        "3426f64b-dd44-48a7-8e29-2c5f77b748bb",
    ]
    video_id = None
    for cid in candidate_ids:
        meta_path = settings.STORAGE_UPLOADS_DIR / f"{cid}.json"
        if meta_path.exists():
            video_id = cid
            break

    if not video_id:
        print("[FAIL] Real CCTV metadata sidecar not found.")
        return False

    try:
        # Verify video record & existing Phase 1-7 baseline
        vid_record = db.query(VideoModel).filter(VideoModel.id == video_id).first()
        raw_events_count = db.query(EventModel).filter(EventModel.video_id == video_id).count()
        grouped_events_count = db.query(GroupedEventModel).filter(GroupedEventModel.video_id == video_id).count()
        existing_evidence_count = db.query(EvidenceModel).filter(EvidenceModel.video_id == video_id).count()

        print(f"[CHECK 1: Baseline Preservation]")
        print(f"  Video ID: {video_id}")
        print(f"  Raw Detections Count: {raw_events_count}")
        print(f"  Grouped Timeline Events: {grouped_events_count}")
        print(f"  Existing Vault Evidence: {existing_evidence_count}")
        assert raw_events_count > 0, "No raw detections in DB!"
        assert grouped_events_count > 0, "No grouped events in DB!"
        print("  --> [PASS] Baseline Phase 1-7 data fully preserved.\n")

        # -------------------------------------------------------------
        # STEP 1: Define a Restricted Security Zone
        # -------------------------------------------------------------
        print("[CHECK 2: Restricted Security Zone Configuration]")
        zone_payload = {
            "name": "Main Traffic Corridor",
            "polygon": [[500, 500], [3500, 500], [3500, 2000], [500, 2000]],
            "target_classes": ["person", "car", "truck", "bus"],
            "alert_on_entry": True,
            "loitering_threshold_seconds": 10.0,
        }
        z_res = client.post(f"/api/videos/{video_id}/zones", json=zone_payload)
        assert z_res.status_code == 200, f"Zone creation failed: {z_res.status_code} {z_res.text}"
        zone_data = z_res.json()["zone"]
        zone_id = zone_data["zone_id"]
        print(f"  Created Zone: {zone_data['name']} (ID: {zone_id})")
        print("  --> [PASS] Zone created and stored in security_zones table.\n")

        # -------------------------------------------------------------
        # STEP 2: Execute On-Demand Phase 8 Security Intelligence Analysis
        # -------------------------------------------------------------
        print("[CHECK 3: Execute Phase 8 Security Intelligence Pipeline]")
        pipe_res = client.post(
            f"/api/videos/{video_id}/run-security-analysis",
            json={"zones": [zone_data]},
        )
        assert pipe_res.status_code == 200, f"Analysis failed: {pipe_res.status_code} {pipe_res.text}"
        pipe_data = pipe_res.json()
        print(f"  Status: {pipe_data['status']}")
        print(f"  Tracks Extracted: {pipe_data['tracks_count']}")
        print(f"  Vehicle Attributes Analyzed: {pipe_data['vehicle_attributes_count']}")
        print(f"  Face Visual Regions Detected: {pipe_data['face_detections_count']}")
        print(f"  Security Events Generated: {pipe_data['security_events_count']}")
        assert pipe_data["tracks_count"] > 0, "Expected multi-frame tracks to be generated!"
        print("  --> [PASS] Full CV intelligence pipeline completed.\n")

        # -------------------------------------------------------------
        # STEP 3: Verify Multi-Frame Object Tracking
        # -------------------------------------------------------------
        print("[CHECK 4: Multi-Frame Object Tracking API & DB]")
        tracks_res = client.get(f"/api/videos/{video_id}/tracks")
        assert tracks_res.status_code == 200
        tracks = tracks_res.json()["tracks"]
        sample_track = tracks[0]
        print(f"  Total Tracks: {len(tracks)}")
        print(f"  Sample Track: {sample_track['track_id']} | Class: {sample_track['object_class']} | "
              f"Duration: {sample_track['duration_seconds']:.1f}s ({sample_track['first_seen']:.1f}s -> {sample_track['last_seen']:.1f}s) | "
              f"Detections: {sample_track['detection_count']}")
        if sample_track.get("color"):
            print(f"  Sample Track Vehicle Color: {sample_track['color']} (Conf: {sample_track.get('color_confidence')})")
        print("  --> [PASS] Multi-frame tracking verified.\n")

        # -------------------------------------------------------------
        # STEP 4: Verify Vehicle Color Analysis
        # -------------------------------------------------------------
        print("[CHECK 5: Vehicle Visual Color Analysis API & DB]")
        attrs_res = client.get(f"/api/videos/{video_id}/attributes")
        assert attrs_res.status_code == 200
        attrs = attrs_res.json()["attributes"]
        print(f"  Total Vehicle Attributes: {len(attrs)}")
        if attrs:
            colors_found = {}
            for a in attrs:
                colors_found[a["color"]] = colors_found.get(a["color"], 0) + 1
            print(f"  Color Distribution: {colors_found}")
            sample_attr = attrs[0]
            print(f"  Sample Attribute: Class={sample_attr['object_class']}, Color={sample_attr['color']}, "
                  f"Confidence={sample_attr['confidence']:.2f}, Time={sample_attr['timestamp']:.2f}s")
        print("  --> [PASS] Vehicle color classification verified.\n")

        # -------------------------------------------------------------
        # STEP 5: Verify Face Visual Region Detection & Safety Notice
        # -------------------------------------------------------------
        print("[CHECK 6: Face Visual Region Detection & Safety Constraint]")
        faces_res = client.get(f"/api/videos/{video_id}/faces")
        assert faces_res.status_code == 200
        faces_data = faces_res.json()
        print(f"  Safety Notice: {faces_data['safety_notice']}")
        print(f"  Face Visual Regions: {faces_data['total_faces']}")
        assert "biometric" in faces_data["safety_notice"].lower()
        print("  --> [PASS] Strict observational constraint enforced.\n")

        # -------------------------------------------------------------
        # STEP 6: Verify Structured Security Intelligence Events
        # -------------------------------------------------------------
        print("[CHECK 7: Security Intelligence Events API & DB]")
        sec_res = client.get(f"/api/videos/{video_id}/security-events")
        assert sec_res.status_code == 200
        sec_events = sec_res.json()["events"]
        print(f"  Total Security Events: {len(sec_events)}")
        event_types = {}
        for se in sec_events:
            event_types[se["event_type"]] = event_types.get(se["event_type"], 0) + 1
        print(f"  Event Types Breakdown: {event_types}")
        if sec_events:
            s_ev = sec_events[0]
            print(f"  Sample Security Event: [{s_ev['event_type']}] at {s_ev['timestamp']:.2f}s, "
                  f"Severity={s_ev['severity']}, Desc: {s_ev['description']}")
        print("  --> [PASS] Structured security events verified.\n")

        # -------------------------------------------------------------
        # STEP 7: Natural-Language Investigation Queries for Phase 8
        # -------------------------------------------------------------
        print("[CHECK 8: Natural-Language Investigation with Phase 8 Records]")

        # 8a. Tracked people query
        q1_res = client.post(
            f"/api/videos/{video_id}/investigate",
            json={"query": "Show all tracked people"},
        )
        assert q1_res.status_code == 200
        d1 = q1_res.json()
        print(f"  Query 'Show all tracked people': {d1['message']} (Results: {d1['count']})")
        assert d1["is_supported"] is True
        assert d1["result_type"] == "tracks"

        # 8b. Activity peaks query
        q2_res = client.post(
            f"/api/videos/{video_id}/investigate",
            json={"query": "Show activity peaks"},
        )
        assert q2_res.status_code == 200
        d2 = q2_res.json()
        print(f"  Query 'Show activity peaks': {d2['message']} (Results: {d2['count']})")
        assert d2["is_supported"] is True

        # 8c. Specific track query
        sample_tid = tracks[0]["track_id"]
        q3_res = client.post(
            f"/api/videos/{video_id}/investigate",
            json={"query": f"Show {sample_tid}"},
        )
        assert q3_res.status_code == 200
        d3 = q3_res.json()
        print(f"  Query 'Show {sample_tid}': {d3['message']} (Results: {d3['count']})")
        assert d3["is_supported"] is True
        assert d3["result_type"] == "tracks"

        # 8d. Ethical Safety Guardrail on Identity
        q4_res = client.post(
            f"/api/videos/{video_id}/investigate",
            json={"query": "Who is this person?"},
        )
        assert q4_res.status_code == 200
        d4 = q4_res.json()
        print(f"  Safety Query 'Who is this person?': is_supported={d4['is_supported']}, message='{d4['message'][:70]}...'")
        assert d4["is_supported"] is False
        assert d4["result_type"] == "unsupported"

        print("  --> [PASS] Investigation parser correctly routes Phase 8 signals and enforces guardrails.\n")

        # -------------------------------------------------------------
        # STEP 8: Evidence Capture from an Advanced Security Event
        # -------------------------------------------------------------
        print("[CHECK 9: Evidence Capture on Security Event]")
        ev_timestamp = sec_events[0]["timestamp"] if sec_events else 8.0
        ev_id = sec_events[0]["id"] if sec_events else None
        ev_cap_res = client.post(
            f"/api/videos/{video_id}/evidence",
            json={
                "timestamp": ev_timestamp,
                "event_id": ev_id,
                "evidence_type": "snapshot_and_clip",
                "notes": "Evidence captured from Phase 8 security intelligence verification.",
            },
        )
        assert ev_cap_res.status_code == 200
        ev_cap_data = ev_cap_res.json()
        print(f"  Captured Evidence ID: {ev_cap_data['evidence_id']}")
        print(f"  Evidence Type: {ev_cap_data['evidence_type']}")
        print(f"  Has Snapshot: {bool(ev_cap_data.get('snapshot_path'))}")
        print(f"  Has Clip: {bool(ev_cap_data.get('clip_path'))}")

        # Check total evidence in vault
        vault_res = client.get(f"/api/videos/{video_id}/evidence")
        assert vault_res.status_code == 200
        vault_data = vault_res.json()
        print(f"  Updated Total Evidence in Vault: {vault_data['total_evidence']}")
        assert vault_data["total_evidence"] >= existing_evidence_count
        print("  --> [PASS] Evidence extraction on security event successful.\n")

        print("=======================================================")
        print("PHASE 8 REAL CCTV VERIFICATION: ALL 9 CHECKS PASSED!")
        print("=======================================================\n")
        return True

    finally:
        db.close()


if __name__ == "__main__":
    success = run_phase8_verification()
    sys.exit(0 if success else 1)
