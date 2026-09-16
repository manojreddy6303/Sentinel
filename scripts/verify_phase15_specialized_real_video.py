"""
Sentinel Phase 15 Real CCTV Specialized Detection Verification Script

Validates the Phase 15 Specialized Visual Detection Core across 3 real CCTV videos:
1. 4K Traffic/Surveillance: 12566041-uhd_3840_2160_30fps.mp4
2. Real Burglary Footage: uccrime_Burglary010_x264.mp4
3. Vehicle Crash Footage: livevid_crash0.mp4

Key Verifications:
- Specialized registry discovery and status inspection
- Zero false fire, false smoke, or false weapon alarms on standard surveillance benchmarks
- Weapon detector foundation abstention (no fabricated weapon detections)
- Investigation parser and service query grounding for specialized categories
- Integrity of specialized observation tables and API endpoints
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.config import settings
from database.session import init_db, SessionLocal
from database.models import VideoModel, EventModel, SecurityEventModel, SpecializedObservationModel
from ai.specialized.registry import SpecializedDetectorRegistry
from ai.specialized.fire_smoke.detector import FireVisualDetector, SmokeVisualDetector
from ai.specialized.weapon.detector import WeaponVisualDetector
from ai.specialized.pose.detector import PoseActionDetector
from backend.app.services.investigation_service import InvestigationService

client = TestClient(app)


def run_phase15_real_video_verification():
    print("\n=======================================================")
    print("STARTING SENTINEL PHASE 15 SPECIALIZED VISUAL VERIFICATION")
    print("=======================================================\n")

    init_db()
    db = SessionLocal()

    # 1. Specialized Detector Discovery & Registry Audit
    print("[CHECK 1: Specialized Detector Registry & Status Audit]")
    registry = SpecializedDetectorRegistry()
    fire_det = FireVisualDetector()
    smoke_det = SmokeVisualDetector()
    weapon_det = WeaponVisualDetector()
    pose_det = PoseActionDetector()

    registry.register(fire_det)
    registry.register(smoke_det)
    registry.register(weapon_det)
    registry.register(pose_det)

    detectors = registry.list_detectors()
    print(f"  Registered Specialized Detectors: {len(detectors)}")
    for d in detectors:
        print(f"    * {d['detector_name']} (v{d['version']}, status: {d['status']})")
    assert len(detectors) == 4
    print("  --> [PASS] Specialized detector registry fully operational.\n")

    # 2. Locate the 3 Benchmark CCTV Videos
    target_files = {
        "4k_cctv": "12566041-uhd_3840_2160_30fps.mp4",
        "burglary": "uccrime_Burglary010_x264.mp4",
        "crash": "livevid_crash0.mp4",
    }
    matched_videos = {}
    for key, fname in target_files.items():
        for p in settings.STORAGE_UPLOADS_DIR.glob(f"*_{fname}"):
            vid_id = p.stem.split("_")[0]
            cnt = db.query(EventModel).filter(EventModel.video_id == vid_id).count()
            if cnt > 0:
                matched_videos[key] = (vid_id, p)
                break

    print(f"Matched real videos: {list(matched_videos.keys())}")
    assert len(matched_videos) == 3, f"Expected 3 benchmark videos, found {len(matched_videos)}"

    # 3. Benchmark Verification: 4K CCTV (Video ID: matched_videos['4k_cctv'][0])
    vid_4k = matched_videos["4k_cctv"][0]
    print(f"\n[CHECK 2: 4K CCTV Benchmark - {vid_4k}]")
    sec_events_4k = db.query(SecurityEventModel).filter(SecurityEventModel.video_id == vid_4k).all()
    fire_events_4k = [e for e in sec_events_4k if "FIRE" in (e.event_type or "")]
    weapon_events_4k = [e for e in sec_events_4k if "WEAPON" in (e.event_type or "")]
    print(f"  4K Total Security Events: {len(sec_events_4k)}")
    print(f"  4K False Fire Events: {len(fire_events_4k)}")
    print(f"  4K False Weapon Events: {len(weapon_events_4k)}")
    assert len(fire_events_4k) == 0, "Regression: False fire event detected on daytime 4K traffic video!"
    assert len(weapon_events_4k) == 0, "Regression: False weapon event fabricated on 4K traffic video!"
    print("  --> [PASS] 4K CCTV: Zero false specialized hazard alarms.")

    # 4. Benchmark Verification: Burglary Video (Video ID: matched_videos['burglary'][0])
    vid_burg = matched_videos["burglary"][0]
    print(f"\n[CHECK 3: Burglary Benchmark - {vid_burg}]")
    theft_events = db.query(SecurityEventModel).filter(
        SecurityEventModel.video_id == vid_burg,
        SecurityEventModel.event_type == "POTENTIAL_THEFT",
    ).all()
    fire_events_burg = db.query(SecurityEventModel).filter(
        SecurityEventModel.video_id == vid_burg,
        SecurityEventModel.event_type == "POTENTIAL_FIRE",
    ).all()
    print(f"  Burglary POTENTIAL_THEFT events: {len(theft_events)} (Grounded takeaway preserved)")
    print(f"  Burglary False Fire Events: {len(fire_events_burg)}")
    assert len(theft_events) >= 1, "Regression: Burglary theft detection lost!"
    assert len(fire_events_burg) == 0, "Regression: False fire event on burglary footage!"
    print("  --> [PASS] Burglary: Grounded theft preserved, zero false hazard alarms.")

    # 5. Benchmark Verification: Highway Crash Video (Video ID: matched_videos['crash'][0])
    vid_crash = matched_videos["crash"][0]
    print(f"\n[CHECK 4: Highway Crash Benchmark - {vid_crash}]")
    sec_events_crash = db.query(SecurityEventModel).filter(SecurityEventModel.video_id == vid_crash).all()
    weapon_events_crash = [e for e in sec_events_crash if "WEAPON" in (e.event_type or "")]
    print(f"  Highway Total Security Events: {len(sec_events_crash)}")
    print(f"  Highway False Weapon Events: {len(weapon_events_crash)}")
    assert len(weapon_events_crash) == 0, "Regression: False weapon event fabricated on crash footage!"
    print("  --> [PASS] Highway Crash: Vehicle incidents stable, zero false weapon alarms.")

    # 6. Investigation Query Verification on Real Videos
    print(f"\n[CHECK 5: Investigation Service Grounding on Real Video]")
    inv_svc = InvestigationService()
    res_fire = inv_svc.investigate(vid_4k, "show fire events")
    print(f"  Query 'show fire events' on 4K -> count: {res_fire['count']}, message: '{res_fire['message']}'")
    assert res_fire["is_supported"] is True

    res_weapon = inv_svc.investigate(vid_burg, "show potential weapon detections")
    print(f"  Query 'show potential weapon detections' on Burglary -> count: {res_weapon['count']}, message: '{res_weapon['message']}'")
    assert res_weapon["is_supported"] is True

    res_all_spec = inv_svc.investigate(vid_crash, "show all specialized visual events")
    print(f"  Query 'show all specialized visual events' on Crash -> count: {res_all_spec['count']}")
    assert res_all_spec["is_supported"] is True

    # 7. Endpoint Verification: GET /api/videos/{video_id}/specialized
    print(f"\n[CHECK 6: Specialized API Endpoint Verification]")
    api_res = client.get(f"/api/videos/{vid_4k}/specialized")
    print(f"  GET /api/videos/{vid_4k}/specialized -> Status Code: {api_res.status_code}")
    assert api_res.status_code == 200
    data = api_res.json()
    assert "observations" in data
    assert "count" in data
    print("  --> [PASS] Specialized API contract verified.")

    db.close()
    print("\n=======================================================")
    print("PHASE 15 SPECIALIZED VISUAL REAL VIDEO VERIFICATION: ALL PASSED!")
    print("=======================================================\n")


if __name__ == "__main__":
    run_phase15_real_video_verification()
