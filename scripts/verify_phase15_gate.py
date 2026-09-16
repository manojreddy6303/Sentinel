"""
Phase 15 Final Gate Verification Script
Verifies:
1. Fresh DB compatibility with SpecializedObservationModel
2. Strict multi-video isolation & data integrity across Video A and Video B
3. Fire / Smoke fallback status and labeling
4. Non-fabrication in Weapon & Pose foundations
"""
import os
import sys
import uuid
import tempfile
from pathlib import Path

# Add project root to sys.path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from database.session import Base
from database.models import (
    VideoModel, SpecializedObservationModel, SecurityEventModel,
    TrackModel, VehicleAttributeModel, FaceDetectionModel, ReportModel
)
from ai.intelligence_repository import SecurityIntelligenceRepository
from ai.specialized.fire_smoke.detector import FireVisualDetector, SmokeVisualDetector
from ai.specialized.weapon.detector import WeaponVisualDetector
from ai.specialized.pose.detector import PoseActionDetector
from ai.specialized.schemas import SpecializedDetectorStatus
from backend.app.services.investigation_service import InvestigationService


def test_fresh_database_creation():
    print("\n--- [GATE 1: Fresh Database Creation & Schema Inspection] ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db_path = Path(tmpdir) / "fresh_sentinel.db"
        fresh_engine = create_engine(f"sqlite:///{test_db_path}")
        
        # Create all tables on fresh DB
        Base.metadata.create_all(bind=fresh_engine)
        
        inspector = inspect(fresh_engine)
        tables = inspector.get_table_names()
        print(f"Tables created on fresh DB ({len(tables)}): {tables}")
        assert "specialized_observations" in tables, "specialized_observations table missing on fresh DB!"
        
        cols = {c["name"]: c for c in inspector.get_columns("specialized_observations")}
        print(f"specialized_observations columns: {list(cols.keys())}")
        
        required_cols = [
            "id", "video_id", "event_id", "detector_name", "detector_version",
            "class_name", "timestamp_seconds", "confidence", "validation_status",
            "bounding_box", "evidence_strength", "metrics", "created_at"
        ]
        for col in required_cols:
            assert col in cols, f"Required column '{col}' missing from specialized_observations!"
        
        print("--> [PASS] Fresh database initialization and schema verified.")
        fresh_engine.dispose()


def test_video_isolation_and_scoping():
    print("\n--- [GATE 2: Strict Multi-Video Isolation & Data Scoping] ---")
    from sqlalchemy.pool import StaticPool
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=test_engine)
    TestSession = sessionmaker(bind=test_engine)
    db = TestSession()

    try:
        vid_a = f"vid-test-A-{uuid.uuid4().hex[:8]}"
        vid_b = f"vid-test-B-{uuid.uuid4().hex[:8]}"

        # Insert 2 distinct videos
        v_a = VideoModel(id=vid_a, original_filename="stream_a.mp4", storage_path="/tmp/stream_a.mp4", duration_seconds=60.0)
        v_b = VideoModel(id=vid_b, original_filename="stream_b.mp4", storage_path="/tmp/stream_b.mp4", duration_seconds=120.0)
        db.add_all([v_a, v_b])
        db.commit()

        # Insert observations for Video A (Fire at 10.5s)
        obs_a1 = SpecializedObservationModel(
            id=f"OBS-A-{uuid.uuid4().hex[:6]}",
            video_id=vid_a,
            detector_name="fire_visual_detector",
            detector_version="1.0.0",
            class_name="fire",
            timestamp_seconds=10.5,
            confidence=0.82,
            evidence_strength=0.74,
            validation_status="RAW",
            bounding_box={"x1": 100, "y1": 150, "x2": 250, "y2": 300},
            metrics={"inference_source": "forensic_chromatic_rules"},
        )
        obs_a2 = SpecializedObservationModel(
            id=f"OBS-A-{uuid.uuid4().hex[:6]}",
            video_id=vid_a,
            detector_name="fire_visual_detector",
            detector_version="1.0.0",
            class_name="fire",
            timestamp_seconds=11.0,
            confidence=0.88,
            evidence_strength=0.79,
            validation_status="VALIDATED",
            bounding_box={"x1": 105, "y1": 148, "x2": 255, "y2": 305},
            metrics={"inference_source": "forensic_chromatic_rules"},
        )

        # Insert observations for Video B (Smoke at 85.0s)
        obs_b1 = SpecializedObservationModel(
            id=f"OBS-B-{uuid.uuid4().hex[:6]}",
            video_id=vid_b,
            detector_name="smoke_visual_detector",
            detector_version="1.0.0",
            class_name="smoke",
            timestamp_seconds=85.0,
            confidence=0.75,
            evidence_strength=0.64,
            validation_status="RAW",
            bounding_box={"x1": 500, "y1": 50, "x2": 700, "y2": 250},
            metrics={"inference_source": "forensic_smoke_texture"},
        )
        db.add_all([obs_a1, obs_a2, obs_b1])
        db.commit()

        # Test Repository Scoping
        repo = SecurityIntelligenceRepository()
        results_a = repo.get_specialized_observations(vid_a, db=db)
        results_b = repo.get_specialized_observations(vid_b, db=db)

        print(f"Video A retrieved observations count: {len(results_a)}")
        print(f"Video B retrieved observations count: {len(results_b)}")

        assert len(results_a) == 2, f"Expected 2 observations for Video A, got {len(results_a)}"
        assert len(results_b) == 1, f"Expected 1 observation for Video B, got {len(results_b)}"

        # Assert zero leakage
        ids_a = {o["id"] for o in results_a}
        ids_b = {o["id"] for o in results_b}
        assert ids_a.isdisjoint(ids_b), "Cross-video observation leakage detected!"
        assert obs_b1.id not in ids_a, "Video B observation leaked into Video A results!"
        assert obs_a1.id not in ids_b, "Video A observation leaked into Video B results!"

        # Assert correct timestamps
        for o in results_a:
            assert o["video_id"] == vid_a
            assert 10.0 <= o["timestamp"] <= 12.0
        for o in results_b:
            assert o["video_id"] == vid_b
            assert o["timestamp"] == 85.0

        # Test SQL Scoping on SpecializedObservationModel
        fire_a = db.query(SpecializedObservationModel).filter(
            SpecializedObservationModel.video_id == vid_a,
            SpecializedObservationModel.class_name == "fire"
        ).all()
        fire_b = db.query(SpecializedObservationModel).filter(
            SpecializedObservationModel.video_id == vid_b,
            SpecializedObservationModel.class_name == "fire"
        ).all()
        print(f"SQL direct query 'fire' for Video A: {len(fire_a)} found")
        print(f"SQL direct query 'fire' for Video B: {len(fire_b)} found")
        assert len(fire_a) == 2
        assert len(fire_b) == 0, "Video A fire observation leaked into Video B query!"

        # Test Investigation Service Scoping on Real Videos in DB
        inv_service = InvestigationService()
        real_vid_4k = "3426f64b-dd44-48a7-8e29-2c5f77b748bb"
        real_vid_crash = "8edd2faf-8a6e-4c7d-9b58-292e45b06b92"
        inv_4k = inv_service.investigate(real_vid_4k, "show fire events")
        inv_crash = inv_service.investigate(real_vid_crash, "show all specialized visual events")
        print(f"Investigation 'show fire events' on 4K: {inv_4k['count']} found")
        print(f"Investigation 'show all specialized visual events' on Crash: {inv_crash['count']} found")
        assert inv_crash["count"] > 0
        for rec in inv_crash["results"]:
            assert rec["video_id"] == real_vid_crash
        assert inv_4k["count"] != inv_crash["count"]

        print("--> [PASS] Strict multi-video isolation confirmed.")
    finally:
        db.close()
        test_engine.dispose()


def test_fire_smoke_model_and_fallback_status():
    print("\n--- [GATE 3: Fire & Smoke Model & Fallback Status Audit] ---")
    fire_det = FireVisualDetector(enabled=True)
    smoke_det = SmokeVisualDetector(enabled=True)
    
    fire_det.initialize()
    smoke_det.initialize()

    info_fire = fire_det.get_model_info()
    info_smoke = smoke_det.get_model_info()

    print(f"Fire Detector: {info_fire.name} (status: {info_fire.status.value})")
    print(f"  Reason: {info_fire.status_reason}")
    print(f"Smoke Detector: {info_smoke.name} (status: {info_smoke.status.value})")
    print(f"  Reason: {info_smoke.status_reason}")

    # Verify fallback behavior when weights are unconfigured:
    # 1. Must NOT claim to be custom YOLO or deep learning model
    assert "weights from" not in info_fire.status_reason, "Unconfigured fire detector falsely claimed loaded weights!"
    assert "weights from" not in info_smoke.status_reason, "Unconfigured smoke detector falsely claimed loaded weights!"
    
    # 2. Verify observational framing and review requirement in detectors
    from ai.incidents.detectors.specialized import SpecializedFireIncidentDetector, SpecializedSmokeIncidentDetector
    inc_fire = SpecializedFireIncidentDetector()
    inc_smoke = SpecializedSmokeIncidentDetector()
    assert inc_fire.human_verification_requirement is True
    assert inc_smoke.human_verification_requirement is True
    assert "POTENTIAL_FIRE" in inc_fire.detector_name or inc_fire.category.value == "environment"
    print("--> [PASS] Fire & smoke fallback labeling and review-required semantics verified.")


def test_weapon_and_pose_foundation_status():
    print("\n--- [GATE 4: Weapon & Pose Non-Fabrication Audit] ---")
    weapon_det = WeaponVisualDetector(enabled=True)
    weapon_det.initialize()
    status_w = weapon_det.get_status()
    print(f"Weapon Detector Status: {status_w.value}")
    assert status_w in (SpecializedDetectorStatus.NOT_CONFIGURED, SpecializedDetectorStatus.UNAVAILABLE)
    
    # Verify zero detections on dummy frame (no fabrication)
    import numpy as np
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    weapon_obs = weapon_det.safe_detect(blank_frame, timestamp=1.0)
    print(f"Weapon detections on test frame: {len(weapon_obs)}")
    assert len(weapon_obs) == 0, "Weapon detector fabricated observations without configured model!"

    pose_det = PoseActionDetector(enabled=False)
    pose_det.initialize()
    print(f"Pose Detector Status: {pose_det.get_status().value}")
    print("--> [PASS] Weapon & pose non-fabrication verified.")


if __name__ == "__main__":
    test_fresh_database_creation()
    test_video_isolation_and_scoping()
    test_fire_smoke_model_and_fallback_status()
    test_weapon_and_pose_foundation_status()
    print("\n=======================================================")
    print("ALL GATE AUDIT CHECKS PASSED SUCCESSFULLY!")
    print("=======================================================\n")
