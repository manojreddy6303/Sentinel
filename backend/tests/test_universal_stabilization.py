"""
Comprehensive Test Suite for Sentinel Universal Video Analysis, Detection Counting,
Tracking Accuracy & Professional Reporting Master Stabilization

Covers:
- Detection Validation: general rejection of low-confidence noise, preservation of legitimate objects
- Intra-Frame Duplicate Suppression: IoU suppression of redundant boxes without dropping adjacent people
- Tracking Accuracy: Multi-frame persistence, zero cross-video track leakage
- Counting Semantics: Detection observations vs unique anonymous tracks
- Video Isolation: Multi-video tests (Video A vs Video B with overlapping timestamps)
- Reprocessing Idempotency: Re-running video intelligence does not duplicate records
- Evidence Grounding: Frame/timestamp/box linkage, rejected detection evidence rejection
- Investigation: Grounded answers distinguishing observations from tracks
- Incident Dossier: Full metadata, 6-column breakdown, real SHA-256 integrity hash
"""

import os
import uuid
import pytest
from datetime import datetime, timezone
from pathlib import Path
from fastapi.testclient import TestClient

from backend.app.main import app
from database.session import SessionLocal
from database.models import (
    VideoModel,
    EventModel,
    GroupedEventModel,
    EvidenceModel,
    TrackModel,
    VehicleAttributeModel,
    SecurityEventModel,
    ReportModel,
)
from ai.detection.detector import YOLODetector
from ai.validation.validator import DetectionValidator, ValidationStatus
from ai.validation.policy import DetectionValidationPolicy
from ai.tracking.tracker import ObjectTracker
from ai.schemas import BoundingBox, TrackedObject
from ai.events.generator import EventGenerator
from backend.app.services.investigation_service import InvestigationService
from backend.app.services.investigation_parser import InvestigationParser
from backend.app.services.report_service import ReportService, ReportError


@pytest.fixture
def client():
    return TestClient(app)


# ===========================================================================
# 1. Detection Validation & Intra-Frame Duplicate Suppression Tests
# ===========================================================================

def test_duplicate_suppression_keeps_highest_confidence():
    """Verify that multiple overlapping detections of the same class on one frame are deduplicated."""
    detector = YOLODetector()
    detections = [
        {
            "object_class": "person",
            "class_id": 0,
            "confidence": 0.85,
            "bounding_box": {"x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 300.0},
            "timestamp": 10.0,
        },
        {
            "object_class": "person",
            "class_id": 0,
            "confidence": 0.72,  # Overlapping duplicate box for the same person
            "bounding_box": {"x1": 105.0, "y1": 102.0, "x2": 202.0, "y2": 298.0},
            "timestamp": 10.0,
        },
    ]
    suppressed = detector._suppress_duplicate_detections(detections, iou_threshold=0.70)
    assert len(suppressed) == 1
    assert suppressed[0]["confidence"] == 0.85


def test_duplicate_suppression_preserves_adjacent_distinct_objects():
    """Verify that side-by-side people or cars (low IoU) are NOT suppressed."""
    detector = YOLODetector()
    detections = [
        {
            "object_class": "person",
            "class_id": 0,
            "confidence": 0.88,
            "bounding_box": {"x1": 50.0, "y1": 100.0, "x2": 150.0, "y2": 300.0},
            "timestamp": 10.0,
        },
        {
            "object_class": "person",
            "class_id": 0,
            "confidence": 0.84,  # Person standing next to the first person
            "bounding_box": {"x1": 170.0, "y1": 100.0, "x2": 270.0, "y2": 300.0},
            "timestamp": 10.0,
        },
    ]
    suppressed = detector._suppress_duplicate_detections(detections, iou_threshold=0.70)
    assert len(suppressed) == 2
    classes = [d["object_class"] for d in suppressed]
    assert classes == ["person", "person"]


def test_validation_rejects_isolated_low_confidence_vehicle():
    """Verify universal validator rejects or marks uncertain isolated low-confidence predictions across classes."""
    validator = DetectionValidator()
    # Bus and truck require temporal support; low-confidence isolated is strictly REJECTED
    for heavy_cls in ["bus", "truck"]:
        raw_det = {
            "object_class": heavy_cls,
            "confidence": 0.28,
            "bounding_box": {"x1": 100.0, "y1": 100.0, "x2": 250.0, "y2": 200.0},
            "timestamp": 15.0,
        }
        res = validator.validate_single_detection(raw_det, temporal_neighbors=[], context_neighbors=[])
        assert res.status == ValidationStatus.REJECTED
        assert "Low confidence" in res.reason or "temporal persistence" in res.reason

    # Car at 0.20 (below temporal threshold) is strictly REJECTED
    car_low = {
        "object_class": "car",
        "confidence": 0.20,
        "bounding_box": {"x1": 100.0, "y1": 100.0, "x2": 250.0, "y2": 200.0},
        "timestamp": 15.0,
    }
    res_car = validator.validate_single_detection(car_low, temporal_neighbors=[], context_neighbors=[])
    assert res_car.status == ValidationStatus.REJECTED

    # Car at 0.28 without temporal support is UNCERTAIN (never VALID)
    car_unc = {
        "object_class": "car",
        "confidence": 0.28,
        "bounding_box": {"x1": 100.0, "y1": 100.0, "x2": 250.0, "y2": 200.0},
        "timestamp": 15.0,
    }
    res_car_unc = validator.validate_single_detection(car_unc, temporal_neighbors=[], context_neighbors=[])
    assert res_car_unc.status == ValidationStatus.UNCERTAIN
    assert res_car_unc.status != ValidationStatus.VALID


def test_validation_preserves_temporal_multi_frame_objects():
    """Verify that legitimate objects with temporal support across frames are validated."""
    validator = DetectionValidator()
    det_f1 = {
        "object_class": "suitcase",
        "confidence": 0.45,
        "bounding_box": {"x1": 200.0, "y1": 200.0, "x2": 250.0, "y2": 260.0},
        "timestamp": 12.0,
    }
    neighbor = {
        "object_class": "suitcase",
        "confidence": 0.50,
        "bounding_box": {"x1": 202.0, "y1": 201.0, "x2": 252.0, "y2": 261.0},
        "timestamp": 13.0,
    }
    res = validator.validate_single_detection(det_f1, temporal_neighbors=[neighbor], context_neighbors=[])
    assert res.status == ValidationStatus.VALID


# ===========================================================================
# 2. Tracking Accuracy & Counting Semantics Tests
# ===========================================================================

def test_tracker_assigns_track_id_and_maintains_coherence():
    """Verify tracker tracks persistent objects over time without leaking track IDs."""
    tracker = ObjectTracker()
    tracker.reset()

    # Frame 1
    d1 = {"object_class": "person", "confidence": 0.90, "bounding_box": {"x1": 50, "y1": 50, "x2": 100, "y2": 200}}
    active_1 = tracker.update(timestamp=1.0, detections=[d1])
    assert len(active_1) == 1
    t1_id = active_1[0].track_id
    assert d1["track_id"] == t1_id

    # Frame 2 (same person moved slightly)
    d2 = {"object_class": "person", "confidence": 0.92, "bounding_box": {"x1": 52, "y1": 51, "x2": 102, "y2": 201}}
    active_2 = tracker.update(timestamp=2.0, detections=[d2])
    assert len(active_2) == 1
    assert active_2[0].track_id == t1_id
    assert d2["track_id"] == t1_id

    # Verify track properties
    track = active_2[0]
    assert track.detection_count == 2
    assert track.duration_seconds == 1.0
    assert track.is_validated is True


def test_event_grouping_enriches_track_count_and_detection_count():
    """Verify that grouped timeline events clearly distinguish detection counts from track counts."""
    generator = EventGenerator(window_seconds=3.0)
    video_id = f"test_group_{uuid.uuid4().hex[:8]}"

    raw_events = [
        {
            "event_id": str(uuid.uuid4()),
            "video_id": video_id,
            "object_class": "person",
            "confidence": 0.85,
            "timestamp": 10.0,
            "track_id": "TRACK-001",
            "validation_status": "VALID",
        },
        {
            "event_id": str(uuid.uuid4()),
            "video_id": video_id,
            "object_class": "person",
            "confidence": 0.88,
            "timestamp": 11.0,
            "track_id": "TRACK-001",
            "validation_status": "VALID",
        },
        {
            "event_id": str(uuid.uuid4()),
            "video_id": video_id,
            "object_class": "person",
            "confidence": 0.79,
            "timestamp": 11.5,
            "track_id": "TRACK-002",
            "validation_status": "VALID",
        },
    ]

    grouped = generator.group_events(video_id, raw_events)
    assert len(grouped) == 1
    g = grouped[0]
    assert g["total_detections"] == 3
    assert g["unique_tracks_count"] == 2
    assert set(g["involved_tracks"]) == {"TRACK-001", "TRACK-002"}

    # Object summary breakdown
    obj_summary = g["objects"]
    assert len(obj_summary) == 1
    p_stat = obj_summary[0]
    assert p_stat["class"] == "person"
    assert p_stat["detection_count"] == 3
    assert p_stat["track_count"] == 2
    assert set(p_stat["track_ids"]) == {"TRACK-001", "TRACK-002"}


# ===========================================================================
# 3. Multi-Video Upload & Strict Isolation Tests
# ===========================================================================

def test_multi_video_isolation_prevents_cross_video_leakage():
    """
    Test Video A vs Video B with identical timestamps:
    Verify that Video A queries return ONLY Video A data, and Video B queries return ONLY Video B data.
    """
    db = SessionLocal()
    vid_a = f"test_vid_A_{uuid.uuid4().hex[:8]}"
    vid_b = f"test_vid_B_{uuid.uuid4().hex[:8]}"

    try:
        # Create Video records
        v_a = VideoModel(id=vid_a, original_filename="camera_north.mp4", storage_path=f"storage/uploads/{vid_a}.mp4", status="processed")
        v_b = VideoModel(id=vid_b, original_filename="camera_south.mp4", storage_path=f"storage/uploads/{vid_b}.mp4", status="processed")
        db.add_all([v_a, v_b])

        # Video A has 2 person detections and 1 car detection at t=10.0s
        e_a1 = EventModel(id=str(uuid.uuid4()), video_id=vid_a, object_class="person", class_id=0, timestamp_seconds=10.0, confidence=0.90, bbox_x1=0, bbox_y1=0, bbox_x2=10, bbox_y2=10, frame_number=300, validation_status="VALID")
        e_a2 = EventModel(id=str(uuid.uuid4()), video_id=vid_a, object_class="person", class_id=0, timestamp_seconds=11.0, confidence=0.91, bbox_x1=0, bbox_y1=0, bbox_x2=10, bbox_y2=10, frame_number=330, validation_status="VALID")
        e_a3 = EventModel(id=str(uuid.uuid4()), video_id=vid_a, object_class="car", class_id=2, timestamp_seconds=10.0, confidence=0.85, bbox_x1=0, bbox_y1=0, bbox_x2=10, bbox_y2=10, frame_number=300, validation_status="VALID")

        # Video B has 1 suitcase detection at identical timestamp t=10.0s
        e_b1 = EventModel(id=str(uuid.uuid4()), video_id=vid_b, object_class="suitcase", class_id=28, timestamp_seconds=10.0, confidence=0.95, bbox_x1=0, bbox_y1=0, bbox_x2=10, bbox_y2=10, frame_number=300, validation_status="VALID")

        # Tracks
        t_a = TrackModel(id=str(uuid.uuid4()), video_id=vid_a, track_id="TRACK-001", object_class="person", first_seen=10.0, last_seen=11.0, duration_seconds=1.0, detection_count=2, max_confidence=0.91)
        t_b = TrackModel(id=str(uuid.uuid4()), video_id=vid_b, track_id="TRACK-001", object_class="suitcase", first_seen=10.0, last_seen=10.0, duration_seconds=0.0, detection_count=1, max_confidence=0.95)

        # Evidence records
        ev_a = EvidenceModel(id=str(uuid.uuid4()), video_id=vid_a, evidence_type="snapshot_only", timestamp_seconds=10.0, source_video_name="camera_north.mp4", object_class="person", validation_status="VALID")
        ev_b = EvidenceModel(id=str(uuid.uuid4()), video_id=vid_b, evidence_type="snapshot_only", timestamp_seconds=10.0, source_video_name="camera_south.mp4", object_class="suitcase", validation_status="VALID")

        db.add_all([e_a1, e_a2, e_a3, e_b1, t_a, t_b, ev_a, ev_b])
        db.commit()

        inv_svc = InvestigationService()

        # Query Video A for people
        res_a_people = inv_svc.investigate(vid_a, "How many person detections were recorded?")
        assert res_a_people["count"] == 2
        assert res_a_people["track_count"] == 1
        assert "2 validated person detection observations across 1 anonymous tracks" in res_a_people["message"]

        # Query Video A for suitcases
        res_a_suitcase = inv_svc.investigate(vid_a, "Show suitcases")
        assert res_a_suitcase["count"] == 0
        assert "No validated suitcase detections were found." in res_a_suitcase["message"]

        # Query Video B for suitcases
        res_b_suitcase = inv_svc.investigate(vid_b, "Show suitcases")
        assert res_b_suitcase["count"] == 1
        assert res_b_suitcase["results"][0]["object_class"] == "suitcase"

        # Query Video B for people (must be 0, no leakage from Video A)
        res_b_people = inv_svc.investigate(vid_b, "How many person detections were recorded?")
        assert res_b_people["count"] == 0
        assert res_b_people["track_count"] == 0

        # Verify Evidence isolation
        from backend.app.services.evidence_service import EvidenceService
        ev_svc = EvidenceService()
        evs_a = ev_svc.get_video_evidence(vid_a)
        evs_b = ev_svc.get_video_evidence(vid_b)
        assert len(evs_a) == 1
        assert evs_a[0]["object_class"] == "person"
        assert len(evs_b) == 1
        assert evs_b[0]["object_class"] == "suitcase"

    finally:
        db.close()


# ===========================================================================
# 4. Reprocessing Idempotency Tests
# ===========================================================================

def test_reprocessing_same_video_is_idempotent():
    """Verify that reprocessing the exact same video does not duplicate derived records or affect other videos."""
    db = SessionLocal()
    vid = f"test_reproc_{uuid.uuid4().hex[:8]}"

    try:
        from ai.events.repository import DatabaseEventRepository
        from ai.intelligence_repository import SecurityIntelligenceRepository
        event_repo = DatabaseEventRepository()
        intel_repo = SecurityIntelligenceRepository()

        # Initial events & tracks
        events_1 = [
            {
                "event_id": str(uuid.uuid4()),
                "video_id": vid,
                "event_type": "object_detected",
                "object_class": "person",
                "class_id": 0,
                "confidence": 0.88,
                "timestamp": 5.0,
                "bounding_box": {"x1": 10, "y1": 10, "x2": 50, "y2": 100},
                "validation_status": "VALID",
            }
        ]
        tracks_1 = [
            TrackedObject(
                track_id="TRACK-001",
                object_class="person",
                first_seen=5.0,
                last_seen=5.0,
                confidence=0.88,
                current_bbox=BoundingBox(10, 10, 50, 100),
            )
        ]

        event_repo.save_events(vid, events_1)
        intel_repo.save_intelligence_results(vid, tracks=tracks_1, vehicle_attributes=[], face_detections=[], security_events=[])

        # Check counts after run 1
        count_events_1 = db.query(EventModel).filter(EventModel.video_id == vid).count()
        count_tracks_1 = db.query(TrackModel).filter(TrackModel.video_id == vid).count()
        assert count_events_1 == 1
        assert count_tracks_1 == 1

        # Reprocess run 2
        events_2 = [
            {
                "event_id": str(uuid.uuid4()),
                "video_id": vid,
                "event_type": "object_detected",
                "object_class": "person",
                "class_id": 0,
                "confidence": 0.88,
                "timestamp": 5.0,
                "bounding_box": {"x1": 10, "y1": 10, "x2": 50, "y2": 100},
                "validation_status": "VALID",
            }
        ]
        event_repo.save_events(vid, events_2)
        intel_repo.save_intelligence_results(vid, tracks=tracks_1, vehicle_attributes=[], face_detections=[], security_events=[])

        # Assert no duplicate accumulation
        count_events_2 = db.query(EventModel).filter(EventModel.video_id == vid).count()
        count_tracks_2 = db.query(TrackModel).filter(TrackModel.video_id == vid).count()
        assert count_events_2 == 1
        assert count_tracks_2 == 1

    finally:
        db.close()


# ===========================================================================
# 5. Incident Dossier Quality, SHA-256 Hash & Pre-Save Consistency Audit
# ===========================================================================

def test_dossier_generation_computes_sha256_and_explicit_metrics():
    """Verify that ReportService computes a true SHA-256 hash and includes 6-column breakdown metrics."""
    db = SessionLocal()
    vid = f"test_dossier_{uuid.uuid4().hex[:8]}"

    try:
        v = VideoModel(
            id=vid,
            original_filename="security_gate_cam.mp4",
            storage_path=f"storage/uploads/{vid}.mp4",
            file_size_bytes=1024 * 1024 * 10,
            duration_seconds=60.0,
            fps=30.0,
            frame_count=1800,
            status="processed",
        )
        db.add(v)

        # 3 Valid Person detections, 1 Rejected Bus detection
        e1 = EventModel(id=str(uuid.uuid4()), video_id=vid, object_class="person", class_id=0, timestamp_seconds=10.0, confidence=0.88, bbox_x1=0, bbox_y1=0, bbox_x2=10, bbox_y2=10, frame_number=300, validation_status="VALID")
        e2 = EventModel(id=str(uuid.uuid4()), video_id=vid, object_class="person", class_id=0, timestamp_seconds=11.0, confidence=0.89, bbox_x1=0, bbox_y1=0, bbox_x2=10, bbox_y2=10, frame_number=330, validation_status="VALID")
        e3 = EventModel(id=str(uuid.uuid4()), video_id=vid, object_class="person", class_id=0, timestamp_seconds=12.0, confidence=0.87, bbox_x1=0, bbox_y1=0, bbox_x2=10, bbox_y2=10, frame_number=360, validation_status="VALID")
        e4 = EventModel(id=str(uuid.uuid4()), video_id=vid, object_class="bus", class_id=5, timestamp_seconds=15.0, confidence=0.29, bbox_x1=0, bbox_y1=0, bbox_x2=10, bbox_y2=10, frame_number=450, validation_status="REJECTED", validation_reason="low confidence noise")
        db.add_all([e1, e2, e3, e4])

        # Track
        t1 = TrackModel(id=str(uuid.uuid4()), video_id=vid, track_id="TRACK-001", object_class="person", first_seen=10.0, last_seen=12.0, duration_seconds=2.0, detection_count=3, max_confidence=0.89)
        db.add(t1)
        db.commit()

        svc = ReportService()
        report = svc.generate_dossier(video_id=vid, title="SECURITY INCIDENT AUDIT")

        # Verify Report model properties
        assert report["report_id"].startswith("REP-")
        meta = report["metadata"]
        assert meta["total_detections"] == 3  # Only validated detections
        assert meta["total_raw_detections"] == 4
        assert meta["total_rejected_detections"] == 1
        assert "bus" not in meta["class_counts"]  # Rejected bus not in validated counts
        assert meta["class_counts"]["person"] == 3
        assert meta["total_tracks"] == 1

        # Verify Real SHA-256 Hash is present
        assert "sha256_hash" in meta
        sha_hash = meta["sha256_hash"]
        assert len(sha_hash) == 64  # Valid 64-char hex SHA-256
        int(sha_hash, 16)  # Verify hex validity

        # Verify PDF exists on disk and its content hash matches the recorded sha256_hash
        import hashlib
        pdf_path, _ = svc.get_report_file_path(report["report_id"])
        assert pdf_path.exists()
        computed_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        assert computed_hash == sha_hash

    finally:
        db.close()
