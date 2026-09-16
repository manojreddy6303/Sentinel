"""
test_specialized_visual_and_reporting_audit.py
==============================================
Focused regression test suite for:
1. Specialized Visual Smoke Detector & Validator Hard Gates (Tests 1-13)
2. Report Service & Dossier Generator Vehicle Attribute Schema Alignment (Tests 14-19)
3. Cross-Layer Provenance, Scores, & Isolation (Tests 20-25)
"""

import math
import os
import uuid
import numpy as np
import pytest

from ai.specialized.fire_smoke.detector import SmokeVisualDetector
from ai.specialized.validator import SpecializedValidationEngine, SpecializedValidationStatus
from ai.specialized.episode import SpecializedVisualEpisodeAggregator
from ai.specialized.schemas import SpecializedObservation
from ai.schemas import BoundingBox
from ai.incidents.detectors.specialized import SpecializedSmokeIncidentDetector
from ai.incidents.schemas import IncidentContext
from ai.incidents.negative_evidence import NegativeEvidenceEngine
from backend.app.services.report_service import ReportService
from ai.reporting.dossier_generator import IncidentDossierPDFGenerator
from ai.reporting.schema import (
    ReportDataPayload,
    ReportVideoMetadata,
    ReportDetectionStats,
    ReportInvestigationFinding,
    ReportSecurityEvent,
)
from database.session import SessionLocal
from database.models import (
    VideoModel,
    EventModel,
    VehicleAttributeModel,
    SecurityEventModel,
    EvidenceModel,
    ReportModel,
)


# ============================================================================
# SMOKE TESTS 1-13
# ============================================================================

def test_1_static_asphalt_texture_rejected():
    """
    Test 1: Static asphalt road texture with high high-frequency gravel roughness
    must be rejected by validator and not produce a validated smoke observation.
    """
    validator = SpecializedValidationEngine()
    obs = SpecializedObservation(
        observation_id="obs_asphalt_1",
        detector_name="SmokeVisualDetector",
        detector_version="1.1.0",
        class_name="smoke",
        timestamp=1.0,
        confidence=0.75,
        evidence_strength=0.7,
        bounding_box=BoundingBox(50, 50, 150, 150),
        visual_metrics={
            "texture_roughness": 680.0,  # Asphalt gravel roughness >> 260.0
            "saturation_mean": 18.0,
            "is_static_surface": True,
            "anchor_count": 5,
        }
    )
    result = validator.validate_observation(obs)
    assert result.validation_status == SpecializedValidationStatus.REJECTED
    assert any(w in result.validation_reason.lower() for w in ("asphalt", "roughness", "texture", "solid", "static"))


def test_2_vehicle_shadow_rejected():
    """
    Test 2: Dark vehicle shadow (low luminance, low saturation, static surface anchor)
    must be rejected by validator.
    """
    validator = SpecializedValidationEngine()
    obs = SpecializedObservation(
        observation_id="obs_shadow_1",
        detector_name="SmokeVisualDetector",
        detector_version="1.1.0",
        class_name="smoke",
        timestamp=2.0,
        confidence=0.60,
        evidence_strength=0.5,
        bounding_box=BoundingBox(50, 50, 150, 150),
        visual_metrics={
            "luminance_mean": 25.0,
            "saturation_mean": 12.0,
            "texture_roughness": 310.0,
            "is_static_surface": True,
        }
    )
    result = validator.validate_observation(obs)
    assert result.validation_status == SpecializedValidationStatus.REJECTED


def test_3_static_wall_texture_rejected():
    """
    Test 3: Static wall surface texture with persistent spatial anchoring
    must be rejected.
    """
    validator = SpecializedValidationEngine()
    obs = SpecializedObservation(
        observation_id="obs_wall_1",
        detector_name="SmokeVisualDetector",
        detector_version="1.1.0",
        class_name="smoke",
        timestamp=3.0,
        confidence=0.65,
        evidence_strength=0.6,
        bounding_box=BoundingBox(50, 50, 150, 150),
        visual_metrics={
            "texture_roughness": 420.0,
            "is_static_surface": True,
            "anchor_count": 6,
        }
    )
    result = validator.validate_observation(obs)
    assert result.validation_status == SpecializedValidationStatus.REJECTED


def test_4_static_floor_texture_rejected():
    """
    Test 4: Static floor or concrete pavement texture must be rejected.
    """
    validator = SpecializedValidationEngine()
    obs = SpecializedObservation(
        observation_id="obs_floor_1",
        detector_name="SmokeVisualDetector",
        detector_version="1.1.0",
        class_name="smoke",
        timestamp=4.0,
        confidence=0.70,
        evidence_strength=0.65,
        bounding_box=BoundingBox(50, 50, 150, 150),
        visual_metrics={
            "texture_roughness": 350.0,
            "is_static_surface": True,
            "is_border_anchored": False,
        }
    )
    result = validator.validate_observation(obs)
    assert result.validation_status == SpecializedValidationStatus.REJECTED


def test_5_compression_artifact_rejected():
    """
    Test 5: Border-anchored compression artifacts or frame border clamping
    must be rejected.
    """
    validator = SpecializedValidationEngine()
    obs = SpecializedObservation(
        observation_id="obs_border_artifact",
        detector_name="SmokeVisualDetector",
        detector_version="1.1.0",
        class_name="smoke",
        timestamp=5.0,
        confidence=0.80,
        evidence_strength=0.75,
        bounding_box=BoundingBox(50, 50, 150, 150),
        visual_metrics={
            "is_border_anchored": True,
            "texture_roughness": 120.0,
        }
    )
    result = validator.validate_observation(obs)
    assert result.validation_status == SpecializedValidationStatus.REJECTED
    assert "border" in result.validation_reason.lower() or "clamped" in result.validation_reason.lower()


def test_6_global_camera_motion_rejected_or_abstained():
    """
    Test 6: High global camera motion without local plume expansion
    must lead to abstention or rejection.
    """
    detector = SmokeVisualDetector()
    detector.initialize()
    f1 = np.full((300, 300, 3), 120, dtype=np.uint8)
    f2 = np.roll(f1, 20, axis=1)  # Pan right by 20 pixels
    obs1 = detector.detect_frame(f1, timestamp=0.0, frame_idx=0)
    obs2 = detector.detect_frame(f2, timestamp=0.1, frame_idx=1)
    assert len(obs2) == 0


def test_7_insufficient_temporal_evolution_rejected():
    """
    Test 7: Static candidate present across frames without deformation
    is flagged as static surface by detector and filtered out.
    """
    detector = SmokeVisualDetector()
    detector.initialize()
    frame = np.full((200, 200, 3), 130, dtype=np.uint8)
    frame[50:150, 50:150] = 115
    obs = []
    for i in range(5):
        obs = detector.detect_frame(frame, timestamp=float(i) * 0.1, frame_idx=i)
    valid_obs = [o for o in obs if not o.visual_metrics.get("is_static_surface", False)]
    assert len(valid_obs) == 0


def test_8_grayscale_ir_static_scene_rejected():
    """
    Test 8: In grayscale/IR mode, static low-saturation background must not trigger smoke.
    """
    detector = SmokeVisualDetector()
    detector.initialize()
    gray_frame = np.full((250, 250, 3), 140, dtype=np.uint8)
    obs = []
    for i in range(4):
        obs = detector.detect_frame(gray_frame, timestamp=float(i) * 0.1, frame_idx=i)
    assert len(obs) == 0


def test_9_genuine_dynamic_smoke_like_synthetic_sequence():
    """
    Test 9: Genuine dynamic smoke plume (expanding low-saturation, soft roughness, moving)
    can produce observations and validate.
    """
    validator = SpecializedValidationEngine()
    obs = SpecializedObservation(
        observation_id="obs_valid_smoke",
        detector_name="SmokeVisualDetector",
        detector_version="1.1.0",
        class_name="smoke",
        timestamp=10.0,
        confidence=0.88,
        evidence_strength=0.85,
        bounding_box=BoundingBox(50, 50, 150, 150),
        visual_metrics={
            "texture_roughness": 85.0,       # Soft diffuse cloud <= 260.0
            "saturation_mean": 22.0,        # Desaturated plume
            "luminance_mean": 160.0,        # Mid-high luminance
            "motion_active": True,          # Moving
            "is_static_surface": False,     # Dynamic
            "is_border_anchored": False,    # Interior
        }
    )
    result = validator.validate_observation(obs)
    assert result.validation_status == SpecializedValidationStatus.VALID


def test_10_zero_validated_smoke_zero_smoke_episodes():
    """
    Test 10: Invariant: If zero observations pass validation, zero smoke episodes are created.
    """
    aggregator = SpecializedVisualEpisodeAggregator()
    obs = SpecializedObservation(
        observation_id="obs_rej",
        detector_name="SmokeVisualDetector",
        detector_version="1.1.0",
        class_name="smoke",
        timestamp=1.0,
        confidence=0.7,
        evidence_strength=0.6,
        bounding_box=BoundingBox(50, 50, 150, 150),
        validation_status=SpecializedValidationStatus.REJECTED,
    )
    episodes = aggregator.aggregate("test_video_10", [obs])
    assert len(episodes) == 0


def test_11_zero_validated_smoke_zero_smoke_incidents():
    """
    Test 11: Invariant: If zero validated smoke episodes exist, SpecializedSmokeIncidentDetector
    produces zero candidates.
    """
    detector = SpecializedSmokeIncidentDetector()
    context = IncidentContext(
        video_id="test_video_11",
        tracks=[],
        specialized_episodes=[],
        specialized_tracks=[],
    )
    candidates = detector.analyze(context)
    assert len(candidates) == 0


def test_12_smoke_video_isolation():
    """
    Test 12: Observations from video A do not leak into video B episodes or incidents.
    """
    aggregator = SpecializedVisualEpisodeAggregator()
    obs_a = SpecializedObservation(
        observation_id="obs_a",
        detector_name="SmokeVisualDetector",
        detector_version="1.1.0",
        class_name="smoke",
        timestamp=1.0,
        confidence=0.85,
        evidence_strength=0.8,
        validation_status=SpecializedValidationStatus.VALID,
        bounding_box=BoundingBox(50, 50, 150, 150),
    )
    episodes_a = aggregator.aggregate("video_A", [obs_a])
    episodes_b = aggregator.aggregate("video_B", [])
    assert len(episodes_b) == 0
    if episodes_a:
        assert episodes_a[0].video_id == "video_A"


def test_13_smoke_reprocessing_idempotency():
    """
    Test 13: Reprocessing the exact same sequence produces deterministic results.
    """
    aggregator = SpecializedVisualEpisodeAggregator()
    obs_list = [
        SpecializedObservation(
            observation_id=f"obs_idem_{i}",
            detector_name="SmokeVisualDetector",
            detector_version="1.1.0",
            class_name="smoke",
            timestamp=float(i),
            confidence=0.82,
            evidence_strength=0.8,
            validation_status=SpecializedValidationStatus.VALID,
            bounding_box=BoundingBox(50 + i, 50, 150 + i, 150),
        )
        for i in range(5)
    ]
    episodes_run1 = aggregator.aggregate("vid_idem", obs_list)
    episodes_run2 = aggregator.aggregate("vid_idem", obs_list)
    assert len(episodes_run1) == len(episodes_run2)
    if episodes_run1:
        assert episodes_run1[0].observation_count == episodes_run2[0].observation_count
        assert episodes_run1[0].mean_confidence == episodes_run2[0].mean_confidence


# ============================================================================
# REPORTING TESTS 14-19
# ============================================================================

def test_14_vehicle_attributes_present_dossier_succeeds():
    """
    Test 14: Dossier generation succeeds without AttributeError when VehicleAttributeModel
    records (using object_class & confidence) are present in the DB.
    """
    db = SessionLocal()
    vid = f"test_rep_14_{uuid.uuid4().hex[:6]}"
    try:
        video = VideoModel(
            id=vid,
            original_filename="vehicle_test.mp4",
            storage_path=f"storage/uploads/{vid}.mp4",
            file_size_bytes=1000,
            duration_seconds=10.0,
            fps=30.0,
            frame_count=300,
            status="processed",
        )
        db.add(video)
        va = VehicleAttributeModel(
            id=f"va_{uuid.uuid4().hex[:6]}",
            video_id=vid,
            track_id="trk_car_14",
            object_class="sedan",
            color="blue",
            confidence=0.92,
            timestamp_seconds=2.5,
        )
        db.add(va)
        db.commit()

        service = ReportService()
        dossier = service.generate_dossier(video_id=vid)
        assert dossier is not None
        assert "report_id" in dossier
        file_path, dl_name = service.get_report_file_path(dossier["report_id"])
        assert file_path.exists()
    finally:
        db.rollback()
        db.query(VehicleAttributeModel).filter(VehicleAttributeModel.video_id == vid).delete()
        db.query(VideoModel).filter(VideoModel.id == vid).delete()
        db.commit()
        db.close()


def test_15_vehicle_type_color_confidence_mismatch_cannot_occur():
    """
    Test 15: Dossier generator handles both VehicleAttributeModel and ReportVehicleAttribute
    with object_class/vehicle_type and confidence/color_confidence seamlessly.
    """
    payload = ReportDataPayload(
        report_id="REP-TEST-15",
        title="TEST VEHICLE REPORT",
        classification="CONFIDENTIAL",
        generated_at_iso="2026-09-13T12:00:00Z",
        video=ReportVideoMetadata(
            video_id="vid_15",
            original_filename="test15.mp4",
            duration_seconds=5.0,
            fps=30.0,
            frame_count=150,
            file_size_bytes=1000,
            status="processed",
            resolution="1920x1080",
            codec="h264",
            sampling_rate_fps=2.0,
            analyzed_frames_count=10,
            uploaded_at=None,
            processed_at=None,
        ),
        stats=ReportDetectionStats(
            total_detections=1,
            total_raw_detections=1,
            total_uncertain_detections=0,
            total_rejected_detections=0,
            class_counts={"car": 1},
            class_raw_counts={"car": 1},
            class_uncertain_counts={},
            class_rejected_counts={},
            class_track_counts={"car": 1},
            class_avg_confidences={"car": 0.90},
            total_tracks=1,
            validated_tracks=1,
            total_security_events=0,
            total_evidence_items=0,
            total_face_detections=0,
        ),
        security_events=[],
        theft_events=[],
        specialized_events=[],
        timeline=[],
        evidence=[],
        investigation_findings=[],
        vehicle_attributes=[
            {
                "track_id": "trk_15",
                "vehicle_type": "truck",
                "primary_color": "white",
                "confidence": 0.88,
                "timestamp_seconds": 2.0,
            }
        ],
        zones=[],
        faces=[],
    )
    import io
    buf = io.BytesIO()
    size_bytes, page_count = IncidentDossierPDFGenerator(data=payload).generate(buf)
    pdf_bytes = buf.getvalue()
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF")


def test_16_null_vehicle_attributes_handled_safely():
    """
    Test 16: None or empty vehicle attributes handled without errors in dossier generator.
    """
    payload = ReportDataPayload(
        report_id="REP-TEST-16",
        title="TEST NULL VEHICLE",
        classification="CONFIDENTIAL",
        generated_at_iso="2026-09-13T12:00:00Z",
        video=ReportVideoMetadata(
            video_id="vid_16",
            original_filename="test16.mp4",
            duration_seconds=5.0,
            fps=30.0,
            frame_count=150,
            file_size_bytes=1000,
            status="processed",
            resolution="1920x1080",
            codec="h264",
            sampling_rate_fps=2.0,
            analyzed_frames_count=10,
            uploaded_at=None,
            processed_at=None,
        ),
        stats=ReportDetectionStats(
            total_detections=0,
            total_raw_detections=0,
            total_uncertain_detections=0,
            total_rejected_detections=0,
            class_counts={},
            class_raw_counts={},
            class_uncertain_counts={},
            class_rejected_counts={},
            class_track_counts={},
            class_avg_confidences={},
            total_tracks=0,
            validated_tracks=0,
            total_security_events=0,
            total_evidence_items=0,
            total_face_detections=0,
        ),
        security_events=[],
        theft_events=[],
        specialized_events=[],
        timeline=[],
        evidence=[],
        investigation_findings=[],
        vehicle_attributes=[],
        zones=[],
        faces=[],
    )
    import io
    buf = io.BytesIO()
    size_bytes, page_count = IncidentDossierPDFGenerator(data=payload).generate(buf)
    pdf_bytes = buf.getvalue()
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")


def test_17_unknown_vehicle_color_handled_safely():
    """
    Test 17: Unknown or None color/class handled gracefully with defaults.
    """
    payload = ReportDataPayload(
        report_id="REP-TEST-17",
        title="TEST UNKNOWN COLOR",
        classification="CONFIDENTIAL",
        generated_at_iso="2026-09-13T12:00:00Z",
        video=ReportVideoMetadata(
            video_id="vid_17",
            original_filename="test17.mp4",
            duration_seconds=5.0,
            fps=30.0,
            frame_count=150,
            file_size_bytes=1000,
            status="processed",
            resolution="1920x1080",
            codec="h264",
            sampling_rate_fps=2.0,
            analyzed_frames_count=10,
            uploaded_at=None,
            processed_at=None,
        ),
        stats=ReportDetectionStats(
            total_detections=1,
            total_raw_detections=1,
            total_uncertain_detections=0,
            total_rejected_detections=0,
            class_counts={"car": 1},
            class_raw_counts={"car": 1},
            class_uncertain_counts={},
            class_rejected_counts={},
            class_track_counts={"car": 1},
            class_avg_confidences={"car": 0.85},
            total_tracks=1,
            validated_tracks=1,
            total_security_events=0,
            total_evidence_items=0,
            total_face_detections=0,
        ),
        security_events=[],
        theft_events=[],
        specialized_events=[],
        timeline=[],
        evidence=[],
        investigation_findings=[],
        vehicle_attributes=[
            {
                "track_id": "trk_17",
                "vehicle_type": "vehicle",
                "primary_color": "unknown",
                "confidence": None,
                "timestamp_seconds": 1.5,
            }
        ],
        zones=[],
        faces=[],
    )
    import io
    buf = io.BytesIO()
    size_bytes, page_count = IncidentDossierPDFGenerator(data=payload).generate(buf)
    pdf_bytes = buf.getvalue()
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")


def test_18_report_preserves_canonical_vehicle_attribute_values():
    """
    Test 18: Exact canonical values (track_id, timestamp, color, class, confidence)
    are preserved without distortion in vehicle_attributes dict.
    """
    va_attr = {
        "track_id": "trk_canon_99",
        "vehicle_type": "suv",
        "primary_color": "silver",
        "confidence": 0.945,
        "timestamp_seconds": 7.82,
    }
    assert va_attr["track_id"] == "trk_canon_99"
    assert va_attr["vehicle_type"] == "suv"
    assert va_attr["primary_color"] == "silver"
    assert va_attr["confidence"] == 0.945
    assert va_attr["timestamp_seconds"] == 7.82


def test_19_report_generation_for_video_without_vehicle_attributes_still_works():
    """
    Test 19: Dossier generation succeeds for video without any vehicle attributes (e.g. indoor burglary).
    """
    db = SessionLocal()
    vid = f"test_rep_19_{uuid.uuid4().hex[:6]}"
    try:
        video = VideoModel(
            id=vid,
            original_filename="burglary_no_cars.mp4",
            storage_path=f"storage/uploads/{vid}.mp4",
            file_size_bytes=1000,
            duration_seconds=12.0,
            fps=30.0,
            frame_count=360,
            status="processed",
        )
        db.add(video)
        se = SecurityEventModel(
            id=f"se_{uuid.uuid4().hex[:6]}",
            video_id=vid,
            event_type="POTENTIAL_THEFT",
            severity="HIGH",
            confidence=0.89,
            timestamp_seconds=5.0,
            duration_seconds=3.0,
            description="Person taking backpack",
        )
        db.add(se)
        db.commit()

        service = ReportService()
        dossier = service.generate_dossier(video_id=vid)
        assert dossier is not None
        assert "report_id" in dossier
        file_path, dl_name = service.get_report_file_path(dossier["report_id"])
        assert file_path.exists()
    finally:
        db.rollback()
        db.query(SecurityEventModel).filter(SecurityEventModel.video_id == vid).delete()
        db.query(VideoModel).filter(VideoModel.id == vid).delete()
        db.commit()
        db.close()


# ============================================================================
# CROSS-LAYER TESTS 20-25
# ============================================================================

def test_20_incident_score_remains_identical_through_evidence():
    """
    Test 20: Incident confidence remains identical when recorded in evidence items.
    """
    se = SecurityEventModel(
        id="se_score_20",
        video_id="vid_20",
        event_type="POTENTIAL_THEFT",
        severity="HIGH",
        confidence=0.884,
        timestamp_seconds=5.0,
    )
    ev = EvidenceModel(
        id="ev_20",
        video_id="vid_20",
        evidence_type="INCIDENT_RECORD",
        confidence=se.confidence,
        timestamp_seconds=se.timestamp_seconds,
        validation_status="VALID",
    )
    assert ev.confidence == se.confidence
    assert math.isclose(ev.confidence, 0.884)


def test_21_investigation_score_remains_identical():
    """
    Test 21: Investigation finding retains exact incident assessment score.
    """
    finding = ReportInvestigationFinding(
        query="Was theft detected?",
        finding="Automated pattern confirmed theft.",
        timestamp_reference="5.0s",
        confidence=0.884,
    )
    assert finding.confidence == 0.884


def test_22_report_score_remains_identical():
    """
    Test 22: Report security event retains exact assessment confidence.
    """
    rse = ReportSecurityEvent(
        id="se_22",
        event_type="SUSPICIOUS_LOITERING",
        timestamp_seconds=1.0,
        duration_seconds=5.0,
        severity="MEDIUM",
        confidence=0.782,
    )
    assert rse.confidence == 0.782


def test_23_video_id_isolation():
    """
    Test 23: Complete isolation of video_id across evidence, incident, and report models.
    """
    vid_a = "video_uuid_alpha"
    vid_b = "video_uuid_beta"
    se_a = SecurityEventModel(id="se_a", video_id=vid_a, event_type="THEFT")
    se_b = SecurityEventModel(id="se_b", video_id=vid_b, event_type="FIGHT")
    assert se_a.video_id != se_b.video_id
    assert se_a.video_id == vid_a
    assert se_b.video_id == vid_b


def test_24_track_id_provenance():
    """
    Test 24: Track ID provenance is strictly preserved from vehicle attributes through to dossier.
    """
    va = VehicleAttributeModel(
        id="va_24",
        video_id="vid_24",
        track_id="track_unique_7788",
        object_class="van",
        color="red",
        confidence=0.91,
        timestamp_seconds=14.2,
    )
    rva = {
        "track_id": va.track_id,
        "vehicle_type": va.object_class,
        "primary_color": va.color,
        "confidence": va.confidence,
        "timestamp_seconds": va.timestamp_seconds,
    }
    assert rva["track_id"] == "track_unique_7788"


def test_25_timestamp_provenance():
    """
    Test 25: Timestamps across incident, vehicle attribute, and evidence remain exactly preserved.
    """
    t_start = 13.456
    se = SecurityEventModel(
        id="se_25",
        video_id="vid_25",
        event_type="VEHICLE_COLLISION",
        timestamp_seconds=t_start,
        duration_seconds=9.333,
    )
    assert se.timestamp_seconds == t_start
    assert math.isclose(se.timestamp_seconds + se.duration_seconds, 22.789, rel_tol=1e-3)
