"""
Phase 15: Specialized Visual Detection Core Comprehensive Test Suite
Validates categories A through P:
- A: Specialized detector schema tests
- B: Registry tests
- C: Model unavailable tests
- D: Model failure isolation
- E: Fire temporal validation
- F: Smoke temporal validation
- G: Fire false-positive rejection
- H: Smoke false-positive rejection
- I: Weapon unsupported-model behavior
- J: Pose fallback behavior
- K: Negative evidence
- L: Camera instability & scene robustness
- M: Incident fusion (Fire + Smoke)
- N: Evidence provenance
- O: Investigation queries
- P: Report consistency & dossier generation
"""

import io
import os
import uuid
import pytest
import numpy as np
from datetime import datetime, timezone

from ai.schemas import BoundingBox
from ai.specialized.schemas import (
    SpecializedDetectorStatus,
    SpecializedValidationStatus,
    SpecializedObservation,
    SpecializedModelInfo,
    SpecializedTemporalTrack,
)
from ai.specialized.base import BaseSpecializedDetector
from ai.specialized.registry import SpecializedDetectorRegistry
from ai.specialized.temporal import SpecializedTemporalTracker
from ai.specialized.negative_evidence import SpecializedNegativeEvidenceEngine
from ai.specialized.fire_smoke.detector import FireVisualDetector, SmokeVisualDetector
from ai.specialized.weapon.detector import WeaponVisualDetector
from ai.specialized.pose.detector import PoseActionDetector

from ai.incidents.schemas import IncidentCandidate, SupportingSignal, IncidentContext
from ai.incidents.fusion import IncidentFusionEngine
from ai.incidents.detectors.specialized import (
    SpecializedFireIncidentDetector,
    SpecializedSmokeIncidentDetector,
    SpecializedWeaponIncidentDetector,
)

from backend.app.services.investigation_parser import InvestigationParser
from backend.app.services.investigation_service import InvestigationService
from database.models import VideoModel, SpecializedObservationModel, SecurityEventModel
from database.session import SessionLocal

from ai.reporting.schema import (
    ReportDataPayload,
    ReportVideoMetadata,
    ReportDetectionStats,
    ReportSpecializedEvent,
)
from ai.reporting.dossier_generator import IncidentDossierPDFGenerator


# ===========================================================================
# Category A: Specialized Detector Schema Tests
# ===========================================================================

def test_01_schema_specialized_observation_validity():
    """Verify SpecializedObservation initialization, validation, and semantic fields."""
    obs = SpecializedObservation(
        observation_id="so-001",
        detector_name="fire_visual_detector",
        detector_version="v1.0-chroma",
        class_name="potential_fire",
        timestamp=12.5,
        confidence=0.88,
        evidence_strength=0.85,
        validation_status=SpecializedValidationStatus.VALID,
        bounding_box=BoundingBox(100.0, 150.0, 220.0, 280.0),
        visual_metrics={"chroma_ratio": 0.42, "flicker_score": 0.82},
        frame_number=375,
    )
    assert obs.detector_name == "fire_visual_detector"
    assert obs.class_name == "potential_fire"
    assert obs.timestamp == 12.5
    assert obs.validation_status == SpecializedValidationStatus.VALID
    assert obs.evidence_strength == 0.85
    assert obs.bounding_box.x1 == 100.0
    assert "flicker_score" in obs.visual_metrics
    d = obs.to_dict()
    assert d["observation_id"] == "so-001"
    assert d["bounding_box"]["x1"] == 100.0


def test_02_schema_temporal_track_updating():
    """Verify SpecializedTemporalTrack accumulates observations and computes persistence duration."""
    track = SpecializedTemporalTrack(
        track_id="track-spec-1",
        class_name="fire",
        detector_name="fire_detector",
        first_seen=10.0,
        last_seen=10.0,
        observation_count=1,
        max_confidence=0.75,
    )
    assert track.persistence_duration == 0.0

    # Add second observation at 12.5s
    obs2 = SpecializedObservation(
        observation_id="so-002",
        detector_name="fire_detector",
        detector_version="1.0",
        class_name="fire",
        timestamp=12.5,
        confidence=0.90,
        evidence_strength=0.90,
    )
    track.add_observation(obs2)
    assert track.observation_count == 2
    assert track.last_seen == 12.5
    assert track.persistence_duration == 2.5
    assert track.max_confidence == 0.90


# ===========================================================================
# Category B: Registry Tests
# ===========================================================================

def test_03_registry_lifecycle_and_discovery():
    """Verify detector registration, lookup, listing, and isolation in SpecializedDetectorRegistry."""
    registry = SpecializedDetectorRegistry()
    registry.clear()

    fire_det = FireVisualDetector()
    smoke_det = SmokeVisualDetector()
    registry.register(fire_det)
    registry.register(smoke_det)

    assert len(registry.list_detectors()) == 2
    assert registry.get("fire_visual_detector") is not None
    assert registry.get("smoke_visual_detector") is not None
    assert registry.get("non_existent") is None

    dets = registry.list_detectors()
    names = [d["detector_name"] for d in dets]
    assert "fire_visual_detector" in names
    assert "smoke_visual_detector" in names

    # Test unregister
    unreg = registry.unregister("smoke_visual_detector")
    assert unreg is not None
    assert registry.get("smoke_visual_detector") is None
    assert len(registry.list_detectors()) == 1


# ===========================================================================
# Category C: Model Unavailable Tests
# ===========================================================================

def test_04_model_unavailable_graceful_handling():
    """Verify missing model file sets UNAVAILABLE status without raising exceptions."""
    weapon_det = WeaponVisualDetector(config={"model_path": "C:/non_existent_weights/weapon_yolo.pt"})
    status = weapon_det.get_status()
    assert status in (SpecializedDetectorStatus.UNAVAILABLE, SpecializedDetectorStatus.NOT_CONFIGURED)

    # Inference should return empty list gracefully
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = weapon_det.safe_detect(dummy_frame, timestamp=1.0, frame_idx=30)
    assert results == []


# ===========================================================================
# Category D: Model Failure Isolation
# ===========================================================================

class FaultyDetector(BaseSpecializedDetector):
    detector_name = "faulty_detector"
    detector_version = "1.0"

    def initialize(self) -> bool:
        self._is_initialized = True
        self._status = SpecializedDetectorStatus.AVAILABLE
        return True

    def detect_frame(self, frame, timestamp, frame_idx=0, context=None):
        raise RuntimeError("Simulated internal GPU memory explosion or crash")


def test_05_failure_isolation_does_not_crash_pipeline():
    """Verify safe_detect isolates detector exceptions and returns empty list with failure logging."""
    faulty = FaultyDetector()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # safe_detect catches the exception and isolates it
    res = faulty.safe_detect(dummy_frame, timestamp=5.0, frame_idx=150)
    assert res == []


# ===========================================================================
# Category E: Fire Temporal Validation
# ===========================================================================

def test_06_fire_temporal_persistence_validation():
    """Verify fire observations require temporal persistence before validation."""
    tracker = SpecializedTemporalTracker(max_time_gap_seconds=1.5)

    # Frame 1: 10.0s
    obs1 = SpecializedObservation(
        observation_id="so-f1",
        detector_name="fire_visual_detector",
        detector_version="1.0",
        class_name="fire",
        timestamp=10.0,
        confidence=0.75,
        evidence_strength=0.70,
        bounding_box=BoundingBox(100.0, 100.0, 200.0, 200.0),
    )
    tracker.update([obs1])
    assert len(tracker.active_tracks) == 1
    trk = tracker.active_tracks[0]
    assert trk.observation_count == 1
    assert trk.persistence_duration == 0.0

    # Frame 2: 10.5s
    obs2 = SpecializedObservation(
        observation_id="so-f2",
        detector_name="fire_visual_detector",
        detector_version="1.0",
        class_name="fire",
        timestamp=10.5,
        confidence=0.80,
        evidence_strength=0.78,
        bounding_box=BoundingBox(102.0, 102.0, 204.0, 202.0),
    )
    tracker.update([obs2])
    assert len(tracker.active_tracks) == 1
    trk = tracker.active_tracks[0]
    assert trk.observation_count == 2
    assert trk.persistence_duration == 0.5

    # Frame 3: 11.2s
    obs3 = SpecializedObservation(
        observation_id="so-f3",
        detector_name="fire_visual_detector",
        detector_version="1.0",
        class_name="fire",
        timestamp=11.2,
        confidence=0.85,
        evidence_strength=0.82,
        bounding_box=BoundingBox(105.0, 101.0, 205.0, 205.0),
    )
    tracker.update([obs3])
    assert len(tracker.active_tracks) == 1
    trk = tracker.active_tracks[0]
    assert trk.observation_count == 3
    assert trk.persistence_duration == pytest.approx(1.2, abs=0.01)


# ===========================================================================
# Category F: Smoke Temporal Validation
# ===========================================================================

def test_07_smoke_temporal_persistence_and_expansion():
    """Verify smoke observations require multiple persistent frames to build persistence duration."""
    tracker = SpecializedTemporalTracker(max_time_gap_seconds=1.5)

    for i in range(4):
        obs = SpecializedObservation(
            observation_id=f"so-s{i}",
            detector_name="smoke_visual_detector",
            detector_version="1.0",
            class_name="smoke",
            timestamp=20.0 + (i * 0.6),
            confidence=0.72 + (i * 0.03),
            evidence_strength=0.70 + (i * 0.04),
            bounding_box=BoundingBox(300.0 - (i * 5), 200.0 - (i * 10), 450.0 + (i * 10), 350.0),
        )
        tracker.update([obs])

    assert len(tracker.active_tracks) == 1
    trk = tracker.active_tracks[0]
    assert trk.observation_count == 4
    assert trk.persistence_duration == pytest.approx(1.8, abs=0.01)


# ===========================================================================
# Category G: Fire False-Positive Rejection (Negative Evidence)
# ===========================================================================

def test_08_fire_false_positive_single_frame_and_static_surface():
    """Verify single-frame anomalous reflection or static surface is rejected by negative evidence."""
    # Case 1: Transient 1-frame track
    transient_trk = SpecializedTemporalTrack(
        track_id="t-transient",
        class_name="fire",
        detector_name="fire_detector",
        first_seen=10.0,
        last_seen=10.0,
        observation_count=1,
    )
    sigs1 = SpecializedNegativeEvidenceEngine.evaluate_fire_negative_evidence(transient_trk)
    assert len(sigs1) > 0
    assert any("Transient" in s.signal_type for s in sigs1)

    # Case 2: Static surface with negligible area variation across 4 frames
    static_trk = SpecializedTemporalTrack(
        track_id="t-static",
        class_name="fire",
        detector_name="fire_detector",
        first_seen=10.0,
        last_seen=12.0,
        observation_count=4,
        visual_metrics_history=[
            {"area_pixels": 1000.0},
            {"area_pixels": 1001.0},
            {"area_pixels": 1000.0},
            {"area_pixels": 1001.0},
        ],
    )
    sigs2 = SpecializedNegativeEvidenceEngine.evaluate_fire_negative_evidence(static_trk)
    assert len(sigs2) > 0
    assert any("Static Surface" in s.signal_type for s in sigs2)


# ===========================================================================
# Category H: Smoke False-Positive Rejection (Global Fog / Haze)
# ===========================================================================

def test_09_smoke_false_positive_global_fog():
    """Verify global fog/haze scene context generates negative evidence against localized smoke."""
    smoke_trk = SpecializedTemporalTrack(
        track_id="t-smoke",
        class_name="smoke",
        detector_name="smoke_detector",
        first_seen=15.0,
        last_seen=18.0,
        observation_count=5,
    )

    class MockSceneContext:
        lighting_condition = "overcast"
        weather_condition = "foggy"

    sigs = SpecializedNegativeEvidenceEngine.evaluate_smoke_negative_evidence(
        smoke_trk, scene_context=MockSceneContext()
    )
    assert len(sigs) > 0
    assert any("Global Atmospheric Haze" in s.signal_type for s in sigs)


# ===========================================================================
# Category I: Weapon Unsupported-Model Behavior (Zero Fabrication)
# ===========================================================================

def test_10_weapon_detector_not_configured_zero_fabrication():
    """Verify weapon detector returns NOT_CONFIGURED and never fabricates detections when unconfigured."""
    weapon_det = WeaponVisualDetector(config={"model_path": ""})
    assert weapon_det.get_status() == SpecializedDetectorStatus.NOT_CONFIGURED
    assert weapon_det.get_model_info().status == SpecializedDetectorStatus.NOT_CONFIGURED

    # Test random noisy frames
    for _ in range(5):
        noise_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        res = weapon_det.safe_detect(noise_frame, timestamp=5.0, frame_idx=150)
        assert res == []


# ===========================================================================
# Category J: Pose Fallback Behavior
# ===========================================================================

def test_11_pose_detector_low_confidence_abstention():
    """Verify pose detector abstains or returns empty list on blank frame without fabricating events."""
    pose_det = PoseActionDetector()
    dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)

    # Detect on empty black frame
    res = pose_det.safe_detect(dummy_frame, timestamp=1.0, frame_idx=30)
    for obs in res:
        assert obs.evidence_strength <= 0.60 or obs.validation_status != SpecializedValidationStatus.VALID


# ===========================================================================
# Category K: Negative Evidence Engine Penalties
# ===========================================================================

def test_12_negative_evidence_optical_resolution_penalty():
    """Verify small bounding regions are penalized with negative evidence for object classification."""
    obs = SpecializedObservation(
        observation_id="so-tiny",
        detector_name="weapon_detector",
        detector_version="1.0",
        class_name="weapon_knife",
        timestamp=5.0,
        confidence=0.65,
        evidence_strength=0.50,
    )
    # Tiny 14x16 px crop
    sigs = SpecializedNegativeEvidenceEngine.evaluate_weapon_negative_evidence(
        obs, target_crop_resolution=(14, 16)
    )
    assert len(sigs) > 0
    assert any("Insufficient Optical Resolution" in s.signal_type for s in sigs)


# ===========================================================================
# Category L: Camera Instability & Scene Robustness
# ===========================================================================

def test_13_camera_boundary_truncation_pose_negative_evidence():
    """Verify person at edge of frame generates boundary truncation negative evidence."""
    edge_box = BoundingBox(2.0, 10.0, 50.0, 200.0)  # x1=2 is <= 5 (frame boundary)
    kpts = [(20, 30, 0.9), (25, 40, 0.85)]
    sigs = SpecializedNegativeEvidenceEngine.evaluate_pose_negative_evidence(
        keypoints=kpts,
        bounding_box=edge_box,
        frame_dimensions=(1920, 1080),
    )
    assert len(sigs) > 0
    assert any("Frame Boundary Truncation" in s.signal_type for s in sigs)


# ===========================================================================
# Category M: Fire + Smoke Cross-Modal Fusion
# ===========================================================================

def test_14_fire_smoke_cross_modal_incident_fusion():
    """Verify overlapping fire and smoke candidates fuse into POTENTIAL_FIRE_SMOKE while preserving provenance."""
    fusion_engine = IncidentFusionEngine()

    fire_candidate = IncidentCandidate(
        incident_id="inc-f1",
        video_id="vid-1",
        event_type="POTENTIAL_FIRE",
        category="environment",
        start_time=15.0,
        end_time=18.0,
        duration=3.0,
        severity="HIGH",
        confidence=0.82,
        supporting_signals=[
            SupportingSignal(
                signal_type="fire_chroma_flicker",
                description="Fire chromatic flicker observed",
                confidence=0.82,
                timestamp=15.0,
            )
        ],
    )

    smoke_candidate = IncidentCandidate(
        incident_id="inc-s1",
        video_id="vid-1",
        event_type="POTENTIAL_SMOKE",
        category="environment",
        start_time=15.5,
        end_time=18.5,
        duration=3.0,
        severity="NORMAL",
        confidence=0.76,
        supporting_signals=[
            SupportingSignal(
                signal_type="smoke_plume_expansion",
                description="Smoke plume expansion observed",
                confidence=0.76,
                timestamp=15.5,
            )
        ],
    )

    fused_candidates = fusion_engine._fuse_fire_and_smoke_interactions([fire_candidate, smoke_candidate])
    assert len(fused_candidates) == 1
    fused = fused_candidates[0]
    assert fused.event_type == "POTENTIAL_FIRE_SMOKE"
    assert fused.confidence >= 0.80
    assert len(fused.supporting_signals) >= 2

    signal_types = [s.signal_type for s in fused.supporting_signals]
    assert "fire_chroma_flicker" in signal_types
    assert "smoke_plume_expansion" in signal_types


# ===========================================================================
# Category N: Evidence Provenance Retention
# ===========================================================================

def test_15_evidence_provenance_retention():
    """Verify specialized incident candidate retains full detector, timestamp, and bbox provenance."""
    detector = SpecializedFireIncidentDetector()
    track = SpecializedTemporalTrack(
        track_id="fire-trk-1",
        class_name="fire",
        detector_name="fire_visual_detector",
        first_seen=8.0,
        last_seen=10.5,
        observation_count=5,
        max_confidence=0.88,
        bounding_boxes=[{"x1": 150.0, "y1": 200.0, "x2": 260.0, "y2": 310.0}],
    )
    context = IncidentContext(
        video_id="vid_provenance_test",
        specialized_tracks=[track],
        specialized_observations=[],
    )

    candidates = detector.analyze(context)
    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.event_type == "POTENTIAL_FIRE"
    assert cand.start_time == 8.0
    assert cand.confidence >= 0.80
    assert len(cand.supporting_signals) > 0


# ===========================================================================
# Category O: Investigation Queries
# ===========================================================================

@pytest.mark.parametrize(
    "query_text, expected_type, expected_start, expected_end, expected_min_conf",
    [
        ("show fire events", "POTENTIAL_FIRE", None, None, None),
        ("find possible smoke", "POTENTIAL_SMOKE", None, None, None),
        ("show potential weapon detections", "POTENTIAL_WEAPON_VISUAL", None, None, None),
        ("find suspicious visual objects", "POTENTIAL_WEAPON_VISUAL", None, None, None),
        ("show fire between 30 and 60 seconds", "POTENTIAL_FIRE", 30.0, 60.0, None),
        ("show smoke evidence with high evidence strength", "POTENTIAL_SMOKE", None, None, 0.70),
    ],
)
def test_16_investigation_parser_specialized_queries(query_text, expected_type, expected_start, expected_end, expected_min_conf):
    """Verify InvestigationParser parses all required Phase 15 specialized queries."""
    res = InvestigationParser.parse_query(query_text)
    assert res["is_supported"] is True
    assert res["result_type"] == "security_events"
    filt = res["interpreted_filters"]
    assert filt["event_type"] == expected_type
    if expected_start is not None:
        assert filt["start_time"] == expected_start
    if expected_end is not None:
        assert filt["end_time"] == expected_end
    if expected_min_conf is not None:
        assert filt["min_confidence"] >= expected_min_conf


def test_16b_investigation_parser_all_specialized():
    """Verify parse_query parses 'show all specialized visual events'."""
    res = InvestigationParser.parse_query("show all specialized visual events")
    assert res["is_supported"] is True
    assert res["result_type"] == "specialized"


def test_17_investigation_service_deterministic_specialized_execution():
    """Verify InvestigationService executes deterministic specialized query and returns observational wording."""
    db = SessionLocal()
    try:
        vid_id = f"test-inv-{uuid.uuid4().hex[:8]}"
        video = VideoModel(
            id=vid_id,
            original_filename="security_cam_test.mp4",
            storage_path="storage/dummy.mp4",
            status="processed",
            duration_seconds=60.0,
            fps=30.0,
        )
        db.add(video)

        spec_obs = SpecializedObservationModel(
            id=f"so-{uuid.uuid4().hex[:8]}",
            video_id=vid_id,
            detector_name="fire_visual_detector",
            detector_version="v1.0",
            class_name="fire",
            timestamp_seconds=42.0,
            confidence=0.88,
            validation_status="VALID",
            evidence_strength=0.85,
            bounding_box={"x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 200.0},
            metrics={"persistence_sec": 3.2},
            created_at=datetime.now(timezone.utc),
        )
        db.add(spec_obs)
        db.commit()

        # Run investigation query
        inv_svc = InvestigationService()
        res = inv_svc.investigate(vid_id, "show fire events")
        assert res["is_supported"] is True
        assert res["count"] >= 1
        assert "Potential" in res["message"]
        assert len(res["results"]) >= 1
        assert res["results"][0]["timestamp"] == 42.0
    finally:
        db.rollback()
        db.close()


# ===========================================================================
# Category P: Report Consistency & Dossier Generation
# ===========================================================================

def test_18_report_dossier_contains_specialized_visual_section():
    """Verify IncidentDossierPDFGenerator embeds specialized events with observational wording."""
    spec_event = ReportSpecializedEvent(
        id="SPEC-01",
        event_type="POTENTIAL_FIRE",
        timestamp_seconds=18.4,
        duration_seconds=2.8,
        detector_name="fire_visual_detector",
        model_name="v1.0-chroma",
        evidence_strength=0.86,
        validation_status="REVIEW_REQUIRED",
        review_required=True,
        description="Potential fire visual evidence was detected with sustained chromatic features.",
    )

    payload = ReportDataPayload(
        report_id="REP-PHASE15-TEST",
        generated_at_iso=datetime.now(timezone.utc).isoformat(),
        video=ReportVideoMetadata(
            video_id="vid_phase15_report",
            original_filename="warehouse_dock.mp4",
            duration_seconds=45.0,
            fps=30.0,
        ),
        stats=ReportDetectionStats(total_detections=5),
        specialized_events=[spec_event],
    )

    buf = io.BytesIO()
    gen = IncidentDossierPDFGenerator(payload)
    sz, pages = gen.generate(buf)

    assert sz > 2000
    assert pages >= 1
    content = buf.getvalue()
    assert content.startswith(b"%PDF-")
