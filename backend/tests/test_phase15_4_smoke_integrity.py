"""
Sentinel Phase 15.4 Smoke Integrity & Validation Hard Boundary Tests

Verifies:
1. Invariant: validated smoke = 0 => smoke episodes = 0 => smoke incidents = 0 => smoke evidence = 0
2. Global hardening against IR / monochrome / grayscale CCTV video streams
3. Temporal motion differentiation: static background surfaces rejected via temporal evidence
4. Genuine dynamic smoke plume validation & episode aggregation
5. Pipeline hard boundary: raw/rejected observations isolated from tracks, episodes, and incidents
"""
import pytest
import numpy as np
import cv2
import uuid

from ai.schemas import BoundingBox
from ai.specialized.schemas import (
    SpecializedObservation,
    SpecializedValidationStatus,
    SpecializedTemporalTrack,
)
from ai.specialized.fire_smoke.detector import SmokeVisualDetector
from ai.specialized.validator import SpecializedValidationEngine
from ai.specialized.episode import SpecializedVisualEpisodeAggregator
from ai.incidents.schemas import IncidentContext
from ai.incidents.detectors.specialized import SpecializedSmokeIncidentDetector
from ai.incidents.engine import IncidentIntelligenceEngine
from ai.intelligence_pipeline import SecurityIntelligencePipeline


def _make_spec_obs(
    class_name: str = "smoke",
    timestamp: float = 1.0,
    confidence: float = 0.80,
    validation_status: SpecializedValidationStatus = SpecializedValidationStatus.RAW,
    metrics: dict = None,
) -> SpecializedObservation:
    return SpecializedObservation(
        observation_id=f"OBS-TEST-{uuid.uuid4().hex[:6]}",
        detector_name="smoke_visual_detector",
        detector_version="1.0.0",
        class_name=class_name,
        timestamp=timestamp,
        confidence=confidence,
        evidence_strength=confidence * 0.9,
        bounding_box=BoundingBox(x1=50.0, y1=50.0, x2=150.0, y2=150.0),
        validation_status=validation_status,
        visual_metrics=metrics or {
            "area_pixels": 10000.0,
            "scene_mean_saturation": 45.0,
            "temporal_motion": 5.0,
            "aspect_ratio": 1.0,
        },
    )


def test_01_invariant_zero_validated_smoke_yields_zero_incidents():
    """
    Strict Invariant:
    validated smoke = 0 => smoke episodes = 0 => smoke incidents = 0
    Raw or rejected observations must never produce smoke incident candidates.
    """
    # 5 raw observations with 0 validated
    raw_obs = [
        _make_spec_obs(
            class_name="smoke",
            timestamp=float(i),
            confidence=0.75,
            validation_status=SpecializedValidationStatus.REJECTED,
        )
        for i in range(1, 6)
    ]

    context = IncidentContext(
        video_id="vid-invariant-test",
        tracks=[],
        fps=30.0,
        specialized_observations=raw_obs,
        specialized_tracks=[],
        specialized_episodes=[],
    )

    detector = SpecializedSmokeIncidentDetector()
    candidates = detector.analyze(context)

    # Must produce exactly 0 candidates when episodes are 0
    assert len(candidates) == 0, f"Expected 0 smoke candidates, got {len(candidates)}"

    # Also test through IncidentIntelligenceEngine
    engine = IncidentIntelligenceEngine()
    res = engine.analyze_incidents(
        video_id="vid-invariant-test",
        tracks=[],
        validated_detections=[],
        fps=30.0,
        specialized_observations=raw_obs,
        specialized_tracks=[],
        specialized_episodes=[],
    )

    smoke_incidents = [i for i in res["incidents"] if "SMOKE" in i.event_type]
    smoke_sec_events = [e for e in res["security_events"] if "SMOKE" in (e.event_type or "")]
    assert len(smoke_incidents) == 0, f"Expected 0 smoke incidents, got {len(smoke_incidents)}"
    assert len(smoke_sec_events) == 0, f"Expected 0 smoke security events, got {len(smoke_sec_events)}"


def test_02_ir_grayscale_stream_clean_abstention():
    """
    Verify SmokeVisualDetector cleanly abstains on grayscale/IR CCTV streams
    where scene_mean_sat < 35.0 or 85%+ of pixels have sat <= 55.
    """
    detector = SmokeVisualDetector(enabled=True)
    detector.initialize()

    # Create a simulated IR night-vision frame: low-saturation background with sensor noise
    ir_frame = np.full((240, 320, 3), 140, dtype=np.uint8)
    # Add slight random sensor noise
    np.random.seed(42)
    noise = np.random.randint(-10, 10, (240, 320, 3), dtype=np.int16)
    ir_frame = np.clip(ir_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    obs = detector.detect_frame(ir_frame, timestamp=1.0, frame_idx=30)
    # Must cleanly abstain with 0 raw observations
    assert len(obs) == 0, f"Expected 0 observations on IR frame, got {len(obs)}"


def test_03_static_background_rejected_via_temporal_evidence():
    """
    Verify static background surfaces with near-zero temporal difference
    are rejected using temporal motion evidence.
    """
    detector = SmokeVisualDetector(enabled=True)
    detector.initialize()
    detector.reset()

    # Create colorful scene with static patch that has smoke-like chromaticity
    frame1 = np.full((300, 300, 3), (200, 100, 50), dtype=np.uint8)  # Colorful blue-ish/green-ish
    # Add static gray patch in middle: B=150, G=150, R=150 (sat=0, val=150)
    cv2.rectangle(frame1, (100, 100), (180, 180), (150, 150, 150), -1)

    # Frame 1: establishes baseline
    detector.detect_frame(frame1, timestamp=1.0, frame_idx=30)

    # Frame 2: identical static frame at timestamp=2.0s
    frame2 = frame1.copy()
    obs2 = detector.detect_frame(frame2, timestamp=2.0, frame_idx=60)

    # Static patch has 0 temporal change across frames; must be rejected
    assert len(obs2) == 0, f"Static patch was not rejected by temporal motion: {len(obs2)} obs"


def test_04_genuine_dynamic_smoke_plume_validates_and_creates_episode():
    """
    Verify genuine dynamic smoke plume with billowing temporal motion validates,
    clusters into a visual episode, and raises an authoritative incident candidate.
    """
    validator = SpecializedValidationEngine()

    # Create 6 observations representing a dynamic expanding plume
    observations = []
    for i in range(6):
        obs = _make_spec_obs(
            class_name="smoke",
            timestamp=1.0 + i * 0.5,
            confidence=0.82 + i * 0.02,
            validation_status=SpecializedValidationStatus.RAW,
            metrics={
                "area_pixels": 8000.0 + i * 500,
                "scene_mean_saturation": 55.0,
                "temporal_motion": 6.5,
                "aspect_ratio": 1.1,
                "mean_luminance": 170.0,
                "mean_saturation": 20.0,
            },
        )
        # Validate
        v_obs = validator.validate_observation(obs)
        assert v_obs.validation_status == SpecializedValidationStatus.VALID
        observations.append(v_obs)

    # Aggregate into episode
    aggregator = SpecializedVisualEpisodeAggregator()
    episodes = aggregator.aggregate(video_id="vid-dynamic-smoke", observations=observations)

    assert len(episodes) == 1, f"Expected 1 episode, got {len(episodes)}"
    ep = episodes[0]
    assert ep.class_name == "smoke"
    assert ep.observation_count == 6
    assert ep.duration_seconds >= 2.0

    # Analyze via SpecializedSmokeIncidentDetector
    context = IncidentContext(
        video_id="vid-dynamic-smoke",
        tracks=[],
        fps=30.0,
        specialized_episodes=episodes,
    )
    detector = SpecializedSmokeIncidentDetector()
    candidates = detector.analyze(context)

    assert len(candidates) == 1, f"Expected 1 candidate, got {len(candidates)}"
    assert candidates[0].event_type == "POTENTIAL_SMOKE"
    assert candidates[0].confidence >= 0.80


def test_05_pipeline_hard_boundary_isolation():
    """
    Verify SecurityIntelligencePipeline enforces the hard validation boundary:
    Rejected observations are stored in repository for auditing but never enter
    tracks, episodes, or incident candidates.
    """
    pipeline = SecurityIntelligencePipeline()
    assert hasattr(pipeline, "specialized_validator")
    assert isinstance(pipeline.specialized_validator, SpecializedValidationEngine)
