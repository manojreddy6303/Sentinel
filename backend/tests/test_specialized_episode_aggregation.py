import pytest
from ai.schemas import BoundingBox
from ai.specialized.schemas import SpecializedObservation, SpecializedTemporalTrack
from ai.specialized.episode import (
    SpecializedVisualEpisodeAggregator,
    SpecializedVisualEpisode,
    SmokeEpisodePolicy,
    FireEpisodePolicy,
)
from ai.incidents.schemas import IncidentContext
from ai.incidents.detectors.specialized import SpecializedSmokeIncidentDetector, SpecializedFireIncidentDetector


def _make_obs(
    obs_id: str,
    class_name: str,
    timestamp: float,
    bbox_coords: tuple,
    confidence: float = 0.85,
    frame_number: int = 1,
) -> SpecializedObservation:
    x1, y1, x2, y2 = bbox_coords
    return SpecializedObservation(
        observation_id=obs_id,
        detector_name="SmokeVisualDetector" if class_name == "smoke" else "FireVisualDetector",
        detector_version="1.0.0",
        class_name=class_name,
        timestamp=timestamp,
        confidence=confidence,
        evidence_strength=confidence * 0.9,
        bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        frame_number=frame_number,
    )


def test_smoke_plume_expanding_bbox():
    """
    Test 1: Same smoke plume with expanding bounding box (low IoU).
    Demonstrates that deformable association correctly groups an expanding plume
    into a single visual episode rather than fracturing into multiple tracks.
    """
    aggregator = SpecializedVisualEpisodeAggregator()
    video_id = "test-video-smoke-expand"

    observations = [
        _make_obs("obs-1", "smoke", 1.0, (100.0, 100.0, 160.0, 180.0), confidence=0.85, frame_number=10),
        _make_obs("obs-2", "smoke", 2.0, (110.0, 95.0, 210.0, 220.0), confidence=0.88, frame_number=20),
        _make_obs("obs-3", "smoke", 3.0, (105.0, 90.0, 280.0, 290.0), confidence=0.91, frame_number=30),
    ]

    episodes = aggregator.aggregate(video_id=video_id, observations=observations)

    assert len(episodes) == 1, f"Expected 1 episode for expanding smoke plume, got {len(episodes)}"
    ep = episodes[0]
    assert ep.class_name == "smoke"
    assert ep.observation_count == 3
    assert ep.start_timestamp == 1.0
    assert ep.end_timestamp == 3.0
    assert ep.duration_seconds == 2.0
    assert ep.peak_confidence == 0.91
    # Check that all observations have the assigned episode_id
    for obs in observations:
        assert obs.episode_id == ep.episode_id


def test_smoke_plume_centroid_drift():
    """
    Test 2: Same smoke plume with centroid drift due to wind.
    Normalized distance relative to plume scale allows association despite drift.
    """
    aggregator = SpecializedVisualEpisodeAggregator()
    video_id = "test-video-smoke-drift"

    observations = [
        _make_obs("obs-1", "smoke", 10.0, (200.0, 200.0, 350.0, 350.0), confidence=0.80, frame_number=100),
        _make_obs("obs-2", "smoke", 11.5, (250.0, 190.0, 400.0, 340.0), confidence=0.82, frame_number=115),
        _make_obs("obs-3", "smoke", 13.0, (300.0, 180.0, 450.0, 330.0), confidence=0.85, frame_number=130),
    ]

    episodes = aggregator.aggregate(video_id=video_id, observations=observations)
    assert len(episodes) == 1
    assert episodes[0].observation_count == 3
    assert episodes[0].start_timestamp == 10.0
    assert episodes[0].end_timestamp == 13.0


def test_smoke_short_detector_gap():
    """
    Test 3: Short detection gap (bounded temporal coasting).
    Smoke is detected at t=5.0s and t=6.0s, detector misses frames until t=9.0s (3s gap <= 4.5s safe window).
    Spatial continuity holds, so it remains ONE episode.
    """
    aggregator = SpecializedVisualEpisodeAggregator()
    video_id = "test-video-smoke-gap"

    observations = [
        _make_obs("obs-1", "smoke", 5.0, (100.0, 100.0, 200.0, 200.0), confidence=0.75, frame_number=50),
        _make_obs("obs-2", "smoke", 6.0, (105.0, 100.0, 205.0, 205.0), confidence=0.78, frame_number=60),
        # Gap of 3.0 seconds (missed detection at t=7 and t=8)
        _make_obs("obs-3", "smoke", 9.0, (110.0, 95.0, 215.0, 210.0), confidence=0.82, frame_number=90),
    ]

    episodes = aggregator.aggregate(video_id=video_id, observations=observations)
    assert len(episodes) == 1
    assert episodes[0].observation_count == 3
    assert episodes[0].start_timestamp == 5.0
    assert episodes[0].end_timestamp == 9.0


def test_two_spatially_separated_smoke_plumes():
    """
    Test 4: Two spatially separated smoke plumes occurring at the exact same timestamp.
    Plume A is on the left (x ~ 50..150), Plume B is on the right (x ~ 800..900).
    They must NOT be merged into a single episode.
    """
    aggregator = SpecializedVisualEpisodeAggregator()
    video_id = "test-video-two-plumes"

    observations = [
        # Plume A (left)
        _make_obs("obs-a1", "smoke", 1.0, (50.0, 100.0, 150.0, 200.0), confidence=0.85, frame_number=10),
        # Plume B (right)
        _make_obs("obs-b1", "smoke", 1.0, (800.0, 100.0, 900.0, 200.0), confidence=0.88, frame_number=10),
        # Plume A next step
        _make_obs("obs-a2", "smoke", 2.0, (55.0, 105.0, 160.0, 210.0), confidence=0.87, frame_number=20),
        # Plume B next step
        _make_obs("obs-b2", "smoke", 2.0, (805.0, 95.0, 910.0, 205.0), confidence=0.90, frame_number=20),
    ]

    episodes = aggregator.aggregate(video_id=video_id, observations=observations)
    assert len(episodes) == 2, f"Expected 2 distinct episodes for spatially separated plumes, got {len(episodes)}"

    # Confirm one episode is left and one is right
    centroids_x = [ep.spatial_envelope["x1"] for ep in episodes]
    assert min(centroids_x) < 200.0
    assert max(centroids_x) > 700.0


def test_two_temporally_separated_smoke_episodes():
    """
    Test 5: Two temporally separated smoke episodes at the same location.
    Episode 1 is at t=1.0..2.0s.
    Episode 2 is at t=62.0..63.0s (60s gap > max_time_gap 4.5s).
    They must remain separate episodes.
    """
    aggregator = SpecializedVisualEpisodeAggregator()
    video_id = "test-video-temp-sep"

    observations = [
        _make_obs("obs-1", "smoke", 1.0, (100.0, 100.0, 200.0, 200.0), confidence=0.80, frame_number=10),
        _make_obs("obs-2", "smoke", 2.0, (105.0, 105.0, 205.0, 205.0), confidence=0.82, frame_number=20),
        # Long gap (60 seconds)
        _make_obs("obs-3", "smoke", 62.0, (102.0, 101.0, 203.0, 202.0), confidence=0.81, frame_number=620),
        _make_obs("obs-4", "smoke", 63.0, (106.0, 104.0, 208.0, 206.0), confidence=0.83, frame_number=630),
    ]

    episodes = aggregator.aggregate(video_id=video_id, observations=observations)
    assert len(episodes) == 2, f"Expected 2 episodes separated by 60s, got {len(episodes)}"
    assert episodes[0].end_timestamp <= 2.0
    assert episodes[1].start_timestamp >= 62.0


def test_fire_plume_deformation():
    """
    Test 6: Fire plume deformation.
    Fire is also a deformable phenomenon and benefits from the same episode aggregator with FireEpisodePolicy.
    """
    aggregator = SpecializedVisualEpisodeAggregator()
    video_id = "test-video-fire"

    observations = [
        _make_obs("obs-1", "fire", 10.0, (300.0, 400.0, 350.0, 460.0), confidence=0.88, frame_number=100),
        _make_obs("obs-2", "fire", 11.0, (290.0, 380.0, 380.0, 490.0), confidence=0.92, frame_number=110),
        _make_obs("obs-3", "fire", 12.0, (295.0, 390.0, 370.0, 480.0), confidence=0.90, frame_number=120),
    ]

    episodes = aggregator.aggregate(video_id=video_id, observations=observations)
    assert len(episodes) == 1
    assert episodes[0].class_name == "fire"
    assert episodes[0].observation_count == 3
    assert episodes[0].peak_confidence == 0.92


def test_counting_semantics_raw_vs_episodes_vs_incidents():
    """
    Test 7 & 8: Enforce counting semantics:
    raw_observation_count != episode_count != incident_count.
    10 raw observations -> 3 temporal tracks -> 1 visual episode -> 1 authoritative incident.
    """
    video_id = "test-video-counting"
    aggregator = SpecializedVisualEpisodeAggregator()

    # Create 10 observations forming 3 temporal track segments due to minor flickering/gaps
    observations = []
    # Segment 1: t=1.0 to 1.3
    for i in range(4):
        observations.append(
            _make_obs(
                f"obs-{i}",
                "smoke",
                round(1.0 + i * 0.1, 2),
                (100.0 + i * 2, 100.0, 200.0 + i * 2, 200.0),
                confidence=round(0.80 + i * 0.02, 2),
                frame_number=10 + i,
            )
        )
    # Segment 2: t=2.0 to 2.2 (0.7s gap from segment 1)
    for i in range(3):
        observations.append(
            _make_obs(
                f"obs-{i+4}",
                "smoke",
                round(2.0 + i * 0.1, 2),
                (110.0 + i * 2, 105.0, 215.0 + i * 2, 210.0),
                confidence=round(0.85 + i * 0.02, 2),
                frame_number=20 + i,
            )
        )
    # Segment 3: t=3.0 to 3.2 (0.8s gap from segment 2)
    for i in range(3):
        observations.append(
            _make_obs(
                f"obs-{i+7}",
                "smoke",
                round(3.0 + i * 0.1, 2),
                (115.0 + i * 2, 110.0, 220.0 + i * 2, 215.0),
                confidence=round(0.89 + i * 0.02, 2),
                frame_number=30 + i,
            )
        )

    # 3 mock tracks
    tracks = [
        SpecializedTemporalTrack(
            track_id="track-1",
            class_name="smoke",
            detector_name="SmokeVisualDetector",
            first_seen=1.0,
            last_seen=1.3,
            observation_count=4,
            max_confidence=0.86,
            mean_confidence=0.83,
            confidence_history=[o.confidence for o in observations[0:4]],
            bounding_boxes=[o.bounding_box.to_dict() for o in observations[0:4]],
        ),
        SpecializedTemporalTrack(
            track_id="track-2",
            class_name="smoke",
            detector_name="SmokeVisualDetector",
            first_seen=2.0,
            last_seen=2.2,
            observation_count=3,
            max_confidence=0.89,
            mean_confidence=0.87,
            confidence_history=[o.confidence for o in observations[4:7]],
            bounding_boxes=[o.bounding_box.to_dict() for o in observations[4:7]],
        ),
        SpecializedTemporalTrack(
            track_id="track-3",
            class_name="smoke",
            detector_name="SmokeVisualDetector",
            first_seen=3.0,
            last_seen=3.2,
            observation_count=3,
            max_confidence=0.93,
            mean_confidence=0.91,
            confidence_history=[o.confidence for o in observations[7:10]],
            bounding_boxes=[o.bounding_box.to_dict() for o in observations[7:10]],
        ),
    ]

    # Aggregate
    episodes = aggregator.aggregate(video_id=video_id, observations=observations, temporal_tracks=tracks)

    raw_count = len(observations)
    segment_count = len(tracks)
    episode_count = len(episodes)

    assert raw_count == 10
    assert segment_count == 3
    assert episode_count == 1

    ep = episodes[0]
    assert ep.observation_count == 10
    assert ep.segment_count == 3
    assert ep.start_timestamp == 1.0
    assert ep.end_timestamp == 3.2
    assert ep.peak_confidence == 0.93
    assert len(ep.supporting_observation_ids) == 10

    # Incident detector evaluation
    context = IncidentContext(
        video_id=video_id,
        tracks=[],
        fps=30.0,
        specialized_episodes=episodes,
    )
    detector = SpecializedSmokeIncidentDetector()
    candidates = detector.analyze(context)

    incident_count = len(candidates)
    assert incident_count == 1, f"Expected 1 authoritative candidate, got {incident_count}"

    cand = candidates[0]
    assert cand.event_type == "POTENTIAL_SMOKE"
    assert cand.incident_metadata["observation_count"] == 10
    assert cand.incident_metadata["segment_count"] == 3
    assert cand.incident_metadata["episode_id"] == ep.episode_id
    assert len(cand.incident_metadata["supporting_observation_ids"]) == 10
    assert cand.incident_metadata["peak_confidence"] == 0.93

    # Also verify the episode meets evidence gate thresholds.
    # This ensures test data quality matches real-world expectations.
    assert ep.observation_count >= SpecializedSmokeIncidentDetector.MINIMUM_SMOKE_OBSERVATIONS
    assert ep.duration_seconds >= SpecializedSmokeIncidentDetector.MINIMUM_SMOKE_DURATION_S


def test_video_isolation():
    """
    Test 9: Video isolation.
    Observations from Video A must NEVER be merged into episodes of Video B,
    even if their timestamps and spatial coordinates perfectly match.
    """
    aggregator = SpecializedVisualEpisodeAggregator()

    obs_video_a = [
        _make_obs("obs-a1", "smoke", 1.0, (100.0, 100.0, 200.0, 200.0), confidence=0.85, frame_number=10),
        _make_obs("obs-a2", "smoke", 2.0, (105.0, 105.0, 205.0, 205.0), confidence=0.88, frame_number=20),
    ]

    obs_video_b = [
        _make_obs("obs-b1", "smoke", 1.5, (102.0, 102.0, 202.0, 202.0), confidence=0.86, frame_number=15),
    ]

    episodes_a = aggregator.aggregate(video_id="video-a", observations=obs_video_a)
    episodes_b = aggregator.aggregate(video_id="video-b", observations=obs_video_b)

    assert len(episodes_a) == 1
    assert len(episodes_b) == 1
    assert episodes_a[0].video_id == "video-a"
    assert episodes_b[0].video_id == "video-b"
    assert episodes_a[0].episode_id != episodes_b[0].episode_id
    assert "video-a" in episodes_a[0].episode_id.lower() or "videoa" in episodes_a[0].episode_id.lower()
    assert "video-b" in episodes_b[0].episode_id.lower() or "videob" in episodes_b[0].episode_id.lower()


def test_reprocessing_idempotency():
    """
    Test 10: Reprocessing idempotency.
    Processing the exact same observations twice with the aggregator yields
    identical deterministic episode groupings and episode IDs.
    """
    aggregator = SpecializedVisualEpisodeAggregator()
    video_id = "test-video-idempotent-001"

    obs_run1 = [
        _make_obs("obs-1", "smoke", 1.0, (100.0, 100.0, 200.0, 200.0), confidence=0.85, frame_number=10),
        _make_obs("obs-2", "smoke", 2.0, (105.0, 105.0, 205.0, 205.0), confidence=0.88, frame_number=20),
    ]
    obs_run2 = [
        _make_obs("obs-1", "smoke", 1.0, (100.0, 100.0, 200.0, 200.0), confidence=0.85, frame_number=10),
        _make_obs("obs-2", "smoke", 2.0, (105.0, 105.0, 205.0, 205.0), confidence=0.88, frame_number=20),
    ]

    episodes_run1 = aggregator.aggregate(video_id=video_id, observations=obs_run1)
    episodes_run2 = aggregator.aggregate(video_id=video_id, observations=obs_run2)

    assert len(episodes_run1) == len(episodes_run2) == 1
    assert episodes_run1[0].episode_id == episodes_run2[0].episode_id
    assert episodes_run1[0].observation_count == episodes_run2[0].observation_count == 2
    assert episodes_run1[0].start_time == episodes_run2[0].start_time == 1.0
    assert episodes_run1[0].end_time == episodes_run2[0].end_time == 2.0


def test_frontend_api_authoritative_contract():
    """
    Test 11: Frontend/API Authoritative Contract.
    Verifies that the generated SecurityEvent payload / incident candidate contains
    authoritative episode metadata fields:
    - observation_count
    - segment_count
    - supporting_observation_ids
    - episode_id
    - peak_confidence
    - representative_timestamps

    Uses 5 observations to satisfy the MINIMUM_SMOKE_OBSERVATIONS gate.
    """
    video_id = "test-video-api-contract"
    aggregator = SpecializedVisualEpisodeAggregator()

    observations = [
        _make_obs("obs-101", "smoke", 10.0, (50.0, 50.0, 150.0, 150.0), confidence=0.80, frame_number=100),
        _make_obs("obs-102", "smoke", 11.0, (55.0, 55.0, 155.0, 155.0), confidence=0.85, frame_number=110),
        _make_obs("obs-103", "smoke", 12.0, (60.0, 60.0, 160.0, 160.0), confidence=0.90, frame_number=120),
        _make_obs("obs-104", "smoke", 13.0, (62.0, 62.0, 162.0, 162.0), confidence=0.88, frame_number=130),
        _make_obs("obs-105", "smoke", 14.0, (65.0, 65.0, 165.0, 165.0), confidence=0.87, frame_number=140),
    ]

    episodes = aggregator.aggregate(video_id=video_id, observations=observations)
    assert len(episodes) == 1

    detector = SpecializedSmokeIncidentDetector()
    context = IncidentContext(
        video_id=video_id,
        tracks=[],
        fps=30.0,
        specialized_episodes=episodes,
    )
    candidates = detector.analyze(context)
    assert len(candidates) == 1

    cand = candidates[0]
    meta = cand.incident_metadata

    assert "observation_count" in meta
    assert meta["observation_count"] == 5
    assert "segment_count" in meta
    assert "supporting_observation_ids" in meta
    assert len(meta["supporting_observation_ids"]) == 5
    assert "episode_id" in meta
    assert meta["episode_id"] == episodes[0].episode_id
    assert "representative_timestamps" in meta


# =============================================================================
# NEW INVARIANT TESTS — Phase 15.3 False-Positive Elimination
# =============================================================================

def test_smoke_episode_below_min_observations_abstains():
    """
    Invariant 12: A smoke episode with fewer observations than the minimum gate
    AND shorter duration than the minimum gate must produce ZERO candidates.
    The detector must ABSTAIN, not emit a speculative candidate.
    """
    video_id = "test-fp-obs-too-few"
    aggregator = SpecializedVisualEpisodeAggregator()

    # 3 observations over 0.3s — below both MINIMUM_SMOKE_OBSERVATIONS (5) and MINIMUM_SMOKE_DURATION_S (1.0)
    observations = [
        _make_obs("obs-1", "smoke", 1.0, (100.0, 100.0, 200.0, 200.0), confidence=0.80, frame_number=10),
        _make_obs("obs-2", "smoke", 1.15, (102.0, 100.0, 202.0, 200.0), confidence=0.82, frame_number=12),
        _make_obs("obs-3", "smoke", 1.3, (104.0, 100.0, 204.0, 200.0), confidence=0.83, frame_number=14),
    ]

    episodes = aggregator.aggregate(video_id=video_id, observations=observations)
    assert len(episodes) == 1  # Aggregation still works
    ep = episodes[0]
    assert ep.observation_count == 3
    assert ep.duration_seconds < SpecializedSmokeIncidentDetector.MINIMUM_SMOKE_DURATION_S

    context = IncidentContext(video_id=video_id, tracks=[], fps=30.0, specialized_episodes=episodes)
    detector = SpecializedSmokeIncidentDetector()
    candidates = detector.analyze(context)

    assert len(candidates) == 0, (
        f"Expected 0 candidates (ABSTAIN) for episode with {ep.observation_count} obs and "
        f"{ep.duration_seconds:.2f}s duration; got {len(candidates)}"
    )


def test_smoke_episode_low_mean_confidence_abstains():
    """
    Invariant 13: A smoke episode that meets observation count and duration
    thresholds but has mean_confidence below MINIMUM_SMOKE_MEAN_CONF must ABSTAIN.
    """
    video_id = "test-fp-low-conf"
    aggregator = SpecializedVisualEpisodeAggregator()

    # 8 observations, 3.5s duration — sufficient count/duration
    # but confidence is uniformly very low (0.30), below the 0.42 gate
    observations = [
        _make_obs(f"obs-{i}", "smoke", float(i) * 0.5,
                  (100.0, 100.0, 200.0, 200.0), confidence=0.30, frame_number=i * 5)
        for i in range(8)
    ]

    episodes = aggregator.aggregate(video_id=video_id, observations=observations)
    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.mean_confidence < SpecializedSmokeIncidentDetector.MINIMUM_SMOKE_MEAN_CONF

    context = IncidentContext(video_id=video_id, tracks=[], fps=30.0, specialized_episodes=episodes)
    detector = SpecializedSmokeIncidentDetector()
    candidates = detector.analyze(context)

    assert len(candidates) == 0, (
        f"Expected 0 candidates (ABSTAIN) for mean_conf={ep.mean_confidence:.3f} "
        f"< {SpecializedSmokeIncidentDetector.MINIMUM_SMOKE_MEAN_CONF}; got {len(candidates)}"
    )


def test_smoke_episode_adequate_evidence_raises_candidate():
    """
    Invariant 14: A smoke episode with sufficient observations, duration,
    and confidence MUST produce exactly 1 candidate. ABSTAIN is wrong here.
    """
    video_id = "test-fp-adequate-evidence"
    aggregator = SpecializedVisualEpisodeAggregator()

    # 10 observations, 4.5s, confidence 0.70-0.85 — clearly above all gates
    observations = [
        _make_obs(f"obs-{i}", "smoke", float(i) * 0.5,
                  (100.0 + i, 100.0, 200.0 + i, 200.0),
                  confidence=round(0.70 + i * 0.015, 3),
                  frame_number=i * 5)
        for i in range(10)
    ]

    episodes = aggregator.aggregate(video_id=video_id, observations=observations)
    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.observation_count >= SpecializedSmokeIncidentDetector.MINIMUM_SMOKE_OBSERVATIONS
    assert ep.duration_seconds >= SpecializedSmokeIncidentDetector.MINIMUM_SMOKE_DURATION_S
    assert ep.mean_confidence >= SpecializedSmokeIncidentDetector.MINIMUM_SMOKE_MEAN_CONF

    context = IncidentContext(video_id=video_id, tracks=[], fps=30.0, specialized_episodes=episodes)
    detector = SpecializedSmokeIncidentDetector()
    candidates = detector.analyze(context)

    assert len(candidates) == 1, (
        f"Expected 1 candidate for episode with obs={ep.observation_count}, "
        f"duration={ep.duration_seconds:.2f}s, mean_conf={ep.mean_confidence:.3f}; "
        f"got {len(candidates)}"
    )


def test_fire_episode_single_observation_abstains():
    """
    Invariant 15: A fire episode with 1 observation and very short duration
    must produce ZERO candidates (transient glitch filter).
    """
    video_id = "test-fp-fire-single-obs"
    aggregator = SpecializedVisualEpisodeAggregator()

    observations = [
        _make_obs("obs-fire-1", "fire", 5.0, (200.0, 200.0, 240.0, 240.0),
                  confidence=0.72, frame_number=50),
    ]

    episodes = aggregator.aggregate(video_id=video_id, observations=observations)
    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.observation_count == 1

    context = IncidentContext(video_id=video_id, tracks=[], fps=30.0, specialized_episodes=episodes)
    detector = SpecializedFireIncidentDetector()
    candidates = detector.analyze(context)

    assert len(candidates) == 0, (
        f"Expected 0 candidates (ABSTAIN) for single-observation fire episode; "
        f"got {len(candidates)}"
    )


def test_frame_quality_gate_blur():
    """
    Invariant 16: FrameQualityGate must flag a heavily blurred frame as not usable.

    Uses a structured gradient source (not pure noise) so that after blurring:
      - std-dev remains non-trivial (passes the uniformity gate)
      - Laplacian variance collapses to near-zero (triggers the blur gate)
    This cleanly isolates the blur rejection reason.
    """
    import numpy as np
    import cv2
    from ai.specialized.media_quality import FrameQualityGate, BLUR_LAPLACIAN_VARIANCE_THRESHOLD

    # Construct a gradient image: pixel values ramp linearly across the frame.
    # This has high std-dev but zero high-frequency content, so after a wide
    # box blur the Laplacian variance drops to essentially zero.
    h, w = 240, 320
    col_gradient = np.tile(np.linspace(80, 200, w, dtype=np.float32), (h, 1))
    row_gradient = np.tile(np.linspace(80, 200, h, dtype=np.float32).reshape(-1, 1), (1, w))
    gray_gradient = ((col_gradient + row_gradient) / 2.0).astype(np.uint8)
    base = np.stack([gray_gradient, gray_gradient, gray_gradient], axis=-1)

    # Apply a moderate box blur (21x21) to simulate camera motion blur.
    # The gradient retains its std-dev but the blur kills fine edge detail.
    blurred = cv2.blur(base, (21, 21))

    result = FrameQualityGate.evaluate(blurred)

    assert not result.is_usable, (
        f"Expected blurred frame to be rejected; "
        f"blur_variance={result.blur_variance:.3f} (threshold={BLUR_LAPLACIAN_VARIANCE_THRESHOLD}), "
        f"std_dev={result.std_dev:.2f}, reasons={result.reasons}"
    )
    assert any("blur" in r.lower() for r in result.reasons), (
        f"Expected 'blur' in rejection reasons; got: {result.reasons}"
    )



def test_frame_quality_gate_dark():
    """
    Invariant 17: FrameQualityGate must flag a near-black frame as not usable.
    Very dark frames arise during scene transitions, night mode start-up,
    or camera shutter lag — specialized detectors produce garbage on them.
    """
    import numpy as np
    from ai.specialized.media_quality import FrameQualityGate, DARKNESS_MEAN_THRESHOLD

    # Frame with mean intensity 8 (much darker than the 18.0 threshold)
    dark_frame = np.full((240, 320, 3), 8, dtype=np.uint8)

    result = FrameQualityGate.evaluate(dark_frame)

    assert not result.is_usable, (
        f"Expected dark frame to be rejected; "
        f"mean_brightness={result.mean_brightness:.1f} (threshold={DARKNESS_MEAN_THRESHOLD}), "
        f"reasons={result.reasons}"
    )
    assert any("dark" in r.lower() for r in result.reasons), (
        f"Expected 'dark' in rejection reasons; got: {result.reasons}"
    )


def test_frame_quality_gate_good_frame_passes():
    """
    Invariant 18: FrameQualityGate must pass a well-lit, sharp frame.
    Ensures the gate does not over-reject valid surveillance frames.
    """
    import numpy as np
    from ai.specialized.media_quality import FrameQualityGate

    # Create a frame with distinct edges (checkerboard) — high Laplacian variance
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    block = 20
    for r in range(0, 480, block * 2):
        for c in range(0, 640, block * 2):
            frame[r:r+block, c:c+block] = 180
            frame[r+block:r+block*2, c+block:c+block*2] = 180

    result = FrameQualityGate.evaluate(frame)

    assert result.is_usable, (
        f"Expected sharp checkerboard frame to pass quality gate; "
        f"reasons={result.reasons}, blur_var={result.blur_variance:.2f}"
    )
    assert result.blur_variance > 10.0, (
        f"Expected Laplacian variance > 10.0 for a high-contrast checkerboard; "
        f"got {result.blur_variance:.3f}"
    )

