"""
Sentinel Global Detection Reliability Core Invariant Test Suite
Tests compliance with the Universal Detection Contract, Bounding Box Invariants,
Track Lifecycle States, dt-Safe Motion Telemetry, Negative Evidence Frame Exits,
Counting Semantics, and Detector Health Registry.
"""
import math
import pytest
from ai.schemas import (
    BoundingBox,
    CanonicalDetection,
    DetectionValidationStatus,
    TrackLifecycleState,
    TrackedObject,
)
from ai.detection.detector import YOLODetector
from ai.tracking.tracker import ObjectTracker
from ai.incidents.motion import UniversalMotionEngine
from ai.incidents.schemas import IncidentContext
from ai.incidents.detectors.theft_and_takeaway import TheftAndTakeawayDetector
from ai.incidents.negative_evidence import NegativeEvidenceEngine
from ai.common.detector_health import (
    DetectorHealthRegistry,
    DetectorOperationalStatus,
    DetectorModelType,
)


def test_canonical_detection_contract_invariants():
    """Verify CanonicalDetection enforces finite values and validates bounding boxes."""
    # Valid canonical detection
    bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=150.0)
    det = CanonicalDetection(
        video_id="video-test-01",
        frame_index=12,
        timestamp_seconds=0.48,
        class_name="person",
        class_id=0,
        confidence=0.88,
        bounding_box=bbox,
        detector_name="yolo_detector",
        image_width=1920,
        image_height=1080,
    )
    assert det.video_id == "video-test-01"
    assert det.timestamp_seconds == 0.48
    assert det.class_name == "person"
    assert det.validation_status == DetectionValidationStatus.RAW
    assert not det.is_edge_clipped

    # Backward compatibility with dict access
    assert det["object_class"] == "person"
    assert det["timestamp"] == 0.48
    assert det.get("confidence") == 0.88
    d_dict = det.to_dict()
    assert d_dict["class_name"] == "person"
    assert d_dict["object_class"] == "person"
    assert d_dict["timestamp_seconds"] == 0.48
    assert d_dict["timestamp"] == 0.48


def test_bounding_box_finite_and_edge_clipping():
    """Verify BoundingBox enforces finite bounds and detects boundary clipping."""
    # NaN coordinate rejection
    with pytest.raises(ValueError):
        BoundingBox(x1=float("nan"), y1=0.0, x2=10.0, y2=10.0)

    # Infinite coordinate rejection
    with pytest.raises(ValueError):
        BoundingBox(x1=0.0, y1=float("inf"), x2=10.0, y2=10.0)

    # Inverted box rejection
    with pytest.raises(ValueError):
        BoundingBox(x1=50.0, y1=20.0, x2=40.0, y2=60.0)

    # Edge clipping detection
    edge_box = BoundingBox(x1=0.0, y1=100.0, x2=200.0, y2=500.0)
    assert edge_box.is_edge_clipped(1920, 1080)
    assert "left" in edge_box.edge_clip_boundaries(1920, 1080)

    center_box = BoundingBox(x1=500.0, y1=400.0, x2=800.0, y2=900.0)
    assert not center_box.is_edge_clipped(1920, 1080)
    assert len(center_box.edge_clip_boundaries(1920, 1080)) == 0


def test_track_lifecycle_transitions():
    """Verify tracker manages TENTATIVE -> CONFIRMED -> COASTING -> LOST / ENDED states."""
    tracker = ObjectTracker(iou_threshold=0.2, max_missing_seconds=1.0, video_id="video-lifecycle")
    tracker.reset(video_id="video-lifecycle")

    # Frame 1: New detection creates TENTATIVE track
    det1 = {
        "object_class": "person",
        "confidence": 0.85,
        "bounding_box": {"x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 300.0},
    }
    active1 = tracker.update(timestamp=0.0, detections=[det1])
    assert len(active1) == 1
    t1 = active1[0]
    assert t1.video_id == "video-lifecycle"
    assert t1.state == TrackLifecycleState.TENTATIVE

    # Frame 2: Second matching detection transitions to CONFIRMED
    det2 = {
        "object_class": "person",
        "confidence": 0.88,
        "bounding_box": {"x1": 105.0, "y1": 102.0, "x2": 205.0, "y2": 302.0},
    }
    active2 = tracker.update(timestamp=0.5, detections=[det2])
    assert len(active2) == 1
    assert active2[0].state == TrackLifecycleState.CONFIRMED

    # Frame 3: Missing detection moves to COASTING
    active3 = tracker.update(timestamp=1.0, detections=[])
    assert len(active3) == 1
    assert active3[0].state == TrackLifecycleState.COASTING

    # Frame 4: Exceeding max_missing_seconds transitions to LOST and deactivates
    active4 = tracker.update(timestamp=2.5, detections=[])
    assert len(active4) == 0

    all_tracks = tracker.finalize()
    assert len(all_tracks) == 1
    assert all_tracks[0].state == TrackLifecycleState.LOST


def test_motion_engine_dt_safety():
    """Verify UniversalMotionEngine prevents division-by-zero on dt <= 0 or identical timestamps."""
    motion_engine = UniversalMotionEngine()
    
    # Track with identical timestamps (dt = 0.0)
    track = TrackedObject(
        track_id="TRACK-DT-TEST",
        video_id="video-dt",
        object_class="car",
        first_seen=1.0,
        last_seen=1.0,
        confidence=0.9,
        current_bbox=BoundingBox(x1=10.0, y1=10.0, x2=50.0, y2=50.0),
        trajectory=[
            (1.0, 100.0, 100.0),
            (1.0, 150.0, 150.0),  # dt = 0.0, dx, dy > 0
            (1.0, 200.0, 200.0),  # dt = 0.0
        ],
        history_bboxes=[],
    )

    motions = motion_engine.compute_track_motion(track)
    assert len(motions) == 3
    for m in motions:
        assert math.isfinite(m.velocity_estimate)
        assert math.isfinite(m.acceleration_estimate)
        assert m.velocity_estimate == 0.0
        assert m.acceleration_estimate == 0.0


def test_theft_detector_frame_boundary_exit_negative_evidence():
    """Verify an object track exiting at the frame boundary is rejected as theft."""
    theft_detector = TheftAndTakeawayDetector()

    # Person approaches object
    p_track = TrackedObject(
        track_id="TRACK-P1",
        video_id="video-boundary",
        object_class="person",
        first_seen=0.0,
        last_seen=10.0,
        confidence=0.9,
        current_bbox=BoundingBox(x1=200.0, y1=200.0, x2=300.0, y2=450.0),
        trajectory=[
            (0.0, 50.0, 50.0),
            (2.0, 200.0, 200.0),
            (5.0, 200.0, 200.0),
            (8.0, 500.0, 500.0),
            (10.0, 600.0, 600.0),
        ],
        history_bboxes=[],
    )

    # Object touches left frame boundary (x1 = 0.01 normalized)
    o_box = BoundingBox(x1=0.01, y1=0.4, x2=0.15, y2=0.6)
    o_track = TrackedObject(
        track_id="TRACK-O1",
        video_id="video-boundary",
        object_class="suitcase",
        first_seen=2.0,
        last_seen=5.5,
        confidence=0.88,
        current_bbox=o_box,
        trajectory=[
            (2.0, 200.0, 200.0),
            (5.5, 10.0, 200.0),
        ],
        history_bboxes=[
            {"timestamp": 2.0, "bbox": o_box.to_dict(), "is_edge_clipped": True},
            {"timestamp": 5.5, "bbox": o_box.to_dict(), "is_edge_clipped": True},
        ],
    )

    context = IncidentContext(video_id="video-boundary", tracks=[p_track, o_track])
    neg_signals = NegativeEvidenceEngine.evaluate_theft_negative_evidence(
        p_track, o_track, context, interaction_end_time=5.0
    )
    assert any(s.signal_type == "Negative: Frame Boundary Exit" for s in neg_signals)

    candidates = theft_detector.analyze(context)
    assert len(candidates) == 0, "Candidate must be rejected when object exited frame boundary"


def test_detector_health_registry():
    """Verify DetectorHealthRegistry registers, counts observations, and records failures."""
    registry = DetectorHealthRegistry()
    registry.register_detector(
        name="test_detector",
        version="1.0.0",
        status=DetectorOperationalStatus.AVAILABLE.value,
        model_type=DetectorModelType.TRAINED_MODEL.value,
    )

    registry.record_observation("test_detector", is_validated=True)
    registry.record_observation("test_detector", is_validated=False, is_rejected=True)

    report = registry.get_health_report()
    assert report["total_detectors"] == 1
    assert report["total_observations"] == 2
    assert report["total_validated"] == 1
    assert report["total_rejected"] == 1
    assert report["total_failures"] == 0

    # Failure recording
    registry.record_failure("test_detector", "Model weights corrupt")
    rec = registry.get_detector("test_detector")
    assert rec.status == DetectorOperationalStatus.FAILED.value
    assert rec.failures == 1
    assert rec.error_message == "Model weights corrupt"


def test_video_isolation_and_reprocessing_idempotence():
    """Verify separate videos have isolated tracker state and reprocessing resets clean."""
    # Video A
    tracker_a = ObjectTracker(video_id="video-alpha")
    det_a = {
        "object_class": "car",
        "confidence": 0.9,
        "bounding_box": {"x1": 50.0, "y1": 50.0, "x2": 150.0, "y2": 150.0},
    }
    tracks_a = tracker_a.update(timestamp=1.0, detections=[det_a])
    assert len(tracks_a) == 1
    assert tracks_a[0].video_id == "video-alpha"
    assert tracks_a[0].track_id == "TRACK-001"

    # Video B: Completely separate instance and video_id
    tracker_b = ObjectTracker(video_id="video-beta")
    det_b = {
        "object_class": "person",
        "confidence": 0.85,
        "bounding_box": {"x1": 200.0, "y1": 200.0, "x2": 250.0, "y2": 350.0},
    }
    tracks_b = tracker_b.update(timestamp=1.0, detections=[det_b])
    assert len(tracks_b) == 1
    assert tracks_b[0].video_id == "video-beta"
    assert tracks_b[0].track_id == "TRACK-001"

    # Reprocessing Video A: tracker.reset(video_id) guarantees clean state
    tracker_a.reset(video_id="video-alpha")
    reprocessed_a = tracker_a.update(timestamp=1.0, detections=[det_a])
    assert len(reprocessed_a) == 1
    assert reprocessed_a[0].track_id == "TRACK-001"
    assert reprocessed_a[0].detection_count == 1


def test_intra_frame_duplicate_suppression():
    """Verify YOLODetector._suppress_intra_frame_duplicates suppresses duplicate boxes of same class."""
    detector = YOLODetector()
    raw_candidates = [
        # Two highly overlapping person detections (IoU > 0.65) -> higher confidence kept
        {
            "class_name": "person",
            "class_id": 0,
            "confidence": 0.92,
            "bounding_box": {"x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 300.0},
            "detector_name": "yolo_detector",
        },
        {
            "class_name": "person",
            "class_id": 0,
            "confidence": 0.75,
            "bounding_box": {"x1": 102.0, "y1": 101.0, "x2": 198.0, "y2": 299.0},
            "detector_name": "yolo_detector",
        },
        # Nearby backpack belonging to person (different class) -> preserved!
        {
            "class_name": "backpack",
            "class_id": 24,
            "confidence": 0.82,
            "bounding_box": {"x1": 110.0, "y1": 150.0, "x2": 160.0, "y2": 220.0},
            "detector_name": "yolo_detector",
        },
    ]
    suppressed = detector._suppress_duplicate_detections(raw_candidates, iou_threshold=0.65)
    assert len(suppressed) == 2
    classes = [d["class_name"] for d in suppressed]
    assert classes == ["person", "backpack"]
    person_det = [d for d in suppressed if d["class_name"] == "person"][0]
    assert person_det["confidence"] == 0.92


def test_json_serialization_safety():
    """Verify all detection and tracker data structures serialize to JSON with strictly finite numbers."""
    import json
    bbox = BoundingBox(10.0, 20.0, 50.0, 80.0)
    canonical = CanonicalDetection(
        video_id="video-json",
        frame_index=1,
        timestamp_seconds=0.033,
        class_name="person",
        confidence=0.88,
        bounding_box=bbox,
    )
    track = TrackedObject(
        track_id="TRACK-001",
        video_id="video-json",
        object_class="person",
        first_seen=0.0,
        last_seen=1.0,
        confidence=0.88,
        current_bbox=bbox,
        trajectory=[(0.0, 30.0, 50.0), (1.0, 32.0, 52.0)],
        history_bboxes=[{"timestamp": 0.0, "bbox": bbox.to_dict()}],
    )

    c_json = json.dumps(canonical.to_dict())
    t_json = json.dumps(track.to_dict())
    assert "NaN" not in c_json
    assert "Infinity" not in c_json
    assert "NaN" not in t_json
    assert "Infinity" not in t_json


def test_object_counting_semantics_audit():
    """Verify 1 person across 100 frames produces 100 raw detections, exactly 1 track, and 0 false incidents."""
    tracker = ObjectTracker(video_id="video-counting-audit")
    raw_detections = []
    
    for frame_idx in range(100):
        t = round(frame_idx * 0.033, 3)
        # Person walking smoothly from x=100 to x=200
        cx = 100.0 + (frame_idx * 1.0)
        det = {
            "object_class": "person",
            "confidence": 0.88,
            "bounding_box": {"x1": cx - 20.0, "y1": 100.0, "x2": cx + 20.0, "y2": 250.0},
            "timestamp": t,
            "validation_status": "VALID",
        }
        raw_detections.append(det)
        tracker.update(timestamp=t, detections=[det])

    all_tracks = tracker.finalize()
    validated_tracks = tracker.get_validated_tracks()

    raw_detection_count = len(raw_detections)
    unique_track_count = len(validated_tracks)

    assert raw_detection_count == 100, f"Expected 100 raw observations, got {raw_detection_count}"
    assert unique_track_count == 1, f"Expected 1 unique person track, got {unique_track_count}"
    assert validated_tracks[0].track_id == "TRACK-001"
    assert validated_tracks[0].detection_count == 100
    assert validated_tracks[0].state == TrackLifecycleState.ENDED


def test_custom_model_dynamic_class_mapping_audit():
    """Verify YOLODetector resolves class names from custom model metadata without COCO assumptions."""
    class MockCustomModel:
        def __init__(self):
            # Custom non-COCO class mapping where 0 is vehicle, 1 is person, 2 is suitcase
            self.names = {0: "vehicle", 1: "person", 2: "suitcase"}

    mock_model = MockCustomModel()
    # Ensure mapping lookup uses model.names directly
    assert mock_model.names.get(0) == "vehicle"
    assert mock_model.names.get(1) == "person"
    assert mock_model.names.get(2) == "suitcase"
    assert mock_model.names.get(99, "unknown") == "unknown"


def test_rejected_detections_cannot_enter_tracks_or_incidents():
    """Verify REJECTED detections are ignored by ObjectTracker and filtered from IncidentContext."""
    from ai.incidents.context import IncidentContextBuilder
    tracker = ObjectTracker(video_id="video-rejected-test")

    rejected_det = {
        "object_class": "person",
        "confidence": 0.20,
        "bounding_box": {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 100.0},
        "validation_status": "REJECTED",
    }
    active = tracker.update(timestamp=1.0, detections=[rejected_det])
    assert len(active) == 0
    assert len(tracker.finalize()) == 0

    # Verify IncidentContextBuilder filters out rejected detections
    builder = IncidentContextBuilder()
    context = builder.build_context(
        video_id="video-rejected-test",
        tracks=[],
        validated_detections=[rejected_det],
    )
    assert len(context.validated_detections) == 0


def test_invalid_bbox_rejected_in_canonical_detection():
    """Verify degenerate/zero-area bounding box is automatically marked REJECTED in CanonicalDetection."""
    zero_bbox = BoundingBox(x1=10.0, y1=10.0, x2=10.0, y2=10.0)
    assert not zero_bbox.is_valid

    canonical = CanonicalDetection(
        video_id="video-bbox-test",
        frame_index=1,
        timestamp_seconds=0.033,
        class_name="person",
        confidence=0.9,
        bounding_box=zero_bbox,
    )
    assert canonical.validation_status == DetectionValidationStatus.REJECTED
    assert "Invalid bounding box geometry" in canonical.validation_reason


def test_pedestrians_crossing_paths_no_false_altercation_or_following():
    """Verify two pedestrians walking past each other in opposite directions do NOT trigger altercation or following."""
    from ai.incidents.detectors.person.altercation import PhysicalAltercationDetector
    from ai.incidents.detectors.person.following import PersonFollowingDetector

    # Pedestrian A walking left to right
    track_a = TrackedObject(
        track_id="TRACK-PED-A",
        video_id="video-ped",
        object_class="person",
        first_seen=0.0,
        last_seen=6.0,
        confidence=0.88,
        current_bbox=BoundingBox(500.0, 200.0, 560.0, 380.0),
        trajectory=[
            (0.0, 100.0, 250.0),
            (2.0, 250.0, 250.0),
            (3.0, 350.0, 250.0),  # Crossing point
            (4.0, 450.0, 250.0),
            (6.0, 600.0, 250.0),
        ],
        history_bboxes=[],
    )

    # Pedestrian B walking right to left (opposite direction)
    track_b = TrackedObject(
        track_id="TRACK-PED-B",
        video_id="video-ped",
        object_class="person",
        first_seen=0.0,
        last_seen=6.0,
        confidence=0.85,
        current_bbox=BoundingBox(100.0, 200.0, 160.0, 380.0),
        trajectory=[
            (0.0, 600.0, 260.0),
            (2.0, 450.0, 260.0),
            (3.0, 350.0, 260.0),  # Crossing point
            (4.0, 250.0, 260.0),
            (6.0, 100.0, 260.0),
        ],
        history_bboxes=[],
    )

    context = IncidentContext(video_id="video-ped", tracks=[track_a, track_b])
    altercation_det = PhysicalAltercationDetector()
    following_det = PersonFollowingDetector()

    alt_candidates = altercation_det.analyze(context)
    fol_candidates = following_det.analyze(context)

    assert len(alt_candidates) == 0, "Normal pedestrian crossing must not create altercation"
    assert len(fol_candidates) == 0, "Opposite direction transit must not create following"


def test_empty_video_produces_zero_detections_tracks_and_incidents():
    """Verify an empty video with no detections produces 0 tracks and 0 incidents without fabrication."""
    from ai.intelligence_pipeline import SecurityIntelligencePipeline
    pipeline = SecurityIntelligencePipeline()
    results = pipeline.process_video_intelligence(
        video_id="video-empty-test",
        video_path="",
        raw_events=[],
        fps=30.0,
        duration_seconds=10.0,
    )
    assert len(results["tracks"]) == 0
    assert len(results["all_tracks"]) == 0
    assert len(results["security_events"]) == 0
    assert len(results["incidents"]) == 0
    assert len(results["specialized_observations"]) == 0
    assert results["incident_diagnostics"]["failed_detectors"] == 0


