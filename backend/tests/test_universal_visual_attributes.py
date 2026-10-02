"""
Universal Visual Attributes and Semantic Isolation Unit Tests (Phase 20.2)

Tests:
- Anatomical region partitioning in PersonAttributeAnalyzer
- Low-illumination and IR uncertainty in RobustColorExtractor
- TemporalColorFilter multi-frame Bayesian voting
- Non-biometric face telemetry and strict privacy boundaries
- Fire semantic isolation (red clothing rejection)
- Theft negative evidence and transient drop rejection
- Pluggable benchmark interfaces and local AI providers
"""
import pytest
import numpy as np
import cv2

from ai.schemas import BoundingBox, ClothingColor, FaceRegionTelemetry, TrackedObject
from ai.attributes.color_analyzer import RobustColorExtractor, TemporalColorFilter, VehicleColorAnalyzer
from ai.attributes.person_analyzer import PersonAttributeAnalyzer
from ai.faces.face_detector import FaceDetector
from ai.specialized.fire_smoke.detector import FireVisualDetector
from ai.incidents.schemas import IncidentContext, SupportingSignal
from ai.incidents.negative_evidence import NegativeEvidenceEngine
from ai.incidents.detectors.theft_and_takeaway import TheftAndTakeawayDetector
from ai.common.benchmark_interfaces import (
    YOLOAdapter,
    ByteTrackAdapter,
    PersonAttributeAdapter,
    VehicleColorAdapter,
)
from ai.investigation.provider import (
    DeterministicFallbackProvider,
    LocalLLMProvider,
    get_llm_provider,
)


def test_anatomical_region_partitioning():
    """Verify PersonAttributeAnalyzer partitions upper body and lower body correctly."""
    analyzer = PersonAttributeAnalyzer()
    
    # Synthetic frame (400x200): Upper body blue, lower body black
    frame = np.full((400, 200, 3), 128, dtype=np.uint8)
    # Upper body region: 18% to 52% of 400 = 72 to 208
    frame[72:208, 30:170] = (220, 50, 30)  # BGR blue
    # Lower body region: 56% to 88% of 400 = 224 to 352
    frame[224:352, 40:160] = (10, 10, 10)  # BGR black/dark

    bbox = BoundingBox(x1=0, y1=0, x2=200, y2=400)
    result = analyzer.analyze(frame_bgr=frame, person_bbox=bbox, timestamp=1.0)

    assert result.track_id in ("ANON", "anonymous")
    assert result.upper_clothing is not None
    assert result.upper_clothing.color == "blue"
    assert result.lower_clothing is not None
    # Luminance 10 is < 30, so RobustColorExtractor correctly marks it uncertain/illumination-uncertain
    assert result.lower_clothing.color in ("black", "grey", "dark_blue", "uncertain")
    if result.lower_clothing.color == "uncertain":
        assert result.lower_clothing.is_illumination_uncertain is True
    assert result.face_telemetry is not None
    assert isinstance(result.face_telemetry, FaceRegionTelemetry)


def test_low_illumination_uncertainty():
    """Verify RobustColorExtractor marks low-illumination crops as uncertain, not black."""
    extractor = RobustColorExtractor()
    
    # Underexposed crop with mean luminance < 30
    dark_crop = np.full((100, 100, 3), 18, dtype=np.uint8)
    color, conf, is_illum_unc = extractor.extract_dominant_color(dark_crop)

    assert color == "uncertain"
    assert is_illum_unc is True
    assert conf <= 0.30


def test_monochromatic_ir_uncertainty():
    """Verify RobustColorExtractor marks monochromatic IR frames as uncertain."""
    extractor = RobustColorExtractor()
    
    # Grayscale image (Saturation = 0 < 15)
    gray_val = 140
    ir_crop = np.full((100, 100, 3), gray_val, dtype=np.uint8)
    color, conf, is_illum_unc = extractor.extract_dominant_color(ir_crop)

    assert color == "uncertain"
    assert is_illum_unc is True


def test_temporal_color_stability():
    """Verify TemporalColorFilter requires repeated observations and resists single-frame noise."""
    temporal = TemporalColorFilter()
    track_id = "TRACK_P01"

    # Frame 1: blue with moderate confidence (not yet confirmed)
    col1, conf1, conf_flag1 = temporal.update(track_id, "blue", 0.62, 1.0)
    assert not conf_flag1

    # Frame 2: blue continues
    col2, conf2, conf_flag2 = temporal.update(track_id, "blue", 0.71, 2.0)
    # Frame 3: blue continues
    col3, conf3, conf_flag3 = temporal.update(track_id, "blue", 0.77, 3.0)
    assert conf_flag3 is True
    assert col3 == "blue"

    # Frame 4: single bad/glare frame of yellow with lower confidence
    col4, conf4, conf_flag4 = temporal.update(track_id, "yellow", 0.35, 4.0)
    # Established color must remain blue!
    assert col4 == "blue"
    assert conf_flag4 is True


def test_face_telemetry_strict_privacy():
    """Verify face telemetry returns optical quality and orientation without biometric identity."""
    face_detector = FaceDetector()
    frame = np.full((300, 200, 3), 120, dtype=np.uint8)
    bbox = BoundingBox(x1=20, y1=20, x2=180, y2=280)

    telemetry = face_detector.analyze_face_telemetry(frame, bbox, timestamp=1.5)
    data = telemetry.to_dict()

    # Must contain observational metadata
    assert "face_present" in data
    assert "visibility_score" in data
    assert "quality_score" in data
    assert "approximate_orientation" in data
    assert "is_occluded" in data

    # Must NOT contain biometric or identification fields
    forbidden_biometrics = ["identity", "name", "embedding", "vector", "face_id", "gallery_id", "similarity"]
    for k in forbidden_biometrics:
        assert k not in data


def test_fire_semantic_isolation_red_clothing():
    """Verify bright red clothing does NOT trigger false fire detection."""
    detector = FireVisualDetector(enabled=True)
    
    # Frame with a bright red shirt (BGR: 20, 20, 230)
    frame = np.full((400, 400, 3), 180, dtype=np.uint8)
    frame[100:250, 150:250] = (20, 20, 230)
    person_bbox = {"x1": 120, "y1": 80, "x2": 280, "y2": 380}

    obs = detector.detect(
        frame=frame,
        timestamp=1.0,
        frame_idx=1,
        context={"person_bounding_boxes": [person_bbox]},
    )
    # Must yield zero fire observations
    assert len(obs) == 0


def test_theft_negative_evidence_transient_drop():
    """Verify single-frame transient object dropout is identified as negative evidence."""
    p_bbox = BoundingBox(x1=100, y1=100, x2=200, y2=300)
    person_track = TrackedObject(
        track_id="P1",
        object_class="person",
        first_seen=1.0,
        last_seen=6.0,
        confidence=0.85,
        current_bbox=p_bbox,
        trajectory=[(1.0, 50.0, 100.0), (3.0, 150.0, 100.0), (6.0, 300.0, 100.0)],
        history_bboxes=[{"timestamp": t, "bbox": p_bbox.to_dict()} for t in range(1, 7)],
    )
    # Transient object: detected in only 1 frame, duration < 0.5s
    o_bbox = BoundingBox(x1=150, y1=100, x2=180, y2=140)
    obj_track = TrackedObject(
        track_id="O1",
        object_class="backpack",
        first_seen=3.0,
        last_seen=3.1,
        confidence=0.85,
        current_bbox=o_bbox,
        trajectory=[(3.0, 160.0, 120.0)],
        history_bboxes=[{"timestamp": 3.0, "bbox": o_bbox.to_dict()}],
    )

    ctx = IncidentContext(
        video_id="v_test",
        tracks=[person_track, obj_track],
        duration_seconds=10.0,
    )

    neg_signals = NegativeEvidenceEngine.evaluate_theft_negative_evidence(
        person_track=person_track,
        object_track=obj_track,
        context=ctx,
        interaction_end_time=3.5,
    )

    # Must contain Transient Object Detection signal
    signal_types = [s.signal_type for s in neg_signals]
    assert "Negative: Transient Object Detection" in signal_types

    # Theft detector must suppress transient dropout
    detector = TheftAndTakeawayDetector()
    candidates = detector.analyze(ctx)
    assert len(candidates) == 0


def test_theft_detector_continuous_walking_no_steal():
    """Verify person walking past an object that remained present produces zero theft candidates."""
    p_bbox = BoundingBox(x1=300, y1=100, x2=350, y2=250)
    person_track = TrackedObject(
        track_id="P1",
        object_class="person",
        first_seen=1.0,
        last_seen=5.0,
        confidence=0.85,
        current_bbox=p_bbox,
        trajectory=[(1.0, 50.0, 100.0), (3.0, 150.0, 100.0), (5.0, 300.0, 100.0)],
        history_bboxes=[{"timestamp": t, "bbox": p_bbox.to_dict()} for t in range(1, 6)],
    )
    # Object that stayed in scene long after person departed
    o_bbox = BoundingBox(x1=150, y1=100, x2=180, y2=140)
    obj_track = TrackedObject(
        track_id="O1",
        object_class="suitcase",
        first_seen=1.0,
        last_seen=10.0,
        confidence=0.85,
        current_bbox=o_bbox,
        trajectory=[(t, 165.0, 120.0) for t in range(1, 11)],
        history_bboxes=[{"timestamp": t, "bbox": o_bbox.to_dict()} for t in range(1, 11)],
    )

    ctx = IncidentContext(
        video_id="v_test",
        tracks=[person_track, obj_track],
        duration_seconds=10.0,
    )

    detector = TheftAndTakeawayDetector()
    candidates = detector.analyze(ctx)
    assert len(candidates) == 0


def test_benchmark_interfaces_instantiation():
    """Verify BaseObjectDetector, BaseMultiObjectTracker, BaseVisualAttributeAnalyzer adapters instantiate."""
    byte_track_adapter = ByteTrackAdapter()
    assert byte_track_adapter is not None

    person_adapter = PersonAttributeAdapter()
    assert person_adapter is not None

    vehicle_adapter = VehicleColorAdapter()
    assert vehicle_adapter is not None


def test_investigation_provider_independent_fallback():
    """Verify investigation agent functions deterministically without Gemini API key or local LLM."""
    prov = get_llm_provider(provider_name="deterministic")
    assert isinstance(prov, DeterministicFallbackProvider)
    assert prov.is_available()

    # Query structured parsing
    intent = prov.generate_structured_intent("find all cars between 5.0 and 15.0s")
    assert intent["intent"] == "investigate"
    assert "car" in intent["object_classes"]

    # Grounded response synthesis
    retrieved = {
        "results": [{"timestamp": 6.5, "object_class": "car", "confidence": 0.88}],
        "count": 1,
        "evidence": [{"has_snapshot": True, "has_clip": False}],
    }
    resp = prov.generate_grounded_response("where was the car?", retrieved)
    assert "Sentinel" in resp
    assert "car" in resp
