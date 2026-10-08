"""
Sentinel Regression Test Suite: Low-Light Color Reliability & Track Continuation (Phases 11 & 12)
===================================================================================================
Covers:
Phase 11: Positive & Negative Color Fixtures (A - I)
  A. True saturated blue clothing -> blue
  B. Dark navy clothing -> distinguish blue only when chromatic evidence is genuinely sufficient
  C. Black clothing under cool/blue illumination -> black/dark grey, NOT blue
  D. Grey clothing under cool illumination -> grey
  E. Black clothing with blue background -> background must not make clothing blue
  F. True blue person at image edge -> remains detectable if evidence is sufficient
  G. One noisy blue frame inside a black track -> temporal consensus remains black
  H. Multiple consistent blue observations -> blue confidence rises
  I. Ambiguous low-light crop -> unknown/low confidence rather than false high-confidence blue

Phase 12: Entity Reconciliation & Continuation Tests
  - Same physical person fragmented across moderate gap -> reconnects with strong continuity
  - Different people passing through same location -> remain separate
  - Three people in similar black clothing -> remain three people
  - Single-frame edge artifact -> raw observation exists, but filtered from verified searchable person entities
  - Same track IDs in different videos -> remain completely isolated
"""

import math
import uuid
import numpy as np
import pytest

from ai.schemas import BoundingBox, ClothingColor, TrackedObject
from ai.attributes.color_analyzer import RobustColorExtractor, TemporalColorFilter
from ai.attributes.person_analyzer import PersonAttributeAnalyzer
from ai.intelligence_pipeline import aggregate_track_clothing_color
from backend.app.services.investigation_service import InvestigationService


# ============================================================================
# PHASE 11: COLOR FIXTURES (A - I)
# ============================================================================

def test_phase11_a_true_saturated_blue_clothing():
    """A. True saturated blue clothing -> classifies as blue with high confidence."""
    # Saturated blue in BGR: high B, low R/G
    blue_roi = np.full((60, 40, 3), (210, 50, 20), dtype=np.uint8)
    res = RobustColorExtractor.extract_from_roi(blue_roi)
    assert res["dominant_color"] == "blue"
    assert res["dominant_confidence"] >= 0.70


def test_phase11_b_dark_navy_clothing():
    """B. Dark navy clothing -> blue only when chromatic saturation/evidence is genuinely sufficient."""
    # Deep navy with sufficient saturation: B=110, G=35, R=15
    navy_roi = np.full((60, 40, 3), (110, 35, 15), dtype=np.uint8)
    res = RobustColorExtractor.extract_from_roi(navy_roi)
    assert res["dominant_color"] == "blue"
    assert res["dominant_confidence"] >= 0.50


def test_phase11_c_black_clothing_under_cool_blue_illumination():
    """C. Black clothing under cool/blue fluorescent illumination -> black/dark grey, NOT blue."""
    # Dark neutral jacket with weak cool ambient bias: B=50, G=42, R=40 (slight B>R/G)
    cool_black_roi = np.full((60, 40, 3), (50, 42, 40), dtype=np.uint8)
    res = RobustColorExtractor.extract_from_roi(cool_black_roi)
    assert res["dominant_color"] in ("black", "grey")
    assert res["dominant_color"] != "blue"


def test_phase11_d_grey_clothing_under_cool_illumination():
    """D. Grey clothing under cool illumination -> grey, NOT blue."""
    # Grey fabric with weak cool camera bias: B=96, G=88, R=86
    cool_grey_roi = np.full((60, 40, 3), (96, 88, 86), dtype=np.uint8)
    res = RobustColorExtractor.extract_from_roi(cool_grey_roi)
    assert res["dominant_color"] in ("grey", "silver")
    assert res["dominant_color"] != "blue"


def test_phase11_e_black_clothing_with_blue_background():
    """E. Black clothing with blue background margin -> torso extraction and achromatic dominance prevent blue."""
    # Synthetic frame: background has blue wall (BGR 180, 70, 40), person has black jacket (BGR 45, 38, 35)
    frame = np.full((300, 200, 3), (180, 70, 40), dtype=np.uint8)
    # Person bbox from (50, 50) to (250, 150)
    # Inner body is black fabric: (80:230, 70:130)
    frame[80:230, 70:130] = (45, 38, 35)

    analyzer = PersonAttributeAnalyzer()
    bbox = BoundingBox(x1=50, y1=50, x2=150, y2=250)
    attr = analyzer.analyze(frame, bbox, timestamp=1.0)
    assert attr.upper_clothing.color in ("black", "grey")
    assert attr.upper_clothing.color != "blue"



def test_phase11_f_true_blue_person_at_image_edge():
    """F. True blue person at image edge remains detectable if chromatic evidence is sufficient."""
    # Frame with person clipped on right margin
    frame = np.full((300, 400, 3), 100, dtype=np.uint8)
    # Person at right edge: bbox from x=350 to x=400 (clipped)
    frame[60:200, 355:398] = (220, 60, 30) # Vibrant blue upper body

    analyzer = PersonAttributeAnalyzer()
    bbox = BoundingBox(x1=350, y1=30, x2=400, y2=260)
    attr = analyzer.analyze(frame, bbox, timestamp=1.0)
    assert attr.upper_clothing.color == "blue"
    assert attr.upper_clothing.confidence >= 0.40


def test_phase11_g_one_noisy_blue_frame_inside_black_track():
    """G. One noisy blue frame inside a black track -> temporal consensus remains black."""
    track = TrackedObject(
        track_id="TRACK-P01",
        object_class="person",
        first_seen=0.0,
        last_seen=5.0,
        confidence=0.85,
        current_bbox=BoundingBox(10.0, 10.0, 50.0, 100.0),
        trajectory=[],
    )
    # 4 black frames + 1 noisy blue frame
    track.attribute_history = [
        {"upper_clothing": {"color_name": "black", "confidence": 0.85, "is_illumination_uncertain": False}},
        {"upper_clothing": {"color_name": "black", "confidence": 0.80, "is_illumination_uncertain": False}},
        {"upper_clothing": {"color_name": "blue", "confidence": 0.55, "is_illumination_uncertain": False}}, # Noisy frame
        {"upper_clothing": {"color_name": "black", "confidence": 0.90, "is_illumination_uncertain": False}},
        {"upper_clothing": {"color_name": "black", "confidence": 0.82, "is_illumination_uncertain": False}},
    ]
    aggregate_track_clothing_color(track)
    assert track.color == "black"
    assert track.color_confidence >= 0.70
    assert track.visual_attributes["clothing_color"]["is_confirmed"] is True


def test_phase11_h_multiple_consistent_blue_observations_rises_confidence():
    """H. Multiple consistent blue observations -> blue confidence rises progressively."""
    t2 = TrackedObject(
        track_id="TRACK-P02",
        object_class="person",
        first_seen=0.0,
        last_seen=2.0,
        confidence=0.80,
        current_bbox=BoundingBox(10.0, 10.0, 50.0, 100.0),
        trajectory=[],
    )
    t2.attribute_history = [
        {"upper_clothing": {"color_name": "blue", "confidence": 0.75, "is_illumination_uncertain": False}},
        {"upper_clothing": {"color_name": "blue", "confidence": 0.75, "is_illumination_uncertain": False}},
    ]
    aggregate_track_clothing_color(t2)
    conf_2_obs = t2.color_confidence

    t5 = TrackedObject(
        track_id="TRACK-P05",
        object_class="person",
        first_seen=0.0,
        last_seen=5.0,
        confidence=0.80,
        current_bbox=BoundingBox(10.0, 10.0, 50.0, 100.0),
        trajectory=[],
    )
    t5.attribute_history = [
        {"upper_clothing": {"color_name": "blue", "confidence": 0.75, "is_illumination_uncertain": False}}
        for _ in range(5)
    ]
    aggregate_track_clothing_color(t5)
    conf_5_obs = t5.color_confidence

    assert t2.color == "blue"
    assert t5.color == "blue"
    assert conf_5_obs > conf_2_obs


def test_phase11_i_ambiguous_low_light_crop():
    """I. Ambiguous low-light crop -> unknown/low confidence rather than false high-confidence blue."""
    # Crop with mean luminance below threshold or highly ambiguous noisy values
    dark_roi = np.full((40, 30, 3), 22, dtype=np.uint8)
    res = RobustColorExtractor.extract_from_roi(dark_roi)
    assert res["dominant_color"] in ("uncertain", "unknown")
    assert res["dominant_confidence"] <= 0.40


# ============================================================================
# PHASE 12: ENTITY RECONCILIATION & CONTINUATION TESTS
# ============================================================================

def test_phase12_same_physical_person_fragmented_across_moderate_gap():
    """Same physical person fragmented across moderate gap reconnects with strong spatial continuity."""
    svc = InvestigationService()
    vid = f"vid_cont_{uuid.uuid4().hex[:6]}"

    # Stationary burglar rummaging behind counter: t1 ends at t=10.0s at (140, 95), t2 starts at t=14.0s at (141, 96)
    t1 = {
        "video_id": vid,
        "track_id": "TRACK-001",
        "object_class": "person",
        "first_seen": 0.0,
        "last_seen": 10.0,
        "duration_seconds": 10.0,
        "detection_count": 10,
        "max_confidence": 0.85,
        "color": "black",
        "color_confidence": 0.75,
        "current_bbox": {"x1": 125, "y1": 60, "x2": 155, "y2": 130},
        "trajectory": [[9.0, 140.0, 94.0], [10.0, 140.0, 95.0]],
    }
    t2 = {
        "video_id": vid,
        "track_id": "TRACK-002",
        "object_class": "person",
        "first_seen": 14.0, # 4-second gap
        "last_seen": 22.0,
        "duration_seconds": 8.0,
        "detection_count": 8,
        "max_confidence": 0.88,
        "color": "black",
        "color_confidence": 0.70,
        "current_bbox": {"x1": 126, "y1": 61, "x2": 156, "y2": 131},
        "trajectory": [[14.0, 141.0, 96.0], [15.0, 142.0, 96.0]],
    }

    canon = svc._reconcile_canonical_entities([t1, t2], video_id=vid)
    assert len(canon) == 1
    assert "TRACK-001" in canon[0]["member_track_ids"]
    assert "TRACK-002" in canon[0]["member_track_ids"]


def test_phase12_different_people_passing_through_same_location_remain_separate():
    """Different people passing through same doorway at different times with infeasible velocity remain separate."""
    svc = InvestigationService()
    vid = f"vid_pass_{uuid.uuid4().hex[:6]}"

    # Person 1 enters door, moves into building towards (300, 150)
    t1 = {
        "video_id": vid,
        "track_id": "TRACK-001",
        "object_class": "person",
        "first_seen": 0.0,
        "last_seen": 5.0,
        "duration_seconds": 5.0,
        "detection_count": 5,
        "max_confidence": 0.85,
        "color": "black",
        "color_confidence": 0.75,
        "current_bbox": {"x1": 280, "y1": 100, "x2": 320, "y2": 200},
        "trajectory": [[4.0, 260.0, 150.0], [5.0, 300.0, 150.0]],
    }
    # Person 2 enters door at (50, 150) 6 seconds later
    t2 = {
        "video_id": vid,
        "track_id": "TRACK-002",
        "object_class": "person",
        "first_seen": 11.0, # 6-second gap
        "last_seen": 16.0,
        "duration_seconds": 5.0,
        "detection_count": 5,
        "max_confidence": 0.80,
        "color": "black",
        "color_confidence": 0.70,
        "current_bbox": {"x1": 40, "y1": 100, "x2": 80, "y2": 200},
        "trajectory": [[11.0, 50.0, 150.0], [12.0, 60.0, 150.0]],
    }

    canon = svc._reconcile_canonical_entities([t1, t2], video_id=vid)
    assert len(canon) == 2


def test_phase12_three_people_in_similar_black_clothing_remain_three_people():
    """Three co-occurring people wearing black in the scene remain three distinct entities."""
    svc = InvestigationService()
    vid = f"vid_three_{uuid.uuid4().hex[:6]}"

    # Burglar A: at counter (x=160)
    t_a = {
        "video_id": vid,
        "track_id": "TRACK-001",
        "object_class": "person",
        "first_seen": 10.0,
        "last_seen": 20.0,
        "duration_seconds": 10.0,
        "detection_count": 10,
        "max_confidence": 0.85,
        "color": "black",
        "color_confidence": 0.80,
        "current_bbox": {"x1": 145, "y1": 60, "x2": 175, "y2": 135},
        "trajectory": [[15.0, 160.0, 95.0]],
    }
    # Burglar B: on floor crawling (x=100)
    t_b = {
        "video_id": vid,
        "track_id": "TRACK-002",
        "object_class": "person",
        "first_seen": 10.0,
        "last_seen": 20.0,
        "duration_seconds": 10.0,
        "detection_count": 10,
        "max_confidence": 0.82,
        "color": "black",
        "color_confidence": 0.75,
        "current_bbox": {"x1": 85, "y1": 70, "x2": 115, "y2": 140},
        "trajectory": [[15.0, 100.0, 105.0]],
    }
    # Burglar C: at doorway (x=50)
    t_c = {
        "video_id": vid,
        "track_id": "TRACK-003",
        "object_class": "person",
        "first_seen": 10.0,
        "last_seen": 20.0,
        "duration_seconds": 10.0,
        "detection_count": 10,
        "max_confidence": 0.78,
        "color": "black",
        "color_confidence": 0.70,
        "current_bbox": {"x1": 35, "y1": 60, "x2": 65, "y2": 130},
        "trajectory": [[15.0, 50.0, 95.0]],
    }

    canon = svc._reconcile_canonical_entities([t_a, t_b, t_c], video_id=vid)
    assert len(canon) == 3


def test_phase12_single_frame_edge_artifact_is_gated_from_canonical_entities():
    """A 1-frame edge artifact (like TRACK-039) is filtered from verified searchable physical entities."""
    svc = InvestigationService()
    vid = f"vid_art_{uuid.uuid4().hex[:6]}"

    # Valid persistent track
    valid_t = {
        "video_id": vid,
        "track_id": "TRACK-001",
        "object_class": "person",
        "first_seen": 5.0,
        "last_seen": 15.0,
        "duration_seconds": 10.0,
        "detection_count": 10,
        "max_confidence": 0.85,
        "color": "black",
        "color_confidence": 0.75,
        "current_bbox": {"x1": 100, "y1": 60, "x2": 140, "y2": 140},
        "trajectory": [[10.0, 120.0, 100.0]],
    }
    # Single-frame edge noise (touching x1=0.04, y2=239.5, 1 detection, confidence 0.28)
    artifact_t = {
        "video_id": vid,
        "track_id": "TRACK-039",
        "object_class": "person",
        "first_seen": 12.0,
        "last_seen": 12.0,
        "duration_seconds": 0.0,
        "detection_count": 1,
        "max_confidence": 0.28,
        "color": "blue",
        "color_confidence": 0.40,
        "current_bbox": {"x1": 0.04, "y1": 187.0, "x2": 41.8, "y2": 239.5},
        "trajectory": [[12.0, 20.0, 210.0]],
    }

    canon = svc._reconcile_canonical_entities([valid_t, artifact_t], video_id=vid)
    assert len(canon) == 1
    assert canon[0]["canonical_id"] == "TRACK-001"
    assert "TRACK-039" not in [ce["canonical_id"] for ce in canon]


def test_phase12_same_track_ids_in_different_videos_remain_isolated():
    """Tracks with identical track_ids in different videos remain completely isolated."""
    svc = InvestigationService()
    vid_a = f"vid_iso_a_{uuid.uuid4().hex[:6]}"
    vid_b = f"vid_iso_b_{uuid.uuid4().hex[:6]}"

    t_a = {
        "video_id": vid_a,
        "track_id": "TRACK-001",
        "object_class": "person",
        "first_seen": 0.0,
        "last_seen": 10.0,
        "duration_seconds": 10.0,
        "detection_count": 10,
        "max_confidence": 0.90,
        "color": "blue",
        "color_confidence": 0.85,
        "current_bbox": {"x1": 100, "y1": 50, "x2": 150, "y2": 150},
        "trajectory": [[5.0, 125.0, 100.0]],
    }
    t_b = {
        "video_id": vid_b,
        "track_id": "TRACK-001",
        "object_class": "person",
        "first_seen": 0.0,
        "last_seen": 10.0,
        "duration_seconds": 10.0,
        "detection_count": 10,
        "max_confidence": 0.90,
        "color": "blue",
        "color_confidence": 0.85,
        "current_bbox": {"x1": 100, "y1": 50, "x2": 150, "y2": 150},
        "trajectory": [[5.0, 125.0, 100.0]],
    }

    # Video A scoped query
    canon_a = svc._reconcile_canonical_entities([t_a, t_b], video_id=vid_a)
    assert len(canon_a) == 1
    assert canon_a[0]["video_id"] == vid_a

    # Video B scoped query
    canon_b = svc._reconcile_canonical_entities([t_a, t_b], video_id=vid_b)
    assert len(canon_b) == 1
    assert canon_b[0]["video_id"] == vid_b
