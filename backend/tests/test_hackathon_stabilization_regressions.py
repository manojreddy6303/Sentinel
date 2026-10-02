"""
Test Hackathon Stabilization Regressions

Covers the three user-facing regressions identified in the Prototype Reliability Audit:
1. Natural-language visual investigation (parser vocabulary, track-level clothing color aggregation, grounded fallback)
2. False potential-fire detection from retail merchandise (core saturation ratio, rigid background motion)
3. Evidence clip browser playback & memory safety (H.264 extraction, cgroup page cache headroom, range playback)
"""

import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest

from backend.app.services.investigation_parser import InvestigationParser
from ai.intelligence_pipeline import aggregate_track_clothing_color
from ai.schemas import TrackedObject, BoundingBox, ClothingColor
from ai.specialized.fire_smoke.detector import FireVisualDetector
from ai.specialized.validator import SpecializedValidationEngine
from ai.specialized.schemas import SpecializedObservation, SpecializedValidationStatus
from ai.investigation.orchestrator import InvestigationOrchestrator
from backend.app.services.playback_service import (
    get_container_memory_headroom_mb,
    is_browser_compatible,
)


# ============================================================================
# REGRESSION 1: Natural-Language Visual Investigation
# ============================================================================

def test_investigation_parser_recognizes_natural_visual_terms():
    """Verify that queries with dress, lady, woman, man, clothing synonyms identify person queries."""
    queries = [
        "what is blue colour dress lady did",
        "what did the lady in the blue dress do",
        "show me woman wearing red jacket",
        "find man in dark pants",
        "person in yellow clothes",
        "what happened with the lady in blue",
    ]
    for q in queries:
        parsed = InvestigationParser.parse_query(q)
        assert parsed["is_supported"] is True, f"Query '{q}' not supported"
        filters = parsed.get("interpreted_filters", {})
        assert filters.get("object_class") == "person", f"Failed to identify person for query: '{q}'"


def test_investigation_parser_extracts_clothing_color_and_action():
    """Verify color extraction and action intent detection for natural queries."""
    parsed = InvestigationParser.parse_query("what is blue colour dress lady did")
    assert parsed["is_supported"] is True
    assert parsed["result_type"] == "tracks"
    filters = parsed["interpreted_filters"]
    assert filters.get("object_class") == "person"
    assert filters.get("color") == "blue"
    assert filters.get("query_action") is True
    assert filters.get("clothing_descriptor") == "dress"

    parsed_red = InvestigationParser.parse_query("person in red dress walking")
    assert parsed_red["is_supported"] is True
    assert parsed_red["result_type"] == "tracks"
    filters_red = parsed_red["interpreted_filters"]
    assert filters_red.get("object_class") == "person"
    assert filters_red.get("color") == "red"
    assert filters_red.get("query_action") is True


def test_clothing_color_track_level_temporal_aggregation():
    """Verify that multiple consistent frame observations aggregate into a confirmed track clothing color."""
    track = TrackedObject(
        track_id="TRACK-001",
        object_class="person",
        first_seen=0.0,
        last_seen=5.0,
        confidence=0.88,
        current_bbox=BoundingBox(10.0, 10.0, 50.0, 100.0),
        trajectory=[],
    )
    # 4 observations with blue clothing
    blue_color = ClothingColor(
        color_name="blue",
        confidence=0.75,
        observation_count=1,
    )
    track.attribute_history = [
        {"clothing_color": blue_color.to_dict()},
        {"clothing_color": blue_color.to_dict()},
        {"clothing_color": blue_color.to_dict()},
        {"clothing_color": blue_color.to_dict()},
    ]

    aggregate_track_clothing_color(track)
    assert track.visual_attributes is not None
    c_color = track.visual_attributes.get("clothing_color", {})
    assert c_color.get("color") == "blue"
    assert c_color.get("observation_count") == 4
    assert c_color.get("is_confirmed") is True


def test_clothing_color_preserves_uncertainty_when_ambiguous():
    """Verify that contradictory or low-confidence color observations do NOT confirm a false color."""
    track = TrackedObject(
        track_id="TRACK-002",
        object_class="person",
        first_seen=0.0,
        last_seen=3.0,
        confidence=0.70,
        current_bbox=BoundingBox(10.0, 10.0, 50.0, 100.0),
        trajectory=[],
    )
    # Contradictory observations: red, green, black
    track.attribute_history = [
        {"clothing_color": ClothingColor(color_name="red", confidence=0.4, observation_count=1).to_dict()},
        {"clothing_color": ClothingColor(color_name="green", confidence=0.4, observation_count=1).to_dict()},
        {"clothing_color": ClothingColor(color_name="black", confidence=0.4, observation_count=1).to_dict()},
    ]

    aggregate_track_clothing_color(track)
    if track.visual_attributes and "clothing_color" in track.visual_attributes:
        assert track.visual_attributes["clothing_color"].get("is_confirmed") is False


def test_deterministic_grounded_fallback_narrative():
    """Verify deterministic fallback produces grounded narrative without hallucination or biometrics."""
    orch = InvestigationOrchestrator()
    search_results = {
        "count": 1,
        "result_type": "tracks",
        "filters": {"object_class": "person", "color": "blue"},
        "results": [
            {
                "track_id": "TRACK-001",
                "object_class": "person",
                "first_seen": 0.0,
                "last_seen": 20.0,
                "duration_seconds": 20.0,
                "detection_count": 20,
                "color": "blue",
                "activity_summary": "Continuous visual presence observed from 0.0s to 20.0s across 20 detections (duration: 20.0s). Associated security event(s): POTENTIAL_PERSON_FALL.",
            }
        ],
    }

    response = orch._format_deterministic_grounded_response(search_results)
    assert "TRACK-001" in response
    assert "blue clothing" in response
    assert "0.0s to 20.0s" in response
    assert "POTENTIAL_PERSON_FALL" in response
    assert "zero personal identity attribution" in response


# ============================================================================
# REGRESSION 2: False Potential Fire From Retail Merchandise
# ============================================================================

def test_merchandise_specular_glare_rejected():
    """
    Verify that an illuminated warm-colored retail shelf object with low incandescent core ratio
    is rejected from being classified as fire.
    """
    detector = FireVisualDetector()
    detector.initialize()
    # Construct a synthetic frame simulating store packaging:
    # Warm YCrCb / HSV patch (orange/yellow packaging) but without an incandescent core (V < 235)
    frame = np.full((300, 300, 3), (20, 100, 200), dtype=np.uint8)  # BGR warm orange
    # Place a 50x50 warm shelf package patch where max V is 210 (below 235 incandescent core)
    frame[100:150, 100:150] = (30, 120, 210)

    observations = detector.detect(frame, timestamp=1.0)
    fire_obs = [obs for obs in observations if obs.class_name == "fire"]
    # Should not produce a raw fire observation due to lack of incandescent core
    assert len(fire_obs) == 0


def test_validator_rejects_rigid_background_and_low_core_ratio():
    """Verify validator suppresses candidates with rigid background motion or low core ratio."""
    engine = SpecializedValidationEngine()
    # Candidate with low core ratio (simulating merchandise)
    cand_low_core = SpecializedObservation(
        observation_id="obs_low_core",
        detector_name="fire_detector",
        detector_version="1.0.0",
        class_name="fire",
        timestamp=17.0,
        confidence=0.55,
        evidence_strength=0.50,
        bounding_box=BoundingBox(228.0, 231.0, 278.0, 270.0),
        visual_metrics={
            "inference_source": "forensic_chromatic_rules",
            "incandescent_core_ratio": 0.02,
            "is_rigid_background": False,
            "max_luminance": 240.0,
        },
    )
    val_obs = engine.validate_observation(cand_low_core)
    assert val_obs.validation_status == SpecializedValidationStatus.REJECTED
    assert "Specular reflection" in (val_obs.validation_reason or "")

    # Candidate moving rigidly with handheld camera
    cand_rigid = SpecializedObservation(
        observation_id="obs_rigid",
        detector_name="fire_detector",
        detector_version="1.0.0",
        class_name="fire",
        timestamp=18.0,
        confidence=0.60,
        evidence_strength=0.55,
        bounding_box=BoundingBox(228.0, 231.0, 278.0, 270.0),
        visual_metrics={
            "inference_source": "forensic_chromatic_rules",
            "incandescent_core_ratio": 0.25,
            "is_rigid_background": True,
            "max_luminance": 240.0,
        },
    )
    val_obs_rigid = engine.validate_observation(cand_rigid)
    assert val_obs_rigid.validation_status == SpecializedValidationStatus.REJECTED
    assert "Static background fixture" in (val_obs_rigid.validation_reason or "")


# ============================================================================
# REGRESSION 3: Evidence Clip Playback & Cgroup Headroom
# ============================================================================

def test_cgroup_reclaimable_page_cache_headroom_calculation(monkeypatch, tmp_path):
    """
    Verify that Linux page-cache accounting does not falsely deplete container headroom.
    Simulate cgroup v2 with 1024 MB limit, 1023.6 MB current usage (leaving 0.4 MB raw),
    but 400 MB of inactive_file reclaimable cache in memory.stat.
    The effective headroom should be ~400.4 MB, well above the 120 MB threshold.
    """
    cg2_dir = tmp_path / "sys_fs_cgroup"
    cg2_dir.mkdir(parents=True)
    cg2_max = cg2_dir / "memory.max"
    cg2_curr = cg2_dir / "memory.current"
    cg2_stat = cg2_dir / "memory.stat"

    # 1024 MB max
    cg2_max.write_text("1073741824\n")
    # 1023.6 MB current (leaves 0.4 MB if raw)
    cg2_curr.write_text("1073322000\n")
    # 400 MB inactive_file + 50 MB slab_reclaimable
    cg2_stat.write_text(
        "anon 573322000\n"
        "file 500000000\n"
        "inactive_file 419430400\n"
        "slab_reclaimable 52428800\n"
    )

    # Monkeypatch paths in playback_service
    import backend.app.services.playback_service as pbs
    monkeypatch.setattr(pbs, "Path", lambda p: cg2_dir / Path(p).name if "sys/fs/cgroup" in str(p) else Path(p))

    headroom = pbs.get_container_memory_headroom_mb()
    # 1073741824 - (1073322000 - (419430400 + 52428800)) = 450 MB approx
    assert headroom > 400.0, f"Headroom calculation failed: expected > 400 MB, got {headroom:.1f} MB"


def test_evidence_clip_extraction_browser_compatible(tmp_path):
    """Verify that evidence clip extraction generates a valid, browser-compatible video file."""
    from backend.app.services.playback_service import get_ffmpeg_binary
    from ai.extraction.extractor import ClipExtractor

    ffmpeg_exe = get_ffmpeg_binary()
    if not ffmpeg_exe:
        pytest.skip("FFmpeg not available in test environment")

    # Use existing benchmark video if available
    bench_video = Path("storage/uploads/5e68d2cc-b315-4534-af53-ec3be34ad076_WhatsApp_Video_2026-09-26_at_06.14.37.mp4")
    if not bench_video.exists():
        pytest.skip(f"Benchmark video {bench_video} not found")

    extractor = ClipExtractor(output_dir=str(tmp_path))
    clip_info = extractor.extract_clip(
        source_video_path=str(bench_video),
        start_time=1.0,
        end_time=3.0,
        clip_id="test_stab_ev",
    )

    clip_file = Path(clip_info["path"])
    assert clip_file.exists()
    assert clip_file.stat().st_size > 0
    assert is_browser_compatible(clip_file) is True
    assert clip_info["duration"] == 2.0
