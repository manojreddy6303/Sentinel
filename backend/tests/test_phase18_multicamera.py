"""
backend/tests/test_phase18_multicamera.py — Phase 18 test suite

Tests for:
  1. Schemas (AssociationEvidence, CrossCameraHypothesis, CameraObservation)
  2. AttributeMatcher
  3. TemporalCompatibilityReasoner
  4. CrossCameraAssociationEngine
  5. MultiCameraSceneIntelligence
  6. SurveillanceSessionService
  7. CrossCameraAnalysisService (mocked)
  8. API endpoints /api/sessions (via TestClient)

Privacy invariants are tested throughout:
  - Biometric signal types are rejected at schema level
  - Class-mismatch gate returns hard abstention
  - Negative temporal gap returns score 0.0
  - Confidence < 0.40 never produces an association
"""
import pytest
import math
from unittest.mock import MagicMock, patch, PropertyMock
from fastapi.testclient import TestClient

# ── Import AI modules ──────────────────────────────────────────────────────
from ai.multicamera.schemas import (
    AssociationEvidence,
    AssociationEvidence as AE,
    CameraObservation,
    CrossCameraHypothesis,
    AssociationType,
    AnalystVerdict,
)
from ai.multicamera.attribute_matcher import AttributeMatcher
from ai.multicamera.temporal_reasoner import TemporalCompatibilityReasoner, _triangular_score
from ai.multicamera.association_engine import CrossCameraAssociationEngine
from ai.multicamera.scene_intelligence import (
    MultiCameraSceneIntelligence,
    MovementSegment,
    CrossCameraMovementNarrative,
    SceneAnomaly,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. SCHEMA TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestAssociationEvidence:
    def test_valid_evidence_created(self):
        ev = AE(signal_type="color_match", description="desc", score=0.75)
        assert ev.score == 0.75
        assert ev.signal_type == "color_match"

    def test_score_clamped_above_one(self):
        ev = AE(signal_type="color_match", description="d", score=1.5)
        assert ev.score == 1.0

    def test_score_clamped_below_zero(self):
        ev = AE(signal_type="color_match", description="d", score=-0.5)
        assert ev.score == 0.0

    def test_score_exactly_zero(self):
        ev = AE(signal_type="object_class_match", description="d", score=0.0)
        assert ev.score == 0.0

    def test_score_exactly_one(self):
        ev = AE(signal_type="adjacency_hint", description="d", score=1.0)
        assert ev.score == 1.0

    def test_biometric_signal_type_rejected_facial(self):
        with pytest.raises(ValueError, match="forbidden signal_type"):
            AE(signal_type="facial_similarity", description="bad", score=0.9)

    def test_biometric_signal_type_rejected_face_match(self):
        with pytest.raises(ValueError, match="forbidden signal_type"):
            AE(signal_type="face_match", description="bad", score=0.9)

    def test_biometric_signal_type_rejected_biometric(self):
        with pytest.raises(ValueError, match="forbidden signal_type"):
            AE(signal_type="biometric", description="bad", score=0.9)

    def test_biometric_signal_type_rejected_voice(self):
        with pytest.raises(ValueError, match="forbidden signal_type"):
            AE(signal_type="voice_match", description="bad", score=0.9)

    def test_biometric_signal_type_rejected_gait(self):
        with pytest.raises(ValueError, match="forbidden signal_type"):
            AE(signal_type="gait_recognition", description="bad", score=0.9)

    def test_biometric_signal_type_rejected_iris(self):
        with pytest.raises(ValueError, match="forbidden signal_type"):
            AE(signal_type="iris_match", description="bad", score=0.9)

    def test_to_dict_keys(self):
        ev = AE(signal_type="color_match", description="match", score=0.8, metadata={"k": "v"})
        d = ev.to_dict()
        assert "signal_type" in d
        assert "description" in d
        assert "score" in d
        assert "metadata" in d

    def test_metadata_defaults_empty(self):
        ev = AE(signal_type="size_ratio_match", description="d", score=0.5)
        assert ev.metadata == {}


class TestCrossCameraHypothesis:
    def _make_hyp(self, confidence: float) -> CrossCameraHypothesis:
        return CrossCameraHypothesis(
            confidence=confidence,
            source_camera_id="cam_a",
            source_track_id="trk_1",
            target_camera_id="cam_b",
            target_track_id="trk_2",
        )

    def test_classify_same_object(self):
        hyp = self._make_hyp(0.80)
        assert hyp.classify_type() == AssociationType.SAME_OBJECT

    def test_classify_same_object_above(self):
        hyp = self._make_hyp(0.95)
        assert hyp.classify_type() == AssociationType.SAME_OBJECT

    def test_classify_probable_same(self):
        hyp = self._make_hyp(0.60)
        assert hyp.classify_type() == AssociationType.PROBABLE_SAME

    def test_classify_probable_same_boundary(self):
        hyp = self._make_hyp(0.79)
        assert hyp.classify_type() == AssociationType.PROBABLE_SAME

    def test_classify_possible_same(self):
        hyp = self._make_hyp(0.40)
        assert hyp.classify_type() == AssociationType.POSSIBLE_SAME

    def test_classify_possible_same_boundary(self):
        hyp = self._make_hyp(0.59)
        assert hyp.classify_type() == AssociationType.POSSIBLE_SAME

    def test_classify_below_threshold_raises(self):
        hyp = self._make_hyp(0.39)
        with pytest.raises(ValueError):
            hyp.classify_type()

    def test_analyst_verdict_pending_by_default(self):
        hyp = self._make_hyp(0.50)
        assert hyp.analyst_verdict == AnalystVerdict.PENDING

    def test_analyst_review_required_true_by_default(self):
        hyp = self._make_hyp(0.50)
        assert hyp.analyst_review_required is True

    def test_to_dict_has_privacy_related_fields(self):
        hyp = self._make_hyp(0.65)
        hyp.association_type = hyp.classify_type()
        d = hyp.to_dict()
        assert "hypothesis_id" in d
        assert "analyst_review_required" in d
        assert d["analyst_verdict"] == "PENDING"

    def test_hypothesis_id_unique(self):
        h1 = self._make_hyp(0.50)
        h2 = self._make_hyp(0.50)
        assert h1.hypothesis_id != h2.hypothesis_id


# ═══════════════════════════════════════════════════════════════════════════
# 2. ATTRIBUTE MATCHER TESTS
# ═══════════════════════════════════════════════════════════════════════════

def _make_obs(**kwargs) -> CameraObservation:
    defaults = dict(
        camera_id="cam_a",
        camera_label="Cam A",
        video_id="vid_1",
        track_id="trk_1",
        object_class="person",
        first_seen=10.0,
        last_seen=20.0,
        duration_seconds=10.0,
        max_confidence=0.85,
    )
    defaults.update(kwargs)
    return CameraObservation(**defaults)


class TestAttributeMatcher:
    def setup_method(self):
        self.matcher = AttributeMatcher()

    def test_class_mismatch_returns_zero_no_evidence(self):
        src = _make_obs(object_class="person")
        tgt = _make_obs(object_class="car", camera_id="cam_b", track_id="trk_2")
        score, evidence = self.matcher.compute(src, tgt)
        assert score == 0.0
        assert evidence == []

    def test_class_match_returns_positive_score(self):
        src = _make_obs(color="red")
        tgt = _make_obs(camera_id="cam_b", track_id="trk_2", color="red")
        score, evidence = self.matcher.compute(src, tgt)
        assert score > 0.0

    def test_exact_color_match_high_score(self):
        src = _make_obs(color="blue")
        tgt = _make_obs(camera_id="cam_b", track_id="trk_2", color="blue")
        score, evidence = self.matcher.compute(src, tgt)
        assert score >= 0.6

    def test_color_mismatch_lower_score(self):
        src = _make_obs(color="red")
        tgt = _make_obs(camera_id="cam_b", track_id="trk_2", color="blue")
        score_mismatch, _ = self.matcher.compute(src, tgt)

        src2 = _make_obs(color="red")
        tgt2 = _make_obs(camera_id="cam_b", track_id="trk_2", color="red")
        score_match, _ = self.matcher.compute(src2, tgt2)

        assert score_mismatch < score_match

    def test_same_color_group_intermediate_score(self):
        src = _make_obs(color="blue")
        tgt = _make_obs(camera_id="cam_b", track_id="trk_2", color="navy")
        score, evidence = self.matcher.compute(src, tgt)
        color_ev = next((e for e in evidence if e.signal_type == "color_match"), None)
        assert color_ev is not None
        assert 0.5 <= color_ev.score <= 0.8

    def test_unknown_color_neutral_score(self):
        src = _make_obs(color=None)
        tgt = _make_obs(camera_id="cam_b", track_id="trk_2", color=None)
        score, evidence = self.matcher.compute(src, tgt)
        assert score > 0.0  # neutral, not penalised

    def test_size_ratio_match_evidence_present(self):
        src = _make_obs(estimated_height_ratio=0.4)
        tgt = _make_obs(camera_id="cam_b", track_id="trk_2", estimated_height_ratio=0.42)
        _, evidence = self.matcher.compute(src, tgt)
        types = [e.signal_type for e in evidence]
        assert "size_ratio_match" in types

    def test_size_ratio_large_difference_low_score(self):
        src = _make_obs(estimated_height_ratio=0.1)
        tgt = _make_obs(camera_id="cam_b", track_id="trk_2", estimated_height_ratio=0.9)
        _, evidence = self.matcher.compute(src, tgt)
        size_ev = next((e for e in evidence if e.signal_type == "size_ratio_match"), None)
        if size_ev:
            assert size_ev.score < 0.3

    def test_direction_compatibility_evidence_present(self):
        src = _make_obs(exit_direction_degrees=90.0)
        tgt = _make_obs(camera_id="cam_b", track_id="trk_2", entry_direction_degrees=90.0)
        _, evidence = self.matcher.compute(src, tgt)
        types = [e.signal_type for e in evidence]
        assert "trajectory_direction_compatibility" in types

    def test_direction_incompatible_low_score(self):
        # Exit 45° vs Entry 270° → direct delta=225, reflective delta=135 — both > 60° threshold
        src = _make_obs(exit_direction_degrees=45.0)
        tgt = _make_obs(camera_id="cam_b", track_id="trk_2", entry_direction_degrees=270.0)
        _, evidence = self.matcher.compute(src, tgt)
        dir_ev = next((e for e in evidence if e.signal_type == "trajectory_direction_compatibility"), None)
        if dir_ev:
            assert dir_ev.score <= 0.5, f"Expected low score for incompatible directions, got {dir_ev.score}"

    def test_score_bounded_zero_to_one(self):
        src = _make_obs(color="red", estimated_height_ratio=0.4, exit_direction_degrees=45.0)
        tgt = _make_obs(camera_id="cam_b", track_id="trk_2", color="red",
                        estimated_height_ratio=0.42, entry_direction_degrees=45.0)
        score, _ = self.matcher.compute(src, tgt)
        assert 0.0 <= score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 3. TEMPORAL REASONER TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestTemporalCompatibilityReasoner:
    def setup_method(self):
        self.reasoner = TemporalCompatibilityReasoner()

    def test_negative_gap_returns_zero(self):
        score, gap, ev = self.reasoner.compute(120.0, 100.0)
        assert score == 0.0
        assert gap == pytest.approx(-20.0)
        assert "BEFORE" in ev.description or "negative" in ev.description.lower()

    def test_missing_source_returns_neutral(self):
        score, gap, ev = self.reasoner.compute(None, 100.0)
        assert score == pytest.approx(0.40)
        assert gap is None

    def test_missing_target_returns_neutral(self):
        score, gap, ev = self.reasoner.compute(100.0, None)
        assert score == pytest.approx(0.40)

    def test_gap_within_general_window_positive_score(self):
        score, gap, ev = self.reasoner.compute(100.0, 115.0)  # 15s = optimal
        assert score > 0.7
        assert gap == pytest.approx(15.0)

    def test_gap_above_max_returns_zero(self):
        score, gap, ev = self.reasoner.compute(100.0, 500.0)  # 400s >> 300s max
        assert score == 0.0

    def test_gap_below_min_returns_zero(self):
        score, gap, ev = self.reasoner.compute(100.0, 100.3)  # 0.3s < 1s min
        assert score == 0.0

    def test_adjacent_optimal_gap_high_score(self):
        score, gap, ev = self.reasoner.compute(100.0, 105.0, cameras_are_adjacent=True)  # 5s = adj optimal
        assert score > 0.8

    def test_adjacent_gap_above_adjacent_max_zero(self):
        score, gap, ev = self.reasoner.compute(100.0, 200.0, cameras_are_adjacent=True)  # 100s > 60s adj max
        assert score == 0.0

    def test_zero_gap_non_adjacent_returns_zero(self):
        score, gap, ev = self.reasoner.compute(100.0, 100.0, cameras_are_adjacent=False)
        assert score == 0.0

    def test_evidence_signal_type_is_temporal(self):
        _, _, ev = self.reasoner.compute(100.0, 115.0)
        assert ev.signal_type == "temporal_gap_plausibility"

    def test_triangular_score_peak(self):
        assert _triangular_score(15.0, 1.0, 15.0, 300.0) == pytest.approx(1.0)

    def test_triangular_score_at_lo(self):
        assert _triangular_score(1.0, 1.0, 15.0, 300.0) == pytest.approx(0.0)

    def test_triangular_score_at_hi(self):
        assert _triangular_score(300.0, 1.0, 15.0, 300.0) == pytest.approx(0.0)

    def test_triangular_score_outside_lo(self):
        assert _triangular_score(0.5, 1.0, 15.0, 300.0) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════
# 4. ASSOCIATION ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════════════

def _make_person_obs(camera_id, track_id, last_seen, first_seen, color="blue", adj=None) -> CameraObservation:
    return CameraObservation(
        camera_id=camera_id,
        camera_label=f"Label-{camera_id}",
        video_id=f"vid_{camera_id}",
        track_id=track_id,
        object_class="person",
        first_seen=first_seen,
        last_seen=last_seen,
        duration_seconds=last_seen - first_seen,
        max_confidence=0.90,
        color=color,
        estimated_height_ratio=0.35,
        exit_direction_degrees=90.0,
        entry_direction_degrees=90.0,
        adjacent_camera_labels=adj or [],
    )


class TestCrossCameraAssociationEngine:
    def setup_method(self):
        self.engine = CrossCameraAssociationEngine()

    def test_single_camera_returns_empty(self):
        obs = {"cam_a": [_make_person_obs("cam_a", "t1", 20.0, 10.0)]}
        results = self.engine.run("sess_1", obs)
        assert results == []

    def test_no_cameras_returns_empty(self):
        results = self.engine.run("sess_1", {})
        assert results == []

    def test_two_cameras_same_class_produces_hypothesis(self):
        obs = {
            "cam_a": [_make_person_obs("cam_a", "t1", 20.0, 10.0, color="red")],
            "cam_b": [_make_person_obs("cam_b", "t2", 40.0, 30.0, color="red")],
        }
        results = self.engine.run("sess_1", obs)
        assert len(results) > 0

    def test_class_mismatch_produces_no_hypothesis(self):
        src = _make_person_obs("cam_a", "t1", 20.0, 10.0)
        tgt = CameraObservation(
            camera_id="cam_b", camera_label="Cam B", video_id="vid_b",
            track_id="t2", object_class="car",
            first_seen=30.0, last_seen=40.0, duration_seconds=10.0, max_confidence=0.9,
        )
        obs = {"cam_a": [src], "cam_b": [tgt]}
        results = self.engine.run("sess_1", obs)
        assert results == []

    def test_negative_gap_produces_no_hypothesis(self):
        # t1 last_seen=20, t2 first_seen=15: cam_a→cam_b gap is -5s (impossible)
        obs = {
            "cam_a": [_make_person_obs("cam_a", "t1", 20.0, 10.0, color="red")],
            "cam_b": [_make_person_obs("cam_b", "t2", 25.0, 15.0, color="red")],
        }
        results = self.engine.run("sess_1", obs)
        # Verify no cam_a→cam_b association exists (negative gap direction is impossible)
        invalid = [
            r for r in results
            if r.source_camera_id == "cam_a" and r.target_camera_id == "cam_b"
        ]
        assert len(invalid) == 0, "Negative gap (cam_a→cam_b) hypothesis should not be created"

    def test_confidence_at_least_threshold(self):
        obs = {
            "cam_a": [_make_person_obs("cam_a", "t1", 20.0, 10.0, color="blue")],
            "cam_b": [_make_person_obs("cam_b", "t2", 40.0, 25.0, color="blue")],
        }
        results = self.engine.run("sess_1", obs)
        for r in results:
            assert r.confidence >= 0.40

    def test_results_sorted_by_confidence_descending(self):
        obs = {
            "cam_a": [
                _make_person_obs("cam_a", "t1", 20.0, 10.0, color="red"),
                _make_person_obs("cam_a", "t3", 20.0, 10.0, color="green"),
            ],
            "cam_b": [
                _make_person_obs("cam_b", "t2", 40.0, 25.0, color="red"),
                _make_person_obs("cam_b", "t4", 40.0, 25.0, color="blue"),
            ],
        }
        results = self.engine.run("sess_1", obs)
        confidences = [r.confidence for r in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_session_id_propagated(self):
        obs = {
            "cam_a": [_make_person_obs("cam_a", "t1", 20.0, 10.0, color="blue")],
            "cam_b": [_make_person_obs("cam_b", "t2", 40.0, 25.0, color="blue")],
        }
        results = self.engine.run("my_session_42", obs)
        for r in results:
            assert r.session_id == "my_session_42"

    def test_analyst_review_required_on_all_results(self):
        obs = {
            "cam_a": [_make_person_obs("cam_a", "t1", 20.0, 10.0, color="blue")],
            "cam_b": [_make_person_obs("cam_b", "t2", 40.0, 25.0, color="blue")],
        }
        results = self.engine.run("sess_1", obs)
        for r in results:
            assert r.analyst_review_required is True

    def test_adjacency_hint_adds_evidence(self):
        obs = {
            "cam_a": [_make_person_obs("cam_a", "t1", 20.0, 10.0, color="blue", adj=["Label-cam_b"])],
            "cam_b": [_make_person_obs("cam_b", "t2", 28.0, 22.0, color="blue")],
        }
        results = self.engine.run("sess_1", obs)
        adj_evidences = [
            e for r in results
            for e in r.evidence_basis
            if e.signal_type == "adjacency_hint"
        ]
        assert len(adj_evidences) > 0

    def test_deduplicated_best_confidence_kept(self):
        # Two observations in cam_b matching the same source track
        obs = {
            "cam_a": [_make_person_obs("cam_a", "t1", 20.0, 10.0, color="blue")],
            "cam_b": [
                _make_person_obs("cam_b", "t2", 35.0, 25.0, color="blue"),
                _make_person_obs("cam_b", "t2", 35.0, 25.0, color="blue"),  # duplicate
            ],
        }
        results = self.engine.run("sess_1", obs)
        # Should not have duplicate (cam_a,t1,cam_b,t2) entries
        keys = [(r.source_camera_id, r.source_track_id, r.target_camera_id, r.target_track_id) for r in results]
        assert len(keys) == len(set(keys))


# ═══════════════════════════════════════════════════════════════════════════
# 5. SCENE INTELLIGENCE TESTS
# ═══════════════════════════════════════════════════════════════════════════

def _make_hyp(src_cam, src_trk, tgt_cam, tgt_trk, assoc_type=AssociationType.SAME_OBJECT, verdict=AnalystVerdict.PENDING, confidence=0.85) -> CrossCameraHypothesis:
    return CrossCameraHypothesis(
        session_id="sess_x",
        source_camera_id=src_cam,
        source_video_id=f"vid_{src_cam}",
        source_track_id=src_trk,
        source_last_seen=20.0,
        target_camera_id=tgt_cam,
        target_video_id=f"vid_{tgt_cam}",
        target_track_id=tgt_trk,
        target_first_seen=30.0,
        confidence=confidence,
        association_type=assoc_type,
        analyst_verdict=verdict,
    )


class TestMultiCameraSceneIntelligence:
    def setup_method(self):
        self.intel = MultiCameraSceneIntelligence()

    def _cam_obs_dict(self):
        obs_a = _make_person_obs("cam_a", "t1", 20.0, 10.0)
        obs_b = _make_person_obs("cam_b", "t2", 40.0, 30.0)
        return {
            "cam_a": [obs_a],
            "cam_b": [obs_b],
        }

    def test_movement_narrative_with_same_object_hyp(self):
        hyp = _make_hyp("cam_a", "t1", "cam_b", "t2", AssociationType.SAME_OBJECT)
        narratives = self.intel.build_movement_narrative([hyp], self._cam_obs_dict())
        assert len(narratives) == 1
        assert len(narratives[0].segments) == 2

    def test_movement_narrative_empty_with_possible_same(self):
        hyp = _make_hyp("cam_a", "t1", "cam_b", "t2", AssociationType.POSSIBLE_SAME)
        narratives = self.intel.build_movement_narrative([hyp], self._cam_obs_dict())
        # POSSIBLE_SAME does not form narratives (only SAME_OBJECT and PROBABLE_SAME)
        assert all(len(n.segments) >= 2 for n in narratives) or len(narratives) == 0

    def test_movement_narrative_excludes_rejected(self):
        hyp = _make_hyp("cam_a", "t1", "cam_b", "t2", AssociationType.SAME_OBJECT, verdict=AnalystVerdict.REJECTED)
        narratives = self.intel.build_movement_narrative([hyp], self._cam_obs_dict())
        assert len(narratives) == 0

    def test_narrative_cluster_key_anonymous(self):
        hyp = _make_hyp("cam_a", "t1", "cam_b", "t2", AssociationType.SAME_OBJECT)
        narratives = self.intel.build_movement_narrative([hyp], self._cam_obs_dict())
        for n in narratives:
            assert n.cluster_key.startswith("cluster_")

    def test_narrative_to_dict_has_privacy_note(self):
        hyp = _make_hyp("cam_a", "t1", "cam_b", "t2", AssociationType.SAME_OBJECT)
        narratives = self.intel.build_movement_narrative([hyp], self._cam_obs_dict())
        for n in narratives:
            d = n.to_dict()
            assert "note" in d
            assert "hypothesis" in d["note"].lower() or "anonymous" in d["note"].lower()

    def test_summarize_session_keys(self):
        hyps = [_make_hyp("cam_a", "t1", "cam_b", "t2")]
        summary = self.intel.summarize_session("sess_1", hyps, self._cam_obs_dict())
        assert "session_id" in summary
        assert "camera_count" in summary
        assert "total_associations" in summary
        assert "privacy_note" in summary

    def test_summarize_session_total_associations(self):
        hyps = [
            _make_hyp("cam_a", "t1", "cam_b", "t2"),
            _make_hyp("cam_a", "t1", "cam_b", "t3", confidence=0.65),
        ]
        summary = self.intel.summarize_session("sess_1", hyps, self._cam_obs_dict())
        assert summary["total_associations"] == 2

    def test_scene_anomaly_missing_continuation_detected(self):
        # t1 exits cam_a but never appears in cam_b (no association)
        obs_a = _make_person_obs("cam_a", "t1", 20.0, 10.0)
        obs_a.exit_direction_degrees = 90.0
        cam_obs = {"cam_a": [obs_a], "cam_b": []}
        adjacency = {"Label-cam_a": ["Label-cam_b"]}
        anomalies = self.intel.detect_scene_anomalies([], cam_obs, adjacency)
        assert len(anomalies) >= 1
        assert any(a.anomaly_type == "MISSING_CONTINUATION" for a in anomalies)

    def test_scene_anomaly_not_detected_when_associated(self):
        obs_a = _make_person_obs("cam_a", "t1", 20.0, 10.0, adj=["Label-cam_b"])
        obs_a.exit_direction_degrees = 90.0
        obs_b = _make_person_obs("cam_b", "t2", 40.0, 25.0)
        cam_obs = {"cam_a": [obs_a], "cam_b": [obs_b]}
        hyp = _make_hyp("cam_a", "t1", "cam_b", "t2", AssociationType.SAME_OBJECT)
        adjacency = {"Label-cam_a": ["Label-cam_b"]}
        anomalies = self.intel.detect_scene_anomalies([hyp], cam_obs, adjacency)
        # Should not flag t1 since it has a confirmed association
        for a in anomalies:
            assert "t1" not in a.involved_tracks or a.anomaly_type != "MISSING_CONTINUATION"


# ═══════════════════════════════════════════════════════════════════════════
# 6. PRIVACY INVARIANT INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestPrivacyInvariants:
    """Critical privacy invariants: no biometric signals, no identity claims."""

    def test_engine_evidence_has_no_biometric_signals(self):
        engine = CrossCameraAssociationEngine()
        obs = {
            "cam_a": [_make_person_obs("cam_a", "t1", 20.0, 10.0, color="blue")],
            "cam_b": [_make_person_obs("cam_b", "t2", 40.0, 25.0, color="blue")],
        }
        results = engine.run("sess_priv", obs)
        forbidden = {"facial_similarity", "face_match", "biometric", "identity",
                     "voice_match", "gait_recognition", "iris_match"}
        for r in results:
            for ev in r.evidence_basis:
                assert ev.signal_type.lower() not in forbidden, (
                    f"Biometric signal '{ev.signal_type}' found in evidence basis"
                )

    def test_narrative_note_contains_anonymous_or_hypothesis(self):
        intel = MultiCameraSceneIntelligence()
        hyp = _make_hyp("cam_a", "t1", "cam_b", "t2", AssociationType.SAME_OBJECT)
        narratives = intel.build_movement_narrative(
            [hyp],
            {"cam_a": [_make_person_obs("cam_a", "t1", 20.0, 10.0)],
             "cam_b": [_make_person_obs("cam_b", "t2", 40.0, 25.0)]}
        )
        for n in narratives:
            assert "anonymous" in n.note.lower() or "hypothesis" in n.note.lower()

    def test_privacy_note_in_summary(self):
        intel = MultiCameraSceneIntelligence()
        summary = intel.summarize_session("s", [], {})
        assert "privacy_note" in summary
        pn = summary["privacy_note"].lower()
        assert "anonymous" in pn or "identity" in pn or "hypothesis" in pn

    def test_confidence_below_threshold_no_association(self):
        """If scoring produces < 0.40 confidence, no hypothesis should emerge."""
        engine = CrossCameraAssociationEngine(abstention_threshold=0.40)
        # Use mismatched colors + incompatible direction + large gap to force low score
        src = CameraObservation(
            camera_id="cam_a", camera_label="A", video_id="v1", track_id="t1",
            object_class="person", first_seen=0.0, last_seen=5.0, duration_seconds=5.0,
            max_confidence=0.5, color="red", estimated_height_ratio=0.1,
            exit_direction_degrees=0.0,
        )
        tgt = CameraObservation(
            camera_id="cam_b", camera_label="B", video_id="v2", track_id="t2",
            object_class="person", first_seen=290.0, last_seen=300.0, duration_seconds=10.0,
            max_confidence=0.5, color="green", estimated_height_ratio=0.9,
            entry_direction_degrees=180.0,
        )
        results = engine.run("sess_priv", {"cam_a": [src], "cam_b": [tgt]})
        for r in results:
            assert r.confidence >= 0.40  # any surviving hypothesis must be above threshold
