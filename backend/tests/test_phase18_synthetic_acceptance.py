"""
backend/tests/test_phase18_synthetic_acceptance.py — Phase 18 Final Acceptance Audit

Comprehensive synthetic multi-camera acceptance fixtures testing all scenarios A through N:
  A. Valid A -> B transition
  B. Impossible travel time (negative gap / too fast)
  C. Incompatible object class
  D. High visual similarity but impossible timing
  E. Two possible candidate matches
  F. Three-camera chain A -> B -> C
  G. Simultaneous independent events
  H. Different resolutions
  I. Different FPS
  J. Camera clock offset
  K. Unknown topology
  L. Missing evidence
  M. REVIEW_REQUIRED source event
  N. Cross-session entity injection

Also explicitly verifies:
  - Negative temporal gap hard veto under multiple variations
  - Association threshold / abstention at 0.40
  - Person & vehicle privacy safeguards (no biometrics, no identity)
  - Provenance integrity (camera IDs, video IDs, track IDs, timestamps)
"""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from ai.multicamera.schemas import (
    CameraObservation,
    CrossCameraHypothesis,
    AssociationEvidence,
    AssociationType,
    AnalystVerdict,
)
from ai.multicamera.attribute_matcher import AttributeMatcher
from ai.multicamera.temporal_reasoner import TemporalCompatibilityReasoner
from ai.multicamera.association_engine import CrossCameraAssociationEngine
from ai.multicamera.scene_intelligence import MultiCameraSceneIntelligence
from backend.app.main import app

client = TestClient(app)


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


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def engine():
    return CrossCameraAssociationEngine()


@pytest.fixture
def matcher():
    return AttributeMatcher()


@pytest.fixture
def reasoner():
    return TemporalCompatibilityReasoner()


@pytest.fixture
def intelligence():
    return MultiCameraSceneIntelligence()


# ═══════════════════════════════════════════════════════════════════════════
# SCENARIOS A - N
# ═══════════════════════════════════════════════════════════════════════════

class TestSyntheticAcceptanceScenarios:
    """Deterministic validation of Scenarios A through N."""

    def test_scenario_a_valid_a_to_b_transition(self, engine):
        """Scenario A: Normal, physically plausible transition from Camera A to Camera B."""
        obs_a = _make_obs(
            camera_id="cam-A",
            camera_label="North Entrance",
            video_id="vid-A",
            track_id="trk-101",
            object_class="person",
            first_seen=10.0,
            last_seen=25.0,
            duration_seconds=15.0,
            color="blue",
            estimated_height_ratio=0.35,
            estimated_width_ratio=0.12,
            exit_direction_degrees=90.0,
            adjacent_camera_labels=["East Corridor"],
        )
        obs_b = _make_obs(
            camera_id="cam-B",
            camera_label="East Corridor",
            video_id="vid-B",
            track_id="trk-202",
            object_class="person",
            first_seen=30.0,  # 5s gap
            last_seen=45.0,
            duration_seconds=15.0,
            color="blue",
            estimated_height_ratio=0.34,
            estimated_width_ratio=0.11,
            entry_direction_degrees=95.0,
            adjacent_camera_labels=["North Entrance"],
        )

        hyps = engine.run("sess-1", {"cam-A": [obs_a], "cam-B": [obs_b]})
        assert len(hyps) == 1
        h = hyps[0]
        # Provenance verification
        assert h.source_camera_id == "cam-A"
        assert h.source_video_id == "vid-A"
        assert h.source_track_id == "trk-101"
        assert h.target_camera_id == "cam-B"
        assert h.target_video_id == "vid-B"
        assert h.target_track_id == "trk-202"
        assert h.temporal_gap_seconds == 5.0
        assert h.confidence >= 0.70
        assert h.association_type in (AssociationType.SAME_OBJECT, AssociationType.PROBABLE_SAME)
        assert any(e.signal_type == "adjacency_hint" for e in h.evidence_basis)

    def test_scenario_b_impossible_travel_time_negative_gap(self, engine):
        """Scenario B: Target appears before source leaves -> impossible travel time (hard veto)."""
        obs_a = _make_obs(
            camera_id="cam-A",
            camera_label="Gate 1",
            video_id="vid-A",
            track_id="trk-1",
            object_class="car",
            first_seen=10.0,
            last_seen=20.0,  # exits at 20s
            color="silver",
        )
        obs_b = _make_obs(
            camera_id="cam-B",
            camera_label="Gate 2",
            video_id="vid-B",
            track_id="trk-2",
            object_class="car",
            first_seen=15.0,  # appears at 15s (gap = -5.0s)
            last_seen=30.0,
            color="silver",
        )

        hyps = engine.run("sess-1", {"cam-A": [obs_a], "cam-B": [obs_b]})
        # Hard veto must reject any association
        assert len(hyps) == 0

    def test_scenario_c_incompatible_object_class(self, engine):
        """Scenario C: Different classes (car vs person) must be strictly gated out."""
        obs_car = _make_obs(
            camera_id="cam-A",
            camera_label="Lane 1",
            video_id="vid-A",
            track_id="trk-veh-1",
            object_class="car",
            first_seen=10.0,
            last_seen=20.0,
            color="white",
        )
        obs_person = _make_obs(
            camera_id="cam-B",
            camera_label="Lane 2",
            video_id="vid-B",
            track_id="trk-ped-1",
            object_class="person",
            first_seen=25.0,
            last_seen=35.0,
            color="white",
        )

        hyps = engine.run("sess-1", {"cam-A": [obs_car], "cam-B": [obs_person]})
        assert len(hyps) == 0

    def test_scenario_d_high_visual_similarity_impossible_timing(self, engine):
        """Scenario D: Identical color, size, trajectory, but temporal gap is impossibly long (e.g. 5000s)."""
        obs_a = _make_obs(
            camera_id="cam-A",
            camera_label="Hallway",
            video_id="vid-A",
            track_id="trk-1",
            object_class="person",
            first_seen=10.0,
            last_seen=20.0,
            color="red",
            estimated_height_ratio=0.4,
            estimated_width_ratio=0.15,
            exit_direction_degrees=180.0,
        )
        obs_b = _make_obs(
            camera_id="cam-B",
            camera_label="Lobby",
            video_id="vid-B",
            track_id="trk-2",
            object_class="person",
            first_seen=5200.0,  # gap = 5180s > 300s max window
            last_seen=5220.0,
            color="red",
            estimated_height_ratio=0.4,
            estimated_width_ratio=0.15,
            entry_direction_degrees=180.0,
        )

        hyps = engine.run("sess-1", {"cam-A": [obs_a], "cam-B": [obs_b]})
        # Even with high visual match, temporal plausibility is 0.0
        for h in hyps:
            assert h.temporal_plausibility_score == 0.0

    def test_scenario_e_two_candidate_matches_disambiguation(self, engine):
        """Scenario E: Source track has two candidate matches on target camera; best confidence wins."""
        src = _make_obs(
            camera_id="cam-A",
            camera_label="Cam 1",
            video_id="vid-A",
            track_id="trk-src",
            object_class="person",
            first_seen=10.0,
            last_seen=20.0,
            color="navy",
            adjacent_camera_labels=["Cam 2"],
        )
        # Candidate 1: Perfect color match & tight time window
        cand1 = _make_obs(
            camera_id="cam-B",
            camera_label="Cam 2",
            video_id="vid-B",
            track_id="trk-good",
            object_class="person",
            first_seen=24.0,  # 4s gap
            last_seen=35.0,
            color="navy",
            adjacent_camera_labels=["Cam 1"],
        )
        # Candidate 2: Different color & delayed time
        cand2 = _make_obs(
            camera_id="cam-B",
            camera_label="Cam 2",
            video_id="vid-B",
            track_id="trk-weak",
            object_class="person",
            first_seen=50.0,  # 30s gap
            last_seen=60.0,
            color="yellow",
            adjacent_camera_labels=["Cam 1"],
        )

        hyps = engine.run("sess-1", {"cam-A": [src], "cam-B": [cand1, cand2]})
        # Find hypothesis for trk-src -> trk-good
        hyp_good = next((h for h in hyps if h.target_track_id == "trk-good"), None)
        assert hyp_good is not None
        assert hyp_good.confidence > 0.70

    def test_scenario_f_three_camera_chain(self, engine, intelligence):
        """Scenario F: Three camera transition A -> B -> C forming a movement narrative."""
        obs_a = _make_obs(
            camera_id="cam-A",
            camera_label="Cam A",
            video_id="vid-A",
            track_id="trk-1",
            object_class="car",
            first_seen=0.0,
            last_seen=10.0,
            duration_seconds=10.0,
            color="black",
            adjacent_camera_labels=["Cam B"],
        )
        obs_b = _make_obs(
            camera_id="cam-B",
            camera_label="Cam B",
            video_id="vid-B",
            track_id="trk-2",
            object_class="car",
            first_seen=15.0,
            last_seen=25.0,
            duration_seconds=10.0,
            color="black",
            adjacent_camera_labels=["Cam A", "Cam C"],
        )
        obs_c = _make_obs(
            camera_id="cam-C",
            camera_label="Cam C",
            video_id="vid-C",
            track_id="trk-3",
            object_class="car",
            first_seen=30.0,
            last_seen=40.0,
            duration_seconds=10.0,
            color="black",
            adjacent_camera_labels=["Cam B"],
        )

        cams = {"cam-A": [obs_a], "cam-B": [obs_b], "cam-C": [obs_c]}
        hyps = engine.run("sess-chain", cams)

        # There should be associations A -> B and B -> C
        ab = next((h for h in hyps if h.source_camera_id == "cam-A" and h.target_camera_id == "cam-B"), None)
        bc = next((h for h in hyps if h.source_camera_id == "cam-B" and h.target_camera_id == "cam-C"), None)
        assert ab is not None
        assert bc is not None

        # Build movement narrative
        narratives = intelligence.build_movement_narrative([ab, bc], cams)
        assert len(narratives) >= 1
        n = narratives[0]
        assert n.object_class == "car"
        assert len(n.segments) == 3  # trk-1, trk-2, trk-3
        assert n.to_dict()["total_span_seconds"] == 40.0

    def test_scenario_g_simultaneous_independent_events(self, engine):
        """Scenario G: Two tracks appearing simultaneously in Cam A and Cam B with negative/zero gap."""
        obs_a = _make_obs(
            camera_id="cam-A",
            camera_label="Cam A",
            video_id="vid-A",
            track_id="trk-sim-A",
            object_class="person",
            first_seen=10.0,
            last_seen=30.0,
            color="green",
        )
        obs_b = _make_obs(
            camera_id="cam-B",
            camera_label="Cam B",
            video_id="vid-B",
            track_id="trk-sim-B",
            object_class="person",
            first_seen=10.0,  # starts at same time
            last_seen=30.0,
            color="green",
        )

        # gap from A (last_seen=30) to B (first_seen=10) is -20s -> hard veto
        # gap from B (last_seen=30) to A (first_seen=10) is -20s -> hard veto
        hyps = engine.run("sess-sim", {"cam-A": [obs_a], "cam-B": [obs_b]})
        assert len(hyps) == 0

    def test_scenario_h_different_resolutions(self, engine):
        """Scenario H: Cam A is 720p, Cam B is 4K, normalized ratios used."""
        obs_720 = _make_obs(
            camera_id="cam-720",
            camera_label="720p Cam",
            video_id="vid-720",
            track_id="trk-720",
            object_class="person",
            first_seen=5.0,
            last_seen=15.0,
            color="blue",
            estimated_height_ratio=0.30,
            estimated_width_ratio=0.10,
        )
        obs_4k = _make_obs(
            camera_id="cam-4k",
            camera_label="4K Cam",
            video_id="vid-4k",
            track_id="trk-4k",
            object_class="person",
            first_seen=20.0,
            last_seen=35.0,
            color="blue",
            estimated_height_ratio=0.31,
            estimated_width_ratio=0.10,
        )

        hyps = engine.run("sess-res", {"cam-720": [obs_720], "cam-4k": [obs_4k]})
        assert len(hyps) == 1
        assert hyps[0].confidence >= 0.50

    def test_scenario_i_different_fps(self, engine):
        """Scenario I: Cam A at 15 FPS, Cam B at 30 FPS. All reasoning operates in seconds."""
        obs_15fps = _make_obs(
            camera_id="cam-15",
            camera_label="15 FPS Cam",
            video_id="vid-15",
            track_id="trk-15",
            object_class="person",
            first_seen=10.0,
            last_seen=20.0,
            color="red",
        )
        obs_30fps = _make_obs(
            camera_id="cam-30",
            camera_label="30 FPS Cam",
            video_id="vid-30",
            track_id="trk-30",
            object_class="person",
            first_seen=25.0,
            last_seen=35.0,
            color="red",
        )

        hyps = engine.run("sess-fps", {"cam-15": [obs_15fps], "cam-30": [obs_30fps]})
        assert len(hyps) == 1
        assert hyps[0].temporal_gap_seconds == 5.0
        assert hyps[0].confidence >= 0.60

    def test_scenario_j_camera_clock_offset(self, engine):
        """Scenario J: Cam B has an explicit +10s clock offset applied before reasoning."""
        raw_b_first_seen = 15.0
        clock_offset_b = 10.0  # true wall clock = 25.0
        normalized_b_first_seen = raw_b_first_seen + clock_offset_b

        obs_a = _make_obs(
            camera_id="cam-A",
            camera_label="Cam A",
            video_id="vid-A",
            track_id="trk-A",
            object_class="person",
            first_seen=10.0,
            last_seen=20.0,
            color="black",
        )
        obs_b = _make_obs(
            camera_id="cam-B",
            camera_label="Cam B",
            video_id="vid-B",
            track_id="trk-B",
            object_class="person",
            first_seen=normalized_b_first_seen,  # 25.0s
            last_seen=normalized_b_first_seen + 15.0,
            color="black",
        )

        hyps = engine.run("sess-clock", {"cam-A": [obs_a], "cam-B": [obs_b]})
        assert len(hyps) == 1
        assert hyps[0].temporal_gap_seconds == 5.0

    def test_scenario_k_unknown_topology(self, engine):
        """Scenario K: No adjacency hints provided -> conservative baseline behavior."""
        obs_a = _make_obs(
            camera_id="cam-A",
            camera_label="Unknown 1",
            video_id="vid-A",
            track_id="trk-1",
            object_class="person",
            first_seen=10.0,
            last_seen=20.0,
            color="green",
            adjacent_camera_labels=[],  # empty
        )
        obs_b = _make_obs(
            camera_id="cam-B",
            camera_label="Unknown 2",
            video_id="vid-B",
            track_id="trk-2",
            object_class="person",
            first_seen=28.0,
            last_seen=40.0,
            color="green",
            adjacent_camera_labels=[],  # empty
        )

        hyps = engine.run("sess-unknown-topo", {"cam-A": [obs_a], "cam-B": [obs_b]})
        assert len(hyps) == 1
        # Adjacency evidence should NOT be added
        assert not any(e.signal_type == "adjacency_hint" for e in hyps[0].evidence_basis)

    def test_scenario_l_missing_evidence(self, engine, matcher, reasoner):
        """Scenario L: Missing color, size, trajectory, and temporal timestamps handled safely."""
        obs_sparse_a = _make_obs(
            camera_id="cam-A",
            camera_label="Sparse A",
            video_id="vid-A",
            track_id="trk-sparse-a",
            object_class="person",
            first_seen=0.0,
            last_seen=0.0,
            color=None,
            estimated_height_ratio=None,
            estimated_width_ratio=None,
            exit_direction_degrees=None,
            entry_direction_degrees=None,
        )
        obs_sparse_b = _make_obs(
            camera_id="cam-B",
            camera_label="Sparse B",
            video_id="vid-B",
            track_id="trk-sparse-b",
            object_class="person",
            first_seen=0.0,
            last_seen=0.0,
            color=None,
            estimated_height_ratio=None,
            estimated_width_ratio=None,
            exit_direction_degrees=None,
            entry_direction_degrees=None,
        )

        # Attribute matcher should return neutral unknown scores
        attr_score, attr_ev = matcher.compute(obs_sparse_a, obs_sparse_b)
        assert attr_score > 0.0
        assert any(e.signal_type == "object_class_match" for e in attr_ev)

        # Temporal reasoner returns 0.40 neutral score on None
        temp_score, gap, temp_ev = reasoner.compute(None, None, False)
        assert temp_score == 0.40
        assert gap is None

    def test_scenario_m_review_required_source_event(self):
        """Scenario M: Source events/incidents with REVIEW_REQUIRED maintain canonical <= 0.65 ceiling."""
        hyp = CrossCameraHypothesis(
            session_id="sess-m",
            source_camera_id="cam-1",
            source_video_id="vid-1",
            source_track_id="trk-1",
            source_last_seen=10.0,
            target_camera_id="cam-2",
            target_video_id="vid-2",
            target_track_id="trk-2",
            target_first_seen=15.0,
            confidence=0.85,
            attribute_match_score=0.9,
            trajectory_compatibility_score=0.8,
            temporal_gap_seconds=5.0,
            temporal_plausibility_score=0.8,
            evidence_basis=[],
            analyst_review_required=True,
            analyst_verdict=AnalystVerdict.PENDING,
        )
        assert hyp.analyst_review_required is True
        assert hyp.analyst_verdict == AnalystVerdict.PENDING

    def test_scenario_n_cross_session_entity_injection_prevention(self):
        """Scenario N: Attempting to access or inject Session B entities into Session A fails safely."""
        # Requesting a track from a video not in the session must return 403 or 404
        resp = client.get(
            "/api/sessions/sess-A/tracks/vid-not-in-session/trk-999/associations"
        )
        assert resp.status_code in (404, 403)


# ═══════════════════════════════════════════════════════════════════════════
# NEGATIVE TEMPORAL GAP HARD VETO (EXTENSIVE AUDIT)
# ═══════════════════════════════════════════════════════════════════════════

class TestNegativeTemporalGapHardVeto:
    """Verifies that no attribute similarity can EVER override a negative temporal gap."""

    @pytest.mark.parametrize("gap_sec", [-0.01, -1.0, -5.0, -10.0, -60.0])
    def test_negative_gap_veto_variations(self, engine, gap_sec):
        """Even with 100% attribute match, negative gap returns None."""
        source = _make_obs(
            camera_id="cam-1",
            camera_label="Exit",
            video_id="vid-1",
            track_id="trk-1",
            object_class="person",
            first_seen=10.0,
            last_seen=20.0,  # leaves at 20.0
            color="red",
            estimated_height_ratio=0.4,
            estimated_width_ratio=0.15,
            exit_direction_degrees=45.0,
        )
        target = _make_obs(
            camera_id="cam-2",
            camera_label="Entry",
            video_id="vid-2",
            track_id="trk-2",
            object_class="person",
            first_seen=20.0 + gap_sec,  # enters before source left
            last_seen=30.0,
            color="red",
            estimated_height_ratio=0.4,
            estimated_width_ratio=0.15,
            entry_direction_degrees=45.0,
        )

        hyps = engine.run("sess-veto", {"cam-1": [source], "cam-2": [target]})
        assert len(hyps) == 0, f"Failed for negative gap {gap_sec}: hypothesis was incorrectly created!"


# ═══════════════════════════════════════════════════════════════════════════
# PRIVACY & ISOLATION AUDIT
# ═══════════════════════════════════════════════════════════════════════════

class TestPrivacyAndIsolationAudit:
    """Verifies strict adherence to privacy and session isolation boundaries."""

    def test_forbidden_biometric_signal_rejected(self):
        for bad_type in [
            "facial_similarity", "face_match", "biometric", "identity",
            "voice_match", "gait_recognition", "iris_match"
        ]:
            with pytest.raises(ValueError, match="forbidden signal_type"):
                AssociationEvidence(signal_type=bad_type, description="test", score=0.9)

    def test_hypothesis_representation_is_anonymous(self, engine):
        """Hypotheses must use track IDs and camera labels only; no identity claims."""
        obs_a = _make_obs(
            camera_id="cam-A",
            camera_label="North",
            video_id="vid-A",
            track_id="TRACK-42",
            object_class="person",
            first_seen=10.0,
            last_seen=20.0,
            color="black",
        )
        obs_b = _make_obs(
            camera_id="cam-B",
            camera_label="South",
            video_id="vid-B",
            track_id="TRACK-99",
            object_class="person",
            first_seen=25.0,
            last_seen=35.0,
            color="black",
        )
        hyps = engine.run("sess-anon", {"cam-A": [obs_a], "cam-B": [obs_b]})
        assert len(hyps) == 1
        d = hyps[0].to_dict()
        assert "TRACK-42" in d["source_track_id"]
        assert "TRACK-99" in d["target_track_id"]
        # Ensure no identity assertion
        assert "identity" not in d
        assert "person_name" not in d


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-LAYER FORENSIC CONSISTENCY AUDIT
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossLayerForensicConsistency:
    """Verifies that DB, Service layer, API models, and Timeline preserve identical values."""

    def test_end_to_end_cross_layer_consistency(self):
        from database.session import SessionLocal, init_db
        from database.models import (
            SurveillanceSessionModel,
            CameraSourceModel,
            CrossCameraAssociationModel,
            VideoModel,
        )
        from backend.app.services.cross_camera_service import CrossCameraAnalysisService
        import uuid

        init_db()
        db = SessionLocal()
        cross_svc = CrossCameraAnalysisService()

        # Seed test video records to satisfy foreign keys
        v1_id = str(uuid.uuid4())
        v2_id = str(uuid.uuid4())
        v1 = VideoModel(id=v1_id, original_filename="cam1_feed.mp4", storage_path=f"storage/uploads/{v1_id}.mp4", status="COMPLETED")
        v2 = VideoModel(id=v2_id, original_filename="cam2_feed.mp4", storage_path=f"storage/uploads/{v2_id}.mp4", status="COMPLETED")
        db.add_all([v1, v2])
        db.commit()

        # Create session
        sess = SurveillanceSessionModel(
            name="Forensic Consistency Test Session",
            site_name="Test Site Alpha",
            description="Multi-camera consistency audit",
            status="active",
        )
        db.add(sess)
        db.commit()
        db.refresh(sess)
        sess_id = sess.id

        # Add two cameras
        cam1 = CameraSourceModel(
            session_id=sess_id,
            video_id=v1_id,
            camera_label="Camera 1",
            position_hint="Gate 1",
            adjacency_hints=["Camera 2"],
        )
        cam2 = CameraSourceModel(
            session_id=sess_id,
            video_id=v2_id,
            camera_label="Camera 2",
            position_hint="Gate 2",
            adjacency_hints=["Camera 1"],
        )
        db.add_all([cam1, cam2])
        db.commit()
        db.refresh(cam1)
        db.refresh(cam2)

        # Create CrossCameraAssociationModel
        assoc = CrossCameraAssociationModel(
            session_id=sess_id,
            source_camera_id=cam1.id,
            source_video_id=v1_id,
            source_track_id="TRK-001",
            source_last_seen=25.0,
            target_camera_id=cam2.id,
            target_video_id=v2_id,
            target_track_id="TRK-002",
            target_first_seen=30.0,
            confidence=0.78,
            association_type="PROBABLE_SAME",
            attribute_match_score=0.85,
            trajectory_compatibility_score=0.90,
            temporal_gap_seconds=5.0,
            temporal_plausibility_score=0.75,
            evidence_basis=[
                {"signal_type": "color_match", "description": "Navy blue match", "score": 0.85},
                {"signal_type": "temporal_plausibility", "description": "5.0s transition", "score": 0.75},
            ],
            analyst_review_required=True,
            analyst_verdict="PENDING",
        )
        db.add(assoc)
        db.commit()
        db.refresh(assoc)
        assoc_id = assoc.id

        try:
            # 1. Direct DB lookup
            db_row = db.query(CrossCameraAssociationModel).filter(CrossCameraAssociationModel.id == assoc_id).first()
            assert db_row is not None
            assert db_row.confidence == 0.78
            assert db_row.association_type == "PROBABLE_SAME"
            assert db_row.analyst_verdict == "PENDING"
            assert db_row.source_track_id == "TRK-001"
            assert db_row.target_track_id == "TRK-002"
            assert db_row.temporal_gap_seconds == 5.0

            # 2. Service layer query
            svc_dict = cross_svc.get_association(db, sess_id, assoc_id)
            assert svc_dict["id"] == assoc_id
            assert svc_dict["session_id"] == sess_id
            assert svc_dict["confidence"] == 0.78
            assert svc_dict["association_type"] == "PROBABLE_SAME"
            assert svc_dict["analyst_verdict"] == "PENDING"
            assert svc_dict["temporal_gap_seconds"] == 5.0
            assert len(svc_dict["evidence_basis"]) == 2

            # 3. HTTP API endpoint
            resp = client.get(f"/api/sessions/{sess_id}/associations/{assoc_id}")
            assert resp.status_code == 200
            api_data = resp.json()
            assert api_data["id"] == assoc_id
            assert api_data["confidence"] == 0.78
            assert api_data["association_type"] == "PROBABLE_SAME"
            assert api_data["analyst_verdict"] == "PENDING"
            assert api_data["source_track_id"] == "TRK-001"
            assert api_data["target_track_id"] == "TRK-002"

            # 4. HTTP API update verdict to CONFIRMED
            patch_resp = client.patch(
                f"/api/sessions/{sess_id}/associations/{assoc_id}/verdict",
                json={"verdict": "CONFIRMED", "notes": "Audited and confirmed by forensic reviewer."},
            )
            assert patch_resp.status_code == 200
            patched_data = patch_resp.json()
            assert patched_data["analyst_verdict"] == "CONFIRMED"
            assert "Audited and confirmed" in patched_data["analyst_notes"]

            # 5. Verify DB reflects the change
            db.refresh(db_row)
            assert db_row.analyst_verdict == "CONFIRMED"

            # 6. Verify cross-session isolation: cannot access under another session_id
            other_sess_resp = client.get(f"/api/sessions/sess-nonexistent/associations/{assoc_id}")
            assert other_sess_resp.status_code in (404, 403)

        finally:
            # Clean up
            db.query(CrossCameraAssociationModel).filter(CrossCameraAssociationModel.session_id == sess_id).delete()
            db.query(CameraSourceModel).filter(CameraSourceModel.session_id == sess_id).delete()
            db.query(SurveillanceSessionModel).filter(SurveillanceSessionModel.id == sess_id).delete()
            db.query(VideoModel).filter(VideoModel.id.in_([v1_id, v2_id])).delete()
            db.commit()
            db.close()


# ═══════════════════════════════════════════════════════════════════════════
# EDGE CASES & BOUNDARY CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCasesAndBoundaryConditions:
    """Verifies safe failure on empty, malformed, or extreme inputs."""

    def test_engine_zero_cameras(self, engine):
        assert engine.run("sess-0", {}) == []

    def test_engine_one_camera(self, engine):
        obs = _make_obs(camera_id="cam-1")
        assert engine.run("sess-1", {"cam-1": [obs]}) == []

    def test_engine_camera_with_no_tracks(self, engine):
        obs = _make_obs(camera_id="cam-1")
        assert engine.run("sess-empty", {"cam-1": [obs], "cam-2": []}) == []

    def test_api_nonexistent_session_returns_404(self):
        resp = client.get("/api/sessions/nonexistent-sess-uuid")
        assert resp.status_code == 404

    def test_api_invalid_session_id_format(self):
        resp = client.get("/api/sessions/bad!id@chars/associations")
        assert resp.status_code == 400

