"""
Phase 14: Crowd, Density & Zone Intelligence Comprehensive Unit & Regression Tests

Validates:
1. Crowd Density Engine (unique active person tracks, deduplication, vehicle separation, clustering)
2. Perspective & Scene Baseline Normalization (image-relative metrics, baseline comparisons)
3. Queue & Flow Engine (linear collinearity, heading alignment, queue detection)
4. Crowd Density Detector (HIGH_PEDESTRIAN_DENSITY, CROWD_DENSITY_INCREASE)
5. Crowd Surge Detector (POTENTIAL_CROWD_SURGE, conservative multivariable gating)
6. Crowd Dispersal Detector (POTENTIAL_CROWD_DISPERSAL, cluster fragmentation)
7. Unusual Crowd Movement Detector (POTENTIAL_UNUSUAL_CROWD_MOVEMENT vs scene baseline)
8. Zone Occupancy & Activity Detector (POTENTIAL_RESTRICTED_ZONE_CROWDING, hysteresis vs boundary jitter)
9. Negative Evidence Engine & Validator Downgrades (queue refutes surge, transient pass refutes crowding)
10. Incident Fusion Crowd Arbitration (prevents duplicate amplification in same episode)
11. Natural-Language Investigation Query Routing
"""
import pytest
import math
from ai.schemas import BoundingBox, TrackedObject, ZoneDefinition
from ai.incidents.schemas import IncidentCandidate, IncidentContext, IncidentCategory, SpatialContext
from ai.incidents.detectors.crowd.density_engine import CrowdDensityEngine
from ai.incidents.detectors.crowd.queue_flow_engine import QueueAndFlowEngine
from ai.incidents.detectors.crowd.crowd_density import CrowdDensityDetector
from ai.incidents.detectors.crowd.crowd_surge import CrowdSurgeDetector
from ai.incidents.detectors.crowd.crowd_dispersal import CrowdDispersalDetector
from ai.incidents.detectors.crowd.unusual_crowd_movement import UnusualCrowdMovementDetector
from ai.incidents.detectors.crowd.zone_occupancy import ZoneOccupancyAndActivityDetector
from ai.incidents.negative_evidence import NegativeEvidenceEngine
from ai.incidents.validator import IncidentCandidateValidator
from ai.incidents.fusion import IncidentFusionEngine
from ai.incidents.scene_context import SceneContextData, NormalBehaviorBaseline
from backend.app.services.investigation_parser import InvestigationParser


def make_track(
    track_id: str,
    object_class: str,
    trajectory: list,
    confidence: float = 0.90,
    bbox_size: tuple = (40.0, 60.0),
) -> TrackedObject:
    """Helper to build TrackedObject with trajectory and history bboxes."""
    history = []
    w, h = bbox_size
    for pt in trajectory:
        t, cx, cy = pt[0], pt[1], pt[2]
        history.append({
            "timestamp": t,
            "bbox": {"x1": cx - w / 2, "y1": cy - h / 2, "x2": cx + w / 2, "y2": cy + h / 2},
            "confidence": confidence,
        })
    last_pt = trajectory[-1]
    curr_bbox = BoundingBox(
        x1=last_pt[1] - w / 2,
        y1=last_pt[2] - h / 2,
        x2=last_pt[1] + w / 2,
        y2=last_pt[2] + h / 2,
    )
    return TrackedObject(
        track_id=track_id,
        object_class=object_class,
        first_seen=trajectory[0][0],
        last_seen=trajectory[-1][0],
        confidence=confidence,
        current_bbox=curr_bbox,
        trajectory=trajectory,
        history_bboxes=history,
    )


# ---------------------------------------------------------------------------
# 1. Test Crowd Density Engine
# ---------------------------------------------------------------------------

class TestCrowdDensityEngine:
    def test_empty_scene(self):
        engine = CrowdDensityEngine()
        ctx = IncidentContext(video_id="test_empty", fps=30.0, duration_seconds=5.0, tracks=[])
        windows = engine.evaluate_windows(ctx)
        assert len(windows) == 0

    def test_single_person_no_crowd(self):
        engine = CrowdDensityEngine()
        p1 = make_track("P1", "person", [(0.0, 100, 100), (2.0, 105, 105)])
        ctx = IncidentContext(video_id="test_single", fps=30.0, duration_seconds=5.0, tracks=[p1])
        windows = engine.evaluate_windows(ctx)
        assert len(windows) >= 1
        assert windows[0].active_person_count == 1
        assert len(windows[0].unique_person_track_ids) == 1
        assert len(windows[0].clusters) == 0

    def test_person_vs_vehicle_separation(self):
        engine = CrowdDensityEngine()
        p1 = make_track("P1", "person", [(0.0, 100, 100), (2.0, 100, 100)])
        v1 = make_track("V1", "car", [(0.0, 110, 100), (2.0, 110, 100)])
        v2 = make_track("V2", "truck", [(0.0, 120, 100), (2.0, 120, 100)])
        ctx = IncidentContext(video_id="test_sep", fps=30.0, duration_seconds=5.0, tracks=[p1, v1, v2])
        windows = engine.evaluate_windows(ctx)
        assert len(windows) >= 1
        assert windows[0].active_person_count == 1
        assert windows[0].active_vehicle_count == 2
        assert len(windows[0].clusters) == 0

    def test_clustering_of_nearby_persons(self):
        engine = CrowdDensityEngine(cluster_distance_px=60.0, min_cluster_size=3)
        # 3 people in close proximity (distance ~20px)
        p1 = make_track("P1", "person", [(0.0, 100, 100), (2.0, 100, 100)])
        p2 = make_track("P2", "person", [(0.0, 120, 100), (2.0, 120, 100)])
        p3 = make_track("P3", "person", [(0.0, 110, 120), (2.0, 110, 120)])
        # 1 isolated person (distance ~300px)
        p4 = make_track("P4", "person", [(0.0, 400, 400), (2.0, 400, 400)])
        ctx = IncidentContext(video_id="test_cluster", fps=30.0, duration_seconds=3.0, tracks=[p1, p2, p3, p4])
        windows = engine.evaluate_windows(ctx)
        assert len(windows) >= 1
        assert windows[0].active_person_count == 4
        assert len(windows[0].clusters) == 1
        assert windows[0].clusters[0].size == 3
        assert set(windows[0].clusters[0].track_ids) == {"P1", "P2", "P3"}


# ---------------------------------------------------------------------------
# 2. Test Queue and Flow Engine
# ---------------------------------------------------------------------------

class TestQueueAndFlowEngine:
    def test_orderly_linear_queue(self):
        # 4 people lined up along X axis with identical motion to the right
        tracks = [
            make_track("P1", "person", [(0.0, 100, 200), (1.0, 120, 200)]),
            make_track("P2", "person", [(0.0, 140, 202), (1.0, 160, 201)]),
            make_track("P3", "person", [(0.0, 180, 199), (1.0, 200, 200)]),
            make_track("P4", "person", [(0.0, 220, 201), (1.0, 240, 200)]),
        ]
        q_engine = QueueAndFlowEngine()
        assessment = q_engine.evaluate_queue_pattern(tracks, eval_time=0.5)
        assert assessment.is_queue_detected is True
        assert assessment.linear_fit_score >= 0.70
        assert assessment.queue_member_count == 4

    def test_scattered_group_not_a_queue(self):
        # 4 people randomly scattered moving in opposing directions
        tracks = [
            make_track("P1", "person", [(0.0, 100, 100), (1.0, 120, 80)]),
            make_track("P2", "person", [(0.0, 200, 300), (1.0, 180, 350)]),
            make_track("P3", "person", [(0.0, 150, 250), (1.0, 140, 200)]),
            make_track("P4", "person", [(0.0, 300, 120), (1.0, 320, 180)]),
        ]
        q_engine = QueueAndFlowEngine()
        assessment = q_engine.evaluate_queue_pattern(tracks, eval_time=0.5)
        assert assessment.is_queue_detected is False


# ---------------------------------------------------------------------------
# 3. Test Crowd Density Detector
# ---------------------------------------------------------------------------

class TestCrowdDensityDetector:
    def test_high_pedestrian_density_detected(self):
        detector = CrowdDensityDetector()
        # 6 people present in scene with normal baseline of 2.0
        tracks = [
            make_track(f"P{i}", "person", [(0.0, 100 + i * 20, 100 + i * 20), (3.0, 105 + i * 20, 105 + i * 20)])
            for i in range(6)
        ]
        scene_ctx = SceneContextData(
            scene_type="pedestrian_area",
            baseline=NormalBehaviorBaseline(average_density_per_second=2.0),
        )
        ctx = IncidentContext(
            video_id="video_crowd_01",
            fps=30.0,
            duration_seconds=5.0,
            tracks=tracks,
            scene_context=scene_ctx,
        )
        candidates = detector.analyze(ctx)
        density_cands = [c for c in candidates if c.event_type == "HIGH_PEDESTRIAN_DENSITY"]
        assert len(density_cands) >= 1
        assert density_cands[0].category == IncidentCategory.CROWD
        assert density_cands[0].confidence >= 0.70

    def test_normal_density_below_baseline_suppressed(self):
        detector = CrowdDensityDetector()
        # 2 people when baseline is 3.0 -> no high density event
        tracks = [
            make_track("P1", "person", [(0.0, 100, 100), (3.0, 110, 110)]),
            make_track("P2", "person", [(0.0, 200, 200), (3.0, 210, 210)]),
        ]
        scene_ctx = SceneContextData(
            scene_type="pedestrian_area",
            baseline=NormalBehaviorBaseline(average_density_per_second=3.0),
        )
        ctx = IncidentContext(
            video_id="video_normal_01",
            fps=30.0,
            duration_seconds=5.0,
            tracks=tracks,
            scene_context=scene_ctx,
        )
        candidates = detector.analyze(ctx)
        assert len(candidates) == 0


# ---------------------------------------------------------------------------
# 4. Test Crowd Surge Detector
# ---------------------------------------------------------------------------

class TestCrowdSurgeDetector:
    def test_potential_crowd_surge_detected(self):
        detector = CrowdSurgeDetector()
        # 5 people in a 2D cluster moving rapidly (30px/s) together in the exact same direction (+X)
        tracks = [
            make_track("P0", "person", [(0.0, 100, 100), (1.0, 130, 100), (2.0, 160, 100)]),
            make_track("P1", "person", [(0.0, 120, 130), (1.0, 150, 130), (2.0, 180, 130)]),
            make_track("P2", "person", [(0.0, 90, 140), (1.0, 120, 140), (2.0, 150, 140)]),
            make_track("P3", "person", [(0.0, 130, 90), (1.0, 160, 90), (2.0, 190, 90)]),
            make_track("P4", "person", [(0.0, 110, 115), (1.0, 140, 115), (2.0, 170, 115)]),
        ]
        ctx = IncidentContext(
            video_id="video_surge_01",
            fps=30.0,
            duration_seconds=3.0,
            tracks=tracks,
        )
        candidates = detector.analyze(ctx)
        surge_cands = [c for c in candidates if c.event_type == "POTENTIAL_CROWD_SURGE"]
        assert len(surge_cands) >= 1
        assert "Surge" in surge_cands[0].explanation

    def test_queue_flow_refutes_crowd_surge(self):
        detector = CrowdSurgeDetector()
        # Orderly linear queue shuffling slowly at 10px/s
        tracks = [
            make_track("P1", "person", [(0.0, 100, 200), (1.0, 110, 200), (2.0, 120, 200)]),
            make_track("P2", "person", [(0.0, 140, 202), (1.0, 150, 201), (2.0, 160, 200)]),
            make_track("P3", "person", [(0.0, 180, 199), (1.0, 190, 200), (2.0, 200, 200)]),
            make_track("P4", "person", [(0.0, 220, 201), (1.0, 230, 200), (2.0, 240, 200)]),
        ]
        ctx = IncidentContext(
            video_id="video_queue_01",
            fps=30.0,
            duration_seconds=3.0,
            tracks=tracks,
        )
        candidates = detector.analyze(ctx)
        # Should be suppressed by negative queue evidence
        surge_cands = [c for c in candidates if c.event_type == "POTENTIAL_CROWD_SURGE"]
        assert len(surge_cands) == 0


# ---------------------------------------------------------------------------
# 5. Test Crowd Dispersal Detector
# ---------------------------------------------------------------------------

class TestCrowdDispersalDetector:
    def test_crowd_dispersal_detected(self):
        detector = CrowdDispersalDetector()
        # Initial: 5 people in dense cluster; Later: 4 people leave FOV or scatter
        tracks = [
            make_track("P1", "person", [(0.0, 100, 100), (1.0, 102, 102), (2.0, 105, 105), (3.0, 108, 108)]),
            make_track("P2", "person", [(0.0, 115, 105), (1.0, 116, 106)]),  # leaves early
            make_track("P3", "person", [(0.0, 105, 115), (1.0, 106, 116)]),  # leaves early
            make_track("P4", "person", [(0.0, 120, 120), (1.0, 121, 121)]),  # leaves early
            make_track("P5", "person", [(0.0, 110, 100), (1.0, 112, 102)]),  # leaves early
        ]
        ctx = IncidentContext(
            video_id="video_dispersal_01",
            fps=30.0,
            duration_seconds=4.0,
            tracks=tracks,
        )
        candidates = detector.analyze(ctx)
        dispersal_cands = [c for c in candidates if c.event_type == "POTENTIAL_CROWD_DISPERSAL"]
        assert len(dispersal_cands) >= 1
        assert "dispersal" in dispersal_cands[0].explanation.lower()


# ---------------------------------------------------------------------------
# 6. Test Unusual Crowd Movement Detector
# ---------------------------------------------------------------------------

class TestUnusualCrowdMovementDetector:
    def test_unusual_crowd_movement_vs_baseline(self):
        detector = UnusualCrowdMovementDetector()
        # Dominant baseline flow is to the right (+X, 0 radians).
        # Group of 4 people moves downwards (+Y, pi/2 radians = 90 deg deviation).
        tracks = [
            make_track(f"P{i}", "person", [
                (0.0, 200 + i * 20, 100),
                (1.0, 200 + i * 20, 130),
                (2.0, 200 + i * 20, 160),
            ])
            for i in range(4)
        ]
        scene_ctx = SceneContextData(
            scene_type="pedestrian_area",
            baseline=NormalBehaviorBaseline(dominant_heading_degrees=0.0),  # Flow along +X
        )
        ctx = IncidentContext(
            video_id="video_unusual_flow",
            fps=30.0,
            duration_seconds=3.0,
            tracks=tracks,
            scene_context=scene_ctx,
        )
        candidates = detector.analyze(ctx)
        unusual_cands = [c for c in candidates if c.event_type == "POTENTIAL_UNUSUAL_CROWD_MOVEMENT"]
        assert len(unusual_cands) >= 1
        assert "deviat" in unusual_cands[0].explanation.lower()


# ---------------------------------------------------------------------------
# 7. Test Zone Occupancy and Activity Detector
# ---------------------------------------------------------------------------

class TestZoneOccupancyAndActivityDetector:
    def test_restricted_zone_crowding(self):
        detector = ZoneOccupancyAndActivityDetector()
        # Define a restricted zone polygon
        zone = ZoneDefinition(
            zone_id="zone_sterile_01",
            name="Restricted Sterile Corridor",
            polygon=[(50.0, 50.0), (300.0, 50.0), (300.0, 300.0), (50.0, 300.0)],
        )
        # 4 people dwelling inside the restricted zone for 4.0 seconds
        tracks = [
            make_track(f"P{i}", "person", [
                (0.0, 100 + i * 25, 100),
                (2.0, 105 + i * 25, 105),
                (4.0, 110 + i * 25, 110),
            ])
            for i in range(4)
        ]
        ctx = IncidentContext(
            video_id="video_restricted_zone",
            fps=30.0,
            duration_seconds=5.0,
            tracks=tracks,
            zones=[zone],
        )
        candidates = detector.analyze(ctx)
        crowd_cands = [c for c in candidates if c.event_type == "POTENTIAL_RESTRICTED_ZONE_CROWDING"]
        assert len(crowd_cands) >= 1
        assert crowd_cands[0].spatial_context.zone_name == "Restricted Sterile Corridor"

    def test_boundary_jitter_suppression(self):
        detector = ZoneOccupancyAndActivityDetector()
        zone = ZoneDefinition(
            zone_id="zone_sterile_01",
            name="Restricted Area",
            polygon=[(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)],
        )
        # People only briefly clipping or oscillating across the boundary (< 1.5s inside)
        tracks = [
            make_track("P1", "person", [(0.0, 95, 150), (0.5, 105, 150), (1.0, 95, 150)]),
            make_track("P2", "person", [(0.0, 98, 160), (0.5, 102, 160), (1.0, 97, 160)]),
        ]
        ctx = IncidentContext(
            video_id="video_boundary_jitter",
            fps=30.0,
            duration_seconds=2.0,
            tracks=tracks,
            zones=[zone],
        )
        candidates = detector.analyze(ctx)
        crowd_cands = [c for c in candidates if c.event_type == "POTENTIAL_RESTRICTED_ZONE_CROWDING"]
        # Hysteresis + minimum dwell prevents brief jitter from registering as crowding
        assert len(crowd_cands) == 0


# ---------------------------------------------------------------------------
# 8. Test Negative Evidence and Validator
# ---------------------------------------------------------------------------

class TestNegativeEvidenceAndValidator:
    def test_queue_negative_evidence_refutes_surge_in_validator(self):
        # A candidate with orderly queue counter-evidence
        tracks = [make_track(f"P{i}", "person", [(0.0, 100 + i * 20, 100)]) for i in range(4)]
        ctx = IncidentContext(video_id="vid_test", fps=30.0, duration_seconds=5.0, tracks=tracks)

        neg_signals = NegativeEvidenceEngine.evaluate_crowd_surge_negative_evidence(
            person_tracks=tracks,
            context=ctx,
            is_queue=True,
            directional_coherence=0.90,
            mean_velocity=20.0,
        )
        candidate = IncidentCandidate(
            incident_id="cand_surge_01",
            video_id="vid_test",
            event_type="POTENTIAL_CROWD_SURGE",
            category=IncidentCategory.CROWD,
            start_time=0.0,
            end_time=3.0,
            duration=3.0,
            severity="HIGH",
            confidence=0.75,
            track_ids=["P0", "P1", "P2", "P3"],
            supporting_signals=[],
            contradictory_signals=neg_signals,
            explanation="Test surge hypothesis",
        )
        validator = IncidentCandidateValidator()
        validated = validator.validate_candidate(candidate, ctx)
        # Severe counter-evidence "Orderly Queue Flow" must reject or downgrade candidate
        assert validated.validation_decision in ["REJECTED", "REVIEW_REQUIRED"]


# ---------------------------------------------------------------------------
# 9. Test Incident Fusion Crowd Arbitration
# ---------------------------------------------------------------------------

class TestIncidentFusionCrowdArbitration:
    def test_crowd_surge_subsumes_high_density_in_same_window(self):
        # Surge and high density in identical window and tracks
        surge = IncidentCandidate(
            incident_id="surge_01",
            video_id="vid_fuse",
            event_type="POTENTIAL_CROWD_SURGE",
            category=IncidentCategory.CROWD,
            start_time=1.0,
            end_time=4.0,
            duration=3.0,
            severity="HIGH",
            confidence=0.80,
            track_ids=["P1", "P2", "P3", "P4"],
            supporting_signals=[],
            explanation="Crowd surge candidate",
        )
        density = IncidentCandidate(
            incident_id="density_01",
            video_id="vid_fuse",
            event_type="HIGH_PEDESTRIAN_DENSITY",
            category=IncidentCategory.CROWD,
            start_time=1.2,
            end_time=3.8,
            duration=2.6,
            severity="NORMAL",
            confidence=0.70,
            track_ids=["P1", "P2", "P3", "P4"],
            supporting_signals=[],
            explanation="High density candidate",
        )
        fusion = IncidentFusionEngine()
        fused = fusion.fuse_candidates([surge, density])
        assert len(fused) == 1
        # POTENTIAL_CROWD_SURGE is higher in hierarchy than HIGH_PEDESTRIAN_DENSITY
        assert fused[0].event_type == "POTENTIAL_CROWD_SURGE"
        assert "HIGH_PEDESTRIAN_DENSITY" in fused[0].incident_metadata.get("alternate_hypotheses", [])


# ---------------------------------------------------------------------------
# 10. Test Investigation Natural Language Routing
# ---------------------------------------------------------------------------

class TestInvestigationRouting:
    def test_crowd_investigation_queries(self):
        parser = InvestigationParser()

        q1 = parser.parse_query("show crowded areas")
        assert q1["is_supported"] is True
        assert q1["result_type"] == "security_events"
        assert q1["interpreted_filters"]["event_type"] == "HIGH_PEDESTRIAN_DENSITY"

        q2 = parser.parse_query("find crowd surges")
        assert q2["is_supported"] is True
        assert q2["result_type"] == "security_events"
        assert q2["interpreted_filters"]["event_type"] == "POTENTIAL_CROWD_SURGE"

        q3 = parser.parse_query("show crowd dispersal")
        assert q3["is_supported"] is True
        assert q3["result_type"] == "security_events"
        assert q3["interpreted_filters"]["event_type"] == "POTENTIAL_CROWD_DISPERSAL"

        q4 = parser.parse_query("show unusual crowd movement")
        assert q4["is_supported"] is True
        assert q4["result_type"] == "security_events"
        assert q4["interpreted_filters"]["event_type"] == "POTENTIAL_UNUSUAL_CROWD_MOVEMENT"

        q5 = parser.parse_query("show restricted zone occupancy")
        assert q5["is_supported"] is True
        assert q5["result_type"] == "security_events"
        assert q5["interpreted_filters"]["event_type"] == "POTENTIAL_RESTRICTED_ZONE_CROWDING"

        q6 = parser.parse_query("which zones had the highest occupancy?")
        assert q6["is_supported"] is True
        assert q6["result_type"] == "security_events"
        assert q6["interpreted_filters"]["event_type"] == "ZONE_OCCUPANCY_OBSERVATION"
