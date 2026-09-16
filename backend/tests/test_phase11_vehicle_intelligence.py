"""
Sentinel Phase 11: Vehicle Incident Intelligence Test Suite
Tests modular vehicle detectors, pairwise interaction model, adaptive sampling,
perspective normalization, negative evidence refutations, candidate validation,
incident fusion, and investigation service compatibility.
"""

import pytest
import math
from typing import List

from ai.schemas import BoundingBox, TrackedObject
from ai.incidents.schemas import (
    IncidentCandidate,
    IncidentContext,
    SpatialContext,
    SupportingSignal,
    ValidationDecision,
)
from ai.incidents.validator import IncidentCandidateValidator
from ai.incidents.fusion import IncidentFusionEngine
from ai.incidents.context import IncidentContextBuilder
from ai.incidents.scene_context import SceneContextData, SceneContextEngine
from ai.incidents.negative_evidence import NegativeEvidenceEngine
from ai.incidents.detectors.vehicle.interaction import VehicleInteractionModel
from ai.incidents.detectors.vehicle.sampling import AdaptiveTemporalSamplingEngine
from ai.incidents.detectors.vehicle.collision import VehicleCollisionDetector
from ai.incidents.detectors.vehicle.near_collision import NearCollisionDetector
from ai.incidents.detectors.vehicle.sudden_stop import SuddenStopDetector
from ai.incidents.detectors.vehicle.wrong_way import WrongWayVehicleDetector
from ai.incidents.detectors.vehicle.unusual_trajectory import UnusualTrajectoryDetector
from ai.incidents.detectors.vehicle.stationary_vehicle import StationaryVehicleDetector

from backend.app.services.investigation_parser import InvestigationParser
from backend.app.services.investigation_service import InvestigationService
from ai.investigation.orchestrator import InvestigationOrchestrator
from database.session import SessionLocal
from database.models import VideoModel, SecurityEventModel


def make_mock_track(
    track_id: str,
    cls: str = "car",
    timestamps: List[float] = None,
    boxes: List[List[float]] = None,
    confidences: List[float] = None,
) -> TrackedObject:
    """Helper to build a valid TrackedObject."""
    if timestamps is None:
        timestamps = [0.0, 0.5, 1.0, 1.5, 2.0]
    if boxes is None:
        boxes = [[100.0, 100.0, 150.0, 150.0] for _ in timestamps]
    if confidences is None:
        confidences = [0.90 for _ in timestamps]

    trajectory = []
    history_bboxes = []
    for ts, b in zip(timestamps, boxes):
        cx = (b[0] + b[2]) / 2.0
        cy = (b[1] + b[3]) / 2.0
        trajectory.append((ts, cx, cy))
        history_bboxes.append({
            "timestamp": ts,
            "bbox": {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]},
        })

    last_box = boxes[-1]
    curr_bbox = BoundingBox(x1=last_box[0], y1=last_box[1], x2=last_box[2], y2=last_box[3])
    avg_conf = sum(confidences) / len(confidences)

    return TrackedObject(
        track_id=track_id,
        object_class=cls,
        first_seen=timestamps[0],
        last_seen=timestamps[-1],
        confidence=avg_conf,
        current_bbox=curr_bbox,
        trajectory=trajectory,
        history_bboxes=history_bboxes,
    )


def make_mock_context(tracks: List[TrackedObject], video_id: str = "vid_test", fps: float = 30.0) -> IncidentContext:
    """Helper to create IncidentContext with full motion telemetry."""
    max_duration = max((t.last_seen for t in tracks), default=5.0)
    builder = IncidentContextBuilder()
    return builder.build_context(
        video_id=video_id,
        tracks=tracks,
        validated_detections=[],
        fps=fps,
        duration_seconds=max_duration,
        sample_rate_fps=2.0,
    )


class TestVehicleInteractionModel:
    """Test pairwise kinematic and geometric interaction features."""

    def test_perspective_normalized_distance(self):
        box_a = [100.0, 100.0, 200.0, 200.0]
        box_b = [300.0, 300.0, 400.0, 400.0]
        norm_dist = VehicleInteractionModel.perspective_normalized_distance(box_a, box_b)
        assert norm_dist > 0.0
        assert VehicleInteractionModel.perspective_normalized_distance(box_a, box_a) == 0.0

    def test_compute_iou(self):
        box_a = [0.0, 0.0, 100.0, 100.0]
        box_b = [50.0, 0.0, 150.0, 100.0]
        iou = VehicleInteractionModel.compute_iou(box_a, box_b)
        assert 0.30 <= iou <= 0.35

        box_c = [200.0, 200.0, 300.0, 300.0]
        assert VehicleInteractionModel.compute_iou(box_a, box_c) == 0.0

    def test_relative_heading_degrees(self):
        v_a = (10.0, 0.0)
        v_b = (15.0, 0.0)
        assert VehicleInteractionModel.relative_heading_degrees(v_a, v_b) == 0.0

        v_c = (-10.0, 0.0)
        assert VehicleInteractionModel.relative_heading_degrees(v_a, v_c) == pytest.approx(180.0, abs=1.0)


class TestAdaptiveTemporalSampling:
    """Test burst sampling triggering and temporal evidence evaluation."""

    def test_trigger_burst_on_rapid_approach(self):
        engine = AdaptiveTemporalSamplingEngine(min_rapid_approach_rate=35.0, min_proximity_threshold=70.0)
        needs_burst, rec_fps, window = engine.evaluate_burst_trigger({
            "max_approach_rate_px_s": 65.0,
            "min_distance_px": 50.0,
            "event_time": 10.0,
        })
        assert needs_burst is True
        assert rec_fps >= 10
        assert window == (8.5, 11.5)

    def test_no_burst_for_normal_following(self):
        engine = AdaptiveTemporalSamplingEngine(min_rapid_approach_rate=35.0, min_proximity_threshold=70.0)
        needs_burst, rec_fps, window = engine.evaluate_burst_trigger({
            "max_approach_rate_px_s": 5.0,
            "min_distance_px": 150.0,
            "event_time": 5.0,
        })
        assert needs_burst is False

    def test_temporal_evidence_limited_flag(self):
        engine = AdaptiveTemporalSamplingEngine()
        ev = engine.assess_temporal_sufficiency(
            actual_sample_rate_fps=1.0,
            target_velocity=85.0,
        )
        assert ev["is_limited"] is True
        assert ev["temporal_evidence_tag"] == "TEMPORAL_EVIDENCE_LIMITED"


class TestVehicleCollisionDetector:
    """Test collision detection and clearance refutation."""

    def test_rejection_when_no_physical_contact(self):
        """Vehicles traveling close but maintaining positive clearance must NOT collide."""
        detector = VehicleCollisionDetector()
        t1 = make_mock_track(
            "TRACK-001", "car",
            timestamps=[1.0, 1.5, 2.0, 2.5],
            boxes=[[100.0, 100.0, 160.0, 160.0], [150.0, 100.0, 210.0, 160.0], [200.0, 100.0, 260.0, 160.0], [250.0, 100.0, 310.0, 160.0]],
        )
        t2 = make_mock_track(
            "TRACK-002", "car",
            timestamps=[1.0, 1.5, 2.0, 2.5],
            boxes=[[100.0, 180.0, 160.0, 240.0], [150.0, 180.0, 210.0, 240.0], [200.0, 180.0, 260.0, 240.0], [250.0, 180.0, 310.0, 240.0]],
        )
        ctx = make_mock_context([t1, t2])
        candidates = detector.analyze(ctx)
        assert len(candidates) == 0

    def test_collision_candidate_with_physical_contact_and_stoppage(self):
        """Verified bounding box overlap followed by sudden velocity drop generates candidate."""
        detector = VehicleCollisionDetector(deceleration_threshold=20.0, min_approach_rate=20.0)
        t1 = make_mock_track(
            "TRACK-001", "car",
            timestamps=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            boxes=[
                [100.0, 100.0, 180.0, 180.0],
                [140.0, 100.0, 220.0, 180.0],
                [180.0, 100.0, 260.0, 180.0],  # overlap at 1.0s
                [180.0, 100.0, 260.0, 180.0],  # arrested post-contact motion
                [180.0, 100.0, 260.0, 180.0],
                [180.0, 100.0, 260.0, 180.0],
            ],
        )
        t2 = make_mock_track(
            "TRACK-002", "car",
            timestamps=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            boxes=[
                [260.0, 100.0, 340.0, 180.0],
                [220.0, 100.0, 300.0, 180.0],
                [180.0, 100.0, 260.0, 180.0],  # direct physical overlap
                [180.0, 100.0, 260.0, 180.0],  # arrested post-contact motion
                [180.0, 100.0, 260.0, 180.0],
                [180.0, 100.0, 260.0, 180.0],
            ],
        )
        ctx = make_mock_context([t1, t2])
        candidates = detector.analyze(ctx)
        assert len(candidates) >= 1
        cand = candidates[0]
        assert cand.event_type == "POTENTIAL_VEHICLE_COLLISION"
        assert cand.human_verification_required is True


class TestNearCollisionDetector:
    """Test near-collision vs normal passing clearance."""

    def test_near_collision_with_evasive_swerve(self):
        detector = NearCollisionDetector(max_proximity_threshold_px=80.0, min_approach_rate=20.0, min_evasive_angle_deg=15.0)
        t1 = make_mock_track(
            "TRACK-001", "car",
            timestamps=[0.0, 0.5, 1.0, 1.5, 2.0],
            boxes=[
                [100.0, 100.0, 160.0, 160.0],
                [140.0, 100.0, 200.0, 160.0],
                [180.0, 100.0, 240.0, 160.0],
                [220.0, 100.0, 280.0, 160.0],
                [260.0, 100.0, 320.0, 160.0],
            ],
        )
        t2 = make_mock_track(
            "TRACK-002", "car",
            timestamps=[0.0, 0.5, 1.0, 1.5, 2.0],
            boxes=[
                [120.0, 190.0, 180.0, 250.0],
                [150.0, 165.0, 210.0, 225.0],
                [180.0, 165.0, 240.0, 225.0],  # closest point of approach without overlap
                [210.0, 215.0, 270.0, 275.0],  # sharp lateral swerve
                [250.0, 250.0, 310.0, 310.0],  # transit continues
            ],
        )
        ctx = make_mock_context([t1, t2])
        candidates = detector.analyze(ctx)
        assert len(candidates) >= 1
        assert candidates[0].event_type == "POTENTIAL_NEAR_COLLISION"

    def test_normal_overtaking_does_not_trigger_near_collision(self):
        detector = NearCollisionDetector()
        t1 = make_mock_track(
            "TRACK-001", "car",
            timestamps=[0.0, 0.5, 1.0, 1.5, 2.0],
            boxes=[[100.0 + i * 20.0, 100.0, 160.0 + i * 20.0, 160.0] for i in range(5)],
        )
        t2 = make_mock_track(
            "TRACK-002", "car",
            timestamps=[0.0, 0.5, 1.0, 1.5, 2.0],
            boxes=[[80.0 + i * 40.0, 220.0, 140.0 + i * 40.0, 280.0] for i in range(5)],
        )
        ctx = make_mock_context([t1, t2])
        candidates = detector.analyze(ctx)
        assert len(candidates) == 0


class TestSuddenStopDetector:
    """Test isolated sudden stop vs synchronized traffic queue."""

    def test_isolated_sudden_stop_detected(self):
        detector = SuddenStopDetector(min_pre_stop_velocity=20.0, deceleration_drop_threshold=15.0, min_stop_duration_seconds=1.0)
        t1 = make_mock_track(
            "TRACK-001", "car",
            timestamps=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            boxes=[
                [100.0, 100.0, 160.0, 160.0],
                [160.0, 100.0, 220.0, 160.0],
                [220.0, 100.0, 280.0, 160.0],
                [225.0, 100.0, 285.0, 160.0],  # sudden stop at 1.5s
                [225.0, 100.0, 285.0, 160.0],
                [225.0, 100.0, 285.0, 160.0],
                [225.0, 100.0, 285.0, 160.0],
            ],
        )
        t2 = make_mock_track(
            "TRACK-002", "car",
            timestamps=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            boxes=[[100.0 + i * 40.0, 250.0, 160.0 + i * 40.0, 310.0] for i in range(7)],
        )
        ctx = make_mock_context([t1, t2])
        candidates = detector.analyze(ctx)
        assert len(candidates) >= 1
        assert candidates[0].event_type == "POTENTIAL_SUDDEN_VEHICLE_STOP"

    def test_synchronized_traffic_slowdown_refuted(self):
        detector = SuddenStopDetector()
        boxes_sync = [
            [100.0, 100.0, 160.0, 160.0],
            [150.0, 100.0, 210.0, 160.0],
            [170.0, 100.0, 230.0, 160.0],
            [175.0, 100.0, 235.0, 160.0],
            [175.0, 100.0, 235.0, 160.0],
        ]
        t1 = make_mock_track("TRACK-001", "car", boxes=boxes_sync)
        t2 = make_mock_track("TRACK-002", "car", boxes=[[b[0], b[1] + 120.0, b[2], b[3] + 120.0] for b in boxes_sync])

        ctx = make_mock_context([t1, t2])
        candidates = detector.analyze(ctx)
        validator = IncidentCandidateValidator()
        for cand in candidates:
            v_res = validator.validate_candidate(cand)
            assert v_res.decision in (ValidationDecision.REJECTED, ValidationDecision.REVIEW_REQUIRED)


class TestWrongWayVehicleDetector:
    """Test opposing traffic direction vs turning vehicle."""

    def test_wrong_way_detected_against_established_flow(self):
        detector = WrongWayVehicleDetector()
        flow_tracks = [
            make_mock_track(f"FLOW-{i}", "car", boxes=[[100.0 + j * 30.0, 100.0 + i * 50.0, 160.0 + j * 30.0, 160.0 + i * 50.0] for j in range(5)])
            for i in range(4)
        ]
        wrong_track = make_mock_track(
            "WRONG-01", "car",
            boxes=[[300.0 - j * 30.0, 120.0, 360.0 - j * 30.0, 180.0] for j in range(5)],
        )
        ctx = make_mock_context(flow_tracks + [wrong_track])
        candidates = detector.analyze(ctx)
        assert len(candidates) >= 1
        assert candidates[0].event_type == "POTENTIAL_WRONG_WAY_VEHICLE"
        assert "WRONG-01" in candidates[0].track_ids

    def test_turning_vehicle_refuted_as_wrong_way(self):
        detector = WrongWayVehicleDetector()
        flow_tracks = [
            make_mock_track(f"FLOW-{i}", "car", boxes=[[100.0 + j * 30.0, 100.0 + i * 50.0, 160.0 + j * 30.0, 160.0 + i * 50.0] for j in range(5)])
            for i in range(4)
        ]
        turn_track = make_mock_track(
            "TURN-01", "car",
            boxes=[[150.0, 100.0 + j * 30.0, 210.0, 160.0 + j * 30.0] for j in range(5)],
        )
        ctx = make_mock_context(flow_tracks + [turn_track])
        candidates = detector.analyze(ctx)
        wrong_cands = [c for c in candidates if "TURN-01" in c.track_ids]
        assert len(wrong_cands) == 0


class TestUnusualTrajectoryDetector:
    """Test lateral weaving vs smooth lane change."""

    def test_erratic_weaving_detected(self):
        detector = UnusualTrajectoryDetector()
        boxes_weave = [
            [100.0, 100.0, 160.0, 160.0],
            [140.0, 140.0, 200.0, 200.0],
            [180.0, 90.0, 240.0, 150.0],
            [220.0, 150.0, 280.0, 210.0],
            [260.0, 95.0, 320.0, 155.0],
            [300.0, 145.0, 360.0, 205.0],
        ]
        t = make_mock_track("WEAVE-01", "car", timestamps=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5], boxes=boxes_weave)
        ctx = make_mock_context([t])
        candidates = detector.analyze(ctx)
        assert len(candidates) >= 1
        assert candidates[0].event_type == "POTENTIAL_UNUSUAL_VEHICLE_TRAJECTORY"

    def test_smooth_lane_change_not_flagged(self):
        detector = UnusualTrajectoryDetector()
        boxes_lane_change = [
            [100.0, 100.0, 160.0, 160.0],
            [130.0, 105.0, 190.0, 165.0],
            [160.0, 115.0, 220.0, 175.0],
            [190.0, 125.0, 250.0, 185.0],
            [220.0, 130.0, 280.0, 190.0],
            [250.0, 130.0, 310.0, 190.0],
        ]
        t = make_mock_track("LANE-01", "car", timestamps=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5], boxes=boxes_lane_change)
        ctx = make_mock_context([t])
        candidates = detector.analyze(ctx)
        assert len(candidates) == 0


class TestStationaryVehicleDetector:
    """Test prolonged stationary vehicle in corridor vs moving traffic."""

    def test_stationary_vehicle_detected(self):
        detector = StationaryVehicleDetector(min_stationary_duration_seconds=3.0)
        t_stopped = make_mock_track(
            "STALLED-01", "car",
            timestamps=[0.0, 1.0, 2.0, 3.0, 4.0],
            boxes=[[200.0, 200.0, 260.0, 260.0] for _ in range(5)],
        )
        t_moving = make_mock_track(
            "FLOW-01", "car",
            timestamps=[0.0, 1.0, 2.0, 3.0, 4.0],
            boxes=[[100.0 + i * 50.0, 100.0, 160.0 + i * 50.0, 160.0] for i in range(5)],
        )
        ctx = make_mock_context([t_stopped, t_moving])
        candidates = detector.analyze(ctx)
        assert len(candidates) >= 1
        assert candidates[0].event_type == "POTENTIAL_STATIONARY_VEHICLE"
        assert "STALLED-01" in candidates[0].track_ids


class TestIncidentFusionAndValidation:
    """Test incident fusion for overlapping vehicle candidates and global validator."""

    def test_fusion_merges_overlapping_vehicle_candidates(self):
        engine = IncidentFusionEngine(time_merge_tolerance_seconds=2.0)
        c1 = IncidentCandidate(
            incident_id="INC-1",
            video_id="vid_test",
            detector_name="VehicleCollisionDetector",
            event_type="POTENTIAL_VEHICLE_COLLISION",
            category="vehicle",
            start_time=12.0,
            end_time=12.5,
            duration=0.5,
            severity="HIGH",
            confidence=0.75,
            track_ids=["TRACK-001", "TRACK-002"],
            supporting_signals=[SupportingSignal("physical_contact", "Contact observed", 0.85)],
        )
        c2 = IncidentCandidate(
            incident_id="INC-2",
            video_id="vid_test",
            detector_name="VehicleCollisionDetector",
            event_type="POTENTIAL_VEHICLE_COLLISION",
            category="vehicle",
            start_time=12.5,
            end_time=13.1,
            duration=0.6,
            severity="HIGH",
            confidence=0.80,
            track_ids=["TRACK-001", "TRACK-002"],
            supporting_signals=[SupportingSignal("velocity_drop", "Velocity dropped", 0.75)],
        )
        fused = engine.fuse_incidents([c1, c2])
        assert len(fused) == 1
        assert fused[0].start_time == 12.0
        assert len(fused[0].supporting_signals) == 2


class TestInvestigationServiceAskSentinel:
    """Test natural language queries for vehicle incidents."""

    def test_parse_vehicle_queries(self):
        parser = InvestigationParser()

        r1 = parser.parse_query("What vehicle incidents occurred?")
        assert r1["is_supported"] is True
        assert r1["result_type"] == "security_events"
        assert r1["interpreted_filters"].get("category") == "vehicle"

        r2 = parser.parse_query("Show potential collisions")
        assert r2["is_supported"] is True
        assert r2["result_type"] == "security_events"
        assert r2["interpreted_filters"].get("event_type") == "POTENTIAL_VEHICLE_COLLISION"

        r3 = parser.parse_query("Show near misses")
        assert r3["is_supported"] is True
        assert r3["interpreted_filters"].get("event_type") == "POTENTIAL_NEAR_COLLISION"

        r4 = parser.parse_query("Were there any wrong way vehicles?")
        assert r4["is_supported"] is True
        assert r4["interpreted_filters"].get("event_type") == "POTENTIAL_WRONG_WAY_VEHICLE"

        r5 = parser.parse_query("Did any car make a sudden stop?")
        assert r5["is_supported"] is True
        assert r5["interpreted_filters"].get("event_type") == "POTENTIAL_SUDDEN_VEHICLE_STOP"

    def test_ask_sentinel_empty_vehicle_response_message(self):
        """When no vehicle incidents exist, response must be exact."""
        db = SessionLocal()
        vid = "vid_unit_test_vehicle_empty"
        try:
            v_entry = db.query(VideoModel).filter(VideoModel.id == vid).first()
            if not v_entry:
                v_entry = VideoModel(
                    id=vid,
                    original_filename="test_vehicle_scene.mp4",
                    storage_path="/tmp/test_vehicle_scene.mp4",
                    file_size_bytes=1000,
                    duration_seconds=10.0,
                    status="COMPLETED",
                )
                db.add(v_entry)
                db.commit()

            service = InvestigationService()
            parsed = service.parser.parse_query("What vehicle incidents occurred?")
            res = service.investigate_filters(
                video_id=vid,
                filters=parsed["interpreted_filters"],
                result_type=parsed["result_type"],
                query_text="What vehicle incidents occurred?",
            )
            assert res["message"] == "No reliable vehicle incident was detected in the available Sentinel data."

            # Also verify deterministic orchestrator synthesizer returns the exact same statement
            orch = InvestigationOrchestrator()
            synth_ans = orch._format_deterministic_grounded_response({
                "count": 0,
                "results": [],
                "evidence": [],
                "filters": parsed["interpreted_filters"],
            })
            assert synth_ans == "No reliable vehicle incident was detected in the available Sentinel data."
        finally:
            db.query(VideoModel).filter(VideoModel.id == vid).delete()
            db.commit()
            db.close()
