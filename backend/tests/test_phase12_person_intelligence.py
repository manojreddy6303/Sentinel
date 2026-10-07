"""
Sentinel Phase 12: Person Incident Intelligence Test Suite
Tests modular person detectors, person motion feature engine, pose feature fallback,
negative evidence refutations, candidate validation, incident fusion, and investigation services.
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
from ai.incidents.negative_evidence import NegativeEvidenceEngine
from ai.incidents.detectors.person.motion_features import PersonMotionFeatureEngine
from ai.incidents.detectors.person.pose_features import PoseFeatureEngine
from ai.incidents.detectors.person.fall import PersonFallDetector
from ai.incidents.detectors.person.person_down import PersonDownDetector
from ai.incidents.detectors.person.panic_running import PanicRunningDetector
from ai.incidents.detectors.person.rapid_movement import UnusualRapidPersonMovementDetector
from ai.incidents.detectors.person.altercation import PhysicalAltercationDetector
from ai.incidents.detectors.person.forced_movement import ForcedMovementDetector
from ai.incidents.detectors.person.following import PersonFollowingDetector
from ai.incidents.detectors.person.coordinated_movement import CoordinatedPersonMovementDetector

from backend.app.services.investigation_parser import InvestigationParser
from backend.app.services.investigation_service import InvestigationService
from ai.investigation.orchestrator import InvestigationOrchestrator
from database.session import SessionLocal
from database.models import VideoModel, SecurityEventModel


def make_person_track(
    track_id: str,
    timestamps: List[float],
    boxes: List[List[float]],
    confidences: List[float] = None,
) -> TrackedObject:
    """Helper to build a valid person TrackedObject."""
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
        object_class="person",
        first_seen=timestamps[0],
        last_seen=timestamps[-1],
        confidence=avg_conf,
        current_bbox=curr_bbox,
        trajectory=trajectory,
        history_bboxes=history_bboxes,
    )


def make_person_context(tracks: List[TrackedObject], video_id: str = "vid_person_test") -> IncidentContext:
    """Helper to build IncidentContext with full motion telemetry."""
    max_duration = max((t.last_seen for t in tracks), default=5.0)
    builder = IncidentContextBuilder()
    return builder.build_context(
        video_id=video_id,
        tracks=tracks,
        validated_detections=[],
        fps=30.0,
        duration_seconds=max_duration,
        sample_rate_fps=2.0,
    )


class TestPersonMotionFeatures:
    """Test person kinematics and aspect-ratio dynamics."""

    def test_aspect_ratio_transition_and_descent(self):
        # Upright person (w=30, h=80, ar=0.375) collapses to ground (w=80, h=30, ar=2.66)
        boxes = [
            [100.0, 100.0, 130.0, 180.0],  # upright
            [100.0, 120.0, 135.0, 185.0],  # starting to slip
            [100.0, 160.0, 180.0, 195.0],  # fallen, horizontal
            [100.0, 165.0, 180.0, 195.0],  # resting
        ]
        t = make_person_track("P-FALL", [0.0, 0.5, 1.0, 1.5], boxes)
        dyn = PersonMotionFeatureEngine.compute_person_dynamics(t)
        assert dyn["has_aspect_ratio_transition"] is True
        assert dyn["max_downward_velocity_px_s"] > 30.0
        assert dyn["final_aspect_ratio"] > 1.0

    def test_upright_standing_aspect_ratio(self):
        # Standing still upright (w=30, h=90, ar=0.33)
        boxes = [[100.0, 100.0, 130.0, 190.0] for _ in range(5)]
        t = make_person_track("P-STAND", [0.0, 1.0, 2.0, 3.0, 4.0], boxes)
        dyn = PersonMotionFeatureEngine.compute_person_dynamics(t)
        assert dyn["has_aspect_ratio_transition"] is False
        assert dyn["final_aspect_ratio"] < 0.50

    def test_pose_engine_graceful_fallback(self):
        status = PoseFeatureEngine.get_status()
        assert isinstance(status["is_available"], bool)
        # Verify extract_pose does not crash and reports pose_available accurately
        res = PoseFeatureEngine.extract_pose(None)
        assert res["pose_available"] is False
        assert res["keypoints"] is None


class TestPersonFallDetector:
    """Test fall candidate generation and recovery refutation."""

    def test_person_fall_candidate_generated(self):
        detector = PersonFallDetector(min_downward_velocity_px_s=25.0)
        boxes = [
            [100.0, 100.0, 130.0, 180.0],  # upright
            [105.0, 130.0, 140.0, 190.0],  # rapid descent
            [105.0, 165.0, 180.0, 198.0],  # on ground, horizontal
            [105.0, 165.0, 180.0, 198.0],  # motionless on ground
        ]
        t = make_person_track("P-FALL-01", [0.0, 0.5, 1.0, 1.5], boxes)
        ctx = make_person_context([t])
        candidates = detector.analyze(ctx)
        assert len(candidates) >= 1
        cand = candidates[0]
        assert cand.event_type == "POTENTIAL_PERSON_FALL"
        assert cand.human_verification_required is True

    def test_fall_refuted_when_walking_resumes(self):
        """If a person slips but quickly stands back up and walks away at speed, refutes fall alert."""
        detector = PersonFallDetector()
        boxes = [
            [100.0, 100.0, 130.0, 180.0],
            [105.0, 140.0, 140.0, 190.0],
            [110.0, 165.0, 180.0, 198.0],  # brief touch
            [130.0, 110.0, 160.0, 190.0],  # stands up
            [170.0, 110.0, 200.0, 190.0],  # walking away
            [210.0, 110.0, 240.0, 190.0],  # walking away
        ]
        t = make_person_track("P-RECOVER", [0.0, 0.5, 1.0, 1.5, 2.0, 2.5], boxes)
        ctx = make_person_context([t])
        candidates = detector.analyze(ctx)
        assert len(candidates) == 0


class TestPersonDownDetector:
    """Test sustained low-mobility ground posture vs upright standing."""

    def test_person_down_detected(self):
        detector = PersonDownDetector(min_down_duration_seconds=3.0)
        # Person remaining horizontal on ground for 4 seconds
        boxes = [[100.0, 200.0, 180.0, 235.0] for _ in range(5)]
        t = make_person_track("P-DOWN", [0.0, 1.0, 2.0, 3.0, 4.0], boxes)
        ctx = make_person_context([t])
        candidates = detector.analyze(ctx)
        assert len(candidates) >= 1
        assert candidates[0].event_type == "POTENTIAL_PERSON_DOWN"

    def test_standing_still_does_not_trigger_person_down(self):
        detector = PersonDownDetector(min_down_duration_seconds=3.0)
        # Person standing still upright (h=90, w=30, ar=0.33)
        boxes = [[100.0, 100.0, 130.0, 190.0] for _ in range(5)]
        t = make_person_track("P-STAND", [0.0, 1.0, 2.0, 3.0, 4.0], boxes)
        ctx = make_person_context([t])
        candidates = detector.analyze(ctx)
        assert len(candidates) == 0


class TestPanicRunningDetector:
    """Test panic running vs normal pedestrian transit."""

    def test_panic_running_detected(self):
        detector = PanicRunningDetector(min_running_speed_bl_s=2.8)
        # Person height = 60px. Running at 240px/s -> 4.0 body-lengths/sec
        boxes = [
            [100.0 + i * 120.0, 100.0, 130.0 + i * 120.0, 160.0]
            for i in range(5)
        ]
        t = make_person_track("P-RUN", [0.0, 0.5, 1.0, 1.5, 2.0], boxes)
        ctx = make_person_context([t])
        candidates = detector.analyze(ctx)
        assert len(candidates) >= 1
        assert candidates[0].event_type == "POTENTIAL_PANIC_RUNNING"

    def test_normal_walking_not_flagged_as_running(self):
        detector = PanicRunningDetector(min_running_speed_bl_s=2.8)
        # Normal walking: moves 20px per 0.5s (40px/s / 60px height = 0.67 bl/s)
        boxes = [
            [100.0 + i * 20.0, 100.0, 130.0 + i * 20.0, 160.0]
            for i in range(5)
        ]
        t = make_person_track("P-WALK", [0.0, 0.5, 1.0, 1.5, 2.0], boxes)
        ctx = make_person_context([t])
        candidates = detector.analyze(ctx)
        assert len(candidates) == 0


class TestUnusualRapidPersonMovementDetector:
    """Test relative speed anomaly against local scene baseline."""

    def test_outlier_speed_relative_to_scene(self):
        detector = UnusualRapidPersonMovementDetector(velocity_disparity_ratio=2.2)
        # Calm walking bystanders (0.5 bl/s)
        bystander1 = make_person_track("P-BY1", [0.0, 1.0, 2.0, 3.0], [[100.0 + i * 15.0, 100.0, 130.0 + i * 15.0, 160.0] for i in range(4)])
        bystander2 = make_person_track("P-BY2", [0.0, 1.0, 2.0, 3.0], [[200.0 + i * 15.0, 100.0, 230.0 + i * 15.0, 160.0] for i in range(4)])
        # Rapid subject (moves at 3.0 bl/s)
        sprint_boxes = [[100.0 + i * 90.0, 200.0, 130.0 + i * 90.0, 260.0] for i in range(4)]
        sprinter = make_person_track("P-SPRINT", [0.0, 0.5, 1.0, 1.5], sprint_boxes)

        ctx = make_person_context([bystander1, bystander2, sprinter])
        candidates = detector.analyze(ctx)
        assert len(candidates) >= 1
        assert candidates[0].event_type == "UNUSUAL_RAPID_PERSON_MOVEMENT"
        assert "P-SPRINT" in candidates[0].track_ids


class TestPhysicalAltercationDetector:
    """Test physical altercation reciprocal motion vs side-by-side walking."""

    def test_physical_altercation_detected(self):
        detector = PhysicalAltercationDetector(min_reciprocal_score=0.40)
        # Two people in tight proximity oscillating back-and-forth
        t1_boxes = [
            [100.0, 100.0, 130.0, 160.0],
            [120.0, 100.0, 150.0, 160.0],  # moves right towards t2
            [100.0, 100.0, 130.0, 160.0],  # moves back left
            [125.0, 100.0, 155.0, 160.0],  # lunges forward
            [105.0, 100.0, 135.0, 160.0],
        ]
        t2_boxes = [
            [135.0, 100.0, 165.0, 160.0],
            [125.0, 100.0, 155.0, 160.0],  # counter-moves left towards t1
            [140.0, 100.0, 170.0, 160.0],  # backs away
            [120.0, 100.0, 150.0, 160.0],  # grapples forward
            [140.0, 100.0, 170.0, 160.0],
        ]
        p1 = make_person_track("FIGHT-01", [0.0, 0.5, 1.0, 1.5, 2.0], t1_boxes)
        p2 = make_person_track("FIGHT-02", [0.0, 0.5, 1.0, 1.5, 2.0], t2_boxes)
        ctx = make_person_context([p1, p2])
        candidates = detector.analyze(ctx)
        assert len(candidates) >= 1
        cand = candidates[0]
        assert cand.event_type == "POTENTIAL_PHYSICAL_ALTERCATION"
        assert cand.validation_decision == "REVIEW_REQUIRED"

    def test_parallel_side_by_side_walking_refuted(self):
        """Two people walking side-by-side in parallel should NOT be flagged as altercation."""
        detector = PhysicalAltercationDetector()
        p1_boxes = [[100.0 + i * 25.0, 100.0, 130.0 + i * 25.0, 160.0] for i in range(5)]
        p2_boxes = [[100.0 + i * 25.0, 140.0, 130.0 + i * 25.0, 200.0] for i in range(5)]
        p1 = make_person_track("PAIR-01", [0.0, 0.5, 1.0, 1.5, 2.0], p1_boxes)
        p2 = make_person_track("PAIR-02", [0.0, 0.5, 1.0, 1.5, 2.0], p2_boxes)
        ctx = make_person_context([p1, p2])
        candidates = detector.analyze(ctx)
        assert len(candidates) == 0


class TestForcedMovementDetector:
    """Test forced movement pattern vs normal pair walking."""

    def test_forced_movement_detected(self):
        # Tests the single-deflection mode (min_deflections=1) — the hardened default is 3.
        # This test documents that a single sharp 90-degree turn in tight contact triggers detection
        # when the deflection threshold is set to legacy mode (1 deflection).
        detector = ForcedMovementDetector(min_duration_seconds=2.0, min_deflections=1)
        # Tight spatial proximity with sharp synchronized 90-deg course deviation
        p1_boxes = [
            [100.0, 100.0, 130.0, 160.0],
            [125.0, 100.0, 155.0, 160.0],
            [150.0, 100.0, 180.0, 160.0],
            [150.0, 135.0, 180.0, 195.0],  # sharp forced turn downward
            [150.0, 170.0, 180.0, 230.0],
            [150.0, 205.0, 180.0, 265.0],
        ]
        p2_boxes = [
            [110.0, 105.0, 140.0, 165.0],
            [135.0, 105.0, 165.0, 165.0],
            [160.0, 105.0, 190.0, 165.0],
            [160.0, 140.0, 190.0, 200.0],  # in lockstep constraint
            [160.0, 175.0, 190.0, 235.0],
            [160.0, 210.0, 190.0, 270.0],
        ]
        p1 = make_person_track("FORCE-01", [0.0, 0.5, 1.0, 1.5, 2.0, 2.5], p1_boxes)
        p2 = make_person_track("FORCE-02", [0.0, 0.5, 1.0, 1.5, 2.0, 2.5], p2_boxes)
        ctx = make_person_context([p1, p2])
        candidates = detector.analyze(ctx)
        assert len(candidates) >= 1
        assert candidates[0].event_type == "POTENTIAL_FORCED_MOVEMENT"
        assert candidates[0].validation_decision == "REVIEW_REQUIRED"


class TestPersonFollowingDetector:
    """Test person following lagged trajectory correlation vs queue."""

    def test_person_following_detected(self):
        detector = PersonFollowingDetector(min_duration_seconds=2.0)
        # Person 1 traverses a path at t=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        p1_boxes = [
            [100.0 + i * 25.0, 100.0, 130.0 + i * 25.0, 160.0]
            for i in range(7)
        ]
        # Person 2 traverses the identical path delayed by ~1.0s (t=[1.0 -> 3.5])
        p2_boxes = [
            [100.0 + (i - 2) * 25.0, 100.0, 130.0 + (i - 2) * 25.0, 160.0]
            for i in range(2, 8)
        ]
        p1 = make_person_track("LEAD-01", [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], p1_boxes)
        p2 = make_person_track("FOLLOW-01", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5], p2_boxes)

        ctx = make_person_context([p1, p2])
        candidates = detector.analyze(ctx)
        assert len(candidates) >= 1
        cand = candidates[0]
        assert cand.event_type == "PERSON_FOLLOWING"
        assert "LEAD-01" in cand.track_ids and "FOLLOW-01" in cand.track_ids


class TestCoordinatedPersonMovementDetector:
    """Test synchronized group trajectory flow."""

    def test_coordinated_group_movement_detected(self):
        # Uses min_group_size=2 and min_duration_seconds=3.0 to match 3-second test tracks.
        # The hardened defaults require min_group_size=3 and min_duration_seconds=4.0.
        # This test documents the 2-person minimum group case (parameterized mode).
        detector = CoordinatedPersonMovementDetector(min_group_size=2, min_duration_seconds=3.0)
        # Three people walking in identical heading and velocity in close spatial cluster
        p1 = make_person_track("GRP-01", [0.0, 1.0, 2.0, 3.0], [[100.0 + i * 30.0, 100.0, 130.0 + i * 30.0, 160.0] for i in range(4)])
        p2 = make_person_track("GRP-02", [0.0, 1.0, 2.0, 3.0], [[105.0 + i * 30.0, 135.0, 135.0 + i * 30.0, 195.0] for i in range(4)])
        p3 = make_person_track("GRP-03", [0.0, 1.0, 2.0, 3.0], [[110.0 + i * 30.0, 170.0, 140.0 + i * 30.0, 230.0] for i in range(4)])

        ctx = make_person_context([p1, p2, p3])
        candidates = detector.analyze(ctx)
        assert len(candidates) >= 1
        assert candidates[0].event_type == "COORDINATED_PERSON_MOVEMENT"
        assert len(candidates[0].track_ids) >= 2


class TestIncidentFusionAndValidation:
    """Test deduplication and fusion for related person candidates."""

    def test_fusion_merges_rapid_movement_and_fall(self):
        engine = IncidentFusionEngine(time_merge_tolerance_seconds=2.5)
        c1 = IncidentCandidate(
            incident_id="INC-P1",
            video_id="vid_person_test",
            detector_name="PanicRunningDetector",
            event_type="POTENTIAL_PANIC_RUNNING",
            category="person",
            start_time=10.0,
            end_time=11.5,
            duration=1.5,
            severity="NORMAL",
            confidence=0.75,
            track_ids=["TRACK-001"],
            supporting_signals=[SupportingSignal("high_speed", "High speed running", 0.80)],
        )
        c2 = IncidentCandidate(
            incident_id="INC-P2",
            video_id="vid_person_test",
            detector_name="PanicRunningDetector",
            event_type="POTENTIAL_PANIC_RUNNING",
            category="person",
            start_time=11.0,
            end_time=12.2,
            duration=1.2,
            severity="NORMAL",
            confidence=0.78,
            track_ids=["TRACK-001"],
            supporting_signals=[SupportingSignal("abrupt_turn", "Abrupt deflection", 0.76)],
        )
        fused = engine.fuse_incidents([c1, c2])
        assert len(fused) == 1
        assert fused[0].start_time == 10.0
        assert len(fused[0].supporting_signals) == 2


class TestInvestigationServiceAskSentinel:
    """Test natural language queries for person incidents."""

    def test_parse_person_queries(self):
        parser = InvestigationParser()

        r1 = parser.parse_query("What person incidents occurred?")
        assert r1["is_supported"] is True
        assert r1["result_type"] == "security_events"
        assert r1["interpreted_filters"].get("category") == "person"

        r2 = parser.parse_query("Show potential falls")
        assert r2["is_supported"] is True
        assert r2["result_type"] == "security_events"
        assert r2["interpreted_filters"].get("event_type") == "POTENTIAL_PERSON_FALL"

        r3 = parser.parse_query("Find person down events")
        assert r3["is_supported"] is True
        assert r3["interpreted_filters"].get("event_type") == "POTENTIAL_PERSON_DOWN"

        r4 = parser.parse_query("Were there any physical altercations?")
        assert r4["is_supported"] is True
        assert r4["interpreted_filters"].get("event_type") == "POTENTIAL_PHYSICAL_ALTERCATION"

        r5 = parser.parse_query("Show people following each other")
        assert r5["is_supported"] is True
        assert r5["interpreted_filters"].get("event_type") == "PERSON_FOLLOWING"

        r6 = parser.parse_query("Find coordinated movement")
        assert r6["is_supported"] is True
        assert r6["interpreted_filters"].get("event_type") == "COORDINATED_PERSON_MOVEMENT"

    def test_ask_sentinel_empty_person_response_message(self):
        """When no person incidents exist, response must be exact."""
        db = SessionLocal()
        vid = "vid_unit_test_person_empty"
        try:
            v_entry = db.query(VideoModel).filter(VideoModel.id == vid).first()
            if not v_entry:
                v_entry = VideoModel(
                    id=vid,
                    original_filename="test_person_scene.mp4",
                    storage_path="/tmp/test_person_scene.mp4",
                    file_size_bytes=1000,
                    duration_seconds=10.0,
                    status="COMPLETED",
                )
                db.add(v_entry)
                db.commit()

            service = InvestigationService()
            parsed = service.parser.parse_query("What person incidents occurred?")
            res = service.investigate_filters(
                video_id=vid,
                filters=parsed["interpreted_filters"],
                result_type=parsed["result_type"],
                query_text="What person incidents occurred?",
            )
            assert res["message"] == "No reliable person incident was detected in the available Sentinel data."

            # Also verify deterministic orchestrator synthesizer returns the exact same statement
            orch = InvestigationOrchestrator()
            synth_ans = orch._format_deterministic_grounded_response({
                "count": 0,
                "results": [],
                "evidence": [],
                "filters": parsed["interpreted_filters"],
            })
            assert synth_ans == "No reliable person incident was detected in the available Sentinel data."
        finally:
            db.query(VideoModel).filter(VideoModel.id == vid).delete()
            db.commit()
            db.close()


class TestPhase12ForensicAuditRegressions:
    """Specific regression tests verifying audit findings and corrections."""

    def test_bidirectional_trajectory_correlation_rejected(self):
        """Two people walking side-by-side in parallel must not produce mutual following events."""
        detector = PersonFollowingDetector(min_duration_seconds=2.0)
        # P1 and P2 walk side-by-side simultaneously (zero lag error is identical to lagged error)
        p1 = make_person_track("SIDE-01", [0.0, 1.0, 2.0, 3.0], [[100.0 + i * 20.0, 100.0, 130.0 + i * 20.0, 180.0] for i in range(4)])
        p2 = make_person_track("SIDE-02", [0.0, 1.0, 2.0, 3.0], [[100.0 + i * 20.0, 135.0, 130.0 + i * 20.0, 215.0] for i in range(4)])
        ctx = make_person_context([p1, p2])
        candidates = detector.analyze(ctx)
        # Must not claim either person is following the other
        assert len(candidates) == 0

    def test_transient_passing_pedestrians_rejected(self):
        """Two people passing each other in opposite directions must not trigger an altercation."""
        detector = PhysicalAltercationDetector(min_interaction_seconds=1.5)
        # P1 walks West to East, P2 walks East to West, crossing at t=1.5s
        p1 = make_person_track("PASS-01", [0.0, 1.0, 2.0, 3.0], [
            [50.0, 100.0, 80.0, 160.0],
            [90.0, 100.0, 120.0, 160.0],
            [130.0, 100.0, 160.0, 160.0],
            [170.0, 100.0, 200.0, 160.0],
        ])
        p2 = make_person_track("PASS-02", [0.0, 1.0, 2.0, 3.0], [
            [170.0, 105.0, 200.0, 165.0],
            [130.0, 105.0, 160.0, 165.0],
            [90.0, 105.0, 120.0, 165.0],
            [50.0, 105.0, 80.0, 165.0],
        ])
        ctx = make_person_context([p1, p2])
        candidates = detector.analyze(ctx)
        assert len(candidates) == 0

    def test_frame_boundary_truncation_suppresses_fall(self):
        """Aspect ratio inversion touching frame boundary (x1 < 25) must be suppressed as edge clipping."""
        detector = PersonFallDetector()
        # Track starts in center, walks near edge at x1=10, y1=10
        boxes = [
            [200.0, 100.0, 230.0, 190.0],
            [150.0, 100.0, 180.0, 190.0],
            [80.0, 100.0, 110.0, 190.0],
            [10.0, 100.0, 75.0, 155.0],  # clipped by border at x1=10!
        ]
        p = make_person_track("EDGE-01", [0.0, 1.0, 2.0, 3.0], boxes)
        ctx = make_person_context([p])
        candidates = detector.analyze(ctx)
        assert len(candidates) == 0

    def test_fusion_arbitration_for_competing_interaction_hypotheses(self):
        """Competing hypotheses for same track pair must be arbitrated into a single candidate."""
        fusion = IncidentFusionEngine()
        c1 = IncidentCandidate(
            incident_id="RAW-ALT-01",
            video_id="vid_test",
            event_type="POTENTIAL_PHYSICAL_ALTERCATION",
            category="person",
            start_time=10.0,
            end_time=13.0,
            duration=3.0,
            confidence=0.65,
            severity="NORMAL",
            track_ids=["P-01", "P-02"],
            object_classes=["person", "person"],
            supporting_signals=[SupportingSignal(signal_type="Sustained Close Proximity", description="Proximity", confidence=0.75, timestamp=11.5)],
            explanation="Altercation hypothesis",
        )
        c2 = IncidentCandidate(
            incident_id="RAW-FORCED-01",
            video_id="vid_test",
            event_type="POTENTIAL_FORCED_MOVEMENT",
            category="person",
            start_time=10.0,
            end_time=13.0,
            duration=3.0,
            confidence=0.60,
            severity="NORMAL",
            track_ids=["P-01", "P-02"],
            object_classes=["person", "person"],
            supporting_signals=[SupportingSignal(signal_type="Synchronized Course Deflection", description="Deflection", confidence=0.70, timestamp=11.5)],
            explanation="Forced movement hypothesis",
        )
        c3 = IncidentCandidate(
            incident_id="RAW-FOLLOW-01",
            video_id="vid_test",
            event_type="PERSON_FOLLOWING",
            category="person",
            start_time=10.0,
            end_time=13.0,
            duration=3.0,
            confidence=0.55,
            severity="NORMAL",
            track_ids=["P-01", "P-02"],
            object_classes=["person", "person"],
            supporting_signals=[SupportingSignal(signal_type="Lagged Trajectory Correlation", description="Lagged", confidence=0.65, timestamp=11.5)],
            explanation="Following hypothesis",
        )

        fused = fusion.fuse_incidents([c1, c2, c3])
        # Must be unified into a single arbitrated incident rather than 3 separate alerts!
        assert len(fused) == 1
        primary = fused[0]
        assert primary.confidence <= 0.65
        assert primary.validation_decision == "REVIEW_REQUIRED"
        assert "alternate_hypotheses" in (primary.incident_metadata or {})
        assert len(primary.incident_metadata["alternate_hypotheses"]) >= 2

    def test_score_independence_deduplication(self):
        """Correlated signals from the same physical family must not double-count bonuses."""
        from ai.incidents.scoring import IncidentScorer
        # 3 signals all belonging to SPATIAL_PROXIMITY family
        spatial_sigs = [
            SupportingSignal(signal_type="Sustained Close Proximity", description="Close", confidence=0.8, timestamp=1.0),
            SupportingSignal(signal_type="Spatial Contact Proximity", description="Contact", confidence=0.8, timestamp=1.0),
            SupportingSignal(signal_type="Clearance Distance Limit", description="Clearance", confidence=0.8, timestamp=1.0),
        ]
        score_single_family = IncidentScorer.calculate_evidence_score(
            base_confidence=0.50,
            supporting_signals=spatial_sigs,
            validation_decision="REVIEW_REQUIRED",
        )
        assert score_single_family["confidence"] <= 0.65
        assert score_single_family["unique_signal_families"] == 1
        assert "evidence_strength" in score_single_family

    def test_validator_downgrades_high_raw_score_on_negative_evidence(self):
        """IncidentCandidateValidator must downgrade raw high confidence when counter-evidence is present."""
        validator = IncidentCandidateValidator()
        cand = IncidentCandidate(
            incident_id="TEST-01",
            video_id="vid_01",
            event_type="POTENTIAL_PHYSICAL_ALTERCATION",
            category="person",
            start_time=5.0,
            end_time=8.0,
            duration=3.0,
            confidence=0.92,
            severity="HIGH",
            track_ids=["T1", "T2"],
            object_classes=["person", "person"],
            supporting_signals=[SupportingSignal(signal_type="Sustained Close Proximity", description="Close", confidence=0.85, timestamp=6.5)],
            contradictory_signals=[SupportingSignal(signal_type="Negative: Parallel Side-by-Side Walking", description="Parallel", confidence=0.90, timestamp=6.5)],
            explanation="Initial raw finding",
        )
        validated = validator.validate_candidate(cand)
        # Must be rejected because Parallel Side-by-Side Walking is in severe contradictions
        assert validated.validation_decision == ValidationDecision.REJECTED.value
        assert validated.confidence <= 0.25


class TestGenericFallFalsePositiveSuppressionAndPositiveDetection:
    """
    Mandatory regression tests for generic multi-signal temporal fall detection:
    A. Person walking out of bottom/side frame -> NO FALL
    B. Temporary bbox becoming wide due to occlusion -> NO FALL
    C. Single downward velocity spike -> NO FALL
    D. Brief crouch/bend then upright -> NO FALL
    E. Track termination with insufficient post-event evidence -> NO CONFIRMED FALL
    F. Upright -> rapid descent -> persistent low/horizontal posture -> FALL CANDIDATE
    """

    def test_scenario_a_person_walking_out_of_bottom_or_side_frame_no_fall(self):
        """A: Person approaching and exiting camera frame border (bottom/side) must not trigger fall."""
        detector = PersonFallDetector()
        # Frame dimensions: 768 x 432
        # Person walks toward bottom border; final bbox is truncated at bottom edge (y2=430 on height 432)
        boxes = [
            [400.0, 100.0, 480.0, 300.0],  # upright in scene (h=200, w=80, ar=0.40)
            [420.0, 150.0, 510.0, 360.0],  # walking forward (h=210, w=90, ar=0.43)
            [450.0, 220.0, 560.0, 410.0],  # approaching bottom (h=190, w=110, ar=0.58)
            [470.0, 260.0, 620.0, 430.0],  # clipped at bottom boundary (dist to 432 is 2px! w=150, h=170, ar=0.88)
        ]
        t = make_person_track("P-EXIT-BOTTOM", [0.0, 1.0, 2.0, 3.0], boxes)
        ctx = make_person_context([t])
        ctx.video_metadata = {"width": 768, "height": 432}
        candidates = detector.analyze(ctx)
        assert len(candidates) == 0, f"Expected 0 fall candidates for frame exit, got {len(candidates)}"

    def test_scenario_b_temporary_bbox_wide_due_to_occlusion_no_fall(self):
        """B: Temporary wide bounding box from occlusion/reassociation must not trigger fall."""
        detector = PersonFallDetector()
        # Person upright, 1 frame temporarily wide, then immediately normal upright
        boxes = [
            [200.0, 200.0, 260.0, 380.0],  # upright (w=60, h=180, ar=0.33)
            [205.0, 205.0, 265.0, 385.0],  # upright (w=60, h=180, ar=0.33)
            [190.0, 210.0, 380.0, 390.0],  # momentary occlusion merge (w=190, h=180, ar=1.05)
            [215.0, 215.0, 275.0, 395.0],  # returned upright (w=60, h=180, ar=0.33)
            [220.0, 220.0, 280.0, 400.0],  # returned upright (w=60, h=180, ar=0.33)
        ]
        t = make_person_track("P-OCCLUDED", [0.0, 1.0, 2.0, 3.0, 4.0], boxes)
        ctx = make_person_context([t])
        candidates = detector.analyze(ctx)
        assert len(candidates) == 0, "Occlusion glitch must not trigger fall"

    def test_scenario_c_single_downward_velocity_spike_no_fall(self):
        """C: Rapid vertical movement without horizontal posture collapse must not trigger fall."""
        detector = PersonFallDetector()
        # Person rapidly steps down or jumps, maintaining upright geometry (ar ~ 0.35)
        boxes = [
            [200.0, 100.0, 250.0, 250.0],  # upright (h=150, w=50, ar=0.33)
            [200.0, 180.0, 250.0, 330.0],  # rapid downward transit (80px in 1s, but upright! ar=0.33)
            [200.0, 200.0, 250.0, 350.0],  # upright (h=150, w=50, ar=0.33)
            [200.0, 205.0, 250.0, 355.0],  # upright (h=150, w=50, ar=0.33)
        ]
        t = make_person_track("P-STEP-DOWN", [0.0, 1.0, 2.0, 3.0], boxes)
        ctx = make_person_context([t])
        candidates = detector.analyze(ctx)
        assert len(candidates) == 0, "Downward velocity without horizontal posture must not trigger fall"

    def test_scenario_d_brief_crouch_bend_then_upright_no_fall(self):
        """D: Person bending down or crouching then resuming upright posture must not trigger fall."""
        detector = PersonFallDetector()
        # Person bends down to tie shoes, then stands back up
        boxes = [
            [200.0, 100.0, 250.0, 250.0],  # upright (w=50, h=150, ar=0.33)
            [200.0, 170.0, 290.0, 260.0],  # crouch/bend (w=90, h=90, ar=1.0)
            [200.0, 105.0, 250.0, 255.0],  # stands back up (w=50, h=150, ar=0.33)
            [200.0, 105.0, 250.0, 255.0],  # stands upright
        ]
        t = make_person_track("P-CROUCH", [0.0, 0.5, 1.0, 1.5], boxes)
        ctx = make_person_context([t])
        candidates = detector.analyze(ctx)
        assert len(candidates) == 0, "Bending/crouching with recovery must not trigger fall"

    def test_scenario_e_track_termination_without_post_event_evidence_no_fall(self):
        """E: Track terminating at geometry change with zero post-event dwell must not confirm fall."""
        detector = PersonFallDetector()
        # Upright person has aspect ratio shift at the final frame, and track terminates immediately
        boxes = [
            [200.0, 100.0, 250.0, 250.0],  # upright (w=50, h=150, ar=0.33)
            [200.0, 110.0, 250.0, 260.0],  # upright (w=50, h=150, ar=0.33)
            [200.0, 160.0, 290.0, 250.0],  # single terminal observation (w=90, h=90, ar=1.0)
        ]
        t = make_person_track("P-DISAPPEAR", [0.0, 1.0, 2.0], boxes)
        ctx = make_person_context([t])
        candidates = detector.analyze(ctx)
        assert len(candidates) == 0, "Single-frame terminal change without post-event evidence must not trigger fall"

    def test_scenario_f_genuine_fall_upright_descent_persistent_ground_dwell(self):
        """F: Upright -> rapid descent -> persistent horizontal low posture on ground triggers fall candidate."""
        detector = PersonFallDetector()
        boxes = [
            [200.0, 100.0, 250.0, 250.0],  # upright (w=50, h=150, ar=0.33)
            [200.0, 170.0, 270.0, 280.0],  # rapid downward descent (h=110, w=70, ar=0.64)
            [200.0, 220.0, 360.0, 280.0],  # horizontal on ground (w=160, h=60, ar=2.67)
            [200.0, 220.0, 360.0, 280.0],  # motionless on ground (persists!)
            [200.0, 220.0, 360.0, 280.0],  # motionless on ground (persists!)
        ]
        t = make_person_track("P-GENUINE-FALL", [0.0, 0.5, 1.0, 1.5, 2.0], boxes)
        ctx = make_person_context([t])
        candidates = detector.analyze(ctx)
        assert len(candidates) >= 1, "Genuine persistent fall must produce fall candidate"
        cand = candidates[0]
        assert cand.event_type == "POTENTIAL_PERSON_FALL"
        assert cand.human_verification_required is True

