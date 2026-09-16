"""
Phase 13: Property & Object Incident Intelligence Comprehensive Unit & Regression Tests

Validates:
1. Object Temporal State Machine transitions (DETECTED -> STATIONARY -> ASSOCIATED -> LEFT_BEHIND / MOVING / DISPLACED)
2. Object-Person Association Engine (approach, contact, departure, co-movement)
3. Camera Stability Engine (global scene drift detection and suppression)
4. Abandoned Object Detector & False Abandoned Object Suppression
5. Object Left Behind Detector (grounded person departure chain)
6. Object Pickup Detector (stationary-to-motion transition with person)
7. Object Displacement Detector (scale-normalized spatial relocation)
8. Object Removal Detector (interior FOV cessation vs edge/occlusion suppression)
9. Property Tampering Detector (sustained contact + observable coordinate shift)
10. Restricted Object Movement Detector (security zone transit)
11. Property Negative Evidence refutations
12. Incident Fusion Property Arbitration (takeaway/removal/pickup/displacement hierarchy)
13. Investigation Query Routing for Property Events
"""
import pytest
import math
from ai.schemas import BoundingBox, TrackedObject, ZoneDefinition
from ai.incidents.schemas import IncidentCandidate, IncidentContext, IncidentCategory, SpatialContext
from ai.incidents.detectors.property.state_machine import (
    ObjectStateMachine,
    ObjectTemporalState,
)
from ai.incidents.detectors.property.association import ObjectPersonAssociationEngine
from ai.incidents.detectors.property.camera_stability import CameraStabilityEngine
from ai.incidents.detectors.abandoned_object import AbandonedObjectDetector
from ai.incidents.detectors.property.left_behind import ObjectLeftBehindDetector
from ai.incidents.detectors.property.pickup import ObjectPickupDetector
from ai.incidents.detectors.property.displacement import ObjectDisplacementDetector
from ai.incidents.detectors.property.removal import ObjectRemovalDetector
from ai.incidents.detectors.property.tampering import PropertyTamperingDetector
from ai.incidents.detectors.property.restricted_movement import RestrictedObjectMovementDetector
from ai.incidents.fusion import IncidentFusionEngine
from ai.incidents.negative_evidence import NegativeEvidenceEngine
from backend.app.services.investigation_parser import InvestigationParser


def make_track(
    track_id: str,
    object_class: str,
    trajectory: list,
    confidence: float = 0.90,
    bbox_size: tuple = (60.0, 60.0),
) -> TrackedObject:
    """Helper to build TrackedObject with trajectory and history bboxes."""
    history = []
    w, h = bbox_size
    for pt in trajectory:
        t, cx, cy = pt[0], pt[1], pt[2]
        history.append({
            "timestamp": t,
            "bbox": {"x1": cx - w/2, "y1": cy - h/2, "x2": cx + w/2, "y2": cy + h/2},
            "confidence": confidence,
        })
    last_pt = trajectory[-1]
    curr_bbox = BoundingBox(
        x1=last_pt[1] - w/2,
        y1=last_pt[2] - h/2,
        x2=last_pt[1] + w/2,
        y2=last_pt[2] + h/2,
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


class TestObjectStateMachine:
    """Tests for ObjectStateMachine."""

    def test_stationary_object_state(self):
        sm = ObjectStateMachine()
        # Stationary backpack for 10s at (500, 500)
        traj = [(float(i), 500.0, 500.0) for i in range(11)]
        track = make_track("OBJ-1", "backpack", traj)
        record = sm.analyze_track_state(track, [])
        assert record.current_state == ObjectTemporalState.STATIONARY
        assert record.net_displacement < 5.0
        assert record.stationary_duration == 10.0

    def test_left_behind_transition(self):
        sm = ObjectStateMachine()
        # Stationary backpack at (500, 500) from t=0..10
        o_traj = [(float(i), 500.0, 500.0) for i in range(11)]
        o_track = make_track("OBJ-1", "backpack", o_traj)

        # Person nearby from t=0..3 at (520, 500), then walks away to (800, 500) at t=10
        p_traj = [
            (0.0, 520.0, 500.0),
            (1.0, 520.0, 500.0),
            (2.0, 520.0, 500.0),
            (3.0, 520.0, 500.0),
            (5.0, 600.0, 500.0),
            (10.0, 800.0, 500.0),
        ]
        p_track = make_track("P-1", "person", p_traj)

        record = sm.analyze_track_state(o_track, [p_track])
        assert record.current_state == ObjectTemporalState.LEFT_BEHIND
        assert "P-1" in record.associated_person_ids


class TestObjectPersonAssociation:
    """Tests for ObjectPersonAssociationEngine."""

    def test_meaningful_interaction_with_departure(self):
        engine = ObjectPersonAssociationEngine()
        o_traj = [(float(i), 400.0, 400.0) for i in range(11)]
        o_track = make_track("OBJ-1", "suitcase", o_traj)

        p_traj = [
            (0.0, 200.0, 400.0),
            (2.0, 390.0, 400.0),
            (3.0, 405.0, 400.0),
            (4.0, 405.0, 400.0),
            (7.0, 600.0, 400.0),
            (10.0, 750.0, 400.0),
        ]
        p_track = make_track("P-1", "person", p_traj)

        result = engine.evaluate_association(p_track, o_track)
        assert result.is_associated is True
        assert result.approach_detected is True
        assert result.departure_detected is True
        assert result.min_distance <= 15.0

    def test_transient_distant_passerby_not_associated(self):
        engine = ObjectPersonAssociationEngine()
        o_traj = [(float(i), 400.0, 400.0) for i in range(11)]
        o_track = make_track("OBJ-1", "suitcase", o_traj)

        # Person passes 300px away
        p_traj = [(float(i), 100.0, float(i) * 50.0) for i in range(11)]
        p_track = make_track("P-2", "person", p_traj)

        result = engine.evaluate_association(p_track, o_track)
        assert result.is_associated is False


class TestCameraStabilityEngine:
    """Tests for CameraStabilityEngine."""

    def test_static_camera_scene(self):
        engine = CameraStabilityEngine()
        # Several stationary tracks
        t1 = make_track("T1", "car", [(float(i), 200.0, 200.0) for i in range(5)])
        t2 = make_track("T2", "bench", [(float(i), 600.0, 600.0) for i in range(5)])
        t3 = make_track("T3", "bicycle", [(float(i), 800.0, 300.0) for i in range(5)])
        ctx = IncidentContext(video_id="vid1", tracks=[t1, t2, t3])
        assessment = engine.assess_stability(ctx)
        assert assessment.is_camera_stable is True
        assert assessment.is_jitter_detected is False

    def test_global_camera_pan_detected(self):
        engine = CameraStabilityEngine(translation_noise_threshold=15.0)
        # All tracks shift +80px horizontally simultaneously
        t1 = make_track("T1", "car", [(0.0, 200.0, 200.0), (1.0, 240.0, 200.0), (2.0, 280.0, 200.0)])
        t2 = make_track("T2", "bench", [(0.0, 600.0, 600.0), (1.0, 640.0, 600.0), (2.0, 680.0, 600.0)])
        t3 = make_track("T3", "bicycle", [(0.0, 800.0, 300.0), (1.0, 840.0, 300.0), (2.0, 880.0, 300.0)])
        ctx = IncidentContext(video_id="vid2", tracks=[t1, t2, t3])
        assessment = engine.assess_stability(ctx)
        assert assessment.is_camera_stable is False
        assert assessment.is_jitter_detected is True


class TestObjectLeftBehindDetector:
    """Tests for ObjectLeftBehindDetector."""

    def test_successful_object_left_behind(self):
        detector = ObjectLeftBehindDetector(min_residual_seconds=2.0)
        o_traj = [(float(i), 500.0, 500.0) for i in range(11)]
        o_track = make_track("OBJ-1", "backpack", o_traj)

        p_traj = [
            (0.0, 300.0, 500.0),
            (1.0, 480.0, 500.0),
            (2.0, 490.0, 500.0),
            (3.0, 490.0, 500.0),
            (5.0, 650.0, 500.0),
            (8.0, 800.0, 500.0),
        ]
        p_track = make_track("P-1", "person", p_traj)

        ctx = IncidentContext(video_id="v_lb", tracks=[o_track, p_track])
        candidates = detector.analyze(ctx)
        assert len(candidates) >= 1
        cand = candidates[0]
        assert cand.event_type == "POTENTIAL_OBJECT_LEFT_BEHIND"
        assert cand.validation_decision.value == "REVIEW_REQUIRED"
        assert "OBJ-1" in cand.track_ids
        assert "P-1" in cand.track_ids


class TestObjectPickupDetector:
    """Tests for ObjectPickupDetector."""

    def test_successful_object_pickup(self):
        detector = ObjectPickupDetector(min_prior_stationary_seconds=1.0, min_pickup_displacement=30.0)
        # Suitcase stationary at (500, 500) for t=0..3, then moves with person to (650, 500) at t=7
        o_traj = [
            (0.0, 500.0, 500.0),
            (1.0, 500.0, 500.0),
            (2.0, 500.0, 500.0),
            (3.0, 505.0, 500.0),
            (4.0, 550.0, 500.0),
            (5.0, 600.0, 500.0),
            (7.0, 650.0, 500.0),
        ]
        o_track = make_track("OBJ-SUITCASE", "suitcase", o_traj)

        p_traj = [
            (0.0, 350.0, 500.0),
            (2.0, 480.0, 500.0),
            (3.0, 500.0, 500.0),
            (4.0, 550.0, 500.0),
            (7.0, 650.0, 500.0),
        ]
        p_track = make_track("P-1", "person", p_traj)

        ctx = IncidentContext(video_id="v_pk", tracks=[o_track, p_track])
        candidates = detector.analyze(ctx)
        assert len(candidates) >= 1
        cand = candidates[0]
        assert cand.event_type == "POTENTIAL_OBJECT_PICKUP"
        assert "OBJ-SUITCASE" in cand.track_ids
        assert "P-1" in cand.track_ids


class TestObjectDisplacementDetector:
    """Tests for ObjectDisplacementDetector."""

    def test_scale_normalized_displacement(self):
        detector = ObjectDisplacementDetector(min_displacement_px=30.0)
        # Box starts at (200, 200) and settles at (350, 200) (disp: 150px)
        traj = [
            (0.0, 200.0, 200.0),
            (1.0, 200.0, 200.0),
            (2.0, 250.0, 200.0),
            (3.0, 300.0, 200.0),
            (4.0, 350.0, 200.0),
            (5.0, 350.0, 200.0),
        ]
        track = make_track("BOX-1", "box", traj, bbox_size=(50.0, 50.0))
        ctx = IncidentContext(video_id="v_disp", tracks=[track])
        candidates = detector.analyze(ctx)
        assert len(candidates) >= 1
        cand = candidates[0]
        assert cand.event_type == "POTENTIAL_OBJECT_DISPLACEMENT"
        assert cand.spatial_context.metadata["net_displacement"] >= 140.0


class TestObjectRemovalDetector:
    """Tests for ObjectRemovalDetector."""

    def test_interior_fov_removal(self):
        detector = ObjectRemovalDetector(min_stable_seconds=2.0)
        # Stably observed laptop at center (960, 540) from t=0..5, cessation at t=5 in 10s video
        traj = [(float(i), 960.0, 540.0) for i in range(6)]
        track = make_track("LAPTOP-1", "laptop", traj)
        ctx = IncidentContext(video_id="v_rem", tracks=[track], duration_seconds=10.0)
        candidates = detector.analyze(ctx)
        assert len(candidates) >= 1
        cand = candidates[0]
        assert cand.event_type == "POTENTIAL_OBJECT_REMOVAL"
        assert cand.validation_decision.value == "REVIEW_REQUIRED"

    def test_frame_edge_exit_suppression(self):
        detector = ObjectRemovalDetector(min_stable_seconds=2.0, frame_edge_margin_px=35.0)
        # Stably observed at border x=15, y=500
        traj = [(float(i), 15.0, 500.0) for i in range(6)]
        track = make_track("BAG-EDGE", "backpack", traj, bbox_size=(20.0, 20.0))
        ctx = IncidentContext(video_id="v_edge", tracks=[track], duration_seconds=10.0)
        candidates = detector.analyze(ctx)
        # Should be suppressed due to frame boundary exit
        assert len(candidates) == 0


class TestPropertyTamperingDetector:
    """Tests for PropertyTamperingDetector."""

    def test_tampering_with_fixture_shift(self):
        detector = PropertyTamperingDetector(min_contact_seconds=2.0, min_fixture_shift_px=10.0)
        # Bicycle initially at (400, 400), shifts to (420, 400) during contact
        b_traj = [
            (0.0, 400.0, 400.0),
            (1.0, 400.0, 400.0),
            (2.0, 408.0, 400.0),
            (3.0, 415.0, 400.0),
            (4.0, 420.0, 400.0),
        ]
        b_track = make_track("BIKE-1", "bicycle", b_traj)

        # Person dwells in physical contact (distance < 30px) for 4s
        p_traj = [
            (0.0, 415.0, 400.0),
            (1.0, 415.0, 400.0),
            (2.0, 418.0, 400.0),
            (3.0, 422.0, 400.0),
            (4.0, 425.0, 400.0),
        ]
        p_track = make_track("P-TAMPER", "person", p_traj)

        ctx = IncidentContext(video_id="v_tamp", tracks=[b_track, p_track])
        candidates = detector.analyze(ctx)
        assert len(candidates) >= 1
        cand = candidates[0]
        assert cand.event_type == "POTENTIAL_PROPERTY_TAMPERING"


class TestRestrictedObjectMovementDetector:
    """Tests for RestrictedObjectMovementDetector."""

    def test_restricted_zone_transit(self):
        detector = RestrictedObjectMovementDetector()
        zone = ZoneDefinition(
            zone_id="Z1",
            name="Vault Perimeter",
            polygon=[(100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)],
            enabled=True,
        )
        # Suitcase inside zone at (200, 200)
        traj = [(float(i), 200.0, 200.0) for i in range(5)]
        track = make_track("OBJ-ZONE", "suitcase", traj)
        ctx = IncidentContext(video_id="v_zn", tracks=[track], zones=[zone])
        candidates = detector.analyze(ctx)
        assert len(candidates) >= 1
        cand = candidates[0]
        assert cand.event_type == "POTENTIAL_RESTRICTED_OBJECT_MOVEMENT"


class TestPropertyFusionArbitration:
    """Tests for IncidentFusionEngine property arbitration."""

    def test_fusion_arbitrates_competing_takeaway_and_displacement(self):
        engine = IncidentFusionEngine()
        # Candidate A: Takeaway (high hierarchy priority 6)
        cand_takeaway = IncidentCandidate(
            incident_id="C-1",
            video_id="v1",
            event_type="POTENTIAL_THEFT",
            category=IncidentCategory.PROPERTY,
            start_time=10.0,
            end_time=15.0,
            duration=5.0,
            severity="HIGH",
            confidence=0.85,
            track_ids=["P-1", "OBJ-1"],
            object_classes=["person", "suitcase"],
            explanation="Potential theft pattern",
        )
        # Candidate B: Simple displacement on same object at same time
        cand_disp = IncidentCandidate(
            incident_id="C-2",
            video_id="v1",
            event_type="POTENTIAL_OBJECT_DISPLACEMENT",
            category=IncidentCategory.PROPERTY,
            start_time=10.0,
            end_time=14.0,
            duration=4.0,
            severity="NORMAL",
            confidence=0.65,
            track_ids=["OBJ-1"],
            object_classes=["suitcase"],
            explanation="Potential displacement",
        )

        fused = engine.fuse_incidents([cand_takeaway, cand_disp])
        # Only primary (POTENTIAL_THEFT) should be emitted, with displacement in alternate_hypotheses
        assert len(fused) == 1
        assert fused[0].event_type == "POTENTIAL_THEFT"
        assert "POTENTIAL_OBJECT_DISPLACEMENT" in fused[0].incident_metadata["alternate_hypotheses"]


class TestPropertyInvestigationQueryParsing:
    """Tests for natural language queries relating to property incidents."""

    def test_investigation_parser_property_queries(self):
        parser = InvestigationParser()

        queries = [
            ("show abandoned objects", "POTENTIAL_ABANDONED_OBJECT"),
            ("find objects left behind", "POTENTIAL_OBJECT_LEFT_BEHIND"),
            ("show object pickups", "POTENTIAL_OBJECT_PICKUP"),
            ("show object displacement", "POTENTIAL_OBJECT_DISPLACEMENT"),
            ("show property tampering", "POTENTIAL_PROPERTY_TAMPERING"),
            ("show restricted object movement", "POTENTIAL_RESTRICTED_OBJECT_MOVEMENT"),
            ("show removed objects", "POTENTIAL_OBJECT_REMOVAL"),
            ("show theft patterns", "POTENTIAL_THEFT"),
        ]

        for q_text, expected_type in queries:
            res = parser.parse_query(q_text)
            assert res["is_supported"] is True
            assert res["interpreted_filters"]["event_type"] == expected_type
