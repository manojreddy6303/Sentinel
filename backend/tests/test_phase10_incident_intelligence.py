"""
Phase 10: Universal Incident Intelligence Engine Comprehensive Test Suite

Tests:
1. Detector interface contract & candidate construction
2. Incident Detector Registry & Failure Isolation
3. Universal Motion Engine calculations
4. Spatial Relationship Engine calculations
5. Temporal Analysis Engine calculations
6. Incident Scoring & calibrated safety language
7. Incident Fusion Engine & duplicate suppression
8. Context-aware Prolonged Presence (vehicles on road vs lingering persons)
9. Database persistence & metadata retrieval
10. Backward compatibility with SecurityEvent
11. Edge cases (empty tracks, single-frame tracks, missing classes, invalid bboxes, zero duration)
"""
import pytest
import math
from typing import List

from ai.schemas import BoundingBox, TrackedObject, ZoneDefinition, SecurityEvent
from ai.incidents.schemas import (
    IncidentCandidate,
    IncidentCategory,
    IncidentContext,
    SupportingSignal,
    EvidenceCandidate,
    SpatialContext,
    TemporalContext,
)
from ai.incidents.base_detector import BaseIncidentDetector
from ai.incidents.registry import IncidentDetectorRegistry
from ai.incidents.motion import UniversalMotionEngine
from ai.incidents.spatial import SpatialRelationshipEngine
from ai.incidents.temporal import TemporalAnalysisEngine
from ai.incidents.context import IncidentContextBuilder
from ai.incidents.scoring import IncidentScorer
from ai.incidents.fusion import IncidentFusionEngine
from ai.incidents.engine import IncidentIntelligenceEngine
from ai.incidents.detectors.zone_intrusion import ZoneIntrusionDetector
from ai.incidents.detectors.prolonged_presence import ProlongedPresenceDetector
from ai.incidents.detectors.abandoned_object import AbandonedObjectDetector
from ai.incidents.detectors.theft_and_takeaway import TheftAndTakeawayDetector
from ai.incidents.detectors.activity_analysis import ActivityAnalysisDetector
from ai.incidents.detectors.observational_anomaly import ObservationalAnomalyDetector
from ai.incidents.detectors.future_stubs import VehicleCollisionDetector


# ===========================================================================
# 1. Motion Engine Tests
# ===========================================================================

class TestUniversalMotionEngine:

    def test_straight_line_motion(self):
        engine = UniversalMotionEngine()
        track = TrackedObject(
            track_id="T-1",
            object_class="car",
            first_seen=0.0,
            last_seen=4.0,
            confidence=0.9,
            current_bbox=BoundingBox(0, 0, 50, 50),
            trajectory=[(0.0, 100.0, 100.0), (1.0, 120.0, 100.0), (2.0, 140.0, 100.0), (3.0, 160.0, 100.0), (4.0, 180.0, 100.0)],
        )
        motions = engine.compute_track_motion(track)
        assert len(motions) == 5

        # Check final frame
        last_m = motions[-1]
        assert last_m.displacement == pytest.approx(80.0, abs=0.1)
        assert last_m.distance_traveled == pytest.approx(80.0, abs=0.1)
        assert last_m.velocity_estimate == pytest.approx(20.0, abs=0.1)
        assert last_m.path_consistency == pytest.approx(1.0, abs=0.01)
        assert not last_m.is_stationary

        summary = engine.summarize_track_motion(motions)
        assert summary["total_duration"] == pytest.approx(4.0, abs=0.01)
        assert summary["net_displacement"] == pytest.approx(80.0, abs=0.1)
        assert not summary["is_predominantly_stationary"]

    def test_stationary_dwell_motion(self):
        engine = UniversalMotionEngine()
        track = TrackedObject(
            track_id="T-STAT",
            object_class="person",
            first_seen=0.0,
            last_seen=5.0,
            confidence=0.88,
            current_bbox=BoundingBox(10, 10, 30, 70),
            trajectory=[(0.0, 100.0, 100.0), (1.0, 100.5, 100.2), (2.0, 100.2, 100.5), (3.0, 100.8, 100.1), (4.0, 100.1, 100.3), (5.0, 100.3, 100.2)],
        )
        motions = engine.compute_track_motion(track)
        summary = engine.summarize_track_motion(motions)
        assert summary["is_predominantly_stationary"]
        assert summary["max_stationary_duration"] >= 4.0
        assert summary["net_displacement"] < 5.0

    def test_inter_track_relative_motion(self):
        t1 = TrackedObject(
            track_id="CAR-A",
            object_class="car",
            first_seen=0.0,
            last_seen=3.0,
            confidence=0.9,
            current_bbox=BoundingBox(0, 0, 40, 40),
            trajectory=[(0.0, 0.0, 100.0), (1.0, 30.0, 100.0), (2.0, 60.0, 100.0), (3.0, 90.0, 100.0)],
        )
        t2 = TrackedObject(
            track_id="CAR-B",
            object_class="car",
            first_seen=0.0,
            last_seen=3.0,
            confidence=0.9,
            current_bbox=BoundingBox(100, 0, 140, 40),
            trajectory=[(0.0, 200.0, 100.0), (1.0, 170.0, 100.0), (2.0, 140.0, 100.0), (3.0, 110.0, 100.0)],
        )
        rel = UniversalMotionEngine.compute_inter_track_relative_motion(t1, t2)
        assert len(rel) == 4
        # At t=0 distance is 200, at t=3 distance is 20
        assert rel[0]["distance"] == pytest.approx(200.0, abs=0.5)
        assert rel[-1]["distance"] == pytest.approx(20.0, abs=0.5)
        # Approach rate should be positive (closing in)
        assert rel[1]["approach_rate"] > 0


# ===========================================================================
# 2. Spatial Relationship Engine Tests
# ===========================================================================

class TestSpatialRelationshipEngine:

    def test_iou_and_overlap(self):
        b1 = BoundingBox(10, 10, 50, 50)
        b2 = BoundingBox(30, 30, 70, 70)
        b3 = BoundingBox(100, 100, 150, 150)

        assert SpatialRelationshipEngine.bbox_overlap(b1, b2)
        assert not SpatialRelationshipEngine.bbox_overlap(b1, b3)
        iou = SpatialRelationshipEngine.iou(b1, b2)
        assert 0.14 < iou < 0.16

    def test_point_in_polygon_and_zone_dwell(self):
        poly = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        assert SpatialRelationshipEngine.point_in_polygon((50.0, 50.0), poly)
        assert not SpatialRelationshipEngine.point_in_polygon((150.0, 50.0), poly)

        zone = ZoneDefinition(
            zone_id="Z-TEST",
            name="Loading Dock",
            polygon=poly,
            target_classes=["person"],
        )
        track = TrackedObject(
            track_id="P-1",
            object_class="person",
            first_seen=1.0,
            last_seen=6.0,
            confidence=0.85,
            current_bbox=BoundingBox(40, 40, 60, 80),
            trajectory=[(1.0, -10.0, 50.0), (2.0, 50.0, 50.0), (3.0, 50.0, 50.0), (4.0, 50.0, 50.0), (5.0, 50.0, 50.0), (6.0, 120.0, 50.0)],
        )
        dwell = SpatialRelationshipEngine.track_zone_dwell_analysis(track, zone)
        assert dwell["entered"]
        assert dwell["entry_time"] == pytest.approx(2.0)
        assert dwell["exited"]
        assert dwell["total_dwell_seconds"] == pytest.approx(3.0)

    def test_trajectory_intersection(self):
        # Path A goes from (0, 50) to (100, 50)
        t_a = TrackedObject(
            track_id="A",
            object_class="car",
            first_seen=0.0,
            last_seen=2.0,
            confidence=0.9,
            current_bbox=BoundingBox(0, 0, 10, 10),
            trajectory=[(0.0, 0.0, 50.0), (2.0, 100.0, 50.0)],
        )
        # Path B goes from (50, 0) to (50, 100)
        t_b = TrackedObject(
            track_id="B",
            object_class="car",
            first_seen=0.0,
            last_seen=2.0,
            confidence=0.9,
            current_bbox=BoundingBox(0, 0, 10, 10),
            trajectory=[(0.0, 50.0, 0.0), (2.0, 50.0, 100.0)],
        )
        assert SpatialRelationshipEngine.check_trajectory_intersection(t_a, t_b)


# ===========================================================================
# 3. Temporal Analysis Engine Tests
# ===========================================================================

class TestTemporalAnalysisEngine:

    def test_disappearance(self):
        # Person enters at 0s, leaves at 10s. Object present from 0s to 3s.
        obj = TrackedObject(
            track_id="OBJ-1",
            object_class="backpack",
            first_seen=0.0,
            last_seen=3.0,
            confidence=0.8,
            current_bbox=BoundingBox(10, 10, 20, 20),
            trajectory=[(0.0, 10.0, 10.0), (3.0, 10.0, 10.0)],
        )
        person = TrackedObject(
            track_id="PER-1",
            object_class="person",
            first_seen=0.0,
            last_seen=10.0,
            confidence=0.9,
            current_bbox=BoundingBox(50, 50, 70, 90),
            trajectory=[(0.0, 12.0, 10.0), (3.0, 12.0, 10.0), (10.0, 80.0, 80.0)],
        )
        assert TemporalAnalysisEngine.detect_disappearance(obj, person, loss_window_seconds=4.0)

    def test_sudden_deceleration(self):
        engine = UniversalMotionEngine()
        track = TrackedObject(
            track_id="C-DECEL",
            object_class="car",
            first_seen=0.0,
            last_seen=3.0,
            confidence=0.95,
            current_bbox=BoundingBox(0, 0, 40, 40),
            trajectory=[(0.0, 0.0, 0.0), (1.0, 100.0, 0.0), (2.0, 102.0, 0.0), (3.0, 103.0, 0.0)],
        )
        motions = engine.compute_track_motion(track)
        stops = TemporalAnalysisEngine.detect_sudden_deceleration(motions, deceleration_threshold=80.0)
        assert len(stops) > 0
        assert stops[0]["prior_velocity"] == pytest.approx(100.0, abs=1.0)


# ===========================================================================
# 4. Detector Registry & Failure Isolation Tests
# ===========================================================================

class FaultyCrashDetector(BaseIncidentDetector):
    detector_name = "faulty_crash_detector"
    detector_version = "0.0.1"
    category = IncidentCategory.GENERAL

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        raise RuntimeError("Simulated crash in faulty detector")


class HealthyDetector(BaseIncidentDetector):
    detector_name = "healthy_detector"
    detector_version = "1.0.0"
    category = IncidentCategory.GENERAL

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        return [
            self.build_candidate(
                video_id=context.video_id,
                event_type="HEALTHY_EVENT",
                start_time=1.0,
                end_time=3.0,
                severity="NORMAL",
                confidence=0.85,
                explanation="Healthy detector executed successfully",
            )
        ]


class TestIncidentDetectorRegistry:

    def test_failure_isolation(self):
        registry = IncidentDetectorRegistry()
        registry.register(FaultyCrashDetector())
        registry.register(HealthyDetector())

        context = IncidentContext(
            video_id="v-iso",
            fps=30.0,
            duration_seconds=10.0,
            sample_rate_fps=1.0,
            validated_detections=[],
            tracks=[],
        )

        candidates, diagnostics = registry.execute_all(context)

        # Faulty detector should NOT crash execution
        assert diagnostics["failed_detectors"] == 1
        assert "faulty_crash_detector" in diagnostics["errors"]
        assert diagnostics["executed_detectors"] == 1
        assert len(candidates) == 1
        assert candidates[0].event_type == "HEALTHY_EVENT"

    def test_enable_disable(self):
        registry = IncidentDetectorRegistry()
        registry.register(HealthyDetector())
        registry.set_enabled("healthy_detector", False)

        context = IncidentContext(
            video_id="v-iso",
            fps=30.0,
            duration_seconds=10.0,
            sample_rate_fps=1.0,
            validated_detections=[],
            tracks=[],
        )
        candidates, diagnostics = registry.execute_all(context)
        assert len(candidates) == 0
        assert diagnostics["execution_details"]["healthy_detector"]["status"] == "disabled"


# ===========================================================================
# 5. Context-Aware Prolonged Presence (Vehicle vs Person)
# ===========================================================================

class TestProlongedPresenceContextAwareness:

    def test_vehicle_moving_on_road_does_not_trigger_loitering(self):
        detector = ProlongedPresenceDetector(
            person_threshold_seconds=4.0,
            vehicle_stationary_threshold_seconds=15.0,
            max_loitering_displacement=120.0,
        )
        # Vehicle traveling 200px over 10 seconds (normal traffic transit)
        car = TrackedObject(
            track_id="CAR-HIGHWAY",
            object_class="car",
            first_seen=0.0,
            last_seen=10.0,
            confidence=0.92,
            current_bbox=BoundingBox(0, 0, 50, 50),
            trajectory=[(t, float(t * 30), 100.0) for t in range(11)],
        )
        motion_engine = UniversalMotionEngine()
        motions = motion_engine.compute_track_motion(car)
        summary = motion_engine.summarize_track_motion(motions)

        context = IncidentContext(
            video_id="v-road",
            fps=30.0,
            duration_seconds=10.0,
            sample_rate_fps=1.0,
            validated_detections=[],
            tracks=[car],
            track_motions={car.track_id: motions},
            motion_summaries={car.track_id: summary},
        )

        candidates = detector.analyze(context)
        # MUST NOT produce prolonged presence for normal traveling car!
        assert len(candidates) == 0

    def test_person_lingering_triggers_loitering(self):
        detector = ProlongedPresenceDetector(
            person_threshold_seconds=4.0,
            max_loitering_displacement=120.0,
        )
        person = TrackedObject(
            track_id="PER-LINGER",
            object_class="person",
            first_seen=0.0,
            last_seen=6.0,
            confidence=0.89,
            current_bbox=BoundingBox(10, 10, 30, 80),
            trajectory=[(t, 50.0 + (t * 2), 50.0) for t in range(7)],
        )
        motion_engine = UniversalMotionEngine()
        motions = motion_engine.compute_track_motion(person)
        summary = motion_engine.summarize_track_motion(motions)

        context = IncidentContext(
            video_id="v-linger",
            fps=30.0,
            duration_seconds=6.0,
            sample_rate_fps=1.0,
            validated_detections=[],
            tracks=[person],
            track_motions={person.track_id: motions},
            motion_summaries={person.track_id: summary},
        )

        candidates = detector.analyze(context)
        assert len(candidates) == 1
        assert candidates[0].event_type == "PROLONGED_PRESENCE"


# ===========================================================================
# 6. Incident Fusion & Deduplication Tests
# ===========================================================================

class TestIncidentFusionEngine:

    def test_duplicate_signal_clustering(self):
        fusion = IncidentFusionEngine(time_merge_tolerance_seconds=2.0, spatial_merge_distance=100.0)
        c1 = IncidentCandidate(
            incident_id="INC-1",
            video_id="v-fuse",
            event_type="POTENTIAL_INTRUSION",
            category="zone",
            start_time=1.0,
            end_time=3.0,
            duration=2.0,
            severity="NORMAL",
            confidence=0.8,
            track_ids=["T-1"],
            object_classes=["person"],
            supporting_signals=[SupportingSignal(signal_type="Zone Breach", description="Breached Zone A", confidence=0.8)],
            explanation="Zone entry 1",
            spatial_context=SpatialContext(centroid=(50.0, 50.0)),
        )
        c2 = IncidentCandidate(
            incident_id="INC-2",
            video_id="v-fuse",
            event_type="POTENTIAL_INTRUSION",
            category="zone",
            start_time=2.5,
            end_time=5.0,
            duration=2.5,
            severity="HIGH",
            confidence=0.88,
            track_ids=["T-1"],
            object_classes=["person"],
            supporting_signals=[SupportingSignal(signal_type="Zone Dwell", description="Lingered in Zone A", confidence=0.88)],
            explanation="Zone entry 2",
            spatial_context=SpatialContext(centroid=(55.0, 52.0)),
        )

        fused = fusion.fuse_incidents([c1, c2])
        assert len(fused) == 1
        f = fused[0]
        assert f.start_time == 1.0
        assert f.end_time == 5.0
        assert f.severity == "HIGH"
        assert len(f.supporting_signals) == 2


# ===========================================================================
# 7. Backward Compatibility & SecurityEvent Conversion
# ===========================================================================

class TestBackwardCompatibility:

    def test_to_security_event_conversion(self):
        cand = IncidentCandidate(
            incident_id="INC-TEST-1",
            video_id="v-1",
            event_type="POTENTIAL_THEFT",
            category="property",
            start_time=5.0,
            end_time=9.0,
            duration=4.0,
            severity="HIGH",
            confidence=0.91,
            track_ids=["TRACK-PERSON", "TRACK-BAG"],
            object_classes=["person", "backpack"],
            supporting_signals=[
                SupportingSignal(signal_type="Interaction", description="Person lingered near bag", confidence=0.9),
            ],
            spatial_context=SpatialContext(bounding_box=BoundingBox(10, 10, 50, 50)),
            explanation="Suspected takeaway",
        )
        sec_ev = cand.to_security_event()
        assert isinstance(sec_ev, SecurityEvent)
        assert sec_ev.event_id == "INC-TEST-1"
        assert sec_ev.event_type == "POTENTIAL_THEFT"
        assert sec_ev.severity == "HIGH"
        assert sec_ev.track_id == "TRACK-PERSON"
        assert sec_ev.object_class == "backpack"
        assert sec_ev.detector_name == "universal_engine"
        assert sec_ev.human_verification_required


# ===========================================================================
# 8. Edge Case Tests
# ===========================================================================

class TestEdgeCases:

    def test_empty_tracks(self):
        engine = IncidentIntelligenceEngine()
        res = engine.analyze_incidents(
            video_id="v-empty",
            tracks=[],
            validated_detections=[],
        )
        assert res["incidents"] == []
        assert res["security_events"] == []
        assert res["diagnostics"]["failed_detectors"] == 0

    def test_single_frame_tracks(self):
        engine = IncidentIntelligenceEngine()
        single_track = TrackedObject(
            track_id="SINGLE",
            object_class="person",
            first_seen=1.0,
            last_seen=1.0,
            confidence=0.8,
            current_bbox=BoundingBox(10, 10, 20, 20),
            trajectory=[(1.0, 15.0, 15.0)],
        )
        res = engine.analyze_incidents(
            video_id="v-single",
            tracks=[single_track],
            validated_detections=[{"timestamp": 1.0, "object_class": "person", "confidence": 0.8}],
        )
        assert isinstance(res["incidents"], list)
        assert res["diagnostics"]["failed_detectors"] == 0

    def test_invalid_and_zero_bboxes(self):
        spatial = SpatialRelationshipEngine()
        b_zero = BoundingBox(10, 10, 10, 10)
        b_normal = BoundingBox(0, 0, 50, 50)
        assert spatial.iou(b_zero, b_normal) == 0.0
