"""
Phase 8: Advanced Security Intelligence & Computer Vision Test Suite
"""
import pytest
from ai.schemas import BoundingBox, TrackedObject
from ai.tracking.tracker import ObjectTracker


class TestObjectTracker:
    """8.1 Object Tracking unit tests."""

    def test_track_creation(self):
        tracker = ObjectTracker()
        dets = [
            {"object_class": "person", "confidence": 0.85, "bounding_box": {"x1": 100, "y1": 100, "x2": 150, "y2": 250}}
        ]
        active = tracker.update(timestamp=1.0, detections=dets)
        assert len(active) == 1
        assert active[0].track_id == "TRACK-001"
        assert active[0].object_class == "person"
        assert active[0].first_seen == 1.0
        assert active[0].last_seen == 1.0
        assert active[0].duration_seconds == 0.0

    def test_track_continuation_and_matching(self):
        tracker = ObjectTracker(iou_threshold=0.3)
        # Frame 1 at 1.0s
        tracker.update(timestamp=1.0, detections=[
            {"object_class": "car", "confidence": 0.9, "bounding_box": {"x1": 200, "y1": 200, "x2": 300, "y2": 280}}
        ])
        # Frame 2 at 2.0s (slight shift)
        active = tracker.update(timestamp=2.0, detections=[
            {"object_class": "car", "confidence": 0.92, "bounding_box": {"x1": 205, "y1": 202, "x2": 308, "y2": 282}}
        ])
        assert len(active) == 1
        assert active[0].track_id == "TRACK-001"
        assert active[0].first_seen == 1.0
        assert active[0].last_seen == 2.0
        assert active[0].duration_seconds == 1.0
        assert len(active[0].trajectory) == 2

    def test_multiple_simultaneous_objects(self):
        tracker = ObjectTracker()
        dets = [
            {"object_class": "person", "confidence": 0.8, "bounding_box": {"x1": 50, "y1": 50, "x2": 80, "y2": 120}},
            {"object_class": "car", "confidence": 0.88, "bounding_box": {"x1": 300, "y1": 300, "x2": 450, "y2": 400}},
            {"object_class": "bus", "confidence": 0.75, "bounding_box": {"x1": 600, "y1": 100, "x2": 800, "y2": 350}},
        ]
        active = tracker.update(timestamp=0.5, detections=dets)
        assert len(active) == 3
        ids = {t.track_id for t in active}
        assert ids == {"TRACK-001", "TRACK-002", "TRACK-003"}

    def test_track_termination_after_missing_frames(self):
        tracker = ObjectTracker(max_missing_seconds=2.0)
        # Active at 1.0s
        tracker.update(timestamp=1.0, detections=[
            {"object_class": "person", "confidence": 0.85, "bounding_box": {"x1": 100, "y1": 100, "x2": 150, "y2": 250}}
        ])
        # Empty frame at 2.0s (within 2.0s coasting window)
        active_2 = tracker.update(timestamp=2.0, detections=[])
        assert len(active_2) == 1
        assert active_2[0].active is True

        # Frame at 3.5s (gap > 2.0s from 1.0s) -> should be deactivated
        active_3 = tracker.update(timestamp=3.5, detections=[])
        assert len(active_3) == 0
        all_tracks = tracker.get_tracks()
        assert len(all_tracks) == 1
        assert all_tracks[0].active is False


class TestVehicleColorAnalyzer:
    """8.2 Vehicle Color Analysis unit tests."""

    def test_red_vehicle_classification(self):
        import numpy as np
        from ai.attributes.color_analyzer import VehicleColorAnalyzer

        analyzer = VehicleColorAnalyzer()
        # Create a synthetic BGR frame with a solid red rectangle (B=0, G=0, R=220)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[20:80, 20:80] = [0, 0, 220]  # BGR Red
        bbox = BoundingBox(x1=20, y1=20, x2=80, y2=80)

        attr = analyzer.analyze(frame, bbox, timestamp=3.5, object_class="car", track_id="TRACK-001")
        assert attr.color == "red"
        assert attr.confidence >= 0.7
        assert attr.track_id == "TRACK-001"

    def test_blue_vehicle_classification(self):
        import numpy as np
        from ai.attributes.color_analyzer import VehicleColorAnalyzer

        analyzer = VehicleColorAnalyzer()
        # Synthetic BGR Blue (B=230, G=20, R=10)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[20:80, 20:80] = [230, 20, 10]
        bbox = BoundingBox(x1=20, y1=20, x2=80, y2=80)

        attr = analyzer.analyze(frame, bbox, timestamp=4.0, object_class="car")
        assert attr.color == "blue"
        assert attr.confidence >= 0.7

    def test_white_vehicle_classification(self):
        import numpy as np
        from ai.attributes.color_analyzer import VehicleColorAnalyzer

        analyzer = VehicleColorAnalyzer()
        # Synthetic BGR White (B=245, G=245, R=245)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[20:80, 20:80] = [245, 245, 245]
        bbox = BoundingBox(x1=20, y1=20, x2=80, y2=80)

        attr = analyzer.analyze(frame, bbox, timestamp=4.5)
        assert attr.color == "white"

    def test_black_vehicle_classification(self):
        import numpy as np
        from ai.attributes.color_analyzer import VehicleColorAnalyzer

        analyzer = VehicleColorAnalyzer()
        # Synthetic BGR Black (B=15, G=15, R=15)
        frame = np.full((100, 100, 3), 200, dtype=np.uint8)  # White background
        frame[20:80, 20:80] = [15, 15, 15]  # Black box
        bbox = BoundingBox(x1=20, y1=20, x2=80, y2=80)

        attr = analyzer.analyze(frame, bbox, timestamp=5.0)
        assert attr.color == "black"

    def test_tiny_crop_falls_back_to_unknown(self):
        import numpy as np
        from ai.attributes.color_analyzer import VehicleColorAnalyzer

        analyzer = VehicleColorAnalyzer()
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        bbox = BoundingBox(x1=2, y1=2, x2=6, y2=6)  # 4x4 pixels < 12x12

        attr = analyzer.analyze(frame, bbox, timestamp=5.5)
        assert attr.color == "unknown"
        assert attr.confidence == 0.0


class TestFaceDetectorSafety:
    """8.3 Face Detection ONLY and Ethical Safety Invariant tests."""

    def test_face_detector_initialization(self):
        from ai.faces.face_detector import FaceDetector

        detector = FaceDetector()
        assert detector is not None
        assert hasattr(detector, "detect_in_person_crop")

    def test_face_detected_in_person_crop_with_skin_tone(self):
        from ai.faces.face_detector import FaceDetector
        import numpy as np

        detector = FaceDetector()
        # Synthetic frame with person: (50, 50) to (150, 300)
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        # Add skin-colored pixels in head region (e.g. y 55-90, x 70-130)
        frame[55:90, 70:130] = [140, 170, 220]  # BGR skin tone

        person_bbox = BoundingBox(x1=50, y1=50, x2=150, y2=300)
        faces = detector.detect_in_person_crop(frame, person_bbox, timestamp=2.4, track_id="TRACK-001")
        assert len(faces) == 1
        assert faces[0].track_id == "TRACK-001"
        assert faces[0].confidence >= 0.5
        assert faces[0].bounding_box.y1 >= 50
        assert faces[0].bounding_box.y2 <= 150

    def test_safety_invariants_no_identity_or_biometrics(self):
        from ai.faces.face_detector import FaceDetector
        import numpy as np

        detector = FaceDetector()
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        frame[55:90, 70:130] = [140, 170, 220]
        person_bbox = BoundingBox(x1=50, y1=50, x2=150, y2=300)

        faces = detector.detect_in_person_crop(frame, person_bbox, timestamp=1.2, track_id="TRACK-001")

        # Verify that schema NEVER contains identity fields
        for f in faces:
            d = f.to_dict()
            assert "name" not in d
            assert "identity" not in d
            assert "person_id" not in d
            assert "embedding" not in d
            assert "criminal" not in d
            assert "face_id" not in d  # Visual bounding box and confidence only

    def test_empty_image_returns_no_faces(self):
        from ai.faces.face_detector import FaceDetector
        import numpy as np

        detector = FaceDetector()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = detector.detect_full_frame(frame, timestamp=0.5)
        assert isinstance(faces, list)


class TestZoneManager:
    """8.4 Restricted-Zone / Intrusion Detection tests."""

    def test_point_in_polygon_geometry(self):
        from ai.zones.zone_manager import ZoneManager

        # Square polygon: (10, 10) to (50, 50)
        poly = [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)]
        assert ZoneManager.is_point_in_polygon((30.0, 30.0), poly) is True
        assert ZoneManager.is_point_in_polygon((5.0, 5.0), poly) is False
        assert ZoneManager.is_point_in_polygon((10.0, 30.0), poly) is True  # on boundary

    def test_zone_management(self):
        from ai.zones.zone_manager import ZoneManager

        zm = ZoneManager()
        z = zm.add_zone(name="Restricted Gate", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)])
        assert z.name == "Restricted Gate"
        assert len(zm.list_zones()) == 1

        removed = zm.remove_zone(z.zone_id)
        assert removed is True
        assert len(zm.list_zones()) == 0

    def test_evaluate_track_intrusion(self):
        from ai.zones.zone_manager import ZoneManager

        zm = ZoneManager()
        zm.add_zone(
            name="Secure Perimeter",
            polygon=[(100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)],
            target_classes=["person"],
        )

        # Track 1: Person inside zone (centroid at 200, 200)
        t_inside = TrackedObject(
            track_id="TRACK-001",
            object_class="person",
            first_seen=1.0,
            last_seen=3.0,
            confidence=0.85,
            current_bbox=BoundingBox(x1=180, y1=150, x2=220, y2=250),
            active=True,
        )
        # Track 2: Car inside zone (should NOT trigger because zone targets 'person' only)
        t_car = TrackedObject(
            track_id="TRACK-002",
            object_class="car",
            first_seen=2.0,
            last_seen=3.0,
            confidence=0.90,
            current_bbox=BoundingBox(x1=150, y1=150, x2=250, y2=250),
            active=True,
        )
        # Track 3: Person outside zone
        t_outside = TrackedObject(
            track_id="TRACK-003",
            object_class="person",
            first_seen=1.0,
            last_seen=3.0,
            confidence=0.80,
            current_bbox=BoundingBox(x1=10, y1=10, x2=40, y2=80),
            active=True,
        )

        events = zm.evaluate_track_intrusions(timestamp=3.0, tracks=[t_inside, t_car, t_outside])
        assert len(events) == 1
        ev = events[0]
        assert ev.event_type == "POTENTIAL_INTRUSION"
        assert ev.track_id == "TRACK-001"
        assert ev.zone_name == "Secure Perimeter"
        assert "criminal" not in ev.description.lower()
        assert "potential restricted-zone intrusion" in ev.description.lower()


class TestActivityAnalyzer:
    """8.7 Crowd & Activity Intelligence unit tests."""

    def test_density_timeline_bucketing(self):
        from ai.activity.activity_analyzer import ActivityAnalyzer

        analyzer = ActivityAnalyzer(window_seconds=2.0)
        events = [
            {"timestamp": 0.5, "object_class": "person"},
            {"timestamp": 1.2, "object_class": "car"},
            {"timestamp": 2.5, "object_class": "person"},
            {"timestamp": 3.1, "object_class": "person"},
        ]
        buckets = analyzer.analyze_timeline_density(events, video_duration=4.0)
        assert len(buckets) == 3
        # First window [0.0 - 2.0s] has 2 detections (1 person, 1 car)
        assert buckets[0]["total_detections"] == 2
        assert buckets[0]["person_count"] == 1
        assert buckets[0]["vehicle_count"] == 1
        # Second window [2.0 - 4.0s] has 2 detections (2 persons)
        assert buckets[1]["total_detections"] == 2
        assert buckets[1]["person_count"] == 2

    def test_high_activity_peak_detection(self):
        from ai.activity.activity_analyzer import ActivityAnalyzer

        analyzer = ActivityAnalyzer(window_seconds=2.0, high_activity_threshold=5, person_density_threshold=3)
        # Create 6 detections within [1.0s, 1.8s]
        events = [
            {"timestamp": 1.0 + i * 0.1, "object_class": "person" if i < 4 else "car"}
            for i in range(6)
        ]
        peaks = analyzer.detect_activity_peaks(events, video_duration=4.0)
        assert len(peaks) >= 1
        peak = peaks[0]
        assert peak.event_type == "HIGH_ACTIVITY_PERIOD"
        assert peak.timestamp == 0.0 or peak.timestamp == 2.0
        assert "elevated activity" in peak.description.lower()


class TestBehaviorAnalyzer:
    """8.5, 8.6, 8.8 Loitering, Abandoned Objects, and Anomaly Intelligence unit tests."""

    def test_prolonged_presence_detection(self):
        from ai.behavior.analyzer import BehaviorAnalyzer

        analyzer = BehaviorAnalyzer(loitering_threshold_seconds=3.0, max_loitering_displacement=50.0)

        # Track 1: Loitering person (stays in same area for 4.0s, displacement 10px)
        t_loiter = TrackedObject(
            track_id="TRACK-001",
            object_class="person",
            first_seen=1.0,
            last_seen=5.0,
            confidence=0.85,
            current_bbox=BoundingBox(x1=100, y1=100, x2=140, y2=200),
            trajectory=[(1.0, 120.0, 150.0), (3.0, 125.0, 152.0), (5.0, 122.0, 151.0)],
            active=False,
        )

        # Track 2: Rapidly moving person (displacement 300px in 4.0s)
        t_walking = TrackedObject(
            track_id="TRACK-002",
            object_class="person",
            first_seen=1.0,
            last_seen=5.0,
            confidence=0.88,
            current_bbox=BoundingBox(x1=400, y1=100, x2=440, y2=200),
            trajectory=[(1.0, 100.0, 150.0), (5.0, 400.0, 150.0)],
            active=False,
        )

        # Track 3: Brief presence (1.0s < 3.0s threshold)
        t_brief = TrackedObject(
            track_id="TRACK-003",
            object_class="person",
            first_seen=1.0,
            last_seen=2.0,
            confidence=0.80,
            current_bbox=BoundingBox(x1=100, y1=100, x2=140, y2=200),
            trajectory=[(1.0, 120.0, 150.0), (2.0, 120.0, 150.0)],
            active=False,
        )

        events = analyzer.detect_prolonged_presence([t_loiter, t_walking, t_brief])
        assert len(events) == 1
        assert events[0].event_type == "PROLONGED_PRESENCE"
        assert events[0].track_id == "TRACK-001"
        assert "criminal" not in events[0].description.lower()
        assert "prolonged presence detected" in events[0].description.lower()

    def test_abandoned_object_detection(self):
        from ai.behavior.analyzer import BehaviorAnalyzer

        analyzer = BehaviorAnalyzer(abandoned_stationary_seconds=3.0, abandoned_separation_distance=80.0)

        # Stationary backpack for 4.0s
        t_bag = TrackedObject(
            track_id="TRACK-BAG",
            object_class="backpack",
            first_seen=1.0,
            last_seen=5.0,
            confidence=0.80,
            current_bbox=BoundingBox(x1=200, y1=200, x2=240, y2=250),
            trajectory=[(1.0, 220.0, 225.0), (3.0, 220.0, 225.0), (5.0, 220.0, 225.0)],
            active=False,
        )

        # Person who was nearby at 1.0s (x=240, y=225, dist=20px), but walked away to x=500 at 5.0s
        t_person = TrackedObject(
            track_id="TRACK-PERSON",
            object_class="person",
            first_seen=1.0,
            last_seen=5.0,
            confidence=0.90,
            current_bbox=BoundingBox(x1=480, y1=150, x2=520, y2=250),
            trajectory=[(1.0, 240.0, 225.0), (3.0, 350.0, 225.0), (5.0, 500.0, 225.0)],
            active=False,
        )

        events = analyzer.detect_abandoned_objects([t_bag, t_person])
        assert len(events) == 1
        assert events[0].event_type == "POTENTIAL_ABANDONED_OBJECT"
        assert events[0].track_id == "TRACK-BAG"
        assert "potential abandoned object" in events[0].description.lower()

    def test_observational_anomalies_aggregation(self):
        from ai.behavior.analyzer import BehaviorAnalyzer
        from ai.schemas import SecurityEvent

        analyzer = BehaviorAnalyzer()

        t_person = TrackedObject(
            track_id="TRACK-001",
            object_class="person",
            first_seen=1.0,
            last_seen=8.0,
            confidence=0.85,
            current_bbox=BoundingBox(x1=100, y1=100, x2=140, y2=200),
            trajectory=[(1.0, 120.0, 150.0), (8.0, 122.0, 151.0)],
            active=False,
        )

        ev_intrusion = SecurityEvent(
            event_id="EV-1",
            event_type="POTENTIAL_INTRUSION",
            severity="HIGH",
            timestamp=2.0,
            duration_seconds=1.0,
            confidence=0.85,
            description="Potential intrusion into Secure Perimeter",
            track_id="TRACK-001",
            object_class="person",
            zone_name="Secure Perimeter",
        )

        ev_loitering = SecurityEvent(
            event_id="EV-2",
            event_type="PROLONGED_PRESENCE",
            severity="NORMAL",
            timestamp=1.0,
            duration_seconds=7.0,
            confidence=0.88,
            description="Prolonged presence detected for 7.0s",
            track_id="TRACK-001",
            object_class="person",
        )

        anomalies = analyzer.detect_observational_anomalies([t_person], [ev_intrusion, ev_loitering])
        assert len(anomalies) == 1
        anom = anomalies[0]
        assert anom.event_type == "OBSERVATIONAL_ANOMALY"
        assert anom.track_id == "TRACK-001"
        assert anom.zone_name == "Secure Perimeter"
        assert len(anom.observable_signals) >= 2
        assert "criminal" not in anom.description.lower()


class TestSecurityIntelligencePipeline:
    """End-to-End Security Intelligence Pipeline tests."""

    def test_pipeline_execution_without_video_file(self):
        from ai.intelligence_pipeline import SecurityIntelligencePipeline
        from ai.schemas import ZoneDefinition

        zone = ZoneDefinition(
            zone_id="Z-1",
            name="Main Entrance",
            polygon=[(50.0, 50.0), (150.0, 50.0), (150.0, 200.0), (50.0, 200.0)],
            target_classes=["person"],
        )

        pipeline = SecurityIntelligencePipeline(zones=[zone])

        raw_events = [
            {"event_id": "e1", "timestamp": 1.0, "object_class": "person", "confidence": 0.85, "bounding_box": {"x1": 80, "y1": 80, "x2": 120, "y2": 180}},
            {"event_id": "e2", "timestamp": 2.0, "object_class": "person", "confidence": 0.88, "bounding_box": {"x1": 82, "y1": 82, "x2": 122, "y2": 182}},
            {"event_id": "e3", "timestamp": 3.0, "object_class": "person", "confidence": 0.90, "bounding_box": {"x1": 85, "y1": 84, "x2": 125, "y2": 184}},
            {"event_id": "e4", "timestamp": 4.0, "object_class": "person", "confidence": 0.89, "bounding_box": {"x1": 85, "y1": 85, "x2": 125, "y2": 185}},
            {"event_id": "e5", "timestamp": 5.0, "object_class": "person", "confidence": 0.91, "bounding_box": {"x1": 86, "y1": 85, "x2": 126, "y2": 185}},
        ]

        result = pipeline.process_video_intelligence(
            video_id="test_vid_123",
            video_path="non_existent_file.mp4",
            raw_events=raw_events,
            fps=30.0,
            duration_seconds=5.0,
        )

        assert result["video_id"] == "test_vid_123"
        assert len(result["tracks"]) == 1
        assert result["tracks"][0].track_id == "TRACK-001"
        assert result["tracks"][0].duration_seconds == 4.0

        # Security events should include:
        # 1. Intrusions into Main Entrance
        # 2. Prolonged presence (4.0s >= 4.0s default threshold)
        ev_types = [e.event_type for e in result["security_events"]]
        assert "POTENTIAL_INTRUSION" in ev_types
        assert "PROLONGED_PRESENCE" in ev_types


class TestSecurityIntelligenceRepository:
    """Repository persistence & querying tests."""

    def test_repository_save_and_query(self):
        from ai.intelligence_repository import get_intelligence_repository
        from ai.schemas import TrackedObject, VehicleAttribute, FaceDetection, SecurityEvent, BoundingBox

        repo = get_intelligence_repository()
        vid = "test_repo_vid_456"

        track = TrackedObject(
            track_id="TRACK-001",
            object_class="car",
            first_seen=1.0,
            last_seen=4.0,
            confidence=0.92,
            current_bbox=BoundingBox(10, 10, 50, 50),
            trajectory=[(1.0, 30.0, 30.0), (4.0, 30.0, 30.0)],
            color="red",
            color_confidence=0.88,
        )

        attr = VehicleAttribute(
            color="red",
            confidence=0.88,
            timestamp=2.0,
            bounding_box=BoundingBox(10, 10, 50, 50),
            track_id="TRACK-001",
        )

        face = FaceDetection(
            timestamp=2.5,
            bounding_box=BoundingBox(100, 50, 140, 90),
            confidence=0.85,
            track_id="TRACK-002",
        )

        sec_event = SecurityEvent(
            event_id="EV-TEST-1",
            event_type="POTENTIAL_INTRUSION",
            severity="HIGH",
            timestamp=2.0,
            duration_seconds=1.0,
            confidence=0.85,
            description="Potential intrusion into Secure Zone",
            track_id="TRACK-001",
            zone_name="Secure Zone",
        )

        repo.save_intelligence_results(
            video_id=vid,
            tracks=[track],
            vehicle_attributes=[attr],
            face_detections=[face],
            security_events=[sec_event],
        )

        # Query back
        tracks = repo.get_tracks(vid)
        assert len(tracks) == 1
        assert tracks[0]["track_id"] == "TRACK-001"
        assert tracks[0]["color"] == "red"

        attrs = repo.get_vehicle_attributes(vid, color="red")
        assert len(attrs) == 1
        assert attrs[0]["color"] == "red"

        faces = repo.get_face_detections(vid)
        assert len(faces) == 1
        assert faces[0]["confidence"] == 0.85

        events = repo.get_security_events(vid, event_type="POTENTIAL_INTRUSION")
        assert len(events) == 1
        assert events[0]["zone_name"] == "Secure Zone"

    def test_zone_crud(self):
        from ai.intelligence_repository import get_intelligence_repository

        repo = get_intelligence_repository()
        z = repo.save_zone(name="Loading Dock", polygon=[[0, 0], [10, 0], [10, 10], [0, 10]], target_classes=["truck"])
        zid = z["zone_id"]
        assert z["name"] == "Loading Dock"

        zones = repo.get_zones()
        assert any(x["zone_id"] == zid for x in zones)

        deleted = repo.delete_zone(zid)
        assert deleted is True


class TestPhase8Investigation:
    """Tests for Phase 8 Natural-Language Investigation integration."""

    @pytest.fixture
    def seeded_intel_video(self):
        import uuid
        from database.session import SessionLocal
        from database.models import VideoModel
        from ai.intelligence_repository import SecurityIntelligenceRepository
        from ai.schemas import (
            TrackedObject,
            VehicleAttribute,
            FaceDetection,
            SecurityEvent,
            BoundingBox,
        )

        vid = f"test_intel_{uuid.uuid4().hex[:8]}"
        db = SessionLocal()
        try:
            v = VideoModel(
                id=vid,
                original_filename="cctv_test.mp4",
                storage_path=f"storage/uploads/{vid}.mp4",
                duration_seconds=30.0,
                fps=30.0,
                status="processed",
            )
            db.add(v)
            db.commit()
        finally:
            db.close()

        # Seed intelligence records
        repo = SecurityIntelligenceRepository()
        t1 = TrackedObject(
            track_id="TRACK-001",
            object_class="person",
            first_seen=2.0,
            last_seen=12.0,
            confidence=0.88,
            current_bbox=BoundingBox(10, 10, 50, 150),
        )
        t4 = TrackedObject(
            track_id="TRACK-004",
            object_class="car",
            first_seen=5.0,
            last_seen=20.0,
            confidence=0.92,
            current_bbox=BoundingBox(100, 100, 300, 250),
            color="blue",
            color_confidence=0.89,
        )
        v_attr = VehicleAttribute(
            track_id="TRACK-004",
            object_class="car",
            color="blue",
            confidence=0.89,
            timestamp=6.0,
            bounding_box=BoundingBox(100, 100, 300, 250),
        )
        face = FaceDetection(
            track_id="TRACK-001",
            timestamp=4.5,
            confidence=0.78,
            bounding_box=BoundingBox(15, 12, 45, 45),
        )
        sec_ev1 = SecurityEvent(
            event_id="sec-001",
            event_type="POTENTIAL_INTRUSION",
            timestamp=7.0,
            duration_seconds=2.0,
            track_id="TRACK-001",
            severity="HIGH",
            confidence=0.85,
            description="Potential restricted-zone intrusion detected.",
            observable_signals=["zone: Restricted Gate", "class: person"],
        )
        sec_ev2 = SecurityEvent(
            event_id="sec-002",
            event_type="PROLONGED_PRESENCE",
            timestamp=15.0,
            duration_seconds=10.0,
            track_id="TRACK-001",
            severity="MEDIUM",
            confidence=0.82,
            description="Prolonged presence detected for 10.0s.",
            observable_signals=["duration: 10.0"],
        )
        sec_ev3 = SecurityEvent(
            event_id="sec-003",
            event_type="HIGH_ACTIVITY_PERIOD",
            timestamp=18.0,
            duration_seconds=5.0,
            severity="MEDIUM",
            confidence=0.75,
            description="Elevated activity detected: 8 objects.",
            observable_signals=["density: 8"],
        )
        sec_ev4 = SecurityEvent(
            event_id="sec-004",
            event_type="POTENTIAL_ABANDONED_OBJECT",
            timestamp=22.0,
            duration_seconds=15.0,
            severity="HIGH",
            confidence=0.80,
            description="Potential abandoned object detected.",
            observable_signals=["stationary_seconds: 15.0"],
        )

        repo.save_intelligence_results(
            video_id=vid,
            tracks=[t1, t4],
            vehicle_attributes=[v_attr],
            face_detections=[face],
            security_events=[sec_ev1, sec_ev2, sec_ev3, sec_ev4],
        )

        return vid

    def test_query_blue_vehicles(self, seeded_intel_video):
        from backend.app.services.investigation_service import InvestigationService
        svc = InvestigationService()
        res = svc.investigate(seeded_intel_video, "Show blue vehicles")
        assert res["is_supported"] is True
        assert res["result_type"] == "vehicle_attributes"
        assert res["count"] == 1
        assert res["results"][0]["color"] == "blue"
        assert res["results"][0]["track_id"] == "TRACK-004"

    def test_query_tracked_people(self, seeded_intel_video):
        from backend.app.services.investigation_service import InvestigationService
        svc = InvestigationService()
        res = svc.investigate(seeded_intel_video, "Show all tracked people")
        assert res["is_supported"] is True
        assert res["result_type"] == "tracks"
        assert res["count"] == 1
        assert res["results"][0]["track_id"] == "TRACK-001"
        assert res["results"][0]["object_class"] == "person"

    def test_query_specific_track(self, seeded_intel_video):
        from backend.app.services.investigation_service import InvestigationService
        svc = InvestigationService()
        res = svc.investigate(seeded_intel_video, "Show Track 004")
        assert res["is_supported"] is True
        assert res["result_type"] == "tracks"
        assert res["count"] == 1
        assert res["results"][0]["track_id"] == "TRACK-004"
        assert res["results"][0]["color"] == "blue"

    def test_query_face_detections(self, seeded_intel_video):
        from backend.app.services.investigation_service import InvestigationService
        svc = InvestigationService()
        res = svc.investigate(seeded_intel_video, "Show face detections")
        assert res["is_supported"] is True
        assert res["result_type"] == "faces"
        assert res["count"] == 1
        assert "zero biometric" in res["message"].lower() or "observational" in res["message"].lower()

    def test_query_security_events(self, seeded_intel_video):
        from backend.app.services.investigation_service import InvestigationService
        svc = InvestigationService()

        # Potential intrusions
        res_int = svc.investigate(seeded_intel_video, "Show potential intrusions")
        assert res_int["is_supported"] is True
        assert res_int["result_type"] == "security_events"
        assert res_int["count"] == 1
        assert res_int["results"][0]["event_type"] == "POTENTIAL_INTRUSION"

        # Prolonged presence
        res_loit = svc.investigate(seeded_intel_video, "Show prolonged presence events")
        assert res_loit["is_supported"] is True
        assert res_loit["count"] == 1
        assert res_loit["results"][0]["event_type"] == "PROLONGED_PRESENCE"

        # Activity peaks
        res_act = svc.investigate(seeded_intel_video, "Show activity peaks")
        assert res_act["is_supported"] is True
        assert res_act["count"] == 1
        assert res_act["results"][0]["event_type"] == "HIGH_ACTIVITY_PERIOD"

        # Abandoned objects
        res_ab = svc.investigate(seeded_intel_video, "Show abandoned object events")
        assert res_ab["is_supported"] is True
        assert res_ab["count"] == 1
        assert res_ab["results"][0]["event_type"] == "POTENTIAL_ABANDONED_OBJECT"


class TestPhase8API:
    """Tests for Phase 8 REST endpoints."""

    @pytest.fixture
    def test_client(self):
        from fastapi.testclient import TestClient
        from backend.app.main import app
        return TestClient(app)

    @pytest.fixture
    def mock_video_files(self):
        import json
        import uuid
        from backend.app.core.config import settings
        vid = f"mock_phase8_{uuid.uuid4().hex[:8]}"
        meta_file = settings.STORAGE_UPLOADS_DIR / f"{vid}.json"
        video_file = settings.STORAGE_UPLOADS_DIR / f"{vid}_test.mp4"
        video_file.write_bytes(b"dummy video content")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({
                "video_id": vid,
                "filename": "test.mp4",
                "storage_path": str(video_file),
                "fps": 30.0,
                "duration_seconds": 15.0,
                "status": "processed",
            }, f)
        yield vid
        try:
            if meta_file.exists():
                meta_file.unlink()
            if video_file.exists():
                video_file.unlink()
        except Exception:
            pass

    def test_get_tracks_empty(self, test_client, mock_video_files):
        resp = test_client.get(f"/api/videos/{mock_video_files}/tracks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["video_id"] == mock_video_files
        assert data["total_tracks"] == 0
        assert data["tracks"] == []

    def test_get_attributes_empty(self, test_client, mock_video_files):
        resp = test_client.get(f"/api/videos/{mock_video_files}/attributes")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_attributes"] == 0

    def test_get_faces_safety_notice(self, test_client, mock_video_files):
        resp = test_client.get(f"/api/videos/{mock_video_files}/faces")
        assert resp.status_code == 200
        data = resp.json()
        assert "safety_notice" in data
        assert "biometric" in data["safety_notice"].lower()

    def test_zones_crud_api(self, test_client, mock_video_files):
        # Create zone
        payload = {
            "name": "Front Gate Area",
            "polygon": [[100, 100], [200, 100], [200, 200], [100, 200]],
            "target_classes": ["person"],
            "alert_on_entry": True,
            "loitering_threshold_seconds": 25.0,
        }
        c_resp = test_client.post(f"/api/videos/{mock_video_files}/zones", json=payload)
        assert c_resp.status_code == 200
        c_data = c_resp.json()
        assert c_data["status"] == "created"
        zone_id = c_data["zone"]["zone_id"]
        assert c_data["zone"]["name"] == "Front Gate Area"

        # List zones
        l_resp = test_client.get(f"/api/videos/{mock_video_files}/zones")
        assert l_resp.status_code == 200
        l_data = l_resp.json()
        assert l_data["total_zones"] >= 1
        assert any(z["zone_id"] == zone_id for z in l_data["zones"])

        # Delete zone
        d_resp = test_client.delete(f"/api/videos/{mock_video_files}/zones/{zone_id}")
        assert d_resp.status_code == 200
        assert d_resp.json()["status"] == "deleted"

    def test_invalid_video_id_format(self, test_client):
        resp = test_client.get("/api/videos/invalid@id!/tracks")
        assert resp.status_code == 400

    def test_not_found_video(self, test_client):
        resp = test_client.get("/api/videos/non_existent_vid_12345/tracks")
        assert resp.status_code == 404







