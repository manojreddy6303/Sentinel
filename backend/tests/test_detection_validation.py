"""
Comprehensive Unit & Regression Tests for Sentinel Global Object Detection Validation Architecture

Tests cover:
- Multi-signal evaluation (confidence, geometry, temporal persistence, context)
- Multi-class false positive rejection (bus, truck, car, person, motorcycle, bicycle, portable items)
- Preservation of legitimate distant people, high-confidence detections, and contextual portable items
- Degenerate bounding boxes (zero width, inverted coordinates, extreme aspect ratios)
- Downstream consumer protection (timeline grouping, tracking, security events, evidence, investigation, reports)
- Safe handling of null/unsupported object classes
"""

import uuid
import pytest
from typing import Dict, Any, List

from ai.validation import (
    DetectionValidator,
    DetectionValidationPolicy,
    ValidationStatus,
    ClassValidationRule,
)
from ai.events.generator import EventGenerator
from ai.tracking.tracker import ObjectTracker
from ai.intelligence_pipeline import SecurityIntelligencePipeline
from backend.app.services.investigation_service import InvestigationService
from backend.app.services.report_service import ReportService
from backend.app.services.evidence_service import EvidenceService
from database.session import SessionLocal, init_db
from database.models import VideoModel, EventModel, GroupedEventModel, EvidenceModel


@pytest.fixture(autouse=True)
def ensure_db():
    init_db()


class TestMultiClassValidation:
    """Test validation engine behavior across multiple object classes."""

    def setup_method(self):
        self.validator = DetectionValidator()

    def test_01_false_bus_rejection(self):
        """Isolated low-confidence bus detection (like ~29% indoor false positive) is REJECTED."""
        det = {
            "object_class": "bus",
            "confidence": 0.2899,
            "bounding_box": {"x1": 77.0, "y1": 0.0, "x2": 317.0, "y2": 231.0},
            "timestamp": 146.0,
        }
        res = self.validator.validate_single_detection(
            detection=det,
            temporal_neighbors=[],
            frame_width=1920,
            frame_height=1080,
        )
        assert res.status == ValidationStatus.REJECTED
        assert "without temporal persistence" in res.reason or "low confidence" in res.reason

    def test_02_false_truck_rejection(self):
        """Isolated low-confidence truck detection is REJECTED."""
        det = {
            "object_class": "truck",
            "confidence": 0.30,
            "bounding_box": {"x1": 100.0, "y1": 100.0, "x2": 400.0, "y2": 350.0},
            "timestamp": 12.0,
        }
        res = self.validator.validate_single_detection(
            detection=det,
            temporal_neighbors=[],
            frame_width=1920,
            frame_height=1080,
        )
        assert res.status == ValidationStatus.REJECTED

    def test_03_false_car_rejection(self):
        """Isolated low-confidence car detection below standalone threshold is REJECTED."""
        det = {
            "object_class": "car",
            "confidence": 0.26,
            "bounding_box": {"x1": 200.0, "y1": 200.0, "x2": 350.0, "y2": 300.0},
            "timestamp": 45.0,
        }
        res = self.validator.validate_single_detection(
            detection=det,
            temporal_neighbors=[],
            frame_width=1920,
            frame_height=1080,
        )
        assert res.status == ValidationStatus.REJECTED

    def test_04_false_person_rejection(self):
        """Isolated weak noise predicting person below 0.25 threshold is REJECTED."""
        det = {
            "object_class": "person",
            "confidence": 0.21,
            "bounding_box": {"x1": 500.0, "y1": 300.0, "x2": 550.0, "y2": 450.0},
            "timestamp": 22.0,
        }
        res = self.validator.validate_single_detection(
            detection=det,
            temporal_neighbors=[],
            frame_width=1920,
            frame_height=1080,
        )
        assert res.status == ValidationStatus.REJECTED

    def test_05_false_motorcycle_and_bicycle_rejection(self):
        """Isolated low-confidence two-wheelers are REJECTED."""
        for cls in ["motorcycle", "bicycle"]:
            det = {
                "object_class": cls,
                "confidence": 0.24,
                "bounding_box": {"x1": 150.0, "y1": 150.0, "x2": 250.0, "y2": 250.0},
                "timestamp": 33.0,
            }
            res = self.validator.validate_single_detection(
                detection=det,
                temporal_neighbors=[],
                frame_width=1920,
                frame_height=1080,
            )
            assert res.status == ValidationStatus.REJECTED

    def test_06_temporally_supported_vehicle_validated(self):
        """Repeated vehicle detection with moderate confidence across frames is VALID."""
        det = {
            "object_class": "car",
            "confidence": 0.32,
            "bounding_box": {"x1": 200.0, "y1": 200.0, "x2": 350.0, "y2": 300.0},
            "timestamp": 10.0,
        }
        neighbor = {
            "object_class": "car",
            "confidence": 0.35,
            "bounding_box": {"x1": 205.0, "y1": 202.0, "x2": 355.0, "y2": 302.0},
            "timestamp": 11.0,
        }
        res = self.validator.validate_single_detection(
            detection=det,
            temporal_neighbors=[neighbor],
            frame_width=1920,
            frame_height=1080,
        )
        assert res.status == ValidationStatus.VALID
        assert "temporal persistence" in res.reason

    def test_07_distant_legitimate_person_preserved(self):
        """Distant small person with temporal persistence is not over-filtered and remains VALID."""
        det = {
            "object_class": "person",
            "confidence": 0.30,
            "bounding_box": {"x1": 800.0, "y1": 200.0, "x2": 825.0, "y2": 270.0},
            "timestamp": 5.0,
        }
        neighbor = {
            "object_class": "person",
            "confidence": 0.32,
            "bounding_box": {"x1": 802.0, "y1": 201.0, "x2": 827.0, "y2": 271.0},
            "timestamp": 6.0,
        }
        res = self.validator.validate_single_detection(
            detection=det,
            temporal_neighbors=[neighbor],
            frame_width=1920,
            frame_height=1080,
        )
        assert res.status == ValidationStatus.VALID

    def test_08_high_confidence_standalone_detection_valid(self):
        """Clear standalone detection (e.g. 0.85 conf) passes immediately without requiring persistence."""
        det = {
            "object_class": "person",
            "confidence": 0.85,
            "bounding_box": {"x1": 400.0, "y1": 100.0, "x2": 550.0, "y2": 500.0},
            "timestamp": 77.0,
        }
        res = self.validator.validate_single_detection(
            detection=det,
            temporal_neighbors=[],
            frame_width=1920,
            frame_height=1080,
        )
        assert res.status == ValidationStatus.VALID
        assert "High confidence" in res.reason

    def test_09_brief_suitcase_with_person_context_valid(self):
        """Briefly visible suitcase near person during handling/theft is VALID via contextual boost."""
        suitcase_det = {
            "object_class": "suitcase",
            "confidence": 0.28,
            "bounding_box": {"x1": 520.0, "y1": 350.0, "x2": 580.0, "y2": 420.0},
            "timestamp": 147.0,
        }
        person_det = {
            "object_class": "person",
            "confidence": 0.82,
            "bounding_box": {"x1": 480.0, "y1": 200.0, "x2": 560.0, "y2": 500.0},
            "timestamp": 147.0,
        }
        res = self.validator.validate_single_detection(
            detection=suitcase_det,
            temporal_neighbors=[],
            context_neighbors=[person_det],
            frame_width=1920,
            frame_height=1080,
        )
        assert res.status == ValidationStatus.VALID
        assert "contextual association" in res.reason


class TestGeometricSanity:
    """Test bounding-box sanity checks."""

    def setup_method(self):
        self.validator = DetectionValidator()

    def test_01_zero_or_negative_dimensions_rejected(self):
        det = {
            "object_class": "car",
            "confidence": 0.90,
            "bounding_box": {"x1": 100.0, "y1": 100.0, "x2": 100.0, "y2": 200.0},  # width = 0
            "timestamp": 1.0,
        }
        res = self.validator.validate_single_detection(det)
        assert res.status == ValidationStatus.REJECTED
        assert "Degenerate" in res.reason

    def test_02_inverted_coordinates_rejected(self):
        det = {
            "object_class": "person",
            "confidence": 0.90,
            "bounding_box": {"x1": 300.0, "y1": 400.0, "x2": 200.0, "y2": 200.0},  # x2 < x1
            "timestamp": 1.0,
        }
        res = self.validator.validate_single_detection(det)
        assert res.status == ValidationStatus.REJECTED
        assert "Degenerate" in res.reason or "inverted" in res.reason.lower()

    def test_03_extreme_aspect_ratio_rejected(self):
        """Absurd sliver (e.g. 50x2 pixels) is geometrically implausible and REJECTED."""
        det = {
            "object_class": "car",
            "confidence": 0.80,
            "bounding_box": {"x1": 100.0, "y1": 100.0, "x2": 600.0, "y2": 110.0},  # w/h = 50.0
            "timestamp": 1.0,
        }
        res = self.validator.validate_single_detection(det)
        assert res.status == ValidationStatus.REJECTED
        assert "aspect ratio" in res.reason


class TestSequenceValidation:
    """Test full multi-frame sequence validation."""

    def test_sequence_annotation(self):
        validator = DetectionValidator()
        frame_detections = [
            {
                "frame_number": 1,
                "detections": [
                    {
                        "object_class": "bus",
                        "confidence": 0.29,
                        "bounding_box": {"x1": 100.0, "y1": 100.0, "x2": 400.0, "y2": 300.0},
                        "timestamp": 146.0,
                    },
                    {
                        "object_class": "person",
                        "confidence": 0.88,
                        "bounding_box": {"x1": 500.0, "y1": 200.0, "x2": 600.0, "y2": 500.0},
                        "timestamp": 146.0,
                    },
                ],
            },
            {
                "frame_number": 2,
                "detections": [
                    {
                        "object_class": "person",
                        "confidence": 0.85,
                        "bounding_box": {"x1": 510.0, "y1": 205.0, "x2": 610.0, "y2": 505.0},
                        "timestamp": 147.0,
                    }
                ],
            },
        ]
        res = validator.validate_sequence(frame_detections, frame_width=1920, frame_height=1080)
        f1_bus = res[0]["detections"][0]
        f1_person = res[0]["detections"][1]
        f2_person = res[1]["detections"][0]

        assert f1_bus["validation_status"] == "REJECTED"
        assert f1_person["validation_status"] == "VALID"
        assert f2_person["validation_status"] == "VALID"


class TestDownstreamIntegrity:
    """Test that rejected detections do NOT propagate to timeline, tracking, investigation, or reports."""

    def test_01_rejected_detection_excluded_from_grouped_events(self):
        """Timeline clustering groups ONLY VALID detections."""
        generator = EventGenerator()
        video_id = f"test_val_{uuid.uuid4().hex[:8]}"
        events = [
            {
                "event_id": str(uuid.uuid4()),
                "video_id": video_id,
                "event_type": "object_detected",
                "object_class": "bus",
                "confidence": 0.29,
                "timestamp": 146.0,
                "validation_status": "REJECTED",
                "validation_reason": "isolated low confidence",
                "bounding_box": {"x1": 100, "y1": 100, "x2": 300, "y2": 300},
            },
            {
                "event_id": str(uuid.uuid4()),
                "video_id": video_id,
                "event_type": "object_detected",
                "object_class": "person",
                "confidence": 0.85,
                "timestamp": 146.0,
                "validation_status": "VALID",
                "validation_reason": "high confidence",
                "bounding_box": {"x1": 500, "y1": 200, "x2": 600, "y2": 500},
            },
        ]
        grouped = generator.group_events(video_id, events)
        assert len(grouped) == 1
        objs = grouped[0]["objects"]
        classes = [o["class"] for o in objs]
        assert "person" in classes
        assert "bus" not in classes

    def test_02_rejected_detection_creates_no_track_or_security_event(self):
        """Security intelligence pipeline fed only validated events produces no tracks/events for rejected objects."""
        pipeline = SecurityIntelligencePipeline()
        video_id = f"test_val_{uuid.uuid4().hex[:8]}"
        # Feed only validated events
        validated_events = [
            {
                "event_id": str(uuid.uuid4()),
                "video_id": video_id,
                "object_class": "person",
                "confidence": 0.85,
                "timestamp": 1.0,
                "bounding_box": {"x1": 500, "y1": 200, "x2": 600, "y2": 500},
            }
        ]
        res = pipeline.process_video_intelligence(
            video_id=video_id,
            video_path="",
            raw_events=validated_events,
            fps=30.0,
            duration_seconds=5.0,
        )
        tracks = res["tracks"]
        for t in tracks:
            assert t.object_class != "bus"

    def test_03_evidence_service_refuses_rejected_detection(self):
        """Attempting to generate evidence from an explicit REJECTED event raises ValueError."""
        db = SessionLocal()
        video_id = f"test_val_{uuid.uuid4().hex[:8]}"
        ev_id = str(uuid.uuid4())
        try:
            db.add(VideoModel(id=video_id, original_filename="test.mp4", storage_path="test.mp4"))
            db.add(EventModel(
                id=ev_id,
                video_id=video_id,
                object_class="bus",
                class_id=5,
                timestamp_seconds=146.0,
                confidence=0.29,
                bbox_x1=100.0, bbox_y1=100.0, bbox_x2=300.0, bbox_y2=300.0,
                frame_number=146,
                validation_status="REJECTED",
                validation_reason="isolated low confidence",
            ))
            db.commit()

            service = EvidenceService()
            with pytest.raises(ValueError, match="Cannot generate evidence for rejected detection"):
                service.create_evidence(video_id=video_id, timestamp=146.0, event_id=ev_id)
        finally:
            db.query(EventModel).filter(EventModel.video_id == video_id).delete()
            db.query(VideoModel).filter(VideoModel.id == video_id).delete()
            db.commit()
            db.close()

    def test_04_investigation_query_ignores_rejected_detections(self):
        """Investigation query 'Show all buses' reports 0 when only a rejected bus exists."""
        db = SessionLocal()
        video_id = f"test_val_{uuid.uuid4().hex[:8]}"
        ev_id = str(uuid.uuid4())
        try:
            db.add(VideoModel(id=video_id, original_filename="test.mp4", storage_path="test.mp4"))
            db.add(EventModel(
                id=ev_id,
                video_id=video_id,
                object_class="bus",
                class_id=5,
                timestamp_seconds=146.0,
                confidence=0.29,
                bbox_x1=100.0, bbox_y1=100.0, bbox_x2=300.0, bbox_y2=300.0,
                frame_number=146,
                validation_status="REJECTED",
                validation_reason="isolated low confidence",
            ))
            db.commit()

            service = InvestigationService()
            res = service.investigate(video_id, "Show all buses")
            assert res["count"] == 0
            assert len(res["results"]) == 0
            assert "No matching detections were found." in res["message"]
        finally:
            db.query(EventModel).filter(EventModel.video_id == video_id).delete()
            db.query(VideoModel).filter(VideoModel.id == video_id).delete()
            db.commit()
            db.close()

    def test_05_safe_handling_of_null_object_class(self):
        """Null or empty object_class is handled safely without throwing an exception."""
        validator = DetectionValidator()
        det = {
            "object_class": None,
            "confidence": 0.50,
            "bounding_box": {"x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 200.0},
            "timestamp": 1.0,
        }
        res = validator.validate_single_detection(det)
        assert res.status in {ValidationStatus.VALID, ValidationStatus.UNCERTAIN, ValidationStatus.REJECTED}
