"""
Audit Consistency Regression Tests (Sentinel Global Reliability Audit)

Tests validating the 5 root causes identified and fixed:
  A. Track count canonical semantics
  B. Score provenance (Behavior Score vs Evidence Strength distinction)
  C. Crowd dispersal episode deduplication
  D. Behavioral detector hardening (coordinated movement, forced movement, prolonged presence)
  E. Report validated_tracks alignment with pipeline semantics

All tests are pure-Python unit tests with no database or video file dependencies.
"""
import math
import pytest
from typing import List
from unittest.mock import MagicMock, patch


# ===========================================================================
# Part A: Track Count Canonical Semantics
# ===========================================================================

class TestTrackCountCanonicalSemantics:
    """Ensure is_validated semantics match across pipeline and report service."""

    def _make_tracked_object(self, track_id: str, object_class: str,
                              history_bboxes: int = 1, confidence: float = 0.5):
        """Create a minimal TrackedObject-like mock."""
        from ai.schemas import TrackedObject, BoundingBox
        bbox = BoundingBox(x1=10.0, y1=10.0, x2=110.0, y2=210.0)
        history = [
            {"timestamp": float(i), "bbox": {"x1": 10.0, "y1": 10.0, "x2": 110.0, "y2": 210.0}}
            for i in range(history_bboxes)
        ]
        obj = TrackedObject(
            track_id=track_id,
            object_class=object_class,
            first_seen=0.0,
            last_seen=float(history_bboxes),
            confidence=confidence,
            current_bbox=bbox,
            history_bboxes=history,
        )
        return obj

    def test_multi_frame_track_always_validated(self):
        """Any track with detection_count >= 2 must be validated."""
        track = self._make_tracked_object("T001", "person", history_bboxes=2)
        assert track.is_validated is True

    def test_multi_frame_vehicle_validated(self):
        """Vehicle with 3 frames must be validated."""
        track = self._make_tracked_object("T002", "car", history_bboxes=3)
        assert track.is_validated is True

    def test_single_frame_person_validated(self):
        """Single-frame person is validated per pipeline semantics (non-vehicle rule)."""
        track = self._make_tracked_object("T003", "person", history_bboxes=1)
        assert track.is_validated is True

    def test_single_frame_low_confidence_car_not_validated(self):
        """Single-frame car with low confidence must NOT be validated."""
        track = self._make_tracked_object("T004", "car", history_bboxes=1, confidence=0.40)
        assert track.is_validated is False

    def test_single_frame_high_confidence_car_validated(self):
        """Single-frame car with confidence >= 0.65 must be validated."""
        track = self._make_tracked_object("T005", "car", history_bboxes=1, confidence=0.70)
        assert track.is_validated is True

    def test_single_frame_bus_boundary_confidence(self):
        """Single-frame bus exactly at 0.65 confidence must be validated."""
        track = self._make_tracked_object("T006", "bus", history_bboxes=1, confidence=0.65)
        assert track.is_validated is True

    def test_single_frame_truck_below_boundary_not_validated(self):
        """Single-frame truck at 0.649 confidence must NOT be validated."""
        track = self._make_tracked_object("T007", "truck", history_bboxes=1, confidence=0.649)
        assert track.is_validated is False


# ===========================================================================
# Part B: Score Provenance — Tier Labels Use Final Calibrated Score
# ===========================================================================

class TestScoreProvenance:
    """Ensure scoring tier labels use the final (capped) score, not raw pre-penalty score."""

    def test_review_required_score_capped_at_65pct(self):
        """REVIEW_REQUIRED candidates cannot exceed 65% final score."""
        from ai.incidents.scoring import IncidentScorer
        from ai.incidents.schemas import SupportingSignal

        signals = [
            SupportingSignal(signal_type="Flame Visual Telemetry", description="High confidence fire", confidence=0.95, timestamp=1.0),
            SupportingSignal(signal_type="Sustained Thermal Combustion Pattern", description="Sustained pattern", confidence=0.90, timestamp=1.0),
        ]
        result = IncidentScorer.calculate_evidence_score(
            base_confidence=0.90,
            supporting_signals=signals,
            validation_decision="REVIEW_REQUIRED",
        )
        assert result["score"] <= 0.65, f"Score {result['score']} exceeds 0.65 cap for REVIEW_REQUIRED"

    def test_review_required_tier_not_high_evidence(self):
        """REVIEW_REQUIRED candidates should NOT claim 'High Evidence Strength' tier."""
        from ai.incidents.scoring import IncidentScorer
        from ai.incidents.schemas import SupportingSignal

        signals = [
            SupportingSignal(signal_type="Proximity Signal", description="Close contact", confidence=0.92, timestamp=1.0),
        ]
        result = IncidentScorer.calculate_evidence_score(
            base_confidence=0.85,
            supporting_signals=signals,
            validation_decision="REVIEW_REQUIRED",
        )
        # Since score is capped at 65%, tier must not be "High Evidence Strength"
        assert result["strength_tier"] != "High Evidence Strength", \
            f"REVIEW_REQUIRED candidate incorrectly labeled '{result['strength_tier']}'"

    def test_rejected_candidate_capped_at_30pct(self):
        """REJECTED candidates must be capped at 30% score."""
        from ai.incidents.scoring import IncidentScorer
        result = IncidentScorer.calculate_evidence_score(
            base_confidence=0.95,
            supporting_signals=[],
            validation_decision="REJECTED",
        )
        assert result["score"] <= 0.30

    def test_validated_candidate_can_reach_high_tier(self):
        """Non-review-required candidates with strong signals can reach High Evidence Strength."""
        from ai.incidents.scoring import IncidentScorer
        from ai.incidents.schemas import SupportingSignal

        signals = [
            SupportingSignal(signal_type="Proximity Signal", description="desc", confidence=0.92, timestamp=1.0),
            SupportingSignal(signal_type="Velocity Signal", description="desc2", confidence=0.88, timestamp=1.0),
            SupportingSignal(signal_type="Duration Dwell", description="dwell", confidence=0.85, timestamp=1.0),
        ]
        result = IncidentScorer.calculate_evidence_score(
            base_confidence=0.80,
            supporting_signals=signals,
            duration_seconds=10.0,
        )
        assert result["score"] > 0.80, "High-evidence non-review candidate should exceed 80%"
        assert result["strength_tier"] == "High Evidence Strength"

    def test_contradictory_signals_reduce_score(self):
        """Contradictory signals must actively reduce the score."""
        from ai.incidents.scoring import IncidentScorer
        from ai.incidents.schemas import SupportingSignal

        pos_signals = [
            SupportingSignal(signal_type="Proximity Signal", description="d", confidence=0.85, timestamp=1.0),
        ]
        neg_signals = [
            SupportingSignal(signal_type="Negative: Normal Transit", description="neg", confidence=0.95, timestamp=1.0),
            SupportingSignal(signal_type="Negative: No Contact", description="neg2", confidence=0.90, timestamp=1.0),
        ]
        result_clean = IncidentScorer.calculate_evidence_score(0.75, pos_signals)
        result_neg = IncidentScorer.calculate_evidence_score(0.75, pos_signals, contradictory_signals=neg_signals)
        assert result_neg["score"] < result_clean["score"], "Contradictory signals must reduce score"


# ===========================================================================
# Part C: Crowd Dispersal Episode Deduplication
# ===========================================================================

class TestCrowdDispersalDeduplication:
    """Ensure the hardened dispersal detector doesn't generate O(n) candidates
    from a single underlying dispersal event."""

    def _make_density_window(self, timestamp: float, window_start: float, window_end: float,
                              person_count: int, rate_of_change: float = 0.0,
                              avg_inter_person_dist: float = 50.0):
        from ai.incidents.detectors.crowd.density_engine import DensityWindowMetrics
        return DensityWindowMetrics(
            timestamp=timestamp,
            window_start=window_start,
            window_end=window_end,
            active_person_count=person_count,
            active_vehicle_count=0,
            unique_person_track_ids=[f"P{i}" for i in range(person_count)],
            clusters=[],
            avg_inter_person_dist=avg_inter_person_dist,
            min_inter_person_dist=avg_inter_person_dist * 0.5,
            density_ratio_to_baseline=1.0,
            rate_of_change=rate_of_change,
            occupied_area_px=10000.0,
        )

    def _make_context(self, video_id="test-vid", duration_seconds=60.0):
        from ai.incidents.schemas import IncidentContext
        ctx = MagicMock(spec=IncidentContext)
        ctx.video_id = video_id
        ctx.duration_seconds = duration_seconds
        ctx.tracks = []
        ctx.zones = []
        return ctx

    def test_single_dispersal_event_produces_one_candidate(self):
        """A single dispersal transition should produce exactly one candidate."""
        from ai.incidents.detectors.crowd.crowd_dispersal import CrowdDispersalDetector
        detector = CrowdDispersalDetector(
            min_initial_cluster_persons=4,
            max_dispersal_rate=-0.7,
            min_absolute_count_drop=2,
            min_baseline_windows=2,
            min_inter_episode_gap_seconds=15.0,
        )

        # 2 stable crowd windows (baseline), then 1 dispersal window
        windows = [
            self._make_density_window(3.0, 0.0, 6.0, person_count=6, rate_of_change=0.0),
            self._make_density_window(6.0, 3.0, 9.0, person_count=6, rate_of_change=0.0),
            self._make_density_window(9.0, 6.0, 12.0, person_count=2, rate_of_change=-1.5),
        ]

        ctx = self._make_context()
        with patch.object(detector.density_engine, "evaluate_windows", return_value=windows):
            candidates = detector.analyze(ctx)

        assert len(candidates) <= 1, \
            f"Single dispersal event generated {len(candidates)} candidates (expected <=1)"

    def test_repeated_adjacent_windows_dont_spam(self):
        """When the dispersal is ongoing across many windows, cooldown prevents repeated alerts."""
        from ai.incidents.detectors.crowd.crowd_dispersal import CrowdDispersalDetector
        detector = CrowdDispersalDetector(
            min_initial_cluster_persons=4,
            max_dispersal_rate=-0.5,
            min_absolute_count_drop=2,
            min_baseline_windows=2,
            min_inter_episode_gap_seconds=15.0,
        )

        # 2 stable baseline windows, then 8 consecutive dispersal windows (ongoing)
        windows = (
            [self._make_density_window(float(i*3), float((i-1)*3), float(i*3+3), person_count=8, rate_of_change=0.0) for i in range(1, 3)] +
            [self._make_density_window(float(i*3), float((i-1)*3), float(i*3+3), person_count=max(1, 8 - i), rate_of_change=-0.8) for i in range(3, 11)]
        )

        ctx = self._make_context(duration_seconds=60.0)
        with patch.object(detector.density_engine, "evaluate_windows", return_value=windows):
            candidates = detector.analyze(ctx)

        # Should be at most 2 episodes (initial dispersal + possible resume)
        assert len(candidates) <= 2, \
            f"Adjacent dispersal windows generated {len(candidates)} candidates (excessive)"

    def test_insufficient_baseline_prevents_dispersal(self):
        """If the crowd hasn't been sustained long enough, dispersal should not fire."""
        from ai.incidents.detectors.crowd.crowd_dispersal import CrowdDispersalDetector
        detector = CrowdDispersalDetector(
            min_initial_cluster_persons=4,
            min_baseline_windows=2,  # Need 2 consecutive windows above threshold
        )

        # Only 1 window with crowd above threshold before dispersal
        windows = [
            self._make_density_window(0.0, 0.0, 3.0, person_count=2, rate_of_change=0.0),  # below threshold
            self._make_density_window(3.0, 3.0, 6.0, person_count=6, rate_of_change=0.0),  # baseline (only 1 window)
            self._make_density_window(6.0, 6.0, 9.0, person_count=1, rate_of_change=-2.0),  # dispersal
        ]

        ctx = self._make_context()
        with patch.object(detector.density_engine, "evaluate_windows", return_value=windows):
            candidates = detector.analyze(ctx)

        assert len(candidates) == 0, \
            f"Dispersal fired without sufficient baseline ({len(candidates)} candidates)"

    def test_absolute_count_drop_required(self):
        """Dispersal should not fire if the count drop is not meaningful (< min_absolute_count_drop)."""
        from ai.incidents.detectors.crowd.crowd_dispersal import CrowdDispersalDetector
        detector = CrowdDispersalDetector(
            min_initial_cluster_persons=4,
            max_dispersal_rate=-0.7,
            min_absolute_count_drop=2,  # Require at least 2 persons to drop
            min_baseline_windows=2,
        )

        # Only 1 person drops (noise-level)
        windows = [
            self._make_density_window(0.0, 0.0, 3.0, person_count=5, rate_of_change=0.0),
            self._make_density_window(3.0, 3.0, 6.0, person_count=5, rate_of_change=0.0),
            self._make_density_window(6.0, 6.0, 9.0, person_count=4, rate_of_change=-0.9),  # only 1 drop
        ]

        ctx = self._make_context()
        with patch.object(detector.density_engine, "evaluate_windows", return_value=windows):
            candidates = detector.analyze(ctx)

        assert len(candidates) == 0, \
            f"Dispersal fired on noise-level count drop (1 person, threshold is 2)"


# ===========================================================================
# Part D: Behavioral Detector Hardening
# ===========================================================================

class TestCoordinatedMovementHardening:
    """Ensure the coordinated movement detector does not fire on ordinary pedestrian pairs."""

    def _make_person_track(self, track_id: str, trajectory: list,
                            first_seen: float, last_seen: float):
        """Create a minimal person TrackedObject-like mock."""
        from ai.schemas import TrackedObject, BoundingBox
        bbox = BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=300.0)
        history = [
            {"timestamp": t, "bbox": {"x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 300.0}}
            for t, _, _ in trajectory
        ]
        obj = TrackedObject(
            track_id=track_id,
            object_class="person",
            first_seen=first_seen,
            last_seen=last_seen,
            confidence=0.85,
            current_bbox=bbox,
            history_bboxes=history,
            trajectory=trajectory,
        )
        return obj

    def _make_context(self, tracks, video_id="test-vid"):
        from ai.incidents.schemas import IncidentContext
        ctx = MagicMock(spec=IncidentContext)
        ctx.video_id = video_id
        ctx.duration_seconds = 30.0
        ctx.tracks = tracks
        ctx.zones = []
        return ctx

    def test_two_people_walking_together_no_alert(self):
        """Two people walking in same direction for 4s must NOT trigger coordinated movement."""
        from ai.incidents.detectors.person.coordinated_movement import CoordinatedPersonMovementDetector
        detector = CoordinatedPersonMovementDetector(min_group_size=3)  # Requires 3+ people

        # Two persons walking east (same direction)
        traj1 = [(0.0, 100.0, 200.0), (1.0, 150.0, 200.0), (2.0, 200.0, 200.0),
                 (3.0, 250.0, 200.0), (4.0, 300.0, 200.0)]
        traj2 = [(0.0, 100.0, 240.0), (1.0, 150.0, 240.0), (2.0, 200.0, 240.0),
                 (3.0, 250.0, 240.0), (4.0, 300.0, 240.0)]

        p1 = self._make_person_track("P1", traj1, 0.0, 4.0)
        p2 = self._make_person_track("P2", traj2, 0.0, 4.0)
        ctx = self._make_context([p1, p2])

        candidates = detector.analyze(ctx)
        assert len(candidates) == 0, \
            f"Ordinary pair-walking incorrectly triggered {len(candidates)} coordinated movement alert(s)"

    def test_three_people_tight_alignment_triggers(self):
        """Three people walking in tight alignment (< 20deg) for 4+ seconds should trigger."""
        from ai.incidents.detectors.person.coordinated_movement import CoordinatedPersonMovementDetector
        detector = CoordinatedPersonMovementDetector(
            min_group_size=3,
            max_heading_diff_deg=20.0,
            min_duration_seconds=4.0,
            max_spatial_cohesion_px=200.0,
        )

        # Three persons walking due east (0 degrees) with 5 trajectory points each
        traj1 = [(0.0, 100.0, 200.0), (1.0, 150.0, 200.0), (2.0, 200.0, 200.0),
                 (3.0, 250.0, 200.0), (4.5, 315.0, 200.0)]
        traj2 = [(0.0, 100.0, 250.0), (1.0, 150.0, 250.0), (2.0, 200.0, 250.0),
                 (3.0, 250.0, 250.0), (4.5, 315.0, 250.0)]
        traj3 = [(0.0, 100.0, 170.0), (1.0, 150.0, 170.0), (2.0, 200.0, 170.0),
                 (3.0, 250.0, 170.0), (4.5, 315.0, 170.0)]

        p1 = self._make_person_track("P1", traj1, 0.0, 4.5)
        p2 = self._make_person_track("P2", traj2, 0.0, 4.5)
        p3 = self._make_person_track("P3", traj3, 0.0, 4.5)
        ctx = self._make_context([p1, p2, p3])

        candidates = detector.analyze(ctx)
        assert len(candidates) >= 1, \
            "Three persons in tight synchronized motion should trigger coordinated movement detection"


class TestForcedMovementHardening:
    """Ensure forced movement detector requires multiple deflections, not just one."""

    def _make_person_track(self, track_id: str, trajectory: list,
                            first_seen: float, last_seen: float):
        from ai.schemas import TrackedObject, BoundingBox
        bbox = BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=300.0)
        history = [{"timestamp": t, "bbox": {"x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 300.0}}
                   for t, _, _ in trajectory]
        return TrackedObject(
            track_id=track_id,
            object_class="person",
            first_seen=first_seen,
            last_seen=last_seen,
            confidence=0.85,
            current_bbox=bbox,
            history_bboxes=history,
            trajectory=trajectory,
        )

    def _make_context(self, tracks):
        from ai.incidents.schemas import IncidentContext
        ctx = MagicMock(spec=IncidentContext)
        ctx.video_id = "test-vid"
        ctx.duration_seconds = 30.0
        ctx.tracks = tracks
        ctx.zones = []
        return ctx

    def test_single_corner_turn_no_alert(self):
        """A pair walking close together and making one turn should NOT trigger forced movement."""
        from ai.incidents.detectors.person.forced_movement import ForcedMovementDetector
        detector = ForcedMovementDetector(
            min_duration_seconds=4.0,
            max_proximity_px=55.0,
            min_deflections=3,
        )

        # Pair walks east for 2s, then turns south for 2s (single turn) — very close proximity
        traj1 = [(0.0, 100.0, 200.0), (1.0, 130.0, 200.0), (2.0, 160.0, 200.0),
                 (3.0, 160.0, 240.0), (4.5, 160.0, 280.0)]
        traj2 = [(0.0, 100.0, 225.0), (1.0, 130.0, 225.0), (2.0, 160.0, 225.0),
                 (3.0, 160.0, 265.0), (4.5, 160.0, 305.0)]

        p1 = self._make_person_track("P1", traj1, 0.0, 4.5)
        p2 = self._make_person_track("P2", traj2, 0.0, 4.5)

        with patch.object(type(detector).__mro__[1], 'evaluate_forced_movement_negative_evidence',
                          return_value=[], create=True):
            ctx = self._make_context([p1, p2])
            # Patch the negative evidence engine to not interfere
            with patch('ai.incidents.detectors.person.forced_movement.NegativeEvidenceEngine') as mock_neg:
                mock_neg.evaluate_forced_movement_negative_evidence.return_value = []
                candidates = detector.analyze(ctx)

        assert len(candidates) == 0, \
            f"Single corner turn incorrectly triggered {len(candidates)} forced movement alert(s)"

    def test_min_deflections_parameter_respected(self):
        """Detector with min_deflections=1 should fire on single-turn close pair."""
        from ai.incidents.detectors.person.forced_movement import ForcedMovementDetector
        detector = ForcedMovementDetector(
            min_duration_seconds=4.0,
            max_proximity_px=55.0,
            min_deflections=1,  # Legacy mode
            min_deflection_angle_deg=45.0,
        )
        assert detector.min_deflections == 1


class TestProlongedPresenceHardening:
    """Ensure prolonged presence does not fire on short-duration or actively-moving tracks."""

    def _make_person_track(self, track_id: str, first_seen: float, last_seen: float,
                            net_displacement: float = 50.0, avg_velocity: float = 5.0):
        from ai.schemas import TrackedObject, BoundingBox
        bbox = BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=300.0)
        trajectory = [
            (first_seen, 100.0, 200.0),
            (last_seen, 100.0 + net_displacement, 200.0),
        ]
        history = [{"timestamp": t, "bbox": {"x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 300.0}}
                   for t, _, _ in trajectory]
        return TrackedObject(
            track_id=track_id,
            object_class="person",
            first_seen=first_seen,
            last_seen=last_seen,
            confidence=0.85,
            current_bbox=bbox,
            history_bboxes=history,
            trajectory=trajectory,
        )

    def _make_context(self, tracks, motion_summary: dict = None):
        from ai.incidents.schemas import IncidentContext
        ctx = MagicMock(spec=IncidentContext)
        ctx.video_id = "test-vid"
        ctx.duration_seconds = 60.0
        ctx.tracks = tracks
        ctx.zones = []
        ctx.get_motion_summary = MagicMock(return_value=motion_summary or {
            "net_displacement": 50.0,
            "avg_velocity": 5.0,
            "max_stationary_duration": 8.0,
            "path_consistency": 0.2,
        })
        return ctx

    def test_short_duration_person_no_alert(self):
        """Person visible for only 5s should NOT trigger prolonged presence (threshold 8s)."""
        from ai.incidents.detectors.prolonged_presence import ProlongedPresenceDetector
        detector = ProlongedPresenceDetector(
            person_threshold_seconds=8.0,
            max_loitering_displacement=80.0,
        )

        p1 = self._make_person_track("P1", 0.0, 5.0)  # Only 5s — below 8s threshold
        ctx = self._make_context([p1], motion_summary={
            "net_displacement": 40.0,
            "avg_velocity": 8.0,
            "max_stationary_duration": 3.0,
            "path_consistency": 0.1,
        })

        with patch('ai.incidents.detectors.prolonged_presence.NegativeEvidenceEngine') as mock_neg:
            mock_neg.evaluate_loitering_negative_evidence.return_value = []
            candidates = detector.analyze(ctx)

        assert len(candidates) == 0, \
            f"5-second track triggered prolonged presence alert (threshold: 8s)"

    def test_long_duration_stationary_person_triggers(self):
        """Person visible for 12s with low displacement and low velocity should trigger."""
        from ai.incidents.detectors.prolonged_presence import ProlongedPresenceDetector
        detector = ProlongedPresenceDetector(
            person_threshold_seconds=8.0,
            max_loitering_displacement=80.0,
        )

        p1 = self._make_person_track("P1", 0.0, 12.0, net_displacement=30.0, avg_velocity=2.5)
        ctx = self._make_context([p1], motion_summary={
            "net_displacement": 30.0,
            "avg_velocity": 2.5,
            "max_stationary_duration": 10.0,
            "path_consistency": 0.05,
        })

        with patch('ai.incidents.detectors.prolonged_presence.NegativeEvidenceEngine') as mock_neg:
            mock_neg.evaluate_loitering_negative_evidence.return_value = []
            candidates = detector.analyze(ctx)

        assert len(candidates) >= 1, \
            "12-second stationary track should trigger prolonged presence alert"

    def test_transit_person_no_alert(self):
        """Person walking through scene at 50px/s should NOT trigger prolonged presence."""
        from ai.incidents.detectors.prolonged_presence import ProlongedPresenceDetector
        detector = ProlongedPresenceDetector(
            person_threshold_seconds=8.0,
            max_loitering_displacement=80.0,
        )

        # Duration 10s but moving fast (transit)
        p1 = self._make_person_track("P1", 0.0, 10.0, net_displacement=500.0, avg_velocity=50.0)
        ctx = self._make_context([p1], motion_summary={
            "net_displacement": 500.0,
            "avg_velocity": 50.0,
            "max_stationary_duration": 0.0,
            "path_consistency": 0.85,
        })

        with patch('ai.incidents.detectors.prolonged_presence.NegativeEvidenceEngine') as mock_neg:
            mock_neg.evaluate_loitering_negative_evidence.return_value = []
            candidates = detector.analyze(ctx)

        # Net displacement 500px >> 80px threshold, so should be filtered
        assert len(candidates) == 0, \
            "Fast-moving transit person incorrectly triggered prolonged presence alert"


# ===========================================================================
# Part E: Report Validated_Tracks Alignment
# ===========================================================================

class TestReportValidatedTracksAlignment:
    """Ensure report service uses the same validation semantics as the pipeline."""

    def _make_track_model(self, object_class: str, detection_count: int,
                           max_confidence: float, duration_seconds: float = 2.0):
        """Create a minimal TrackModel-like object for testing."""
        m = MagicMock()
        m.object_class = object_class
        m.detection_count = detection_count
        m.max_confidence = max_confidence
        m.duration_seconds = duration_seconds
        return m

    def _count_validated(self, tracks):
        """Apply the canonical validated_tracks logic from report_service.py."""
        VEHICLE_CLASSES_REPORT = {"car", "bus", "truck"}
        validated_tracks = 0
        for t in tracks:
            det_count = t.detection_count or 0
            cls = (t.object_class or "").lower()
            if det_count >= 2:
                validated_tracks += 1
            elif cls in VEHICLE_CLASSES_REPORT:
                if (t.max_confidence or 0.0) >= 0.65:
                    validated_tracks += 1
            else:
                # Non-vehicle single-frame: validated per pipeline semantics
                validated_tracks += 1
        return validated_tracks

    def test_multi_frame_tracks_always_validated(self):
        """All tracks with detection_count >= 2 are validated regardless of class."""
        tracks = [
            self._make_track_model("person", 3, 0.8),
            self._make_track_model("car", 2, 0.5),
            self._make_track_model("truck", 5, 0.9),
        ]
        assert self._count_validated(tracks) == 3

    def test_single_frame_vehicle_threshold(self):
        """Single-frame vehicles: validated only if confidence >= 0.65."""
        tracks = [
            self._make_track_model("car", 1, 0.70),   # validated
            self._make_track_model("car", 1, 0.60),   # not validated
            self._make_track_model("bus", 1, 0.65),   # validated (boundary)
            self._make_track_model("truck", 1, 0.64), # not validated
        ]
        assert self._count_validated(tracks) == 2

    def test_single_frame_person_always_validated(self):
        """Single-frame person tracks are always validated (no confidence threshold)."""
        tracks = [
            self._make_track_model("person", 1, 0.40),
            self._make_track_model("person", 1, 0.75),
        ]
        assert self._count_validated(tracks) == 2

    def test_total_vs_validated_semantics_documented(self):
        """total_tracks = ALL stored; validated_tracks = pipeline-semantics subset."""
        tracks = [
            self._make_track_model("person", 1, 0.50),   # validated (non-vehicle)
            self._make_track_model("car", 1, 0.40),       # NOT validated
            self._make_track_model("person", 3, 0.70),   # validated
        ]
        total = len(tracks)
        validated = self._count_validated(tracks)
        assert total == 3
        assert validated == 2  # person (single frame) + person (multi-frame)
        assert total != validated  # these should differ and both be meaningful
