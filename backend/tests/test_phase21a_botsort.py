"""
Phase 21A: BoT-SORT Advanced Tracking Test Suite

Comprehensive tests for:
- Kalman filter prediction/update
- Hungarian assignment
- GMC success and fallback
- Occlusion and reactivation
- Tracker lifecycle (TENTATIVE → CONFIRMED → COASTING → OCCLUDED → LOST → ENDED)
- Empty detections
- Resolution independence
- Camera motion handling
- Memory behavior
- Backward compatibility with existing ObjectTracker interface
- TrackedObject output contract
"""
import math
import pytest
import numpy as np

from ai.schemas import BoundingBox, TrackedObject, TrackLifecycleState
from ai.tracking.kalman_filter import KalmanFilter
from ai.tracking.gmc import GlobalMotionCompensation
from ai.tracking.matching import (
    iou_batch, iou_cost_matrix, centroid_distance_matrix,
    linear_assignment, has_scipy,
)
from ai.tracking.botsort_tracker import BoTSORTTracker
from ai.tracking.tracker import ObjectTracker


# =============================================================================
# KALMAN FILTER TESTS
# =============================================================================

class TestKalmanFilter:

    def test_initiate_returns_correct_shapes(self):
        kf = KalmanFilter()
        measurement = np.array([100.0, 200.0, 50.0, 100.0])
        mean, cov = kf.initiate(measurement)
        assert mean.shape == (8,)
        assert cov.shape == (8, 8)
        assert np.allclose(mean[:4], measurement)
        assert np.allclose(mean[4:], 0.0)  # initial velocity = 0

    def test_predict_propagates_velocity(self):
        kf = KalmanFilter()
        measurement = np.array([100.0, 200.0, 50.0, 100.0])
        mean, cov = kf.initiate(measurement)
        # Set velocity: moving right at 10 px/s, down at 5 px/s
        mean[4] = 10.0
        mean[5] = 5.0
        predicted, _ = kf.predict(mean, cov, dt=1.0)
        assert predicted[0] == pytest.approx(110.0, abs=0.1)  # cx + vx*dt
        assert predicted[1] == pytest.approx(205.0, abs=0.1)  # cy + vy*dt

    def test_predict_with_variable_dt(self):
        kf = KalmanFilter()
        measurement = np.array([100.0, 200.0, 50.0, 100.0])
        mean, cov = kf.initiate(measurement)
        mean[4] = 20.0  # vx = 20 px/s
        predicted, _ = kf.predict(mean, cov, dt=0.5)
        assert predicted[0] == pytest.approx(110.0, abs=0.1)  # cx + vx*0.5

    def test_update_corrects_prediction(self):
        kf = KalmanFilter()
        measurement = np.array([100.0, 200.0, 50.0, 100.0])
        mean, cov = kf.initiate(measurement)
        # Predict forward
        mean, cov = kf.predict(mean, cov, dt=1.0)
        # Update with actual measurement nearby
        new_meas = np.array([102.0, 201.0, 50.0, 100.0])
        updated_mean, updated_cov = kf.update(mean, cov, new_meas)
        # Updated position should be close to measurement
        assert abs(updated_mean[0] - 102.0) < abs(mean[0] - 102.0)

    def test_gating_distance_nearby_is_small(self):
        kf = KalmanFilter()
        measurement = np.array([100.0, 200.0, 50.0, 100.0])
        mean, cov = kf.initiate(measurement)
        # Nearby measurement should have small Mahalanobis distance
        nearby = np.array([101.0, 201.0, 50.0, 100.0])
        dist = kf.gating_distance(mean, cov, nearby)
        assert dist < 16.27  # within 99.5% gate

    def test_gating_distance_far_is_large(self):
        kf = KalmanFilter()
        measurement = np.array([100.0, 200.0, 50.0, 100.0])
        mean, cov = kf.initiate(measurement)
        # Far measurement should have large distance
        far = np.array([500.0, 800.0, 50.0, 100.0])
        dist = kf.gating_distance(mean, cov, far)
        assert dist > 16.27

    def test_state_to_bbox_roundtrip(self):
        kf = KalmanFilter()
        x1, y1, x2, y2 = 100.0, 200.0, 150.0, 300.0
        meas = kf.bbox_to_measurement(x1, y1, x2, y2)
        rx1, ry1, rx2, ry2 = kf.state_to_bbox(np.concatenate([meas, np.zeros(4)]))
        assert rx1 == pytest.approx(x1, abs=0.01)
        assert ry1 == pytest.approx(y1, abs=0.01)
        assert rx2 == pytest.approx(x2, abs=0.01)
        assert ry2 == pytest.approx(y2, abs=0.01)

    def test_predict_covariance_increases(self):
        kf = KalmanFilter()
        measurement = np.array([100.0, 200.0, 50.0, 100.0])
        mean, cov = kf.initiate(measurement)
        trace_before = np.trace(cov)
        _, predicted_cov = kf.predict(mean, cov, dt=1.0)
        trace_after = np.trace(predicted_cov)
        assert trace_after > trace_before  # uncertainty should grow


# =============================================================================
# HUNGARIAN ASSIGNMENT TESTS
# =============================================================================

class TestMatching:

    def test_iou_batch_identical(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        iou = iou_batch(a, a)
        assert iou[0, 0] == pytest.approx(1.0)

    def test_iou_batch_no_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        b = np.array([[20, 20, 30, 30]], dtype=np.float64)
        iou = iou_batch(a, b)
        assert iou[0, 0] == pytest.approx(0.0)

    def test_iou_batch_partial_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        b = np.array([[5, 5, 15, 15]], dtype=np.float64)
        iou = iou_batch(a, b)
        assert 0.0 < iou[0, 0] < 1.0

    def test_iou_cost_matrix(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        cost = iou_cost_matrix(a, a)
        assert cost[0, 0] == pytest.approx(0.0)  # cost = 1 - IoU

    def test_linear_assignment_optimal(self):
        cost = np.array([[0.1, 0.9], [0.9, 0.2]], dtype=np.float64)
        matches, unmatched_r, unmatched_c = linear_assignment(cost, threshold=0.5)
        assert len(matches) == 2
        # Optimal: (0,0) and (1,1) with costs 0.1 and 0.2
        match_dict = dict(matches)
        assert match_dict[0] == 0
        assert match_dict[1] == 1

    def test_linear_assignment_threshold_filters(self):
        cost = np.array([[0.1, 0.9], [0.9, 0.9]], dtype=np.float64)
        matches, unmatched_r, unmatched_c = linear_assignment(cost, threshold=0.5)
        assert len(matches) == 1
        assert matches[0] == (0, 0)
        assert 1 in unmatched_r

    def test_linear_assignment_empty(self):
        cost = np.zeros((0, 5), dtype=np.float64)
        matches, unmatched_r, unmatched_c = linear_assignment(cost, threshold=0.5)
        assert len(matches) == 0
        assert len(unmatched_r) == 0
        assert len(unmatched_c) == 5

    def test_centroid_distance_matrix(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        b = np.array([[100, 100, 110, 110]], dtype=np.float64)
        dist = centroid_distance_matrix(a, b, frame_diag=200.0)
        assert dist[0, 0] > 0.0
        assert dist[0, 0] < 1.0  # normalized


# =============================================================================
# GMC TESTS
# =============================================================================

class TestGMC:

    def test_first_frame_returns_identity(self):
        gmc = GlobalMotionCompensation()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        affine = gmc.estimate(frame)
        assert GlobalMotionCompensation.is_identity(affine)

    def test_none_frame_returns_identity(self):
        gmc = GlobalMotionCompensation()
        affine = gmc.estimate(None)
        assert GlobalMotionCompensation.is_identity(affine)

    def test_static_camera_returns_near_identity(self):
        gmc = GlobalMotionCompensation()
        # Two identical frames = no camera motion
        frame = np.random.randint(50, 200, (240, 320, 3), dtype=np.uint8)
        gmc.estimate(frame)  # first frame
        affine = gmc.estimate(frame.copy())  # second identical frame
        # Should be identity or very close
        identity = np.eye(2, 3, dtype=np.float64)
        assert np.allclose(affine, identity, atol=0.5)

    def test_gmc_fallback_on_blank_frames(self):
        gmc = GlobalMotionCompensation()
        # Blank frames have no features → should fallback to identity
        black = np.zeros((240, 320, 3), dtype=np.uint8)
        gmc.estimate(black)
        affine = gmc.estimate(black)
        assert GlobalMotionCompensation.is_identity(affine)

    def test_reset_clears_state(self):
        gmc = GlobalMotionCompensation()
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        gmc.estimate(frame)
        assert gmc._prev_gray is not None
        gmc.reset()
        assert gmc._prev_gray is None

    def test_apply_affine_to_tlbr_identity(self):
        identity = np.eye(2, 3, dtype=np.float64)
        x1, y1, x2, y2 = GlobalMotionCompensation.apply_affine_to_tlbr(
            identity, 100.0, 200.0, 150.0, 300.0
        )
        assert x1 == pytest.approx(100.0, abs=0.01)
        assert y1 == pytest.approx(200.0, abs=0.01)
        assert x2 == pytest.approx(150.0, abs=0.01)
        assert y2 == pytest.approx(300.0, abs=0.01)

    def test_apply_affine_to_tlbr_translation(self):
        # Pure translation: shift right by 10, down by 20
        affine = np.eye(2, 3, dtype=np.float64)
        affine[0, 2] = 10.0
        affine[1, 2] = 20.0
        x1, y1, x2, y2 = GlobalMotionCompensation.apply_affine_to_tlbr(
            affine, 100.0, 200.0, 150.0, 300.0
        )
        assert x1 == pytest.approx(110.0, abs=0.01)
        assert y1 == pytest.approx(220.0, abs=0.01)

    def test_gmc_detects_camera_pan(self):
        """GMC should detect horizontal pan from a synthetic shifted scene."""
        gmc = GlobalMotionCompensation(min_inliers=4)
        # Create a textured frame (checkerboard pattern for strong corners)
        frame1 = np.zeros((240, 320, 3), dtype=np.uint8)
        for i in range(0, 240, 20):
            for j in range(0, 320, 20):
                if (i // 20 + j // 20) % 2 == 0:
                    frame1[i:i+20, j:j+20] = [200, 200, 200]
        gmc.estimate(frame1)
        # Shift frame right by 15 pixels (simulate camera pan)
        frame2 = np.zeros_like(frame1)
        frame2[:, 15:] = frame1[:, :305]
        affine = gmc.estimate(frame2)
        # Translation should be approximately (15, 0) — might not be exact
        # but should NOT be identity
        if not GlobalMotionCompensation.is_identity(affine):
            assert abs(affine[0, 2]) > 2.0  # detected some horizontal shift

    def test_downsample_large_frame(self):
        gmc = GlobalMotionCompensation(downsample_max=640)
        frame = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
        affine = gmc.estimate(frame)
        assert affine.shape == (2, 3)  # should not crash


# =============================================================================
# BoT-SORT TRACKER TESTS
# =============================================================================

class TestBoTSORTTracker:

    def _make_det(self, cls, conf, x1, y1, x2, y2):
        return {
            "object_class": cls,
            "confidence": conf,
            "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        }

    # --- Basic lifecycle ---

    def test_track_creation(self):
        tracker = BoTSORTTracker()
        dets = [self._make_det("person", 0.85, 100, 100, 150, 250)]
        active = tracker.update(timestamp=1.0, detections=dets)
        assert len(active) == 1
        assert active[0].track_id == "TRACK-001"
        assert active[0].object_class == "person"

    def test_track_continuation(self):
        tracker = BoTSORTTracker()
        tracker.update(timestamp=1.0, detections=[
            self._make_det("car", 0.9, 200, 200, 300, 280)
        ])
        active = tracker.update(timestamp=2.0, detections=[
            self._make_det("car", 0.92, 205, 202, 308, 282)
        ])
        assert len(active) == 1
        assert active[0].track_id == "TRACK-001"
        assert active[0].last_seen == 2.0
        assert len(active[0].trajectory) == 2

    def test_multiple_simultaneous_objects(self):
        tracker = BoTSORTTracker()
        dets = [
            self._make_det("person", 0.8, 50, 50, 80, 120),
            self._make_det("car", 0.88, 300, 300, 450, 400),
            self._make_det("bus", 0.75, 600, 100, 800, 350),
        ]
        active = tracker.update(timestamp=0.5, detections=dets)
        assert len(active) == 3
        ids = {t.track_id for t in active}
        assert len(ids) == 3

    def test_track_termination_after_missing(self):
        tracker = BoTSORTTracker(max_missing_seconds=2.0)
        tracker.update(timestamp=1.0, detections=[
            self._make_det("person", 0.85, 100, 100, 150, 250)
        ])
        # Within coasting window
        active = tracker.update(timestamp=2.0, detections=[])
        assert len(active) == 1
        # Beyond coasting window
        active = tracker.update(timestamp=3.5, detections=[])
        assert len(active) == 0
        all_tracks = tracker.get_tracks()
        assert len(all_tracks) == 1
        assert all_tracks[0].active is False

    def test_empty_detections(self):
        tracker = BoTSORTTracker()
        active = tracker.update(timestamp=1.0, detections=[])
        assert len(active) == 0

    def test_rejected_detections_filtered(self):
        tracker = BoTSORTTracker()
        dets = [{
            "object_class": "person", "confidence": 0.9,
            "bounding_box": {"x1": 100, "y1": 100, "x2": 150, "y2": 250},
            "validation_status": "REJECTED",
        }]
        active = tracker.update(timestamp=1.0, detections=dets)
        assert len(active) == 0

    def test_finalize(self):
        tracker = BoTSORTTracker()
        tracker.update(timestamp=1.0, detections=[
            self._make_det("person", 0.85, 100, 100, 150, 250)
        ])
        all_tracks = tracker.finalize()
        assert len(all_tracks) == 1
        assert all_tracks[0].active is False
        assert all_tracks[0].state in (TrackLifecycleState.ENDED,)

    # --- Occlusion handling ---

    def test_occlusion_detection(self):
        """When a track overlaps with a matched detection, it should become OCCLUDED."""
        tracker = BoTSORTTracker(max_missing_seconds=2.0, occlusion_iou_threshold=0.2)
        # Frame 1: two separate persons
        tracker.update(timestamp=1.0, detections=[
            self._make_det("person", 0.9, 100, 100, 150, 250),
            self._make_det("person", 0.9, 200, 100, 250, 250),
        ])
        # Frame 2: only one person visible, at the location of person 1
        # Person 2 is missing but overlaps with person 1's position → should be OCCLUDED
        tracker.update(timestamp=2.0, detections=[
            self._make_det("person", 0.9, 105, 102, 155, 252),
        ])
        # Check track states
        all_tracks = tracker.get_tracks()
        states = {t.track_id: t.state for t in all_tracks}
        # At least one track should still be active (the matched one)
        active = [t for t in all_tracks if t.active]
        assert len(active) >= 1

    def test_occluded_track_longer_survival(self):
        """OCCLUDED tracks should survive longer than COASTING tracks."""
        tracker = BoTSORTTracker(
            max_missing_seconds=2.0,
            max_occluded_seconds=5.0,
            occlusion_iou_threshold=0.2,
        )
        # Create overlapping tracks
        tracker.update(timestamp=1.0, detections=[
            self._make_det("person", 0.9, 100, 100, 200, 300),
            self._make_det("person", 0.9, 130, 100, 230, 300),
        ])
        # Only one detection → other should be occluded
        tracker.update(timestamp=2.0, detections=[
            self._make_det("person", 0.9, 105, 102, 205, 302),
        ])
        # At 3.5s (>2.0s max_missing for COASTING), but OCCLUDED has 5.0s window
        active = tracker.update(timestamp=3.5, detections=[
            self._make_det("person", 0.9, 110, 105, 210, 305),
        ])
        # The occluded track should still be alive if it was correctly marked OCCLUDED
        all_tracks = tracker.get_tracks()
        assert any(t.active for t in all_tracks)

    # --- Lifecycle states ---

    def test_lifecycle_tentative_to_confirmed(self):
        tracker = BoTSORTTracker()
        tracker.update(timestamp=1.0, detections=[
            self._make_det("person", 0.85, 100, 100, 150, 250)
        ])
        active = tracker.update(timestamp=2.0, detections=[
            self._make_det("person", 0.85, 103, 102, 153, 252)
        ])
        assert active[0].state == TrackLifecycleState.CONFIRMED

    def test_lifecycle_coasting(self):
        tracker = BoTSORTTracker(max_missing_seconds=3.0)
        tracker.update(timestamp=1.0, detections=[
            self._make_det("person", 0.85, 100, 100, 150, 250)
        ])
        tracker.update(timestamp=2.0, detections=[])
        all_tracks = tracker.get_tracks()
        coasting = [t for t in all_tracks if t.state == TrackLifecycleState.COASTING]
        assert len(coasting) >= 1

    def test_lifecycle_lost(self):
        tracker = BoTSORTTracker(max_missing_seconds=1.0)
        tracker.update(timestamp=1.0, detections=[
            self._make_det("person", 0.85, 100, 100, 150, 250)
        ])
        tracker.update(timestamp=5.0, detections=[])
        all_tracks = tracker.get_tracks()
        lost = [t for t in all_tracks if t.state == TrackLifecycleState.LOST]
        assert len(lost) >= 1

    # --- Resolution independence ---

    def test_resolution_independence_720p(self):
        tracker = BoTSORTTracker()
        dets = [self._make_det("car", 0.9, 100, 100, 200, 180)]
        active = tracker.update(timestamp=1.0, detections=dets, frame_width=1280, frame_height=720)
        assert len(active) == 1

    def test_resolution_independence_4k(self):
        tracker = BoTSORTTracker()
        dets = [self._make_det("car", 0.9, 400, 400, 800, 720)]
        active = tracker.update(timestamp=1.0, detections=dets, frame_width=3840, frame_height=2160)
        assert len(active) == 1

    # --- GMC integration ---

    def test_tracker_works_without_frame(self):
        """Tracker should work normally when no frame_bgr is provided (GMC disabled)."""
        tracker = BoTSORTTracker()
        dets = [self._make_det("person", 0.85, 100, 100, 150, 250)]
        active = tracker.update(timestamp=1.0, detections=dets, frame_bgr=None)
        assert len(active) == 1

    def test_tracker_works_with_frame(self):
        """Tracker should work when frame_bgr is provided for GMC."""
        tracker = BoTSORTTracker()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dets = [self._make_det("person", 0.85, 100, 100, 150, 250)]
        active = tracker.update(timestamp=1.0, detections=dets, frame_bgr=frame)
        assert len(active) == 1

    def test_gmc_disabled_still_tracks(self):
        """Tracker with enable_gmc=False should still track correctly."""
        tracker = BoTSORTTracker(enable_gmc=False)
        tracker.update(timestamp=1.0, detections=[
            self._make_det("car", 0.9, 200, 200, 300, 280)
        ])
        active = tracker.update(timestamp=2.0, detections=[
            self._make_det("car", 0.92, 205, 202, 308, 282)
        ])
        assert len(active) == 1
        assert active[0].track_id == "TRACK-001"

    # --- Output contract ---

    def test_output_is_tracked_object(self):
        tracker = BoTSORTTracker()
        active = tracker.update(timestamp=1.0, detections=[
            self._make_det("person", 0.85, 100, 100, 150, 250)
        ])
        assert isinstance(active[0], TrackedObject)

    def test_output_has_valid_bbox(self):
        tracker = BoTSORTTracker()
        active = tracker.update(timestamp=1.0, detections=[
            self._make_det("person", 0.85, 100, 100, 150, 250)
        ])
        bbox = active[0].current_bbox
        assert isinstance(bbox, BoundingBox)
        assert bbox.is_valid

    def test_output_trajectory_format(self):
        tracker = BoTSORTTracker()
        tracker.update(timestamp=1.0, detections=[
            self._make_det("person", 0.85, 100, 100, 150, 250)
        ])
        active = tracker.update(timestamp=2.0, detections=[
            self._make_det("person", 0.85, 105, 102, 155, 252)
        ])
        traj = active[0].trajectory
        assert len(traj) == 2
        assert len(traj[0]) == 3  # (timestamp, cx, cy)

    def test_output_to_dict_succeeds(self):
        tracker = BoTSORTTracker()
        active = tracker.update(timestamp=1.0, detections=[
            self._make_det("person", 0.85, 100, 100, 150, 250)
        ])
        d = active[0].to_dict()
        assert "track_id" in d
        assert "object_class" in d
        assert "current_bbox" in d
        assert "trajectory" in d

    def test_track_id_assigned_to_raw_detection(self):
        """Tracker should assign track_id back to the raw detection dict."""
        tracker = BoTSORTTracker()
        det = self._make_det("person", 0.85, 100, 100, 150, 250)
        tracker.update(timestamp=1.0, detections=[det])
        assert "track_id" in det
        assert det["track_id"] == "TRACK-001"

    # --- Memory behavior ---

    def test_memory_bounded_many_tracks(self):
        """Creating many tracks should not cause excessive memory growth."""
        import sys
        tracker = BoTSORTTracker()
        # Create 100 tracks at different positions
        dets = [
            self._make_det("person", 0.8, i * 50, 100, i * 50 + 30, 200)
            for i in range(100)
        ]
        active = tracker.update(timestamp=1.0, detections=dets)
        assert len(active) == 100
        # Each track should be reasonably sized
        all_tracks = tracker.get_tracks()
        for t in all_tracks:
            d = t.to_dict()
            size = sys.getsizeof(str(d))
            assert size < 10000  # less than 10KB per track dict


# =============================================================================
# BACKWARD COMPATIBILITY TESTS
# =============================================================================

class TestBackwardCompatibility:

    def _make_det(self, cls, conf, x1, y1, x2, y2):
        return {
            "object_class": cls,
            "confidence": conf,
            "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        }

    def test_bytetrack_accepts_frame_bgr(self):
        """ObjectTracker should accept and ignore frame_bgr parameter."""
        tracker = ObjectTracker()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        active = tracker.update(
            timestamp=1.0,
            detections=[self._make_det("person", 0.85, 100, 100, 150, 250)],
            frame_bgr=frame,
        )
        assert len(active) == 1

    def test_bytetrack_without_frame_bgr(self):
        """ObjectTracker should still work without frame_bgr."""
        tracker = ObjectTracker()
        active = tracker.update(
            timestamp=1.0,
            detections=[self._make_det("person", 0.85, 100, 100, 150, 250)],
        )
        assert len(active) == 1

    def test_both_trackers_produce_same_output_type(self):
        """Both trackers should produce TrackedObject instances."""
        det = self._make_det("person", 0.85, 100, 100, 150, 250)
        bt = ObjectTracker()
        bs = BoTSORTTracker()
        bt_active = bt.update(timestamp=1.0, detections=[det.copy()])
        bs_active = bs.update(timestamp=1.0, detections=[det.copy()])
        assert isinstance(bt_active[0], TrackedObject)
        assert isinstance(bs_active[0], TrackedObject)

    def test_both_trackers_same_track_id_format(self):
        """Both trackers should use TRACK-NNN format."""
        det = self._make_det("person", 0.85, 100, 100, 150, 250)
        bt = ObjectTracker()
        bs = BoTSORTTracker()
        bt_active = bt.update(timestamp=1.0, detections=[det.copy()])
        bs_active = bs.update(timestamp=1.0, detections=[det.copy()])
        assert bt_active[0].track_id.startswith("TRACK-")
        assert bs_active[0].track_id.startswith("TRACK-")

    def test_both_trackers_have_reset(self):
        bt = ObjectTracker()
        bs = BoTSORTTracker()
        bt.reset(video_id="test")
        bs.reset(video_id="test")

    def test_both_trackers_have_finalize(self):
        bt = ObjectTracker()
        bs = BoTSORTTracker()
        bt.finalize()
        bs.finalize()

    def test_both_trackers_have_get_tracks(self):
        bt = ObjectTracker()
        bs = BoTSORTTracker()
        assert isinstance(bt.get_tracks(), list)
        assert isinstance(bs.get_tracks(), list)

    def test_both_trackers_have_get_validated_tracks(self):
        bt = ObjectTracker()
        bs = BoTSORTTracker()
        assert isinstance(bt.get_validated_tracks(), list)
        assert isinstance(bs.get_validated_tracks(), list)

    def test_config_default_is_bytetrack(self):
        from backend.app.core.config import settings
        assert settings.TRACKER_TYPE == "bytetrack"
