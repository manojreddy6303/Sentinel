"""
Kalman Filter for Multi-Object Tracking (Phase 21A)

8-dimensional state vector: [cx, cy, w, h, vx, vy, vw, vh]
- (cx, cy): bounding box center
- (w, h): bounding box width and height
- (vx, vy, vw, vh): respective velocities

Constant velocity model with linear observation.
Supports variable time steps (dt) for non-uniform frame sampling.

SAFETY: No biometrics, no identity inference — purely kinematic state estimation.
"""
import numpy as np
from typing import Tuple


class KalmanFilter:
    """
    Standard Kalman filter for bounding box tracking.

    State: [cx, cy, w, h, vx, vy, vw, vh]
    Measurement: [cx, cy, w, h]
    """

    # Observation matrix H (4x8): extracts [cx, cy, w, h] from state
    _observation_mat = np.eye(4, 8, dtype=np.float64)

    # Process noise scaling factors (relative to object size)
    _std_weight_position = 1.0 / 20
    _std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize track state from first measurement.

        Args:
            measurement: [cx, cy, w, h] array

        Returns:
            (mean, covariance) tuple — mean shape (8,), covariance shape (8, 8)
        """
        mean_pos = measurement.astype(np.float64).copy()
        mean_vel = np.zeros(4, dtype=np.float64)
        mean = np.concatenate([mean_pos, mean_vel])

        # Initial uncertainty: large for velocity (unknown), moderate for position
        w, h = max(abs(measurement[2]), 1.0), max(abs(measurement[3]), 1.0)
        std = [
            2 * self._std_weight_position * w,
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * w,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * w,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * w,
            10 * self._std_weight_velocity * h,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray, dt: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next state using constant-velocity model.

        Args:
            mean: current state mean (8,)
            covariance: current state covariance (8, 8)
            dt: time step in seconds since last update

        Returns:
            (predicted_mean, predicted_covariance)
        """
        # Build transition matrix F with variable dt
        F = np.eye(8, dtype=np.float64)
        F[0, 4] = dt
        F[1, 5] = dt
        F[2, 6] = dt
        F[3, 7] = dt

        # Process noise scales with object size and dt
        w = max(abs(mean[2]), 1.0)
        h = max(abs(mean[3]), 1.0)
        dt_factor = max(dt, 0.01)  # prevent zero noise
        std = [
            self._std_weight_position * w * dt_factor,
            self._std_weight_position * h * dt_factor,
            self._std_weight_position * w * dt_factor,
            self._std_weight_position * h * dt_factor,
            self._std_weight_velocity * w * dt_factor,
            self._std_weight_velocity * h * dt_factor,
            self._std_weight_velocity * w * dt_factor,
            self._std_weight_velocity * h * dt_factor,
        ]
        motion_cov = np.diag(np.square(std))

        predicted_mean = F @ mean
        predicted_cov = F @ covariance @ F.T + motion_cov
        return predicted_mean, predicted_cov

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update state with new measurement [cx, cy, w, h].

        Returns:
            (updated_mean, updated_covariance)
        """
        H = self._observation_mat
        w = max(abs(mean[2]), 1.0)
        h = max(abs(mean[3]), 1.0)
        std = [
            self._std_weight_position * w,
            self._std_weight_position * h,
            self._std_weight_position * w,
            self._std_weight_position * h,
        ]
        innovation_cov = np.diag(np.square(std))

        # Project state to measurement space
        projected_mean = H @ mean
        projected_cov = H @ covariance @ H.T + innovation_cov

        # Kalman gain via Cholesky decomposition for numerical stability
        try:
            chol = np.linalg.cholesky(projected_cov)
            kalman_gain = np.linalg.solve(
                chol @ chol.T, (covariance @ H.T).T
            ).T
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if Cholesky fails
            kalman_gain = covariance @ H.T @ np.linalg.pinv(projected_cov)

        innovation = measurement.astype(np.float64) - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance

    def project(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project state to measurement space.

        Returns:
            (projected_mean, projected_covariance) in [cx, cy, w, h] space
        """
        H = self._observation_mat
        w = max(abs(mean[2]), 1.0)
        h = max(abs(mean[3]), 1.0)
        std = [
            self._std_weight_position * w,
            self._std_weight_position * h,
            self._std_weight_position * w,
            self._std_weight_position * h,
        ]
        innovation_cov = np.diag(np.square(std))
        projected_mean = H @ mean
        projected_cov = H @ covariance @ H.T + innovation_cov
        return projected_mean, projected_cov

    def gating_distance(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> float:
        """
        Compute squared Mahalanobis distance between state and measurement.

        Used for association gating: reject matches above chi-squared threshold.
        For 4-DOF measurement space, 95% gate ≈ 9.49, 99% gate ≈ 13.28.
        """
        projected_mean, projected_cov = self.project(mean, covariance)
        diff = measurement.astype(np.float64) - projected_mean
        try:
            chol = np.linalg.cholesky(projected_cov)
            z = np.linalg.solve(chol, diff)
            return float(z @ z)
        except np.linalg.LinAlgError:
            return float("inf")

    @staticmethod
    def state_to_bbox(mean: np.ndarray) -> Tuple[float, float, float, float]:
        """Convert Kalman state [cx, cy, w, h, ...] to [x1, y1, x2, y2]."""
        cx, cy, w, h = mean[:4]
        w = max(w, 1.0)
        h = max(h, 1.0)
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    @staticmethod
    def bbox_to_measurement(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to Kalman measurement [cx, cy, w, h]."""
        return np.array([
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
            x2 - x1,
            y2 - y1,
        ], dtype=np.float64)
