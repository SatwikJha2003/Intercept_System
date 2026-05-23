"""
tracker.py
----------
Tracking algorithms for estimating the true target position from noisy
sensor measurements.

Provides:
- BaseTracker: abstract interface for any tracker implementation
- ExponentialSmoothingTracker: simple first-order smoothing
- KalmanFilterTracker: placeholder for future implementation
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from config import DEFAULT_SENSOR_NOISE, DEFAULT_SMOOTHING_ALPHA, TIMESTEP


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseTracker(ABC):
    """
    Abstract base class defining the tracker interface.

    Any tracker must implement:
    - update(measurement) -> smoothed position estimate
    - get_estimated_velocity() -> current velocity estimate
    - reset() -> reinitialize internal state
    """

    @abstractmethod
    def update(self, measurement: NDArray[np.float64]) -> NDArray[np.float64]:
        """Process a new noisy measurement and return the smoothed estimate."""
        ...

    @abstractmethod
    def get_estimated_velocity(self) -> NDArray[np.float64]:
        """Return the current estimated velocity vector."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the tracker to its initial state."""
        ...


# ---------------------------------------------------------------------------
# Exponential Smoothing Tracker
# ---------------------------------------------------------------------------

class ExponentialSmoothingTracker(BaseTracker):
    """
    First-order exponential smoothing tracker.

    The smoothed estimate is:
        s_t = alpha * measurement + (1 - alpha) * s_{t-1}

    Velocity is estimated from successive smoothed positions.

    Parameters
    ----------
    alpha : smoothing factor in (0, 1]. Higher = more responsive, noisier.
    dt    : timestep between updates (seconds).
    """

    def __init__(self, alpha: float = DEFAULT_SMOOTHING_ALPHA, dt: float = TIMESTEP):
        self.alpha = alpha
        self.dt = dt
        self._position: NDArray[np.float64] | None = None
        self._prev_position: NDArray[np.float64] | None = None
        self._velocity: NDArray[np.float64] = np.zeros(3)

    def update(self, measurement: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Incorporate a new measurement and return the smoothed position.

        On the first call the measurement is accepted directly (no history).
        """
        if self._position is None:
            # First measurement — initialize
            self._position = measurement.copy()
            self._prev_position = measurement.copy()
            self._velocity = np.zeros(3)
            return self._position.copy()

        # Store previous for velocity estimation
        self._prev_position = self._position.copy()

        # Exponential smoothing
        self._position = self.alpha * measurement + (1 - self.alpha) * self._position

        # Velocity from smoothed position change
        if self.dt > 0:
            self._velocity = (self._position - self._prev_position) / self.dt

        return self._position.copy()

    def get_estimated_velocity(self) -> NDArray[np.float64]:
        """Return the latest velocity estimate."""
        return self._velocity.copy()

    def reset(self) -> None:
        """Clear all internal state."""
        self._position = None
        self._prev_position = None
        self._velocity = np.zeros(3)


# ---------------------------------------------------------------------------
# Kalman Filter Tracker
# ---------------------------------------------------------------------------

class KalmanFilterTracker(BaseTracker):
    """
    Linear Kalman filter for 3D position tracking with a constant-velocity
    motion model.

    State vector (6D): [x, y, z, vx, vy, vz]

    The filter assumes:
    - Constant-velocity motion between updates (process model)
    - Direct position observations with Gaussian noise (measurement model)

    Parameters
    ----------
    dt : float
        Timestep between updates (seconds).
    process_noise_std : float
        Standard deviation of process noise (acceleration uncertainty).
    measurement_noise_std : float
        Standard deviation of measurement noise per axis.
    """

    def __init__(
        self,
        dt: float = TIMESTEP,
        process_noise_std: float = 2.0,
        measurement_noise_std: float = DEFAULT_SENSOR_NOISE,
    ):
        self.dt = dt
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std

        # State vector [x, y, z, vx, vy, vz]
        self._x: NDArray[np.float64] = np.zeros(6)

        # State covariance matrix (6x6)
        self._P: NDArray[np.float64] = np.eye(6) * 500.0

        # Initialized flag
        self._initialized: bool = False

    @property
    def _F(self) -> NDArray[np.float64]:
        """State transition matrix for constant-velocity model."""
        dt = self.dt
        F = np.eye(6)
        F[0, 3] = dt  # x += vx * dt
        F[1, 4] = dt  # y += vy * dt
        F[2, 5] = dt  # z += vz * dt
        return F

    @property
    def _H(self) -> NDArray[np.float64]:
        """Measurement matrix: we observe position only."""
        H = np.zeros((3, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        return H

    @property
    def _Q(self) -> NDArray[np.float64]:
        """
        Process noise covariance matrix.

        Uses a piecewise-constant white noise acceleration model.
        """
        dt = self.dt
        q = self.process_noise_std ** 2

        # For each axis, the block is:
        # [[dt^4/4, dt^3/2],
        #  [dt^3/2, dt^2  ]] * q
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[i, i] = (dt ** 4) / 4.0 * q
            Q[i, i + 3] = (dt ** 3) / 2.0 * q
            Q[i + 3, i] = (dt ** 3) / 2.0 * q
            Q[i + 3, i + 3] = (dt ** 2) * q
        return Q

    @property
    def _R(self) -> NDArray[np.float64]:
        """Measurement noise covariance matrix."""
        r = self.measurement_noise_std ** 2
        return np.eye(3) * r

    def update(self, measurement: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Run one predict-update cycle of the Kalman filter.

        Parameters
        ----------
        measurement : 3D position measurement [x, y, z]

        Returns
        -------
        Filtered position estimate [x, y, z]
        """
        if not self._initialized:
            # Initialize state from first measurement
            self._x[:3] = measurement.copy()
            self._x[3:] = 0.0  # Unknown velocity
            self._P = np.eye(6) * 500.0
            self._initialized = True
            return self._x[:3].copy()

        F = self._F
        H = self._H
        Q = self._Q
        R = self._R

        # --- Predict step ---
        x_pred = F @ self._x
        P_pred = F @ self._P @ F.T + Q

        # --- Update step ---
        z = measurement
        y = z - H @ x_pred  # Innovation (measurement residual)
        S = H @ P_pred @ H.T + R  # Innovation covariance
        K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain

        self._x = x_pred + K @ y
        self._P = (np.eye(6) - K @ H) @ P_pred

        return self._x[:3].copy()

    def get_estimated_velocity(self) -> NDArray[np.float64]:
        """Return the Kalman filter's velocity estimate [vx, vy, vz]."""
        return self._x[3:].copy()

    def reset(self) -> None:
        """Reset the filter to uninitialized state."""
        self._x = np.zeros(6)
        self._P = np.eye(6) * 500.0
        self._initialized = False


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def tracking_error(
    estimated: NDArray[np.float64], true_pos: NDArray[np.float64]
) -> float:
    """Compute the Euclidean tracking error between estimate and truth."""
    return float(np.linalg.norm(estimated - true_pos))
