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

from config import DEFAULT_SMOOTHING_ALPHA, TIMESTEP


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
# Kalman Filter Tracker (placeholder)
# ---------------------------------------------------------------------------

class KalmanFilterTracker(BaseTracker):
    """
    Placeholder for a future Kalman filter-based tracker.

    This class satisfies the BaseTracker interface but raises
    NotImplementedError on use. It exists to show where a more
    advanced tracker would slot in.
    """

    def update(self, measurement: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError("KalmanFilterTracker is not yet implemented.")

    def get_estimated_velocity(self) -> NDArray[np.float64]:
        raise NotImplementedError("KalmanFilterTracker is not yet implemented.")

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def tracking_error(
    estimated: NDArray[np.float64], true_pos: NDArray[np.float64]
) -> float:
    """Compute the Euclidean tracking error between estimate and truth."""
    return float(np.linalg.norm(estimated - true_pos))
