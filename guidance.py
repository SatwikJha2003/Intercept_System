"""
guidance.py
-----------
Vector helper functions, future target prediction, interceptor movement
direction computation, and intercept detection logic.
"""

import numpy as np
from numpy.typing import NDArray

from config import INTERCEPT_RADIUS, MAX_PREDICTION_TIME


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def normalize(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the unit vector of *v*. Returns zero vector if magnitude is ~0."""
    mag = np.linalg.norm(v)
    if mag < 1e-9:
        return np.zeros(3)
    return v / mag


def distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Euclidean distance between two 3D points."""
    return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_future_position(
    estimated_position: NDArray[np.float64],
    estimated_velocity: NDArray[np.float64],
    prediction_time: float,
) -> NDArray[np.float64]:
    """
    Predict where the target will be after *prediction_time* seconds,
    using a simple linear extrapolation from the estimated state.

    Parameters
    ----------
    estimated_position : current smoothed position estimate
    estimated_velocity : current estimated velocity vector
    prediction_time    : how far ahead to predict (seconds)

    Returns
    -------
    Predicted 3D position as numpy array.
    """
    # Clamp prediction time to avoid wild extrapolation
    t = min(max(prediction_time, 0.0), MAX_PREDICTION_TIME)
    return estimated_position + estimated_velocity * t


def compute_intercept_direction(
    interceptor_pos: NDArray[np.float64],
    predicted_target_pos: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute the unit direction vector the interceptor should move toward.

    Returns a zero vector if the interceptor is already at the target.
    """
    diff = predicted_target_pos - interceptor_pos
    return normalize(diff)


# ---------------------------------------------------------------------------
# Intercept detection
# ---------------------------------------------------------------------------

def check_intercept(
    interceptor_pos: NDArray[np.float64],
    target_pos: NDArray[np.float64],
    radius: float = INTERCEPT_RADIUS,
) -> bool:
    """Return True if the interceptor is within *radius* of the target."""
    return distance(interceptor_pos, target_pos) <= radius
