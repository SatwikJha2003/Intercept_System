"""
test_guidance.py
----------------
Unit tests for the guidance module: vector helpers, prediction, and intercept detection.
"""

import sys
import os

# Allow imports from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from guidance import (
    check_intercept,
    compute_intercept_direction,
    distance,
    normalize,
    predict_future_position,
)


class TestNormalize:
    """Tests for the normalize() vector helper."""

    def test_unit_vector(self):
        v = np.array([3.0, 0.0, 0.0])
        result = normalize(v)
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0])

    def test_arbitrary_vector(self):
        v = np.array([1.0, 1.0, 1.0])
        result = normalize(v)
        expected_mag = np.linalg.norm(result)
        assert abs(expected_mag - 1.0) < 1e-9

    def test_zero_vector(self):
        v = np.array([0.0, 0.0, 0.0])
        result = normalize(v)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])

    def test_very_small_vector(self):
        v = np.array([1e-12, 0.0, 0.0])
        result = normalize(v)
        # Should return zero vector since magnitude < 1e-9
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])


class TestDistance:
    """Tests for the distance() function."""

    def test_same_point(self):
        a = np.array([1.0, 2.0, 3.0])
        assert distance(a, a) == 0.0

    def test_known_distance(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([3.0, 4.0, 0.0])
        assert abs(distance(a, b) - 5.0) < 1e-9


class TestPredictFuturePosition:
    """Tests for predict_future_position()."""

    def test_stationary(self):
        pos = np.array([10.0, 20.0, 30.0])
        vel = np.array([0.0, 0.0, 0.0])
        result = predict_future_position(pos, vel, 5.0)
        np.testing.assert_allclose(result, pos)

    def test_linear_motion(self):
        pos = np.array([0.0, 0.0, 0.0])
        vel = np.array([10.0, 0.0, 0.0])
        result = predict_future_position(pos, vel, 2.0)
        np.testing.assert_allclose(result, [20.0, 0.0, 0.0])

    def test_prediction_time_clamped(self):
        """Prediction time should be clamped to MAX_PREDICTION_TIME."""
        pos = np.array([0.0, 0.0, 0.0])
        vel = np.array([1.0, 0.0, 0.0])
        # Request 100 seconds, should be clamped to MAX_PREDICTION_TIME (5.0)
        result = predict_future_position(pos, vel, 100.0)
        np.testing.assert_allclose(result, [5.0, 0.0, 0.0])

    def test_negative_time_clamped_to_zero(self):
        pos = np.array([5.0, 5.0, 5.0])
        vel = np.array([10.0, 10.0, 10.0])
        result = predict_future_position(pos, vel, -3.0)
        np.testing.assert_allclose(result, pos)


class TestCheckIntercept:
    """Tests for check_intercept()."""

    def test_within_radius(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert check_intercept(a, b, radius=2.0) is True

    def test_outside_radius(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([10.0, 0.0, 0.0])
        assert check_intercept(a, b, radius=2.0) is False

    def test_exactly_on_boundary(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([3.0, 0.0, 0.0])
        assert check_intercept(a, b, radius=3.0) is True


class TestComputeInterceptDirection:
    """Tests for compute_intercept_direction()."""

    def test_direction_toward_target(self):
        interceptor = np.array([0.0, 0.0, 0.0])
        target = np.array([10.0, 0.0, 0.0])
        direction = compute_intercept_direction(interceptor, target)
        np.testing.assert_allclose(direction, [1.0, 0.0, 0.0])

    def test_same_position_returns_zero(self):
        pos = np.array([5.0, 5.0, 5.0])
        direction = compute_intercept_direction(pos, pos)
        np.testing.assert_allclose(direction, [0.0, 0.0, 0.0])
