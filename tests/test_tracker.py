"""
test_tracker.py
---------------
Unit tests for the tracker module: ExponentialSmoothingTracker behavior.
"""

import sys
import os

# Allow imports from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from tracker import ExponentialSmoothingTracker, tracking_error


class TestExponentialSmoothingTracker:
    """Tests for ExponentialSmoothingTracker."""

    def test_first_measurement_accepted_directly(self):
        """First measurement should be returned as-is."""
        tracker = ExponentialSmoothingTracker(alpha=0.5, dt=0.1)
        measurement = np.array([10.0, 20.0, 30.0])
        result = tracker.update(measurement)
        np.testing.assert_allclose(result, measurement)

    def test_smoothing_reduces_noise(self):
        """After many noisy measurements around a point, estimate should be close."""
        tracker = ExponentialSmoothingTracker(alpha=0.3, dt=0.1)
        true_pos = np.array([50.0, 50.0, 50.0])

        # Feed 100 noisy measurements
        np.random.seed(42)
        for _ in range(100):
            noisy = true_pos + np.random.randn(3) * 5.0
            estimate = tracker.update(noisy)

        # Estimate should be within a reasonable range of true position
        error = np.linalg.norm(estimate - true_pos)
        assert error < 10.0, f"Tracking error too large: {error}"

    def test_velocity_estimation(self):
        """Velocity should reflect consistent movement direction."""
        tracker = ExponentialSmoothingTracker(alpha=0.8, dt=1.0)

        # Move consistently in +X direction
        for i in range(20):
            measurement = np.array([float(i), 0.0, 0.0])
            tracker.update(measurement)

        vel = tracker.get_estimated_velocity()
        # Velocity X component should be positive and close to 1.0
        assert vel[0] > 0.5, f"Expected positive X velocity, got {vel[0]}"

    def test_reset_clears_state(self):
        """After reset, tracker should behave like a fresh instance."""
        tracker = ExponentialSmoothingTracker(alpha=0.5, dt=0.1)

        # Feed some data
        tracker.update(np.array([10.0, 10.0, 10.0]))
        tracker.update(np.array([20.0, 20.0, 20.0]))

        # Reset
        tracker.reset()

        # Next measurement should be accepted directly
        result = tracker.update(np.array([5.0, 5.0, 5.0]))
        np.testing.assert_allclose(result, [5.0, 5.0, 5.0])

    def test_alpha_one_follows_measurement_exactly(self):
        """With alpha=1.0, the tracker should follow measurements exactly."""
        tracker = ExponentialSmoothingTracker(alpha=1.0, dt=0.1)
        tracker.update(np.array([0.0, 0.0, 0.0]))

        measurement = np.array([99.0, 88.0, 77.0])
        result = tracker.update(measurement)
        np.testing.assert_allclose(result, measurement)

    def test_alpha_zero_stays_at_first(self):
        """With alpha=0.0 (after init), tracker ignores new measurements."""
        tracker = ExponentialSmoothingTracker(alpha=0.0, dt=0.1)
        first = np.array([10.0, 10.0, 10.0])
        tracker.update(first)

        # Subsequent measurements should be ignored
        result = tracker.update(np.array([99.0, 99.0, 99.0]))
        np.testing.assert_allclose(result, first)


class TestTrackingError:
    """Tests for the tracking_error utility function."""

    def test_zero_error(self):
        pos = np.array([1.0, 2.0, 3.0])
        assert tracking_error(pos, pos) == 0.0

    def test_known_error(self):
        estimated = np.array([0.0, 0.0, 0.0])
        true_pos = np.array([3.0, 4.0, 0.0])
        assert abs(tracking_error(estimated, true_pos) - 5.0) < 1e-9
