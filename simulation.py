"""
simulation.py
-------------
Core simulation state and update logic.

Manages:
- Target motion with pseudo-random direction changes and boundary bouncing
- Noisy sensor measurements
- Interceptor position updates (driven by guidance module)
- History / trail storage
- Reset logic
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from config import (
    ARENA_SIZE,
    DEFAULT_INTERCEPTOR_SPEED,
    DEFAULT_SENSOR_NOISE,
    DEFAULT_TARGET_SPEED,
    MAX_TRAIL_LENGTH,
    TARGET_DIRECTION_CHANGE_RATE,
    TIMESTEP,
)
from guidance import (
    check_intercept,
    compute_intercept_direction,
    distance,
    normalize,
    predict_future_position,
)
from tracker import ExponentialSmoothingTracker, tracking_error


# ---------------------------------------------------------------------------
# Simulation State
# ---------------------------------------------------------------------------

@dataclass
class SimulationState:
    """
    Holds all mutable state for one simulation run.

    Create via SimulationState.create_default() for a fresh start.
    """

    # Positions
    target_pos: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    target_vel: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    interceptor_pos: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))

    # Sensor
    sensor_measurement: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))

    # Tracker
    estimated_pos: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    estimated_vel: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    predicted_intercept_pos: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))

    # Trails (lists of 3-element arrays)
    target_trail: list = field(default_factory=list)
    interceptor_trail: list = field(default_factory=list)
    estimated_trail: list = field(default_factory=list)

    # Counters
    time: float = 0.0
    intercept_count: int = 0

    # Tracker instance
    tracker: ExponentialSmoothingTracker = field(
        default_factory=ExponentialSmoothingTracker
    )

    # Runtime parameters (can be changed via UI)
    target_speed: float = DEFAULT_TARGET_SPEED
    interceptor_speed: float = DEFAULT_INTERCEPTOR_SPEED
    sensor_noise: float = DEFAULT_SENSOR_NOISE
    dt: float = TIMESTEP
    paused: bool = False

    @classmethod
    def create_default(cls) -> "SimulationState":
        """Factory method to create a fresh simulation with randomized start."""
        state = cls()
        state.reset()
        return state

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reinitialize all state to starting conditions."""
        # Random target start inside arena
        self.target_pos = np.random.uniform(-ARENA_SIZE * 0.5, ARENA_SIZE * 0.5, size=3)

        # Random initial direction
        direction = normalize(np.random.randn(3))
        self.target_vel = direction * self.target_speed

        # Interceptor starts at origin
        self.interceptor_pos = np.zeros(3)

        # Clear sensor / tracker state
        self.sensor_measurement = self.target_pos.copy()
        self.estimated_pos = self.target_pos.copy()
        self.estimated_vel = np.zeros(3)
        self.predicted_intercept_pos = self.target_pos.copy()

        # Clear trails
        self.target_trail = []
        self.interceptor_trail = []
        self.estimated_trail = []

        # Reset counters
        self.time = 0.0
        self.intercept_count = 0

        # Reset tracker
        self.tracker.reset()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance the simulation by one timestep."""
        if self.paused:
            return

        dt = self.dt

        # 1. Update target motion
        self._update_target(dt)

        # 2. Generate noisy sensor measurement
        self._generate_measurement()

        # 3. Update tracker with measurement
        self._update_tracker()

        # 4. Predict future target position
        self._predict_intercept()

        # 5. Move interceptor toward predicted position
        self._update_interceptor(dt)

        # 6. Check for intercept
        self._check_intercept()

        # 7. Record trails
        self._record_trails()

        # 8. Advance time
        self.time += dt

    # ------------------------------------------------------------------
    # Internal update methods
    # ------------------------------------------------------------------

    def _update_target(self, dt: float) -> None:
        """Move the target and handle boundary bouncing."""
        # Slightly drift the direction for organic motion
        drift = np.random.randn(3) * TARGET_DIRECTION_CHANGE_RATE
        self.target_vel = normalize(self.target_vel + drift) * self.target_speed

        # Move
        self.target_pos = self.target_pos + self.target_vel * dt

        # Bounce off arena walls
        for axis in range(3):
            if self.target_pos[axis] > ARENA_SIZE:
                self.target_pos[axis] = ARENA_SIZE
                self.target_vel[axis] *= -1
            elif self.target_pos[axis] < -ARENA_SIZE:
                self.target_pos[axis] = -ARENA_SIZE
                self.target_vel[axis] *= -1

    def _generate_measurement(self) -> None:
        """Simulate a noisy sensor reading of the target position."""
        noise = np.random.randn(3) * self.sensor_noise
        self.sensor_measurement = self.target_pos + noise

    def _update_tracker(self) -> None:
        """Feed the noisy measurement into the tracker."""
        self.estimated_pos = self.tracker.update(self.sensor_measurement)
        self.estimated_vel = self.tracker.get_estimated_velocity()

    def _predict_intercept(self) -> None:
        """Predict where the target will be when the interceptor could arrive."""
        # Estimate time-to-intercept based on current distance and interceptor speed
        dist = distance(self.interceptor_pos, self.estimated_pos)
        if self.interceptor_speed > 0:
            prediction_time = dist / self.interceptor_speed
        else:
            prediction_time = 0.0

        self.predicted_intercept_pos = predict_future_position(
            self.estimated_pos, self.estimated_vel, prediction_time
        )

    def _update_interceptor(self, dt: float) -> None:
        """Move the interceptor toward the predicted intercept point."""
        direction = compute_intercept_direction(
            self.interceptor_pos, self.predicted_intercept_pos
        )
        self.interceptor_pos = self.interceptor_pos + direction * self.interceptor_speed * dt

    def _check_intercept(self) -> None:
        """Detect intercept and reset interceptor if successful."""
        if check_intercept(self.interceptor_pos, self.target_pos):
            self.intercept_count += 1
            # Reset interceptor to origin after successful intercept
            self.interceptor_pos = np.zeros(3)

    def _record_trails(self) -> None:
        """Append current positions to trail histories, trimming if needed."""
        self.target_trail.append(self.target_pos.copy())
        self.interceptor_trail.append(self.interceptor_pos.copy())
        self.estimated_trail.append(self.estimated_pos.copy())

        # Trim to max length
        if len(self.target_trail) > MAX_TRAIL_LENGTH:
            self.target_trail = self.target_trail[-MAX_TRAIL_LENGTH:]
        if len(self.interceptor_trail) > MAX_TRAIL_LENGTH:
            self.interceptor_trail = self.interceptor_trail[-MAX_TRAIL_LENGTH:]
        if len(self.estimated_trail) > MAX_TRAIL_LENGTH:
            self.estimated_trail = self.estimated_trail[-MAX_TRAIL_LENGTH:]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_distance_to_target(self) -> float:
        """Current distance between interceptor and true target."""
        return distance(self.interceptor_pos, self.target_pos)

    def get_tracking_error(self) -> float:
        """Current Euclidean error between estimate and true target."""
        return tracking_error(self.estimated_pos, self.target_pos)
