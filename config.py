"""
config.py
---------
Default constants for the 3D Object Tracking and Intercept Simulation.
All values can be overridden at runtime via the Dash frontend controls.
"""

# Arena ------------------------------------------------------------------
ARENA_SIZE: float = 100.0  # Half-extent of the cubic arena (-SIZE to +SIZE)

# Timing -----------------------------------------------------------------
TIMESTEP: float = 0.1  # Seconds per simulation step
UPDATE_INTERVAL_MS: int = 100  # Dash interval callback period (ms)

# Target -----------------------------------------------------------------
DEFAULT_TARGET_SPEED: float = 15.0  # Units per second
TARGET_DIRECTION_CHANGE_RATE: float = 0.05  # How often direction drifts

# Interceptor ------------------------------------------------------------
DEFAULT_INTERCEPTOR_SPEED: float = 25.0  # Units per second

# Sensor -----------------------------------------------------------------
DEFAULT_SENSOR_NOISE: float = 5.0  # Std-dev of Gaussian noise per axis

# Tracking ---------------------------------------------------------------
DEFAULT_SMOOTHING_ALPHA: float = 0.3  # Exponential smoothing factor (0-1)

# Guidance / Intercept ---------------------------------------------------
INTERCEPT_RADIUS: float = 3.0  # Distance threshold for intercept detection
MAX_PREDICTION_TIME: float = 5.0  # Clamp prediction horizon (seconds)

# Visualization ----------------------------------------------------------
MAX_TRAIL_LENGTH: int = 200  # Max number of trail points stored
