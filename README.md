# 3D Object Tracking and Intercept Simulation

A browser-based educational simulation demonstrating 3D object tracking, prediction, and interception. Built entirely in Python using Dash and Plotly.

> **Note:** This is strictly a simulated, educational project. It contains no real-world deployment, hardware integration, or physical guidance system logic.

---

## Overview

The simulation features:

- A **target** moving with pseudo-random 3D motion inside a bounded arena
- A **sensor model** that produces noisy position measurements
- A **tracker** with two selectable algorithms:
  - **Exponential Smoothing** — simple first-order filter
  - **Kalman Filter** — optimal linear state estimator with constant-velocity model
- A **predictor** that extrapolates the target's future position
- An **interceptor** that navigates toward the predicted intercept point
- A **multi-page browser UI** with:
  - Real-time 3D visualization and interactive controls
  - A dedicated Theory & Math page explaining the system's physics

---

## System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────────────┐
│   Target    │────▶│    Sensor    │────▶│        Tracker           │
│  (motion)   │     │ (adds noise) │     │ (Exp. Smooth / Kalman)   │
└─────────────┘     └──────────────┘     └────────────┬─────────────┘
                                                      │
                                                      ▼
                                              ┌──────────────┐
                                              │  Predictor   │
                                              │(extrapolate) │
                                              └──────┬───────┘
                                                     │
                                                     ▼
                                              ┌──────────────┐
                                              │ Interceptor  │
                                              │  (guidance)  │
                                              └──────────────┘
```

### Tracking / Prediction Loop

1. **Target moves** — smooth pseudo-random 3D motion with boundary bouncing
2. **Sensor measures** — true position + Gaussian noise
3. **Tracker updates** — exponential smoothing or Kalman filter processes the measurement
4. **Velocity estimated** — from smoothed positions (exp.) or directly from state vector (Kalman)
5. **Future position predicted** — linear extrapolation using estimated velocity
6. **Interceptor steers** — moves toward the predicted intercept point
7. **Intercept check** — if interceptor is within threshold distance of target, count it and reset

---

## Project Structure

```
Intercept_System/
├── app.py              # Dash multi-page app, Plotly visualization, callbacks
├── theory_page.py      # Theory & Math page layout (guidance/interception math)
├── simulation.py       # SimulationState, target/interceptor update logic
├── tracker.py          # BaseTracker, ExponentialSmoothingTracker, KalmanFilterTracker
├── guidance.py         # Vector helpers, prediction, intercept detection
├── config.py           # Default constants and parameters
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── tests/
    ├── test_guidance.py    # Tests for vector math and intercept logic
    └── test_tracker.py     # Tests for the smoothing tracker
```

---

## Setup

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
pip install -r requirements.txt
```

---

## Running the Simulation

```bash
python app.py
```

Then open your browser to: **http://127.0.0.1:8050**

### Pages

| URL | Description |
|-----|-------------|
| `/` | Live 3D simulation with controls |
| `/theory` | Math & physics explanation of the system |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Controls

| Control | Description |
|---------|-------------|
| Target Speed | How fast the target moves |
| Interceptor Speed | How fast the interceptor moves |
| Sensor Noise (σ) | Standard deviation of measurement noise |
| Timestep | Simulation time per update step |
| Show Trails | Toggle trail visualization |
| Kalman Filter | Toggle between Kalman filter and exponential smoothing |
| Start | Begin the simulation (starts paused) |
| Pause / Resume | Pause or resume the running simulation |
| Reset | Restart with fresh random conditions (pauses until Start) |

---

## Live Status Display

- **Time** — elapsed simulation time
- **Distance to target** — current interceptor-to-target distance
- **Tracking error** — distance between estimate and true target
- **Intercepts** — number of successful interceptions
- **Tracker** — which tracking algorithm is active

---

## Kalman Filter

The Kalman filter implementation uses a 6-state constant-velocity model:

- **State:** `[x, y, z, vx, vy, vz]`
- **Process model:** constant velocity with white noise acceleration
- **Measurement model:** direct position observation with Gaussian noise
- **Advantages over exponential smoothing:**
  - Optimal noise rejection given the model assumptions
  - Direct velocity estimation (no finite differencing lag)
  - Adapts gain automatically based on noise levels

Toggle it on via the checkbox in the controls panel to compare performance.

---

## Suggested Future Improvements

1. ~~Kalman Filter tracker~~ ✅ Implemented
2. **Missed detections** — simulate sensor dropouts where no measurement is available
3. **Sensor field of view** — limit sensor to a cone or range, requiring search behavior
4. **Multiple targets** — track and prioritize multiple moving objects
5. **Interceptor dynamics** — add acceleration limits and turning radius constraints
6. **3D arena obstacles** — add static or moving obstacles the interceptor must avoid
7. **Performance metrics** — plot tracking error and intercept rate over time
8. ~~Configurable tracker switching~~ ✅ Implemented
9. **Replay / recording** — save and replay simulation runs for analysis
10. **Extended Kalman Filter** — handle non-linear target motion models

---

## License

Educational use only. No real-world deployment intended.
