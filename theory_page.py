"""
theory_page.py
--------------
Layout for the Theory & Math page explaining the guidance and interception
system's underlying physics and mathematics.
"""

from dash import dcc, html


# ---------------------------------------------------------------------------
# Reusable styles
# ---------------------------------------------------------------------------

_SECTION_STYLE = {
    "backgroundColor": "#f9f9f9",
    "padding": "25px 30px",
    "borderRadius": "8px",
    "marginBottom": "25px",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
}

_EQ_STYLE = {
    "backgroundColor": "#fff",
    "border": "1px solid #e0e0e0",
    "borderRadius": "6px",
    "padding": "12px 18px",
    "margin": "12px 0",
    "fontFamily": "monospace",
    "fontSize": "15px",
    "overflowX": "auto",
}

_H2_STYLE = {"marginTop": "0", "color": "#2c3e50"}
_H3_STYLE = {"color": "#34495e", "marginTop": "20px"}


# ---------------------------------------------------------------------------
# Theory page layout
# ---------------------------------------------------------------------------

theory_layout = html.Div(
    style={"padding": "30px", "maxWidth": "900px", "margin": "0 auto"},
    children=[
        html.H1(
            "Theory: Math & Physics of Guidance and Interception",
            style={"textAlign": "center", "marginBottom": "10px", "color": "#2c3e50"},
        ),
        html.P(
            "This page explains the mathematical foundations behind each subsystem "
            "in the simulation: target motion, sensor modeling, state estimation "
            "(tracking), prediction, and interceptor guidance.",
            style={"textAlign": "center", "color": "#555", "marginBottom": "35px"},
        ),

        # ===================================================================
        # Section 1: Target Motion Model
        # ===================================================================
        html.Div(
            style=_SECTION_STYLE,
            children=[
                html.H2("1. Target Motion Model", style=_H2_STYLE),
                html.P(
                    "The target moves through 3D space with a constant speed but "
                    "a slowly drifting direction, producing smooth, organic trajectories."
                ),
                html.H3("Kinematics", style=_H3_STYLE),
                html.P("Position update (Euler integration):"),
                html.Div("p(t + Δt) = p(t) + v(t) · Δt", style=_EQ_STYLE),
                html.P("Velocity direction drifts each step:"),
                html.Div(
                    "v(t + Δt) = ‖v(t) + η(t)‖̂ · speed,   η ~ N(0, σ_drift²·I₃)",
                    style=_EQ_STYLE,
                ),
                html.P(
                    "where ‖·‖̂ denotes normalization to a unit vector. The drift "
                    "noise η introduces gradual direction changes while maintaining "
                    "constant speed magnitude."
                ),
                html.H3("Boundary Conditions", style=_H3_STYLE),
                html.P(
                    "The arena is a cube [−L, L]³. When the target reaches a wall, "
                    "the velocity component along that axis is reflected:"
                ),
                html.Div(
                    "if |p_i| ≥ L:  p_i ← L·sign(p_i),  v_i ← −v_i",
                    style=_EQ_STYLE,
                ),
            ],
        ),

        # ===================================================================
        # Section 2: Sensor Model
        # ===================================================================
        html.Div(
            style=_SECTION_STYLE,
            children=[
                html.H2("2. Sensor Model", style=_H2_STYLE),
                html.P(
                    "The sensor provides noisy position measurements of the target. "
                    "We model sensor noise as additive isotropic Gaussian noise:"
                ),
                html.Div("z(t) = p_target(t) + ε,   ε ~ N(0, σ²·I₃)", style=_EQ_STYLE),
                html.P([
                    "where ",
                    html.B("z"),
                    " is the measurement vector, ",
                    html.B("σ"),
                    " is the sensor noise standard deviation (controllable via the UI slider), "
                    "and I₃ is the 3×3 identity matrix."
                ]),
                html.P(
                    "This means each axis (x, y, z) is corrupted independently by "
                    "zero-mean Gaussian noise with the same variance σ². Higher σ "
                    "makes the raw measurements scatter further from the true position."
                ),
            ],
        ),

        # ===================================================================
        # Section 3: Exponential Smoothing Tracker
        # ===================================================================
        html.Div(
            style=_SECTION_STYLE,
            children=[
                html.H2("3. State Estimation — Exponential Smoothing", style=_H2_STYLE),
                html.P(
                    "The simplest tracker uses first-order exponential smoothing "
                    "(also called an Exponentially Weighted Moving Average)."
                ),
                html.H3("Position Estimate", style=_H3_STYLE),
                html.Div(
                    "ŝ(t) = α · z(t) + (1 − α) · ŝ(t−1)",
                    style=_EQ_STYLE,
                ),
                html.P([
                    html.B("α ∈ (0, 1]"),
                    " is the smoothing factor. α = 1 means no smoothing (follow "
                    "measurements exactly). α → 0 means heavy smoothing (slow response)."
                ]),
                html.H3("Velocity Estimate", style=_H3_STYLE),
                html.P("Velocity is derived from successive smoothed positions:"),
                html.Div("v̂(t) = (ŝ(t) − ŝ(t−1)) / Δt", style=_EQ_STYLE),
                html.H3("Trade-offs", style=_H3_STYLE),
                html.P(
                    "Exponential smoothing is computationally trivial and has a single "
                    "tuning parameter. However, it has no formal model of the system "
                    "dynamics, cannot optimally fuse process and measurement uncertainty, "
                    "and introduces lag proportional to (1 − α)."
                ),
            ],
        ),

        # ===================================================================
        # Section 4: Kalman Filter
        # ===================================================================
        html.Div(
            style=_SECTION_STYLE,
            children=[
                html.H2("4. State Estimation — Kalman Filter", style=_H2_STYLE),
                html.P(
                    "The Kalman filter is the optimal linear estimator for systems "
                    "with Gaussian noise. It maintains a full probabilistic state "
                    "estimate (mean + covariance) and recursively updates it."
                ),
                html.H3("State Vector", style=_H3_STYLE),
                html.Div("x = [x, y, z, vx, vy, vz]ᵀ  ∈ ℝ⁶", style=_EQ_STYLE),
                html.P(
                    "We track both position and velocity, giving the filter the "
                    "ability to predict ahead even without a new measurement."
                ),
                html.H3("Constant-Velocity Process Model", style=_H3_STYLE),
                html.P("State transition (prediction):"),
                html.Div(
                    children=[
                        html.Div("x(t|t−1) = F · x(t−1|t−1)", style=_EQ_STYLE),
                        html.Div(
                            "F = [[I₃, Δt·I₃], [0₃, I₃]]   (6×6 block matrix)",
                            style=_EQ_STYLE,
                        ),
                    ]
                ),
                html.P(
                    "This encodes the assumption that velocity is approximately "
                    "constant between steps: position advances by v·Δt."
                ),
                html.H3("Process Noise (Q)", style=_H3_STYLE),
                html.P(
                    "We use a piecewise-constant white noise acceleration model. "
                    "For each axis i, the 2×2 block is:"
                ),
                html.Div(
                    "Q_i = q · [[Δt⁴/4, Δt³/2], [Δt³/2, Δt²]]",
                    style=_EQ_STYLE,
                ),
                html.P(
                    "where q = σ_process² represents the variance of unmodeled "
                    "accelerations. Larger q tells the filter to trust measurements "
                    "more relative to the motion model."
                ),
                html.H3("Measurement Model", style=_H3_STYLE),
                html.Div(
                    children=[
                        html.Div("z = H · x + ε,   ε ~ N(0, R)", style=_EQ_STYLE),
                        html.Div("H = [I₃, 0₃]   (3×6 — observes position only)", style=_EQ_STYLE),
                        html.Div("R = σ_sensor² · I₃", style=_EQ_STYLE),
                    ]
                ),
                html.H3("Predict–Update Cycle", style=_H3_STYLE),
                html.P(html.B("Predict:")),
                html.Div(
                    children=[
                        html.Div("x⁻ = F · x", style=_EQ_STYLE),
                        html.Div("P⁻ = F · P · Fᵀ + Q", style=_EQ_STYLE),
                    ]
                ),
                html.P(html.B("Update (correction):")),
                html.Div(
                    children=[
                        html.Div("y = z − H · x⁻          (innovation)", style=_EQ_STYLE),
                        html.Div("S = H · P⁻ · Hᵀ + R     (innovation covariance)", style=_EQ_STYLE),
                        html.Div("K = P⁻ · Hᵀ · S⁻¹       (Kalman gain)", style=_EQ_STYLE),
                        html.Div("x = x⁻ + K · y           (corrected state)", style=_EQ_STYLE),
                        html.Div("P = (I − K · H) · P⁻     (corrected covariance)", style=_EQ_STYLE),
                    ]
                ),
                html.H3("Why It's Better", style=_H3_STYLE),
                html.P(
                    "The Kalman filter optimally balances process model predictions "
                    "against noisy measurements. When noise is high, the gain K shrinks "
                    "and the filter trusts its motion model more. When noise is low, "
                    "K grows and measurements dominate. It also directly estimates "
                    "velocity without finite-differencing, reducing lag."
                ),
            ],
        ),

        # ===================================================================
        # Section 5: Prediction (Linear Extrapolation)
        # ===================================================================
        html.Div(
            style=_SECTION_STYLE,
            children=[
                html.H2("5. Target Position Prediction", style=_H2_STYLE),
                html.P(
                    "To intercept a moving target, the interceptor must aim not at "
                    "where the target is now, but where it will be when the interceptor "
                    "arrives. We use linear extrapolation from the estimated state:"
                ),
                html.Div(
                    "p_predicted = p̂(t) + v̂(t) · t_intercept",
                    style=_EQ_STYLE,
                ),
                html.H3("Estimating Time-to-Intercept", style=_H3_STYLE),
                html.P(
                    "The prediction horizon is estimated from the current geometry:"
                ),
                html.Div(
                    "t_intercept = ‖p_interceptor − p̂_target‖ / speed_interceptor",
                    style=_EQ_STYLE,
                ),
                html.P(
                    "This is a first-order approximation — it assumes the interceptor "
                    "will close the current distance at its full speed. The prediction "
                    "time is clamped to a maximum value to prevent wild extrapolation "
                    "when the interceptor is far away."
                ),
                html.H3("Limitations", style=_H3_STYLE),
                html.P(
                    "Linear prediction works well for short horizons and targets with "
                    "slowly changing velocity. For highly maneuvering targets, higher-order "
                    "models (e.g., constant-acceleration or curvilinear prediction) would "
                    "improve accuracy at the cost of complexity."
                ),
            ],
        ),

        # ===================================================================
        # Section 6: Interceptor Guidance Law
        # ===================================================================
        html.Div(
            style=_SECTION_STYLE,
            children=[
                html.H2("6. Interceptor Guidance Law", style=_H2_STYLE),
                html.P(
                    "The interceptor uses a pursuit-style guidance law: it always "
                    "flies directly toward the predicted intercept point at maximum speed."
                ),
                html.H3("Direction Computation", style=_H3_STYLE),
                html.Div(
                    "d̂ = (p_predicted − p_interceptor) / ‖p_predicted − p_interceptor‖",
                    style=_EQ_STYLE,
                ),
                html.H3("Position Update", style=_H3_STYLE),
                html.Div(
                    "p_interceptor(t + Δt) = p_interceptor(t) + d̂ · speed_interceptor · Δt",
                    style=_EQ_STYLE,
                ),
                html.H3("Relation to Classical Guidance", style=_H3_STYLE),
                html.P(
                    "This is a form of deviated pursuit — rather than chasing the "
                    "target's current position (pure pursuit), the interceptor leads "
                    "the target by aiming at the predicted future position. This is "
                    "conceptually similar to Proportional Navigation (PN) guidance "
                    "used in real missile systems, though PN formally commands "
                    "acceleration proportional to the line-of-sight rotation rate:"
                ),
                html.Div(
                    "a_cmd = N · V_c · (dλ/dt)",
                    style=_EQ_STYLE,
                ),
                html.P([
                    "where ",
                    html.B("N"),
                    " is the navigation constant (typically 3–5), ",
                    html.B("V_c"),
                    " is the closing velocity, and ",
                    html.B("dλ/dt"),
                    " is the line-of-sight rate. Our simplified model achieves a "
                    "similar lead-pursuit effect through explicit position prediction."
                ]),
            ],
        ),

        # ===================================================================
        # Section 7: Intercept Detection
        # ===================================================================
        html.Div(
            style=_SECTION_STYLE,
            children=[
                html.H2("7. Intercept Detection", style=_H2_STYLE),
                html.P(
                    "An intercept is declared when the interceptor enters a sphere "
                    "of radius r centered on the target:"
                ),
                html.Div(
                    "‖p_interceptor − p_target‖ ≤ r",
                    style=_EQ_STYLE,
                ),
                html.P(
                    "The intercept radius r represents the effective kill/capture "
                    "radius. After a successful intercept, the interceptor resets "
                    "to the origin and begins pursuing the target again."
                ),
            ],
        ),

        # ===================================================================
        # Section 8: System Pipeline Summary
        # ===================================================================
        html.Div(
            style=_SECTION_STYLE,
            children=[
                html.H2("8. Full System Pipeline", style=_H2_STYLE),
                html.P("Each simulation timestep executes the following pipeline:"),
                html.Ol([
                    html.Li("Target motion update (kinematics + boundary bounce)"),
                    html.Li("Sensor measurement generation (true position + Gaussian noise)"),
                    html.Li("State estimation (exponential smoothing or Kalman filter)"),
                    html.Li("Future position prediction (linear extrapolation)"),
                    html.Li("Interceptor guidance (pursuit toward predicted point)"),
                    html.Li("Intercept detection (distance threshold check)"),
                ]),
                html.P(
                    "The interplay between sensor noise, tracker quality, and "
                    "prediction accuracy determines how effectively the interceptor "
                    "can close on the target. The Kalman filter's superior noise "
                    "rejection and velocity estimation typically yield faster and "
                    "more reliable intercepts, especially at high noise levels."
                ),
            ],
        ),
    ],
)
