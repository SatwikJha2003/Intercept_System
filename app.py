"""
app.py
------
Dash-based browser frontend for the 3D Object Tracking and Intercept Simulation.

Run with:
    python app.py

Then open http://127.0.0.1:8050 in your browser.
"""

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html

from config import (
    ARENA_SIZE,
    DEFAULT_INTERCEPTOR_SPEED,
    DEFAULT_SENSOR_NOISE,
    DEFAULT_TARGET_SPEED,
    TIMESTEP,
    UPDATE_INTERVAL_MS,
)
from simulation import SimulationState
from theory_page import theory_layout

# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------

app = Dash(__name__, title="3D Object Tracking & Intercept Simulation", suppress_callback_exceptions=True)

# Global simulation state
sim = SimulationState.create_default()
sim.paused = True  # Start paused — user clicks "Start" to begin

# ---------------------------------------------------------------------------
# Navigation bar
# ---------------------------------------------------------------------------

navbar = html.Div(
    style={
        "display": "flex",
        "gap": "20px",
        "padding": "12px 20px",
        "backgroundColor": "#2c3e50",
        "alignItems": "center",
    },
    children=[
        html.H3(
            "Intercept System",
            style={"color": "white", "margin": "0", "marginRight": "30px"},
        ),
        dcc.Link(
            "Simulation",
            href="/",
            style={"color": "#ecf0f1", "textDecoration": "none", "fontSize": "16px"},
        ),
        dcc.Link(
            "Theory & Math",
            href="/theory",
            style={"color": "#ecf0f1", "textDecoration": "none", "fontSize": "16px"},
        ),
    ],
)

# ---------------------------------------------------------------------------
# Simulation page layout
# ---------------------------------------------------------------------------

simulation_layout = html.Div(
    style={"padding": "20px"},
    children=[
        html.H1(
            "3D Object Tracking & Intercept Simulation",
            style={"textAlign": "center", "marginBottom": "10px"},
        ),
        html.P(
            "Educational simulation of target tracking, prediction, and interception in 3D space.",
            style={"textAlign": "center", "color": "#555", "marginBottom": "20px"},
        ),
        # Main content: graph + controls side by side
        html.Div(
            style={"display": "flex", "gap": "20px", "flexWrap": "wrap"},
            children=[
                # 3D Graph
                html.Div(
                    style={"flex": "3", "minWidth": "500px"},
                    children=[
                        dcc.Graph(
                            id="graph-3d",
                            style={"height": "650px"},
                            config={"displayModeBar": True},
                        ),
                    ],
                ),
                # Controls panel
                html.Div(
                    style={
                        "flex": "1",
                        "minWidth": "260px",
                        "backgroundColor": "#f9f9f9",
                        "padding": "20px",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                    },
                    children=[
                        html.H3("Controls", style={"marginTop": "0"}),
                        # Target speed
                        html.Label("Target Speed"),
                        dcc.Slider(
                            id="slider-target-speed",
                            min=1,
                            max=50,
                            step=1,
                            value=DEFAULT_TARGET_SPEED,
                            marks={1: "1", 25: "25", 50: "50"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Br(),
                        # Interceptor speed
                        html.Label("Interceptor Speed"),
                        dcc.Slider(
                            id="slider-interceptor-speed",
                            min=1,
                            max=80,
                            step=1,
                            value=DEFAULT_INTERCEPTOR_SPEED,
                            marks={1: "1", 40: "40", 80: "80"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Br(),
                        # Sensor noise
                        html.Label("Sensor Noise (σ)"),
                        dcc.Slider(
                            id="slider-sensor-noise",
                            min=0,
                            max=20,
                            step=0.5,
                            value=DEFAULT_SENSOR_NOISE,
                            marks={0: "0", 10: "10", 20: "20"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Br(),
                        # Timestep
                        html.Label("Timestep (s)"),
                        dcc.Slider(
                            id="slider-timestep",
                            min=0.01,
                            max=0.5,
                            step=0.01,
                            value=TIMESTEP,
                            marks={0.01: "0.01", 0.25: "0.25", 0.5: "0.5"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Br(),
                        # Show trails toggle
                        dcc.Checklist(
                            id="toggle-trails",
                            options=[{"label": " Show Trails", "value": "trails"}],
                            value=["trails"],
                            style={"marginBottom": "15px"},
                        ),
                        # Kalman filter toggle
                        dcc.Checklist(
                            id="toggle-kalman",
                            options=[{"label": " Kalman Filter", "value": "kalman"}],
                            value=[],
                            style={"marginBottom": "15px"},
                        ),
                        # Buttons
                        html.Div(
                            style={"display": "flex", "gap": "10px", "marginTop": "10px"},
                            children=[
                                html.Button(
                                    "Start",
                                    id="btn-start",
                                    n_clicks=0,
                                    style={
                                        "flex": "1",
                                        "padding": "8px",
                                        "cursor": "pointer",
                                        "backgroundColor": "#27ae60",
                                        "color": "white",
                                        "border": "none",
                                        "borderRadius": "4px",
                                    },
                                ),
                                html.Button(
                                    "Pause",
                                    id="btn-pause",
                                    n_clicks=0,
                                    style={
                                        "flex": "1",
                                        "padding": "8px",
                                        "cursor": "pointer",
                                    },
                                ),
                                html.Button(
                                    "Reset",
                                    id="btn-reset",
                                    n_clicks=0,
                                    style={
                                        "flex": "1",
                                        "padding": "8px",
                                        "cursor": "pointer",
                                    },
                                ),
                            ],
                        ),
                        html.Hr(),
                        # Status display
                        html.H3("Status", style={"marginBottom": "10px"}),
                        html.Div(id="status-display"),
                    ],
                ),
            ],
        ),
        # Interval timer for animation
        dcc.Interval(
            id="interval-timer",
            interval=UPDATE_INTERVAL_MS,
            n_intervals=0,
        ),
    ],
)

# ---------------------------------------------------------------------------
# App layout with routing
# ---------------------------------------------------------------------------

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif"},
    children=[
        dcc.Location(id="url", refresh=False),
        navbar,
        html.Div(id="page-content"),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def display_page(pathname):
    """Route to the correct page based on URL."""
    if pathname == "/theory":
        return theory_layout
    return simulation_layout


@app.callback(
    Output("graph-3d", "figure"),
    Output("status-display", "children"),
    Output("btn-pause", "children"),
    Input("interval-timer", "n_intervals"),
    Input("btn-start", "n_clicks"),
    Input("btn-reset", "n_clicks"),
    Input("btn-pause", "n_clicks"),
    State("slider-target-speed", "value"),
    State("slider-interceptor-speed", "value"),
    State("slider-sensor-noise", "value"),
    State("slider-timestep", "value"),
    State("toggle-trails", "value"),
    State("toggle-kalman", "value"),
)
def update_simulation(
    n_intervals,
    start_clicks,
    reset_clicks,
    pause_clicks,
    target_speed,
    interceptor_speed,
    sensor_noise,
    timestep,
    trail_toggle,
    kalman_toggle,
):
    """Main callback: advances simulation and updates the 3D plot + status."""
    global sim

    triggered = callback_context.triggered
    trigger_id = triggered[0]["prop_id"].split(".")[0] if triggered else ""

    # Handle start
    if trigger_id == "btn-start":
        sim.paused = False

    # Handle reset
    if trigger_id == "btn-reset":
        sim.reset()
        sim.paused = True  # Pause after reset so user must click Start again

    # Handle pause toggle
    if trigger_id == "btn-pause":
        sim.paused = not sim.paused

    # Update runtime parameters from sliders
    sim.target_speed = target_speed
    sim.interceptor_speed = interceptor_speed
    sim.sensor_noise = sensor_noise
    sim.dt = timestep
    sim.tracker.dt = timestep

    # Handle Kalman filter toggle
    use_kalman = "kalman" in (kalman_toggle or [])
    sim.set_tracker(use_kalman)

    # Advance simulation
    sim.step()

    # Build figure
    show_trails = "trails" in (trail_toggle or [])
    fig = build_figure(sim, show_trails)

    # Build status text
    status = build_status(sim)

    # Pause button label
    pause_label = "Resume" if sim.paused else "Pause"

    return fig, status, pause_label


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------


def build_figure(state: SimulationState, show_trails: bool) -> go.Figure:
    """Construct the Plotly 3D scatter figure from current simulation state."""
    data = []

    # Target marker
    data.append(
        go.Scatter3d(
            x=[state.target_pos[0]],
            y=[state.target_pos[1]],
            z=[state.target_pos[2]],
            mode="markers",
            marker=dict(size=8, color="red"),
            name="Target",
        )
    )

    # Interceptor marker
    data.append(
        go.Scatter3d(
            x=[state.interceptor_pos[0]],
            y=[state.interceptor_pos[1]],
            z=[state.interceptor_pos[2]],
            mode="markers",
            marker=dict(size=8, color="blue", symbol="diamond"),
            name="Interceptor",
        )
    )

    # Sensor measurement marker
    data.append(
        go.Scatter3d(
            x=[state.sensor_measurement[0]],
            y=[state.sensor_measurement[1]],
            z=[state.sensor_measurement[2]],
            mode="markers",
            marker=dict(size=5, color="orange", opacity=0.7),
            name="Sensor Measurement",
        )
    )

    # Estimated position marker
    data.append(
        go.Scatter3d(
            x=[state.estimated_pos[0]],
            y=[state.estimated_pos[1]],
            z=[state.estimated_pos[2]],
            mode="markers",
            marker=dict(size=6, color="green"),
            name="Estimated Position",
        )
    )

    # Predicted intercept point
    data.append(
        go.Scatter3d(
            x=[state.predicted_intercept_pos[0]],
            y=[state.predicted_intercept_pos[1]],
            z=[state.predicted_intercept_pos[2]],
            mode="markers",
            marker=dict(size=6, color="purple", symbol="x"),
            name="Predicted Intercept",
        )
    )

    # Trails
    if show_trails and len(state.target_trail) > 1:
        trail_arr = np.array(state.target_trail)
        data.append(
            go.Scatter3d(
                x=trail_arr[:, 0],
                y=trail_arr[:, 1],
                z=trail_arr[:, 2],
                mode="lines",
                line=dict(color="red", width=2),
                opacity=0.4,
                name="Target Trail",
                showlegend=False,
            )
        )

    if show_trails and len(state.interceptor_trail) > 1:
        trail_arr = np.array(state.interceptor_trail)
        data.append(
            go.Scatter3d(
                x=trail_arr[:, 0],
                y=trail_arr[:, 1],
                z=trail_arr[:, 2],
                mode="lines",
                line=dict(color="blue", width=2),
                opacity=0.4,
                name="Interceptor Trail",
                showlegend=False,
            )
        )

    if show_trails and len(state.estimated_trail) > 1:
        trail_arr = np.array(state.estimated_trail)
        data.append(
            go.Scatter3d(
                x=trail_arr[:, 0],
                y=trail_arr[:, 1],
                z=trail_arr[:, 2],
                mode="lines",
                line=dict(color="green", width=2),
                opacity=0.3,
                name="Estimate Trail",
                showlegend=False,
            )
        )

    fig = go.Figure(data=data)

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-ARENA_SIZE, ARENA_SIZE], title="X"),
            yaxis=dict(range=[-ARENA_SIZE, ARENA_SIZE], title="Y"),
            zaxis=dict(range=[-ARENA_SIZE, ARENA_SIZE], title="Z"),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(x=0.01, y=0.99),
        uirevision="constant",  # Preserve camera angle between updates
    )

    return fig


# ---------------------------------------------------------------------------
# Status builder
# ---------------------------------------------------------------------------


def build_status(state: SimulationState) -> html.Div:
    """Create the status display showing live metrics."""
    dist = state.get_distance_to_target()
    error = state.get_tracking_error()

    return html.Div(
        [
            html.P(f"⏱ Time: {state.time:.1f} s"),
            html.P(f"📏 Distance to target: {dist:.1f}"),
            html.P(f"📡 Tracking error: {error:.1f}"),
            html.P(f"🎯 Intercepts: {state.intercept_count}"),
            html.P(f"🔬 Tracker: {'Kalman Filter' if state.use_kalman else 'Exp. Smoothing'}"),
            html.P(f"{'⏸ PAUSED' if state.paused else '▶ RUNNING'}"),
        ]
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting 3D Object Tracking & Intercept Simulation...")
    print("Open http://127.0.0.1:8050 in your browser.")
    app.run(debug=False, host="127.0.0.1", port=8050)
