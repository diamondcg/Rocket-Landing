# CLAUDE.md

Guidance for AI assistants (and humans) working in this repository.

## Project Overview

Rocket Landing GNC (Guidance, Navigation & Control) simulation: a 1D
(vertical-axis only) closed-loop simulation of a rocket descending and
landing under thrust control. It models:

- **Dynamics**: rocket state `[position_z, velocity_z, mass]` propagated
  via Euler integration under gravity and thrust, with fuel consumption.
- **Navigation**: a linear Kalman Filter estimates `[z, v, mass]` from
  noisy altitude measurements.
- **Control**: a PID controller (with gravity feed-forward) computes a
  thrust command to drive `z -> 0`, `v -> 0`.
- **Simulation**: ties dynamics + estimator + controller together into a
  single closed-loop run.
- **Monte Carlo**: runs many randomized simulations to estimate landing
  success rate and error statistics.
- **Visualization**: replays a recorded simulation log in an SDL2/OpenGL
  window, rendering the rocket's altitude and thrust over time, with a
  diagnostics HUD (telemetry, force arrows, fuel gauge).

## Repository Layout

```
config.yaml        # All tunable parameters (sim, rocket, controller, estimator, noise, MC, visualization)
dynamics.py         # RocketDynamics: Euler-integrated equations of motion
controller.py       # PIDController: thrust command from state estimate
estimator.py        # KalmanFilter: linear KF over [z, v, mass]
simulation.py       # load_config() + run_simulation(): single closed-loop run
monte_carlo.py      # run_monte_carlo(): batch of randomized runs + stats
viz_transform.py    # pure altitude<->NDC/screen coordinate transforms
viz_geometry.py     # pure vertex generation (rocket, flame, ground, axis ticks, arrows, progress bars)
viz_diagnostics.py  # pure HUD diagnostic-value computation (forces, fuel fraction, ...)
viz_playback.py     # pure playback timing (PlaybackClock, index_for_time, ...)
viz_window.py       # GLWindow: SDL2 window + OpenGL context (thin, not unit tested)
viz_text.py         # TextRenderer: SDL_ttf text -> OpenGL texture (thin, not unit tested)
viz_renderer.py     # SceneRenderer: OpenGL draw calls + HUD overlay (thin, not unit tested)
visualize.py        # entry point: runs a simulation and plays it back
tests/test_gnc.py   # pytest suite covering dynamics/controller/estimator/simulation/MC
tests/test_viz.py   # pytest suite covering viz_transform/viz_geometry/viz_playback/viz_diagnostics/config
requirements.txt    # numpy, pyyaml, pytest, PySDL2, PyOpenGL, pysdl2-dll
```

## State & Conventions

- **State vector**: `x = [z, v, mass]` — altitude `z` [m] (positive up,
  ground at `z = 0`), vertical velocity `v` [m/s] (negative = descending),
  and vehicle `mass` [kg].
- **Control input**: `thrust` [N], scalar, clipped to
  `[thrust_min, thrust_max]`.
- **Equations of motion** (see `dynamics.py`):
  - `dz/dt = v`
  - `dv/dt = (thrust / m) - g`
  - `dm/dt = -alpha * thrust`
- Mass is clamped to a minimum of `1e-6` to avoid division by zero.
- Landing is detected when `state[0] <= 0.0`; the simulation log is
  truncated at that step.
- All physical constants (`gravity`, `alpha`, etc.) are passed as
  constructor parameters — never hard-code them inside module logic.
  They flow from `config.yaml` through `load_config()`.

## Configuration (`config.yaml`)

Single source of truth for all parameters, grouped by section:
`simulation`, `rocket`, `controller` (`pid` and `lqr` sub-blocks),
`estimator`, `noise`, `monte_carlo`, `visualization`.

The `visualization` section configures `visualize.py`: window size/title,
`fps`, `playback_speed`, `loop`, `view_padding_frac` (altitude view-bound
padding), rocket size in NDC units, a `hud` sub-block, and a `colors` map
(RGB triples in `[0, 1]`) for background/ground/rocket/flame/text/arrows/
fuel bar.

The `visualization.hud` sub-block configures the diagnostics overlay:
`enabled`, `font_path` (absolute path to a TrueType/OpenType font — by
default the Caskaydia Cove Nerd Font; this path is machine-specific and may
need updating on other systems), `font_size_px`, `text_height` and `margin`
(NDC), `arrow_max_force`/`arrow_max_length`/`arrow_shaft_width` (force-arrow
scaling, NDC), and `fuel_bar_width`/`fuel_bar_height` (NDC).

Note: `controller.type` includes an `lqr` option in the config schema,
but `simulation.py` currently always instantiates `PIDController` using
`controller.pid`. If adding LQR support, wire it through
`run_simulation` based on `ctrl_cfg["type"]`.

## Development Workflow

### Setup
```bash
pip install -r requirements.txt
```

The visualization depends on `pysdl2-dll`, which bundles prebuilt SDL2 /
SDL2_ttf shared libraries for Linux x86_64 — this avoids any system
`apt install`. Install dependencies into the project's `.venv` (gitignored)
rather than globally:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run a single simulation
```bash
python simulation.py
```
Prints landing status, final altitude, and step count using `seed=42`.

### Run the Monte Carlo campaign
```bash
python monte_carlo.py
```
Prints success rate and error statistics over `config.yaml`'s
`monte_carlo.n_runs` (default 1000) runs.

### Run the visualization
```bash
python visualize.py
```
Runs a simulation (`seed=42`) and replays it in an SDL2/OpenGL window:
rocket altitude over time, with an exhaust flame that scales with thrust,
and a diagnostics HUD overlay (top-left telemetry panel for altitude/
velocity/mass/time, bottom-left weight/thrust/net-force arrows with N
labels, right-edge fuel gauge). Close the window or press Esc to quit.

### Run tests
```bash
pytest
# or
pytest tests/test_gnc.py -v
pytest tests/test_viz.py -v
```

## Testing Conventions

- Tests live in `tests/test_gnc.py`, organized into one `Test*` class per
  module (`TestRocketDynamics`, `TestPIDController`, `TestKalmanFilter`,
  `TestSimulation`, `TestMonteCarlo`).
- Tests insert the project root onto `sys.path` so modules can be
  imported without packaging — keep all `.py` modules at the repo root
  (no `src/` layout).
- `_default_cfg()` loads `config.yaml` relative to the test file; use it
  when a test needs realistic, end-to-end parameters.
- Key invariants enforced by tests — preserve these when refactoring:
  - Thrust output always within `[thrust_min, thrust_max]`.
  - Mass never drops below `1e-6`.
  - `KalmanFilter.state` returns a copy (not a reference to internal state).
  - KF covariance `P` stays positive semi-definite.
  - Default simulation lands within `t_max` and final `|z| <= 0.1` m.
  - Monte Carlo with 1000 runs achieves `success_rate >= 0.8`.
- Tests for the visualization live in `tests/test_viz.py`, organized into
  `TestVizTransform`, `TestVizGeometry`, `TestVizPlayback`, `TestVizDiagnostics`,
  `TestVizConfig`. Only `viz_transform.py`, `viz_geometry.py`,
  `viz_playback.py`, and `viz_diagnostics.py` are imported/tested — they have
  no SDL/OpenGL dependency. `viz_window.py`, `viz_text.py`, and
  `viz_renderer.py` require a display/GL context (and SDL_ttf) and are not
  unit tested.

## Style Notes

- Numpy-style docstrings on classes/functions (see existing modules for
  the pattern) with explicit `Parameters`/`Returns` sections.
- Type hints throughout; `from __future__ import annotations` used where
  needed for `str | Path` style unions (project targets Python with PEP
  604 union syntax).
- Keep modules independent and importable on their own — `simulation.py`
  composes `dynamics.py`, `controller.py`, and `estimator.py`;
  `monte_carlo.py` composes `simulation.py`. Avoid circular imports.
