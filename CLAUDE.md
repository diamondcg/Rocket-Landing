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

## Repository Layout

```
config.yaml        # All tunable parameters (sim, rocket, controller, estimator, noise, MC)
dynamics.py         # RocketDynamics: Euler-integrated equations of motion
controller.py       # PIDController: thrust command from state estimate
estimator.py        # KalmanFilter: linear KF over [z, v, mass]
simulation.py       # load_config() + run_simulation(): single closed-loop run
monte_carlo.py      # run_monte_carlo(): batch of randomized runs + stats
tests/test_gnc.py   # pytest suite covering all modules above
requirements.txt    # numpy, pyyaml, pytest
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
`estimator`, `noise`, `monte_carlo`.

Note: `controller.type` includes an `lqr` option in the config schema,
but `simulation.py` currently always instantiates `PIDController` using
`controller.pid`. If adding LQR support, wire it through
`run_simulation` based on `ctrl_cfg["type"]`.

## Development Workflow

### Setup
```bash
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

### Run tests
```bash
pytest
# or
pytest tests/test_gnc.py -v
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

## Style Notes

- Numpy-style docstrings on classes/functions (see existing modules for
  the pattern) with explicit `Parameters`/`Returns` sections.
- Type hints throughout; `from __future__ import annotations` used where
  needed for `str | Path` style unions (project targets Python with PEP
  604 union syntax).
- Keep modules independent and importable on their own — `simulation.py`
  composes `dynamics.py`, `controller.py`, and `estimator.py`;
  `monte_carlo.py` composes `simulation.py`. Avoid circular imports.
