# Rocket-Landing

Rocket Landing GNC Simulation

A 1D (vertical-axis) closed-loop simulation of a rocket descending and
landing under thrust control. The project models the full
Guidance, Navigation & Control (GNC) loop:

- **Dynamics** (`dynamics.py`): Euler-integrated equations of motion for
  state `[position_z, velocity_z, mass]` under gravity, thrust, and fuel
  consumption.
- **Navigation** (`estimator.py`): a linear Kalman Filter estimates
  `[z, v, mass]` from noisy altitude measurements.
- **Control** (`controller.py`): a PID controller with gravity
  feed-forward computes a thrust command to drive `z -> 0`, `v -> 0`.
- **Simulation** (`simulation.py`): ties dynamics, estimator, and
  controller together into a single closed-loop run.
- **Monte Carlo** (`monte_carlo.py`): runs many randomized simulations to
  estimate landing success rate and error statistics.

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt` (`numpy`, `pyyaml`, `pytest`)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Run a single simulation

```bash
python simulation.py
```

This loads `config.yaml`, runs a single closed-loop landing simulation
with a fixed random seed, and prints whether the rocket landed, the
final altitude, and the number of simulation steps.

### Run the Monte Carlo campaign

```bash
python monte_carlo.py
```

This runs a batch of randomized landing simulations (default 1000, see
`config.yaml`) with randomized initial conditions and noise seeds, and
prints the success rate and landing-error statistics.

### Run the test suite

```bash
pytest
```

## Configuration

All tunable parameters live in `config.yaml`, grouped by section:

- `simulation` — time step (`dt`) and max simulation time (`t_max`)
- `rocket` — gravity, fuel-consumption coefficient (`alpha`), masses,
  initial altitude/velocity, and thrust limits
- `controller` — controller type and gains (`pid` block; `lqr` block is
  reserved for future use)
- `estimator` — Kalman Filter process/measurement noise and initial
  covariance
- `noise` — measurement noise standard deviation
- `monte_carlo` — number of runs, initial-condition distributions, and
  the landing success threshold

## State Vector & Conventions

- **State**: `x = [z, v, mass]` — altitude `z` [m] (positive up, ground
  at `z = 0`), vertical velocity `v` [m/s] (negative = descending), and
  vehicle `mass` [kg].
- **Control input**: `thrust` [N], clipped to `[thrust_min, thrust_max]`.
- **Equations of motion**:
  - `dz/dt = v`
  - `dv/dt = (thrust / m) - g`
  - `dm/dt = -alpha * thrust`
- Landing is detected when `z <= 0`; the simulation log is truncated at
  that step.

## Project Layout

```
config.yaml        # All tunable parameters
dynamics.py         # RocketDynamics: equations of motion
controller.py       # PIDController: thrust command from state estimate
estimator.py        # KalmanFilter: linear KF over [z, v, mass]
simulation.py       # load_config() + run_simulation(): single closed-loop run
monte_carlo.py      # run_monte_carlo(): batch of randomized runs + stats
tests/test_gnc.py   # pytest suite covering all modules above
requirements.txt    # Python dependencies
```

For more detailed development conventions and guidance for AI assistants,
see [CLAUDE.md](CLAUDE.md).
