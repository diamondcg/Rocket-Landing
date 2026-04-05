"""Simulation module.

Runs a single closed-loop rocket-landing simulation.

Order per step:
  1. Propagate dynamics
  2. Generate noisy measurement
  3. Update estimator (Kalman Filter)
  4. Compute control
  5. Log data
"""

from __future__ import annotations

import yaml
import numpy as np
from pathlib import Path

from dynamics import RocketDynamics
from controller import PIDController
from estimator import KalmanFilter


def load_config(path: str | Path = "config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def run_simulation(
    cfg: dict,
    z_init: float | None = None,
    v_init: float | None = None,
    mass_init: float | None = None,
    seed: int | None = None,
) -> dict:
    """Run a single closed-loop landing simulation.

    Parameters
    ----------
    cfg:
        Configuration dictionary (from ``load_config``).
    z_init:
        Override initial altitude [m].
    v_init:
        Override initial velocity [m/s].
    mass_init:
        Override initial mass [kg].
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict
        Simulation log with keys:
        ``time``, ``z``, ``v``, ``mass``, ``z_hat``, ``v_hat``,
        ``thrust``, ``landed``, ``final_z``.
    """
    rng = np.random.default_rng(seed)

    sim_cfg = cfg["simulation"]
    rocket_cfg = cfg["rocket"]
    ctrl_cfg = cfg["controller"]
    est_cfg = cfg["estimator"]
    noise_cfg = cfg["noise"]

    dt = sim_cfg["dt"]
    t_max = sim_cfg["t_max"]
    n_steps = int(t_max / dt)

    # Initial conditions
    z0 = z_init if z_init is not None else rocket_cfg["z_init"]
    v0 = v_init if v_init is not None else rocket_cfg["v_init"]
    m0 = mass_init if mass_init is not None else rocket_cfg["mass_init"]
    state = np.array([z0, v0, m0])

    # Dynamics
    dynamics = RocketDynamics(
        gravity=rocket_cfg["gravity"],
        alpha=rocket_cfg["alpha"],
        dt=dt,
    )

    # Controller
    pid_cfg = ctrl_cfg["pid"]
    controller = PIDController(
        Kp=pid_cfg["Kp"],
        Ki=pid_cfg["Ki"],
        Kd=pid_cfg["Kd"],
        thrust_min=rocket_cfg["thrust_min"],
        thrust_max=rocket_cfg["thrust_max"],
        dt=dt,
    )

    # Estimator
    Q_diag = est_cfg["Q_proc"]
    Q = np.diag(Q_diag)
    R = est_cfg["R_meas"]
    P_init = np.diag(est_cfg["P_init"])
    kf = KalmanFilter(dt=dt, Q=Q, R=R, P_init=P_init)
    kf.initialise(state)

    meas_std = noise_cfg["measurement_std"]

    # Logging
    times = np.zeros(n_steps)
    z_log = np.zeros(n_steps)
    v_log = np.zeros(n_steps)
    mass_log = np.zeros(n_steps)
    z_hat_log = np.zeros(n_steps)
    v_hat_log = np.zeros(n_steps)
    thrust_log = np.zeros(n_steps)

    thrust = 0.0
    landed = False
    landing_step = n_steps

    for k in range(n_steps):
        t = k * dt
        times[k] = t
        z_log[k] = state[0]
        v_log[k] = state[1]
        mass_log[k] = state[2]
        z_hat_log[k] = kf.state[0]
        v_hat_log[k] = kf.state[1]
        thrust_log[k] = thrust

        # Check landing condition
        if state[0] <= 0.0 and not landed:
            landed = True
            landing_step = k
            break

        # 1. Propagate dynamics
        state = dynamics.step(state, thrust)

        # 2. Noisy measurement
        measurement = state[0] + rng.normal(0.0, meas_std)

        # 3. Estimator predict + update
        kf.predict(thrust)
        kf.update(measurement)

        # 4. Compute control
        thrust = controller.compute(kf.state)

    final_z = state[0]

    return {
        "time": times[:landing_step],
        "z": z_log[:landing_step],
        "v": v_log[:landing_step],
        "mass": mass_log[:landing_step],
        "z_hat": z_hat_log[:landing_step],
        "v_hat": v_hat_log[:landing_step],
        "thrust": thrust_log[:landing_step],
        "landed": landed,
        "final_z": final_z,
    }


if __name__ == "__main__":
    cfg = load_config()
    result = run_simulation(cfg, seed=42)
    print(f"Landed: {result['landed']}")
    print(f"Final altitude: {result['final_z']:.4f} m")
    print(f"Simulation steps: {len(result['time'])}")
