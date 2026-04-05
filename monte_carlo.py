"""Monte Carlo module.

Runs >= 1000 landing simulations with randomised initial conditions and
noise seeds, then reports success rate and error distributions.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any

from simulation import load_config, run_simulation


def run_monte_carlo(
    cfg: dict | None = None,
    config_path: str | Path = "config.yaml",
    n_runs: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Run a Monte Carlo campaign of landing simulations.

    Parameters
    ----------
    cfg:
        Pre-loaded configuration dictionary.  If *None*, the file at
        ``config_path`` is loaded automatically.
    config_path:
        Path to the YAML configuration file (used when ``cfg`` is None).
    n_runs:
        Override the number of runs from the config.
    seed:
        Master random seed for reproducibility.

    Returns
    -------
    dict
        Results dictionary with keys:
        ``n_runs``, ``success_rate``, ``final_z_errors``,
        ``success_count``, ``failed_count``.
    """
    if cfg is None:
        cfg = load_config(config_path)

    mc_cfg = cfg["monte_carlo"]
    n = n_runs if n_runs is not None else mc_cfg["n_runs"]
    threshold = mc_cfg["success_position_threshold"]

    rng = np.random.default_rng(seed)

    z_samples = rng.normal(mc_cfg["z_init_mean"], mc_cfg["z_init_std"], n)
    v_samples = rng.normal(mc_cfg["v_init_mean"], mc_cfg["v_init_std"], n)
    mass_samples = rng.normal(mc_cfg["mass_init_mean"], mc_cfg["mass_init_std"], n)

    final_z_errors: list[float] = []
    success_count = 0

    for i in range(n):
        result = run_simulation(
            cfg,
            z_init=float(z_samples[i]),
            v_init=float(v_samples[i]),
            mass_init=float(max(mass_samples[i], cfg["rocket"]["mass_dry"] + 0.1)),
            seed=int(rng.integers(0, 2**31)),
        )
        final_z = abs(result["final_z"])
        final_z_errors.append(final_z)

        if result["landed"] and final_z <= threshold:
            success_count += 1

    errors = np.array(final_z_errors)
    success_rate = success_count / n

    return {
        "n_runs": n,
        "success_count": success_count,
        "failed_count": n - success_count,
        "success_rate": success_rate,
        "final_z_errors": errors,
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "max_error": float(np.max(errors)),
    }


if __name__ == "__main__":
    cfg = load_config()
    print("Running Monte Carlo simulation...")
    results = run_monte_carlo(cfg)
    print(f"Runs:          {results['n_runs']}")
    print(f"Successes:     {results['success_count']}")
    print(f"Success rate:  {results['success_rate'] * 100:.1f}%")
    print(f"Mean |z| error: {results['mean_error']:.4f} m")
    print(f"Std  |z| error: {results['std_error']:.4f} m")
    print(f"Max  |z| error: {results['max_error']:.4f} m")
