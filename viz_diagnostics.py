"""Diagnostic-value computation for the simulation visualization HUD.

Pure math only -- no SDL/OpenGL dependencies, so this module is safe to
import and unit test without a display or GL context.
"""

from __future__ import annotations


def compute_diagnostics(z: float, v: float, mass: float, thrust: float,
                         t: float, gravity: float, mass_dry: float,
                         mass_init: float, thrust_max: float) -> dict:
    """Compute HUD diagnostic values for one simulation step.

    Parameters
    ----------
    z:
        Altitude [m].
    v:
        Vertical velocity [m/s] (negative = descending).
    mass:
        Current total vehicle mass (dry + fuel) [kg].
    thrust:
        Current thrust command [N].
    t:
        Simulation time [s].
    gravity:
        Gravitational acceleration [m/s^2].
    mass_dry:
        Dry vehicle mass (no fuel) [kg].
    mass_init:
        Initial total mass (dry + fuel) [kg].
    thrust_max:
        Maximum thrust [N].

    Returns
    -------
    dict
        Diagnostic values:
        ``altitude``, ``velocity``, ``speed``, ``time``, ``mass``,
        ``fuel_mass``, ``fuel_frac``, ``weight``, ``thrust_force``,
        ``net_force``, ``thrust_max``.
    """
    fuel_mass = mass - mass_dry
    fuel_capacity = mass_init - mass_dry
    fuel_frac = fuel_mass / fuel_capacity if fuel_capacity > 0 else 0.0
    fuel_frac = max(0.0, min(fuel_frac, 1.0))

    weight = mass * gravity
    net_force = thrust - weight

    return {
        "altitude": z,
        "velocity": v,
        "speed": abs(v),
        "time": t,
        "mass": mass,
        "fuel_mass": fuel_mass,
        "fuel_frac": fuel_frac,
        "weight": weight,
        "thrust_force": thrust,
        "net_force": net_force,
        "thrust_max": thrust_max,
    }
