"""Rocket dynamics module.

State vector: x = [position_z, velocity_z, mass]
Input: u = thrust [N]

Equations of motion:
  dz/dt  =  v
  dv/dt  =  (u / m) - g
  dm/dt  = -alpha * u
"""

import numpy as np


class RocketDynamics:
    """Deterministic 2D vertical rocket dynamics."""

    def __init__(self, gravity: float = 9.81, alpha: float = 0.0005,
                 dt: float = 0.01) -> None:
        """Initialise dynamics parameters.

        Parameters
        ----------
        gravity:
            Gravitational acceleration [m/s^2].
        alpha:
            Fuel-consumption coefficient [kg/(N·s)].
        dt:
            Integration time step [s].
        """
        self.g = gravity
        self.alpha = alpha
        self.dt = dt

    def step(self, state: np.ndarray, thrust: float) -> np.ndarray:
        """Propagate the state by one time step using Euler integration.

        Parameters
        ----------
        state:
            Current state [z, v, mass].
        thrust:
            Thrust command [N].

        Returns
        -------
        np.ndarray
            Next state [z, v, mass].
        """
        z, v, m = state

        # Guard against zero/negative mass
        m = max(m, 1e-6)

        dz = v
        dv = (thrust / m) - self.g
        dm = -self.alpha * thrust

        z_next = z + dz * self.dt
        v_next = v + dv * self.dt
        m_next = m + dm * self.dt

        # Mass cannot drop below a small positive value
        m_next = max(m_next, 1e-6)

        return np.array([z_next, v_next, m_next])
