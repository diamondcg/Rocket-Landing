"""Controller module.

Provides a PID controller that commands thrust to drive the rocket to z = 0
with v = 0.
"""

import numpy as np


class PIDController:
    """PID controller for vertical rocket landing.

    The controller targets z = 0, v = 0.  The error is defined as the
    negative of the current altitude so that a positive altitude produces a
    positive (upward) thrust demand.

    Parameters
    ----------
    Kp, Ki, Kd:
        Proportional, integral, and derivative gains.
    thrust_min, thrust_max:
        Saturation bounds for the thrust output [N].
    dt:
        Control time step [s].
    """

    def __init__(self, Kp: float, Ki: float, Kd: float,
                 thrust_min: float = 0.0, thrust_max: float = 500.0,
                 dt: float = 0.01, gravity: float = 9.81) -> None:
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.thrust_min = thrust_min
        self.thrust_max = thrust_max
        self.dt = dt
        self.gravity = gravity

        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._first_step: bool = True

    def reset(self) -> None:
        """Reset internal integrator and derivative state."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._first_step = True

    def compute(self, state_estimate: np.ndarray) -> float:
        """Compute the thrust command from the estimated state.

        Parameters
        ----------
        state_estimate:
            Estimated state [z, v, mass].

        Returns
        -------
        float
            Clipped thrust command [N].
        """
        z, v, m = state_estimate

        # Error: we want z → 0
        error = -z  # positive when above ground

        self._integral += error * self.dt

        if self._first_step:
            derivative = 0.0
            self._first_step = False
        else:
            derivative = (error - self._prev_error) / self.dt

        self._prev_error = error

        # Gravity feed-forward so the rocket can hover at rest
        gravity_ff = m * self.gravity

        u = gravity_ff + self.Kp * error + self.Ki * self._integral + self.Kd * derivative
        return float(np.clip(u, self.thrust_min, self.thrust_max))
