"""Kalman Filter estimator module.

State: x = [position_z, velocity_z, mass]
Measurement: y = position_z + noise
"""

import numpy as np


class KalmanFilter:
    """Linear Kalman Filter for the vertical rocket landing problem.

    The dynamics are linearised around constant mass for the prediction step.
    Mass is treated as a near-constant with small process noise.

    Parameters
    ----------
    dt:
        Time step [s].
    Q:
        Process-noise covariance matrix (3×3).
    R:
        Measurement-noise variance (scalar).
    P_init:
        Initial state-error covariance matrix (3×3).
    """

    def __init__(self, dt: float, Q: np.ndarray, R: float,
                 P_init: np.ndarray) -> None:
        self.dt = dt
        self.Q = np.asarray(Q, dtype=float)
        self.R = float(R)

        # State transition matrix (z, v, mass) – mass is modelled as constant
        self.F = np.array([
            [1.0, dt, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        # Observation matrix: we observe position only
        self.H = np.array([[1.0, 0.0, 0.0]])

        self.x_hat: np.ndarray = np.zeros(3)
        self.P: np.ndarray = np.asarray(P_init, dtype=float).copy()

    def initialise(self, x0: np.ndarray) -> None:
        """Set the initial state estimate."""
        self.x_hat = np.asarray(x0, dtype=float).copy()

    def predict(self, thrust: float) -> None:
        """Prediction step.

        Parameters
        ----------
        thrust:
            Thrust applied during the previous step [N].
        """
        m = max(self.x_hat[2], 1e-6)
        g = 9.81
        alpha = 0.0005

        # Non-linear prediction of state mean
        z = self.x_hat[0] + self.x_hat[1] * self.dt
        v = self.x_hat[1] + ((thrust / m) - g) * self.dt
        mass = max(self.x_hat[2] - alpha * thrust * self.dt, 1e-6)
        self.x_hat = np.array([z, v, mass])

        # Covariance prediction using linearised F
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement: float) -> None:
        """Measurement update step.

        Parameters
        ----------
        measurement:
            Noisy position measurement [m].
        """
        y = measurement - (self.H @ self.x_hat)[0]  # innovation
        S = (self.H @ self.P @ self.H.T)[0, 0] + self.R  # innovation covariance
        K = (self.P @ self.H.T) / S  # Kalman gain (3×1)

        self.x_hat = self.x_hat + K[:, 0] * y
        I_KH = np.eye(3) - K @ self.H
        self.P = I_KH @ self.P

    @property
    def state(self) -> np.ndarray:
        """Current state estimate [z, v, mass]."""
        return self.x_hat.copy()
