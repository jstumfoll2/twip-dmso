"""General-purpose state estimators (the thesis DMSO observers are in :mod:`twip.legacy.observers`).

* :class:`LinearKalman` - standard discrete Kalman filter on the linear model.
* :class:`AngleBiasKalman` - 2-state tilt/gyro-bias Kalman (``kalmanFilter.m``,
  ``twip_v4/filters.ino``).
* :func:`complementary_filter` - ``complementaryFilter.m``.
"""

from __future__ import annotations

import numpy as np

RAD2DEG = 180 / np.pi


class LinearKalman:
    """Discrete Kalman filter; the update order matches the thesis scripts.

    ``P- = F P F' + Q``; ``K = P- H' (H P- H' + R)^-1``; ``x = F x + G u``;
    ``x += K (y - H x)``; ``P = (I - K H) P-``.

    ``R`` is added with numpy broadcasting, as MATLAB does.  A scalar ``R``
    therefore adds to *every* element of ``H P H'``.  That reproduces
    ``lqrkalmantest1v2.m``, where ``R = .1`` (meant for another filter)
    overwrites the 4x4 R before the loop.
    """

    def __init__(self, F, G, Q, R, H=None, x0=None, P0=None):
        n = F.shape[0]
        self.F, self.G, self.Q, self.R = F, G.reshape(n), Q, R
        self.H = H if H is not None else np.eye(n)
        self.x = np.zeros(n) if x0 is None else np.asarray(x0, float)
        self.P = 0.001 * np.eye(n) if P0 is None else P0

    def step(self, y: np.ndarray, u: float = 0.0) -> np.ndarray:
        F, H = self.F, self.H
        Pm = F @ self.P @ F.T + self.Q
        S = H @ Pm @ H.T + self.R
        K = Pm @ H.T @ np.linalg.inv(S)
        x = F @ self.x + self.G * u
        self.x = x + K @ (y - H @ x)
        self.P = (np.eye(len(x)) - K @ H) @ Pm
        return self.x.copy()


class AngleBiasKalman:
    """Tilt angle + gyro bias Kalman filter (``kalmanFilter.m`` / ``filters.ino``).

    Works in degrees.  With ``literal=True`` the covariance update is a literal port of the
    in-place C code.  ``P[1][0]`` and ``P[1][1]`` are updated using the
    *already updated* ``P[0][0]`` and ``P[0][1]``, which differs from the
    textbook ``(I - K H) P``.  The default is the textbook update; use ``literal=True``
    to reproduce the robot logs.
    """

    def __init__(
        self,
        Q_angle: float = 0.001,
        Q_gyro_bias: float = 0.003,
        R_angle: float = 0.005,
        literal: bool = False,
    ):
        self.Q_angle, self.Q_gyro_bias, self.R_angle = Q_angle, Q_gyro_bias, R_angle
        self.literal = literal
        self.angle = 0.0
        self.bias = 0.0
        self.P = 0.01 * np.eye(2)

    def step(self, gy: float, ax: float, az: float, dt: float) -> tuple[float, float]:
        P = self.P
        self.angle += dt * (gy - self.bias)
        P[0, 0] += dt * (dt * P[1, 1] - P[0, 1] - P[1, 0] + self.Q_angle)
        P[0, 1] -= dt * P[1, 1]
        P[1, 0] -= dt * P[1, 1]
        P[1, 1] += self.Q_gyro_bias * dt

        S = P[0, 0] + self.R_angle
        K0, K1 = P[0, 0] / S, P[1, 0] / S
        acc = np.arctan2(-ax, az) * RAD2DEG
        ytilde = acc - self.angle
        self.angle += K0 * ytilde
        self.bias += K1 * ytilde

        if self.literal:
            P[0, 0] -= K0 * P[0, 0]
            P[0, 1] -= K0 * P[0, 1]
            P[1, 0] -= K1 * P[0, 0]
            P[1, 1] -= K1 * P[0, 1]
        else:
            P00, P01 = P[0, 0], P[0, 1]
            P[0, 0] -= K0 * P00
            P[0, 1] -= K0 * P01
            P[1, 0] -= K1 * P00
            P[1, 1] -= K1 * P01
        return self.angle, self.bias


def complementary_filter(prev: float, gy: float, ax: float, az: float, dt: float, alpha: float = 0.98) -> float:
    """``alpha*(prev + gy*dt) + (1-alpha)*atan2(-ax, az)`` in degrees."""
    return alpha * (prev + gy * dt) + (1 - alpha) * np.arctan2(-ax, az) * RAD2DEG
